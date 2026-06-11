"""API routes for batch evaluation of prompts against UC datasets."""

import logging
import asyncio
import mlflow
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field
from server.cache import TtlCache
from server.eval_jobs import create_job, get_job, run_job, update_job
from server.mlflow_client import get_prompt_template, list_prompts
from server.templates import render_template, parse_system_user
from server.mlflow_helpers import (
    configure_mlflow,
    configured_mlflow_experiment_name,
    get_experiment_id,
    experiment_url as make_experiment_url,
    get_mlflow_client,
)
from server.llm import call_model, EvalAbortError, TokenLimitError, RateLimitError
from server.warehouse import list_eval_tables, get_table_columns, read_table_rows, count_table_rows
from server.evaluation import mlflow_genai_evaluate, _extract_row_scores
from server.settings import get_effective_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eval", tags=["evaluate"])

# Cap workspace-wide experiment scans so the header dropdown stays responsive on shared workspaces.
_EXPERIMENT_BROWSE_MAX = 50
_EXPERIMENT_FILTER_CANDIDATE_MAX = 100
_EXPERIMENT_SEARCH_MIN_LEN = 2
_EXPERIMENT_PROMPTS_CACHE = TtlCache(ttl_sec=120)
_EXPERIMENT_BROWSE_CACHE = TtlCache(ttl_sec=300)


def _experiment_payload(exp) -> dict:
    return {
        "name": exp.name,
        "experiment_id": exp.experiment_id,
        "url": make_experiment_url(exp.experiment_id),
    }


def _active_experiments(experiments) -> list:
    return [e for e in experiments if e.lifecycle_stage == "active"]


def _get_active_experiment_by_name(client, name: str):
    if not name:
        return None
    try:
        exp = client.get_experiment_by_name(name)
    except Exception:
        return None
    if exp and exp.lifecycle_stage == "active":
        return exp
    return None


def _browse_experiments(client, *, max_results: int) -> list:
    experiments = client.search_experiments(max_results=max_results)
    return _active_experiments(experiments)


def _cached_browse_experiments(client, *, max_results: int) -> list:
    cache_key = f"browse:{max_results}"
    cached = _EXPERIMENT_BROWSE_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    result = _browse_experiments(client, max_results=max_results)
    _EXPERIMENT_BROWSE_CACHE.set(cache_key, result)
    return result


def _configured_experiment_payload(client) -> list[dict]:
    configured_name = configured_mlflow_experiment_name()
    exp = _get_active_experiment_by_name(client, configured_name)
    return [_experiment_payload(exp)] if exp else []


def _get_warehouse_id() -> str:
    """Resolve the effective SQL warehouse ID from persisted settings or env var."""
    wid = get_effective_config().get("sql_warehouse_id", "")
    if not wid:
        raise HTTPException(
            status_code=400,
            detail="SQL warehouse not configured. Open Settings (gear icon) to select a warehouse.",
        )
    return wid


# --- Discovery routes ---

@router.post("/experiments/warm")
async def warm_experiment_cache():
    """Populate the experiment browse cache in the background (page load / dropdown open)."""
    async def _warm() -> None:
        try:
            client = get_mlflow_client()
            await asyncio.to_thread(
                _cached_browse_experiments,
                client,
                max_results=_EXPERIMENT_BROWSE_MAX,
            )
        except Exception as e:
            logger.debug("experiment cache warm failed (non-fatal): %s", e)

    asyncio.create_task(_warm())
    return {"status": "warming"}


@router.get("/experiments")
async def api_list_experiments(
    catalog: str | None = None,
    schema: str | None = None,
    q: str | None = Query(None, description="Case-insensitive substring filter (min 2 chars)"),
    configured_only: bool = Query(
        False,
        description="Return only the experiment saved in Settings (instant)",
    ),
    browse: bool = Query(
        False,
        description="Return a capped workspace browse (header dropdown), not the Settings default only",
    ),
):
    """List active MLflow experiments for the dropdown.

    Uses the configured default experiment when set (fast path). Otherwise returns a
    capped recent slice of workspace experiments — never the full unbounded list.
    Results are cached server-side for several minutes after the first browse.

    When catalog and schema are provided, filters to experiments with runs tagged
    for that catalog.schema prompt prefix, searching only the default experiment or
    a capped candidate set instead of the full workspace.
    """
    try:
        import re

        client = get_mlflow_client()
        configured_name = configured_mlflow_experiment_name()

        def _browse_payload(max_results: int):
            return [
                _experiment_payload(e)
                for e in _cached_browse_experiments(client, max_results=max_results)
            ]

        if configured_only:
            configured = await asyncio.to_thread(_configured_experiment_payload, client)
            return {"experiments": configured}

        query = (q or "").strip()
        if not catalog or not schema:
            if (
                not browse
                and configured_name
                and len(query) < _EXPERIMENT_SEARCH_MIN_LEN
            ):
                configured = await asyncio.to_thread(_configured_experiment_payload, client)
                if configured:
                    return {"experiments": configured}
            browsed = await asyncio.to_thread(_browse_payload, _EXPERIMENT_BROWSE_MAX)
            if len(query) >= _EXPERIMENT_SEARCH_MIN_LEN:
                needle = query.lower()
                browsed = [e for e in browsed if needle in e["name"].lower()]
            return {"experiments": browsed}

        if not re.match(r"^[\w\-]+$", catalog) or not re.match(r"^[\w\-]+$", schema):
            if (
                not browse
                and configured_name
                and len(query) < _EXPERIMENT_SEARCH_MIN_LEN
            ):
                configured = await asyncio.to_thread(_configured_experiment_payload, client)
                if configured:
                    return {"experiments": configured}
            browsed = await asyncio.to_thread(_browse_payload, _EXPERIMENT_BROWSE_MAX)
            if len(query) >= _EXPERIMENT_SEARCH_MIN_LEN:
                needle = query.lower()
                browsed = [e for e in browsed if needle in e["name"].lower()]
            return {"experiments": browsed}

        prefix = f"{catalog}.{schema}."
        filter_string = f"tags.prompt_name LIKE '{prefix}%'"

        def _filter_candidates():
            configured_exp = _get_active_experiment_by_name(client, configured_name)
            if configured_exp:
                candidate_ids = [configured_exp.experiment_id]
                candidate_exps = [configured_exp]
            else:
                candidate_exps = _cached_browse_experiments(
                    client, max_results=_EXPERIMENT_FILTER_CANDIDATE_MAX
                )
                candidate_ids = [e.experiment_id for e in candidate_exps]

            matched_ids: set[str] = set()
            chunk_size = 100
            for i in range(0, len(candidate_ids), chunk_size):
                chunk = candidate_ids[i : i + chunk_size]
                if not chunk:
                    continue
                runs = client.search_runs(chunk, filter_string, max_results=500)
                matched_ids.update(run.info.experiment_id for run in runs)

            filtered = [e for e in candidate_exps if e.experiment_id in matched_ids]
            if filtered:
                payloads = [_experiment_payload(e) for e in filtered]
            elif configured_exp:
                payloads = [_experiment_payload(configured_exp)]
            else:
                payloads = [_experiment_payload(e) for e in candidate_exps[:_EXPERIMENT_BROWSE_MAX]]

            if len(query) >= _EXPERIMENT_SEARCH_MIN_LEN:
                needle = query.lower()
                payloads = [e for e in payloads if needle in e["name"].lower()]
            return payloads

        return {"experiments": await asyncio.to_thread(_filter_candidates)}
    except HTTPException:
        raise
    except Exception as e:
        from server.routes.setup import _raise_from_setup_error

        _raise_from_setup_error(e)


@router.get("/experiments/prompts")
async def api_get_experiment_prompts(
    experiment_name: str = Query(..., min_length=1),
    catalog: str = Query(..., min_length=1),
    schema: str = Query(..., min_length=1),
):
    """Return prompt names associated with the given experiment.

    Uses the _mlflow_experiment_ids tag on each prompt (set automatically
    by MLflow when a prompt is evaluated, or by our app on creation).
    Falls back to searching runs if no tag-based matches are found.
    """
    try:
        configure_mlflow()
        client = get_mlflow_client()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            return {"prompt_names": []}
        exp_id = experiment.experiment_id
        cache_key = f"{catalog}.{schema}:{exp_id}"

        def _tagged_from_cache():
            cached = _EXPERIMENT_PROMPTS_CACHE.get(cache_key)
            if cached is not None:
                return cached
            all_prompts = list_prompts(catalog, schema)
            tagged = sorted(
                p["name"]
                for p in all_prompts
                if f",{exp_id}," in p.get("tags", {}).get("_mlflow_experiment_ids", "")
            )
            if tagged:
                _EXPERIMENT_PROMPTS_CACHE.set(cache_key, tagged)
            return tagged

        tagged = await asyncio.to_thread(_tagged_from_cache)
        if tagged:
            return {"prompt_names": tagged}

        # Fallback: search runs for prompt_name tags (covers older prompts
        # that were evaluated before the tag-based approach was added)
        runs = await asyncio.to_thread(
            client.search_runs,
            experiment_ids=[exp_id],
            filter_string="tags.prompt_name != ''",
            max_results=1000,
        )
        prompt_names = sorted({
            run.data.tags["prompt_name"]
            for run in runs
            if "prompt_name" in run.data.tags
        })
        return {"prompt_names": prompt_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tables")
async def api_list_eval_tables(
    catalog: str = Query(..., min_length=1),
    schema: str = Query(..., min_length=1),
):
    """List tables available as eval datasets."""
    try:
        warehouse_id = _get_warehouse_id()
        tables = await asyncio.to_thread(list_eval_tables, catalog, schema, warehouse_id)
        return {"tables": tables}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/judges")
async def api_list_judges(experiment_name: str | None = None):
    """List LLM judges (registered scorers) for the given experiment."""
    try:
        configure_mlflow()
        exp_name = (experiment_name or "").strip() or configured_mlflow_experiment_name()
        mlflow.set_experiment(exp_name)
        exp = mlflow.get_experiment_by_name(exp_name)
        exp_id = exp.experiment_id if exp else None

        from mlflow.genai.scorers import list_scorers
        scorers = list_scorers(experiment_id=exp_id)

        return {"judges": [
            {"name": s.name}
            for s in scorers
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/judges/detail")
async def api_get_judge_detail(name: str):
    """Get the details (instructions/guidelines) of a registered judge."""
    try:
        configure_mlflow()
        from mlflow.genai.scorers import get_scorer
        scorer = get_scorer(name=name)
        data = scorer.model_dump() if hasattr(scorer, 'model_dump') else {}
        # Check both model_dump and attribute — Databricks-created scorers may only serialize
        # into model_dump; also check truthiness so None/[] doesn't trigger guidelines branch.
        raw_guidelines = data.get('guidelines') or getattr(scorer, 'guidelines', None)
        raw_instructions = data.get('instructions') or getattr(scorer, 'instructions', None)

        if raw_guidelines:
            judge_type = "guidelines"
            guidelines = raw_guidelines
            instructions = None
        else:
            judge_type = "custom"
            instructions = raw_instructions
            guidelines = None
        return {
            "name": name,
            "type": judge_type,
            "instructions": instructions,
            "guidelines": guidelines,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CreateJudgeRequest(BaseModel):
    name: str = Field(min_length=1)
    type: str = "custom"  # "custom" | "guidelines"
    instructions: str | None = None       # for type="custom"
    guidelines: list[str] | None = None   # for type="guidelines"
    experiment_name: str | None = None
    is_update: bool = False


@router.post("/judges")
async def api_create_judge(request: CreateJudgeRequest):
    """Create a custom LLM judge and register it on the experiment."""
    try:
        # Validate based on type
        if request.type == "custom":
            if not request.instructions:
                raise HTTPException(
                    status_code=400,
                    detail="instructions must be provided for type='custom'",
                )
        elif request.type == "guidelines":
            if not request.guidelines:
                raise HTTPException(
                    status_code=400,
                    detail="guidelines must be provided and non-empty for type='guidelines'",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported judge type: {request.type}. Must be 'custom' or 'guidelines'.",
            )

        configure_mlflow()
        exp_name = (request.experiment_name or "").strip() or configured_mlflow_experiment_name()
        mlflow.set_experiment(exp_name)
        exp = mlflow.get_experiment_by_name(exp_name)
        exp_id = exp.experiment_id if exp else None

        if request.type == "custom":
            from mlflow.genai.judges import make_judge

            assert request.instructions is not None
            judge = make_judge(
                name=request.name,
                instructions=request.instructions,
            )
        else:  # guidelines
            from mlflow.genai.scorers import Guidelines

            assert request.guidelines is not None
            judge = Guidelines(name=request.name, guidelines=request.guidelines)

        if request.is_update:
            from mlflow.genai.scorers import delete_scorer
            delete_scorer(name=request.name)

        judge.register(experiment_id=exp_id)

        return {"name": request.name, "status": "updated" if request.is_update else "registered"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/judges")
async def api_delete_judge(name: str, experiment_name: str | None = None):  # noqa: ARG001 - reserved for future use
    """Delete a registered judge/scorer."""
    try:
        configure_mlflow()
        from mlflow.genai.scorers import delete_scorer
        delete_scorer(name=name)
        return {"name": name, "status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/columns")
async def api_get_columns(catalog: str, schema: str, table: str):
    """Get column names for a table so the user can map them to prompt variables."""
    try:
        warehouse_id = _get_warehouse_id()
        cols = await asyncio.to_thread(get_table_columns, catalog, schema, table, warehouse_id)
        return {"columns": cols}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def api_eval_history(
    prompt_name: str,
    prompt_version: str | None = None,
    experiment_name: str | None = None,
    limit: int = 10,
):
    """Return past batch eval runs for a prompt (optionally filtered to one version), newest first."""
    try:
        configure_mlflow()
        client = get_mlflow_client()
        exp_name = (experiment_name or "").strip() or configured_mlflow_experiment_name()
        exp = await asyncio.to_thread(client.get_experiment_by_name, exp_name)
        if not exp:
            return {"runs": []}

        filter_parts = [
            f"tags.prompt_name = '{prompt_name}'",
            "tags.eval_type = 'batch'",
        ]
        if prompt_version is not None:
            filter_parts.insert(1, f"tags.prompt_version = '{prompt_version}'")
        filter_string = " AND ".join(filter_parts)

        effective_limit = limit if prompt_version is not None else max(limit, 50)
        runs = await asyncio.to_thread(
            client.search_runs,
            [exp.experiment_id],
            filter_string,
            max_results=effective_limit,
            order_by=["attribute.start_time DESC"],
        )

        exp_url_base = make_experiment_url(exp.experiment_id)
        result = []
        for run in runs:
            tags = run.data.tags
            metrics = run.data.metrics
            scorer = tags.get("scorer", "response_quality")

            avg_score = None
            for key in [f"{scorer}/mean", scorer, "response_quality/mean"]:
                if key in metrics:
                    avg_score = round(metrics[key], 2)
                    break

            result.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or "",
                "created_at": run.info.start_time,
                "avg_score": avg_score,
                "model": tags.get("model", ""),
                "dataset": tags.get("dataset", ""),
                "scorer": scorer,
                "prompt_version": tags.get("prompt_version", ""),
                "total_rows": int(tags["total_rows"]) if tags.get("total_rows") else None,
                "run_url": f"{exp_url_base}/runs/{run.info.run_id}",
            })

        return {"runs": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/run-traces")
async def api_run_traces(run_id: str):
    """Return per-row scores and rationales extracted from MLflow traces for a historical eval run."""
    try:
        configure_mlflow()
        client = get_mlflow_client()
        run = await asyncio.to_thread(client.get_run, run_id)
        scorer_name = run.data.tags.get("scorer", "response_quality")
        row_scores = await asyncio.to_thread(_extract_row_scores, run_id, scorer_name)

        rows = []
        for row_idx in sorted(row_scores.keys()):
            score, rationale, details = row_scores[row_idx]
            rows.append({
                "row_index": row_idx,
                "score": score,
                "rationale": rationale,
                "details": details,
            })
        return {"scorer": scorer_name, "rows": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/table-preview")
async def api_table_preview(
    catalog: str,
    schema: str,
    table: str,
    limit: int = 20,
    include_count: bool = Query(False, description="Run COUNT(*) — slower on large tables"),
):
    """Return column names and a sample of rows for the dataset preview UI."""
    try:
        warehouse_id = _get_warehouse_id()
        cols, rows = await asyncio.gather(
            asyncio.to_thread(get_table_columns, catalog, schema, table, warehouse_id),
            asyncio.to_thread(read_table_rows, catalog, schema, table, warehouse_id, limit=limit),
        )
        total_rows = None
        if include_count:
            total_rows = await asyncio.to_thread(
                count_table_rows, catalog, schema, table, warehouse_id
            )
        return {"columns": cols, "rows": rows, "total_rows": total_rows}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Evaluation run ---

class EvalRequest(BaseModel):
    prompt_name: str
    prompt_version: str
    model_name: str
    dataset_catalog: str
    dataset_schema: str
    dataset_table: str
    column_mapping: dict[str, str]  # {prompt_variable: table_column}
    max_rows: int = 20
    temperature: float = 1.0
    experiment_name: str | None = None
    scorer_name: str | None = None  # registered MLflow judge name; falls back to built-in quality scorer
    judge_model: str | None = None  # model for the default quality scorer; falls back to model_name if not set
    judge_temperature: float = 0.0  # temperature for the default quality scorer; lower = more consistent
    expectations_column: str | None = None  # dataset column with ground-truth expected responses (for Correctness scorer)


class ScoreDetail(BaseModel):
    name: str
    value: float | str | None = None
    rationale: str | None = None


class EvalRowResult(BaseModel):
    row_index: int
    variables: dict[str, str]
    rendered_prompt: str
    rendered_system_prompt: str | None = None
    response: str
    score: float | str | None = None
    score_rationale: str | None = None
    score_details: list[ScoreDetail] | None = None


class EvalResponse(BaseModel):
    prompt_name: str
    prompt_version: str
    model_name: str
    dataset: str
    total_rows: int
    results: list[EvalRowResult]
    avg_score: float | None = None
    run_id: str | None = None
    experiment_url: str | None = None


class EvalJobStartResponse(BaseModel):
    job_id: str
    status: str = "pending"


class EvalJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    total: int = 0
    message: str = ""
    result: EvalResponse | None = None
    error: str | None = None


async def _execute_evaluation(
    request: EvalRequest,
    *,
    job_id: str | None = None,
) -> EvalResponse:
    """Run a prompt version against every row in an eval dataset and score with mlflow.genai.evaluate()."""

    # Load prompt template
    try:
        prompt_data = await asyncio.to_thread(
            get_prompt_template, request.prompt_name, request.prompt_version
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading prompt: {e}")

    # Read dataset rows
    try:
        warehouse_id = _get_warehouse_id()
        rows = await asyncio.to_thread(
            read_table_rows,
            request.dataset_catalog,
            request.dataset_schema,
            request.dataset_table,
            warehouse_id,
            limit=request.max_rows,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {e}")

    if not rows:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    total_rows = len(rows)
    if job_id:
        await update_job(job_id, total=total_rows, progress=0, message="Running model on rows…")

    # Pre-flight: verify all mapped columns (including expectations) actually exist in the dataset
    available_cols = set(rows[0].keys())
    all_required_cols = set(request.column_mapping.values())
    if request.expectations_column:
        all_required_cols.add(request.expectations_column)
    missing_cols = {col for col in all_required_cols if col not in available_cols}
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column(s) not found in dataset: {', '.join(sorted(missing_cols))}. "
                f"Available columns: {', '.join(sorted(available_cols))}"
            ),
        )

    dataset_full_name = f"{request.dataset_catalog}.{request.dataset_schema}.{request.dataset_table}"

    system_prompt_raw = prompt_data.get("system_prompt")

    # Run model against each row concurrently (max 10 in-flight at once)
    sem = asyncio.Semaphore(10)

    async def run_row(i: int, row: dict) -> tuple:
        variables = {
            var: str(row.get(col, "")) for var, col in request.column_mapping.items()
        }
        rendered = render_template(prompt_data["template"], variables)
        rendered_system = render_template(system_prompt_raw, variables) if system_prompt_raw else None
        expectations_val = (
            str(row.get(request.expectations_column, "")) if request.expectations_column else None
        )
        async with sem:
            try:
                model_result = await call_model(
                    endpoint_name=request.model_name,
                    prompt=rendered,
                    temperature=request.temperature,
                    system_prompt=rendered_system,
                )
                response_text = model_result["content"]
            except EvalAbortError:
                raise  # propagate to abort the entire eval
            except Exception as e:
                response_text = f"[ERROR: {e}]"
        if job_id:
            pct = int(((i + 1) / total_rows) * 70)
            await update_job(job_id, progress=pct, message=f"Row {i + 1} of {total_rows}")
        return (i, variables, rendered, rendered_system, response_text, expectations_val)

    try:
        results_raw = await asyncio.gather(*[run_row(i, row) for i, row in enumerate(rows)])
    except TokenLimitError as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{e} Consider reducing the prompt length or the Max Rows setting "
                "to stay within the model's context window."
            ),
        )
    except RateLimitError as e:
        raise HTTPException(status_code=422, detail=str(e))
    sorted_results = sorted(results_raw)
    row_data: list[tuple[dict, str, str]] = [
        (variables, rendered, response_text)
        for _, variables, rendered, _sys, response_text, _ in sorted_results
    ]
    rendered_systems: list[str | None] = [
        _sys for _, _, _, _sys, _, _ in sorted_results
    ]
    expectations_data: list[str | None] | None = (
        [expectations_val for _, _, _, _, _, expectations_val in sorted_results]
        if request.expectations_column else None
    )

    # Skip mlflow evaluation if all rows errored (nothing useful to score)
    all_errored = all(resp.startswith("[ERROR:") for _, _, resp in row_data)

    # Run mlflow.genai.evaluate() in a thread (all MLflow calls are synchronous)
    run_id = None
    exp_url = None
    row_scores: dict[int, tuple[float | str | None, str | None]] = {}
    if all_errored:
        logger.warning("All %d rows errored — skipping mlflow_genai_evaluate", len(row_data))
    else:
        if job_id:
            await update_job(job_id, progress=75, message="Scoring with MLflow…")
        try:
            run_name = f"eval-{request.prompt_name.split('.')[-1]}-v{request.prompt_version}"
            run_id, row_scores = await asyncio.to_thread(
                mlflow_genai_evaluate,
                row_data,
                request.model_name,
                run_name,
                request.prompt_name,
                request.prompt_version,
                dataset_full_name,
                request.experiment_name,
                request.scorer_name,
                request.judge_model,
                request.judge_temperature,
                expectations_data,
            )
            if run_id:
                exp_id = get_experiment_id(request.experiment_name)
                if exp_id:
                    exp_url = make_experiment_url(exp_id)
        except Exception as e:
            logger.warning("MLflow eval failed (non-fatal): %s", e)

    # Build final results, merging in per-row scores extracted from traces
    results: list[EvalRowResult] = []
    for i, (variables, rendered, response_text) in enumerate(row_data):
        score, rationale, details = row_scores.get(i, (None, None, None))
        results.append(EvalRowResult(
            row_index=i,
            variables=variables,
            rendered_prompt=rendered,
            rendered_system_prompt=rendered_systems[i] if i < len(rendered_systems) else None,
            response=response_text,
            score=score,
            score_rationale=rationale,
            score_details=[ScoreDetail(**d) for d in details] if details else None,
        ))

    numeric_scores: list[float] = []
    for r in results:
        if isinstance(r.score, (int, float)):
            numeric_scores.append(float(r.score))
        elif isinstance(r.score, str) and '/' in r.score:
            parts = r.score.split('/')
            if len(parts) == 2:
                try:
                    numeric_scores.append(float(parts[0]) / float(parts[1]))
                except (ValueError, ZeroDivisionError):
                    pass
    avg_score = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None

    # Log avg_score as a metric so eval history can find it for all scorer types
    if run_id and avg_score is not None:
        try:
            scorer_key = request.scorer_name or "response_quality"
            client = get_mlflow_client()
            client.log_metric(run_id, f"{scorer_key}/mean", avg_score)
        except Exception as e:
            logger.warning("Failed to log avg_score metric (non-fatal): %s", e)

    if job_id:
        await update_job(job_id, progress=100, message="Complete")

    return EvalResponse(
        prompt_name=request.prompt_name,
        prompt_version=request.prompt_version,
        model_name=request.model_name,
        dataset=dataset_full_name,
        total_rows=len(results),
        results=results,
        avg_score=avg_score,
        run_id=run_id,
        experiment_url=exp_url,
    )


@router.post("/run", response_model=EvalJobStartResponse)
async def api_run_evaluation(request: EvalRequest, background_tasks: BackgroundTasks):
    """Enqueue a batch evaluation job and return a job_id for polling."""
    job_id = await create_job()

    async def _runner(active_job_id: str):
        try:
            result = await _execute_evaluation(request, job_id=active_job_id)
            return result.model_dump()
        except HTTPException as e:
            raise RuntimeError(e.detail) from e

    background_tasks.add_task(run_job, job_id, _runner)
    return EvalJobStartResponse(job_id=job_id)


@router.get("/run/status", response_model=EvalJobStatusResponse)
async def api_eval_job_status(job_id: str = Query(..., min_length=1)):
    """Poll background eval job status. Result is present when status is completed."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    result = None
    if job.get("result"):
        result = EvalResponse(**job["result"])
    return EvalJobStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=job.get("progress", 0),
        total=job.get("total", 0),
        message=job.get("message", ""),
        result=result,
        error=job.get("error"),
    )
