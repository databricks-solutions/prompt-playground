"""API routes for batch evaluation of prompts against UC datasets."""

import logging
import asyncio
import mlflow
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from server.mlflow_client import get_prompt_template, list_prompts
from server.templates import render_template, parse_system_user
from server.mlflow_helpers import configure_mlflow, get_experiment_id, experiment_url as make_experiment_url, get_mlflow_client, EXPERIMENT_NAME
from server.llm import call_model, EvalAbortError, TokenLimitError, RateLimitError
from server.warehouse import list_eval_tables, get_table_columns, read_table_rows, count_table_rows
from server.evaluation import mlflow_genai_evaluate, _extract_row_scores
from server.settings import get_effective_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eval", tags=["evaluate"])


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

@router.get("/experiments")
async def api_list_experiments(catalog: str | None = None, schema: str | None = None):
    """List active MLflow experiments for the dropdown.

    When catalog and schema are provided, filters to only experiments that contain
    runs logged from prompts in that catalog.schema (using the prompt_name run tag).
    Falls back to all experiments if no matches found.
    """
    try:
        client = get_mlflow_client()
        experiments = await asyncio.to_thread(client.search_experiments)
        active = [e for e in experiments if e.lifecycle_stage == "active"]

        if not catalog or not schema:
            return {"experiments": [
                {"name": e.name, "experiment_id": e.experiment_id, "url": make_experiment_url(e.experiment_id)} for e in active
            ]}

        # Sanitize: only allow word chars and hyphens in catalog/schema names
        import re
        if not re.match(r'^[\w\-]+$', catalog) or not re.match(r'^[\w\-]+$', schema):
            return {"experiments": [
                {"name": e.name, "experiment_id": e.experiment_id, "url": make_experiment_url(e.experiment_id)} for e in active
            ]}

        all_ids = [e.experiment_id for e in active]
        prefix = f"{catalog}.{schema}."
        # MLflow caps search_runs at 100 experiment IDs per call — batch as needed
        relevant_ids: set[str] = set()
        chunk_size = 100
        for i in range(0, len(all_ids), chunk_size):
            chunk = all_ids[i:i + chunk_size]
            runs = await asyncio.to_thread(
                client.search_runs,
                chunk,
                f"tags.prompt_name LIKE '{prefix}%'",
                max_results=500,
            )
            relevant_ids.update(run.info.experiment_id for run in runs)
        filtered = [e for e in active if e.experiment_id in relevant_ids]

        # Fall back to all experiments if nothing matched (e.g. fresh workspace)
        return {"experiments": [
            {"name": e.name, "experiment_id": e.experiment_id, "url": make_experiment_url(e.experiment_id)}
            for e in (filtered if filtered else active)
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/prompts")
async def api_get_experiment_prompts(
    experiment_name: str,
    catalog: str = "main",
    schema: str = "prompts",
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

        # Primary: filter prompts by _mlflow_experiment_ids tag
        all_prompts = await asyncio.to_thread(list_prompts, catalog, schema)
        tagged = sorted(
            p["name"]
            for p in all_prompts
            if f",{exp_id}," in p.get("tags", {}).get("_mlflow_experiment_ids", "")
        )
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
async def api_list_eval_tables(catalog: str = "main", schema: str = "eval_data"):
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
        exp_name = experiment_name or EXPERIMENT_NAME
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
        exp_name = request.experiment_name or EXPERIMENT_NAME
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
        exp_name = experiment_name or EXPERIMENT_NAME
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
async def api_table_preview(catalog: str, schema: str, table: str, limit: int = 20):
    """Return column names, a sample of rows, and total row count for the dataset preview UI."""
    try:
        warehouse_id = _get_warehouse_id()
        cols, rows, total_rows = await asyncio.gather(
            asyncio.to_thread(get_table_columns, catalog, schema, table, warehouse_id),
            asyncio.to_thread(read_table_rows, catalog, schema, table, warehouse_id, limit=limit),
            asyncio.to_thread(count_table_rows, catalog, schema, table, warehouse_id),
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


@router.post("/run", response_model=EvalResponse)
async def api_run_evaluation(request: EvalRequest):
    """Run a prompt version against every row in an eval dataset and score with mlflow.genai.evaluate()."""

    # Load prompt template
    try:
        prompt_data = get_prompt_template(request.prompt_name, request.prompt_version)
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
