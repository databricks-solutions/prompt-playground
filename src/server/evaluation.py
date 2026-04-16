"""MLflow GenAI evaluation orchestration.

Runs mlflow.genai.evaluate() with a predict_fn so that model calls happen
inside evaluate's tracing context.  Combined with mlflow.openai.autolog(),
each eval trace automatically captures token usage, latencies, and model params.
"""

import asyncio
import re
import logging
import mlflow
import mlflow.genai
from mlflow.genai.scorers import Scorer
from server.scoring import QualityScorer
from server.llm import call_model, EvalAbortError
from server.mlflow_helpers import configure_mlflow, get_mlflow_client, EXPERIMENT_NAME

logger = logging.getLogger(__name__)

BUILTIN_SCORERS = {
    "safety": "Safety",
    "relevance_to_query": "RelevanceToQuery",
    "fluency": "Fluency",
    "completeness": "Completeness",
    "summarization": "Summarization",
    "correctness": "Correctness",
}

# Type alias: row_index -> (summary_score, rationale, per-guideline details or None)
RowScore = tuple[float | str | None, str | None, list[dict] | None]


def mlflow_genai_evaluate(
    eval_rows: list[dict],
    predict_config: dict,
    run_name: str,
    prompt_name: str,
    prompt_version: str,
    dataset: str,
    experiment_name: str | None = None,
    scorer_name: str | None = None,
    judge_model: str | None = None,
    judge_temperature: float = 0.0,
) -> tuple[str | None, dict[int, RowScore], list[str]]:
    """Run mlflow.genai.evaluate() with a predict_fn that calls the model.

    eval_rows: list of dicts with keys "request", "system_prompt" (optional),
               and optionally "expected_response".
    predict_config: dict with "model_name" and "temperature".

    Returns (run_id, row_scores, responses) where:
      - row_scores maps row_index -> (score, rationale, details)
      - responses is the list of model response strings in row order
    """
    configure_mlflow()
    exp_name = experiment_name or EXPERIMENT_NAME
    mlflow.set_experiment(exp_name)

    model_name = predict_config["model_name"]
    temperature = predict_config["temperature"]

    # Collect responses as predict_fn runs so the route can return them
    responses: list[str | None] = [None] * len(eval_rows)

    def predict_fn(request: str, _row_index: int = -1, system_prompt: str | None = None) -> str:
        """Sync predict function for evaluate() — calls model via OpenAI SDK.

        evaluate() runs this in a ThreadPoolExecutor (default 10 workers),
        and autologging captures token usage on each trace.
        Parameter names must match the keys in the eval_data "inputs" dicts.
        """
        row_idx = _row_index
        request_text = request

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                call_model(
                    endpoint_name=model_name,
                    prompt=request_text,
                    temperature=temperature,
                    system_prompt=system_prompt,
                )
            )
            response_text = result["content"]
        except EvalAbortError:
            raise
        except Exception as e:
            response_text = f"[ERROR: {e}]"
        finally:
            loop.close()

        if row_idx is not None:
            responses[row_idx] = response_text
        return response_text

    # Build eval dataset in MLflow 3 GenAI format
    eval_data = []
    for idx, row in enumerate(eval_rows):
        entry: dict = {
            "inputs": {
                "request": row["request"],
                "_row_index": idx,
            },
        }
        if row.get("system_prompt"):
            entry["inputs"]["system_prompt"] = row["system_prompt"]
        if row.get("expected_response"):
            entry["expectations"] = {"expected_response": row["expected_response"]}
        eval_data.append(entry)

    scorers = _resolve_scorers(scorer_name, model_name, judge_model, judge_temperature)

    # Wrap scorers to capture per-row scores in memory.  With UC-backed trace
    # experiments, result_df is often None and search_traces may fail, so this
    # in-memory capture is the most reliable way to get per-row scores.
    captured_scores: dict[int, RowScore] = {}
    scorers = _wrap_scorers_for_capture(scorers, captured_scores)

    try:
        eval_result = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        run_id = eval_result.run_id
    except Exception as e:
        logger.warning("mlflow.genai.evaluate failed: %s", e)
        # Collect any responses that were produced before failure
        return None, {}, [r or "[ERROR: evaluate failed]" for r in responses]

    _log_run_metadata(run_id, run_name, prompt_name, prompt_version, model_name, scorer_name, dataset, len(eval_rows))
    _link_prompt_version(run_id, prompt_name, prompt_version)
    _link_prompt_to_traces(run_id, prompt_name, prompt_version)
    _log_dataset_input(run_id, dataset)

    expected_name = scorer_name or "response_quality"

    # Primary: use scores captured in memory during evaluation
    row_scores = captured_scores
    if not row_scores:
        # Fallback 1: extract from result_df (works for non-UC experiments)
        row_scores = _extract_scores_from_result(eval_result, expected_name)
    has_details = any(rs[2] is not None for rs in row_scores.values())
    if not row_scores or not has_details:
        # Fallback 2: extract from traces
        trace_scores = _extract_row_scores(run_id, expected_name)
        if trace_scores:
            row_scores = trace_scores

    return run_id, row_scores, [r or "[ERROR: no response]" for r in responses]


class _CapturingScorer(Scorer):
    """Wraps an MLflow scorer to capture per-row scores in memory.

    With UC-backed trace experiments (MLflow 3.10+), result_df is often None
    and search_traces may fail. Capturing scores in memory during evaluation
    is the most reliable way to get per-row results.

    Uses object.__setattr__ to store _inner and _captured because Scorer is a
    Pydantic BaseModel that rejects undeclared fields.
    """

    name: str = "capturing_scorer"

    def __init__(self, inner, captured: dict[int, RowScore]):
        super().__init__(name=getattr(inner, 'name', 'scorer'))
        object.__setattr__(self, '_inner', inner)
        object.__setattr__(self, '_captured', captured)

    def run(self, *, inputs=None, outputs=None, expectations=None, trace=None, session=None):
        """Override run() to intercept scorer results.

        evaluate() calls scorer.run(), which inspects __call__'s signature
        and delegates. We override run() directly to capture results.
        """
        from mlflow.entities import Feedback
        result = self._inner.run(
            inputs=inputs, outputs=outputs, expectations=expectations,
            trace=trace, session=session,
        )

        # Extract row index from inputs
        row_idx = inputs.get('_row_index') if isinstance(inputs, dict) else None
        if row_idx is None:
            return result

        # Parse Feedback result into RowScore format
        if isinstance(result, Feedback):
            score_val = result.value
            if isinstance(score_val, str):
                score_val = _normalize_pass_fail(score_val)
            self._captured[int(row_idx)] = (score_val, result.rationale, None)
        elif isinstance(result, list):
            # Guidelines scorer returns a list of Feedback objects
            details = []
            for fb in result:
                if isinstance(fb, Feedback):
                    val = fb.value
                    if isinstance(val, str):
                        val = _normalize_pass_fail(val)
                    details.append({"name": fb.name, "value": val, "rationale": fb.rationale})
            if details:
                passes = sum(1 for d in details if _is_pass(d["value"]))
                self._captured[int(row_idx)] = (f"{passes}/{len(details)}", None, details)

        return result

    def __call__(self, **kwargs):
        # Needed to satisfy Scorer ABC; run() is what evaluate() actually calls
        return self._inner(**kwargs)


def _wrap_scorers_for_capture(scorers: list, captured: dict[int, RowScore]) -> list:
    """Wrap each scorer in a _CapturingScorer to capture per-row scores in memory."""
    return [_CapturingScorer(s, captured) for s in scorers]


def _resolve_scorers(scorer_name: str | None, model_name: str, judge_model: str | None = None, judge_temperature: float = 0.0) -> list:
    """Load a registered scorer by name, or fall back to the built-in QualityScorer.

    judge_model and judge_temperature only apply to the built-in QualityScorer.
    Built-in MLflow scorers and registered judges manage their own model configuration.
    """
    if scorer_name and scorer_name in BUILTIN_SCORERS:
        from mlflow.genai import scorers as _scorers_mod
        cls = getattr(_scorers_mod, BUILTIN_SCORERS[scorer_name])
        return [cls()]
    if scorer_name:
        from mlflow.genai.scorers import get_scorer
        try:
            return [get_scorer(name=scorer_name)]
        except Exception as e:
            logger.warning("Could not load scorer '%s': %s — falling back to QualityScorer", scorer_name, e)
    effective_judge_model = judge_model or model_name
    return [QualityScorer(judge_model=effective_judge_model, judge_temperature=judge_temperature)]


def _log_dataset_input(run_id: str, dataset: str) -> None:
    """Register the UC table as an MLflow dataset input so it appears in the Experiments Datasets tab."""
    try:
        ds = mlflow.data.load_delta(table_name=dataset, name=dataset)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_input(ds, context="eval")
    except Exception as e:
        logger.warning("MLflow dataset input logging failed (non-fatal): %s", e)


def _log_run_metadata(
    run_id: str, run_name: str, prompt_name: str, prompt_version: str,
    model_name: str, scorer_name: str | None, dataset: str, total_rows: int,
) -> None:
    """Add descriptive tags and params to the MLflow run."""
    client = get_mlflow_client()
    try:
        client.update_run(run_id=run_id, name=run_name)
        client.set_tag(run_id, "eval_type", "batch")
        client.set_tag(run_id, "prompt_name", prompt_name)
        client.set_tag(run_id, "prompt_version", prompt_version)
        client.set_tag(run_id, "model", model_name)
        client.set_tag(run_id, "scorer", scorer_name or "response_quality")
        client.set_tag(run_id, "dataset", dataset)
        client.log_param(run_id, "prompt_version", prompt_version)
        client.log_param(run_id, "model_name", model_name)
        client.log_param(run_id, "dataset", dataset)
        client.log_param(run_id, "total_rows", str(total_rows))
    except Exception as e:
        logger.warning("MLflow metadata logging failed (non-fatal): %s", e)


def _link_prompt_version(run_id: str, prompt_name: str, prompt_version: str) -> None:
    """Link the eval run to the prompt version in the Prompt Registry."""
    client = get_mlflow_client()
    try:
        pv = client.get_prompt_version(name=prompt_name, version=prompt_version)
        client.link_prompt_version_to_run(run_id=run_id, prompt=pv)
    except Exception as e:
        logger.warning("link_prompt_version_to_run failed (non-fatal): %s", e)


def _link_prompt_to_traces(run_id: str, prompt_name: str, prompt_version: str) -> None:
    """Link prompt version to each trace in the eval run for the Traces UI."""
    client = get_mlflow_client()
    try:
        pv = client.get_prompt_version(name=prompt_name, version=prompt_version)
        traces = mlflow.search_traces(run_id=run_id, return_type="list")
        for trace in traces:
            trace_id = trace.info.request_id if hasattr(trace, 'info') else None
            if trace_id:
                try:
                    client.link_prompt_versions_to_trace(
                        prompt_versions=[pv],
                        trace_id=trace_id,
                    )
                except Exception:
                    pass
    except Exception as e:
        logger.warning("link_prompt_to_traces failed (non-fatal): %s", e)


_PASS_STRINGS = frozenset({
    "true", "yes", "pass", "1",
    # MLflow builtin scorer return values
    "fluent", "safe", "relevant", "complete", "correct",
})
_FAIL_STRINGS = frozenset({
    "false", "no", "fail", "0",
    # MLflow builtin scorer return values
    "not fluent", "not safe", "unsafe", "not relevant", "irrelevant",
    "not complete", "incomplete", "not correct", "incorrect",
})


def _is_pass(val: object) -> bool:
    """Determine if an assessment value represents a pass/true."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val >= 1
    if isinstance(val, str):
        return val.lower() in _PASS_STRINGS
    return False


def _normalize_pass_fail(val: float | str | None) -> float | str | None:
    """Convert known pass/fail string values to 1.0/0.0 for binary scorers."""
    if isinstance(val, str) and val.lower() in _PASS_STRINGS | _FAIL_STRINGS:
        return 1.0 if val.lower() in _PASS_STRINGS else 0.0
    return val


def _extract_scores_from_result(eval_result: object, expected_name: str) -> dict[int, RowScore]:
    """Extract per-row scores from the EvaluationResult's result_df (primary method)."""
    row_scores: dict[int, RowScore] = {}
    try:
        df = getattr(eval_result, 'result_df', None)
        if df is None or not hasattr(df, 'iterrows'):
            # Some versions use .tables dict
            tables = getattr(eval_result, 'tables', None)
            if isinstance(tables, dict):
                for _key, tdf in tables.items():
                    if hasattr(tdf, 'columns') and any(expected_name in str(c) for c in tdf.columns):
                        df = tdf
                        break
            if df is None or not hasattr(df, 'iterrows'):
                return row_scores

        # Detect Guidelines sub-columns: any column starting with {expected_name}/
        # that is not itself a rationale/justification column.
        # MLflow may use numeric indices (/0, /1) or guideline text as the suffix.
        guideline_cols: list[tuple[str, str | None]] = []  # (score_col, rationale_col or None)
        for col in df.columns:
            col_str = str(col)
            col_lower = col_str.lower()
            if not col_str.startswith(f"{expected_name}/"):
                continue
            if any(col_lower.endswith(s) for s in ('/rationale', '/justification', '_rationale', '_justification')):
                continue
            # Strip trailing /value to get the base name.
            # If base == expected_name, this column is the aggregated overall score (e.g.
            # "Guidelines/value") — NOT a per-rule sub-column. Skip it.
            base = col_str.removesuffix('/value')
            if base == expected_name:
                continue
            rat_col = next(
                (str(c) for c in df.columns
                 if str(c) in (f"{base}/rationale", f"{base}/justification", f"{base}_rationale")),
                None,
            )
            guideline_cols.append((col_str, rat_col))

        logger.debug("result_df guideline columns for '%s': %s", expected_name, guideline_cols)

        if guideline_cols:
            # Guidelines judge: collect per-guideline results into score_details
            for idx, row in df.iterrows():
                row_idx = idx

                details = []
                for score_col, rat_col in guideline_cols:
                    raw_val = row.get(score_col) if hasattr(row, 'get') else None
                    raw_rat = row.get(rat_col) if rat_col and hasattr(row, 'get') else None
                    try:
                        sv: float | str | None = float(raw_val) if raw_val is not None else None
                    except (TypeError, ValueError):
                        sv = _safe_str(raw_val)
                    details.append({"name": score_col, "value": sv, "rationale": _safe_str(raw_rat)})

                passes = sum(1 for d in details if _is_pass(d["value"]))
                row_scores[int(row_idx)] = (f"{passes}/{len(details)}", None, details)
            return row_scores

        # Single-score judge: find score and rationale columns
        score_col = None
        rationale_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if col == expected_name or col == f"{expected_name}/value" or col == f"{expected_name}/score":
                score_col = col
            if (col == f"{expected_name}/rationale"
                    or col_lower.endswith("/rationale")
                    or col_lower.endswith("_rationale")
                    or col == f"{expected_name}/justification"
                    or col_lower.endswith("/justification")):
                rationale_col = col

        if score_col is None:
            for col in df.columns:
                col_lower = str(col).lower()
                if (expected_name in str(col)
                        and "rationale" not in col_lower
                        and "justification" not in col_lower):
                    score_col = col
                    break

        if score_col is None:
            logger.debug("No score column found for '%s' in columns: %s", expected_name, list(df.columns))
            return row_scores

        for idx, row in df.iterrows():
            row_idx = idx

            raw_value = row.get(score_col) if hasattr(row, 'get') else None
            raw_rationale = row.get(rationale_col) if rationale_col and hasattr(row, 'get') else None

            try:
                score: float | str | None = float(raw_value) if raw_value is not None else None
            except (TypeError, ValueError):
                score = _safe_str(raw_value)
            score = _normalize_pass_fail(score)

            row_scores[int(row_idx)] = (score, _safe_str(raw_rationale), None)

    except Exception as e:
        logger.warning("Score extraction from result_df failed (non-fatal): %s", e, exc_info=True)

    return row_scores


def _safe_str(val: object) -> str | None:
    """Convert a value to str, returning None for None/NaN/empty."""
    if val is None:
        return None
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return s if s and s.lower() not in ("none", "nan") else None


def _extract_row_scores(run_id: str, expected_name: str) -> dict[int, RowScore]:
    """Fallback: extract per-row scores from evaluation traces."""
    # First pass: collect all matching assessments per row
    raw: dict[int, list[dict]] = {}
    traces: list = []

    try:
        # Always request a list of Trace objects — the default returns a pandas DataFrame
        # when pandas is installed, which iterates over column names, not rows.
        traces = mlflow.search_traces(run_id=run_id, return_type="list")

        for trace_idx, trace in enumerate(traces):
            row_idx = trace_idx
            assessments: list = []

            if hasattr(trace, 'info'):
                info = trace.info
                assessments = getattr(info, 'assessments', None) or []
                if not assessments:
                    data = getattr(trace, 'data', None)
                    if data:
                        assessments = getattr(data, 'assessments', []) or []
            elif isinstance(trace, dict):
                assessments = trace.get("assessments") or []

            for assessment in assessments:
                if hasattr(assessment, 'name'):
                    aname = assessment.name
                    # assessment.rationale is a top-level str|None field on Assessment
                    rationale = (getattr(assessment, 'rationale', None)
                                 or getattr(assessment, 'justification', None)
                                 or getattr(assessment, 'reasoning', None))
                    # Feedback subclass exposes .value as a property returning feedback.value.
                    # Use that directly; fall back to feedback.value for the base Assessment case.
                    if hasattr(assessment, 'value'):
                        raw_value = assessment.value
                    else:
                        feedback_obj = getattr(assessment, 'feedback', None)
                        raw_value = getattr(feedback_obj, 'value', None) if feedback_obj else None
                elif isinstance(assessment, dict):
                    aname = assessment.get("name") or assessment.get("assessment_name")
                    feedback = assessment.get("feedback") or {}
                    raw_value = feedback.get("value") or assessment.get("value")
                    rationale = (feedback.get("rationale") or feedback.get("justification")
                                 or assessment.get("rationale") or assessment.get("justification"))
                else:
                    continue

                is_match = (aname == expected_name
                            or (aname and aname.startswith(f"{expected_name}/")))
                if not is_match:
                    continue

                try:
                    score_val: float | str | None = float(raw_value) if raw_value is not None else None
                except (TypeError, ValueError):
                    score_val = _safe_str(raw_value)
                score_val = _normalize_pass_fail(score_val)

                raw.setdefault(int(row_idx), []).append({
                    "name": aname,
                    "value": score_val,
                    "rationale": _safe_str(rationale),
                })

    except Exception as e:
        logger.warning("Trace score extraction failed (non-fatal): %s", e, exc_info=True)

    logger.debug("Trace extraction: %d traces → raw assessments for %d rows: %s",
                 len(traces) if hasattr(traces, '__len__') else '?',  # type: ignore[union-attr]
                 len(raw),
                 {k: [(d["name"], d["value"]) for d in v] for k, v in raw.items()})

    # Second pass: build final row_scores from collected assessments
    row_scores: dict[int, RowScore] = {}
    for row_idx, details in raw.items():
        if len(details) == 1 and details[0]["name"] == expected_name:
            # Single non-guidelines scorer (QualityScorer, custom judge)
            row_scores[row_idx] = (details[0]["value"], details[0]["rationale"], None)
        elif details:
            # Guidelines: multiple sub-assessments — compute pass summary
            passes = sum(1 for d in details if _is_pass(d["value"]))
            row_scores[row_idx] = (f"{passes}/{len(details)}", None, details)

    return row_scores
