"""API routes for running prompts against models."""

import re
import logging
import mlflow
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from server.mlflow_client import get_prompt_template
from server.templates import render_template, parse_template_variables, parse_system_user
from server.mlflow_helpers import configure_mlflow, get_experiment_id, experiment_url, get_mlflow_client
from server.llm import call_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["run"])

_VAR_MAX_LEN = 50_000
_TEMPLATE_PATTERN = re.compile(r"\{\{\s*\w+\s*\}\}")


def _validate_variables(variables: dict[str, str]) -> None:
    """Raise HTTPException if any variable value is too long or contains template syntax."""
    for key, value in variables.items():
        if len(value) > _VAR_MAX_LEN:
            raise HTTPException(
                status_code=400,
                detail=f"Variable '{key}' exceeds the {_VAR_MAX_LEN:,}-character limit.",
            )
        if _TEMPLATE_PATTERN.search(value):
            raise HTTPException(
                status_code=400,
                detail=f"Variable '{key}' contains template syntax ({{{{...}}}}). Variable values cannot contain {{{{variable}}}} patterns.",
            )


class RunRequest(BaseModel):
    prompt_name: str
    prompt_version: str
    variables: dict[str, str] = {}
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    experiment_name: str | None = None
    draft_template: str | None = None


class RunResponse(BaseModel):
    rendered_prompt: str
    system_prompt: str | None = None
    response: str
    model: str
    usage: dict = {}
    run_id: str | None = None
    experiment_url: str | None = None


def _load_prompt_data(request: RunRequest) -> dict:
    """Load prompt template from registry or use draft."""
    if request.draft_template is not None:
        system_prompt, user_template = parse_system_user(request.draft_template)
        return {
            "template": user_template,
            "system_prompt": system_prompt,
            "variables": parse_template_variables(request.draft_template),
        }
    try:
        return get_prompt_template(request.prompt_name, request.prompt_version)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading prompt: {e}")


def _log_run_artifacts(run_id: str, rendered: str, rendered_system: str | None, result: dict, request: RunRequest):
    """Log artifacts, metrics, and link prompt version to the MLflow run."""
    if rendered_system:
        mlflow.log_text(rendered_system, "system_prompt.txt")
    mlflow.log_text(rendered, "rendered_prompt.txt")
    mlflow.log_text(result["content"], "response.txt")
    usage = result.get("usage", {})
    if usage:
        mlflow.log_metrics({k: v for k, v in usage.items() if isinstance(v, (int, float))})

    if request.draft_template is None:
        try:
            client = get_mlflow_client()
            prompt_version_obj = client.get_prompt_version(
                name=request.prompt_name,
                version=request.prompt_version,
            )
            client.link_prompt_version_to_run(
                run_id=run_id,
                prompt=prompt_version_obj,
            )
        except Exception as e:
            logger.warning("link_prompt_version_to_run failed (non-fatal): %s", e)


@router.post("/run", response_model=RunResponse)
async def api_run_prompt(request: RunRequest):
    """Run a prompt with variable substitution against a selected model."""
    _validate_variables(request.variables)
    prompt_data = _load_prompt_data(request)
    rendered = render_template(prompt_data["template"], request.variables)
    system_prompt_raw = prompt_data.get("system_prompt")
    rendered_system = render_template(system_prompt_raw, request.variables) if system_prompt_raw else None

    # Call model inside an MLflow trace so it appears in the Traces tab
    run_id = None
    exp_url = None
    result = None

    try:
        configure_mlflow()
        exp_id = get_experiment_id(request.experiment_name)
        run_name = f"{request.prompt_name.split('.')[-1]}-v{request.prompt_version}"

        with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
            mlflow.set_tags({
                "mlflow.runName": run_name,
                "prompt_name": request.prompt_name,
                "prompt_version": request.prompt_version,
                "model": request.model_name,
                "is_draft": str(request.draft_template is not None).lower(),
            })
            mlflow.log_params({k: v[:250] for k, v in request.variables.items()})
            mlflow.log_param("model_name", request.model_name)
            mlflow.log_param("prompt_version", request.prompt_version)

            # call_model() uses the OpenAI SDK — mlflow.openai.autolog()
            # automatically creates a traced span with token usage, latencies,
            # and structured inputs/outputs. No manual span plumbing needed.
            try:
                result = await call_model(
                    endpoint_name=request.model_name,
                    prompt=rendered,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    system_prompt=rendered_system,
                )
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Model call failed: {e}")

            # Link prompt version to the trace so it appears in the
            # "Linked prompts" tab in the Databricks Traces UI.
            if request.draft_template is None:
                try:
                    client = get_mlflow_client()
                    pv = client.get_prompt_version(
                        name=request.prompt_name,
                        version=request.prompt_version,
                    )
                    active_trace = mlflow.get_current_active_span()
                    if active_trace:
                        client.link_prompt_versions_to_trace(
                            prompt_versions=[pv],
                            trace_id=active_trace.request_id,
                        )
                except Exception as e:
                    logger.warning("link_prompt_versions_to_trace failed (non-fatal): %s", e)

            _log_run_artifacts(run.info.run_id, rendered, rendered_system, result, request)

            run_id = run.info.run_id

        if exp_id:
            exp_url = experiment_url(exp_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal): %s", e)
        if result is None:
            try:
                result = await call_model(
                    endpoint_name=request.model_name,
                    prompt=rendered,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    system_prompt=rendered_system,
                )
            except Exception as model_err:
                raise HTTPException(status_code=502, detail=f"Model call failed: {model_err}")

    return RunResponse(
        rendered_prompt=rendered,
        system_prompt=rendered_system,
        response=result["content"],
        model=result["model"],
        usage=result["usage"],
        run_id=run_id,
        experiment_url=exp_url,
    )


class PreviewRequest(BaseModel):
    prompt_name: str
    prompt_version: str
    variables: dict[str, str] = {}


@router.post("/preview")
async def api_preview_prompt(request: PreviewRequest):
    """Preview a rendered prompt without calling a model."""
    try:
        prompt_data = get_prompt_template(request.prompt_name, request.prompt_version)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading prompt: {e}")

    rendered = render_template(prompt_data["template"], request.variables)
    system_prompt_raw = prompt_data.get("system_prompt")
    rendered_system = render_template(system_prompt_raw, request.variables) if system_prompt_raw else None
    return {
        "rendered_prompt": rendered,
        "system_prompt": rendered_system,
        "template": prompt_data["template"],
        "variables": prompt_data["variables"],
    }
