"""API routes for running prompts against models."""

import asyncio
import re
import json
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


def _load_prompt_data_sync(request: RunRequest) -> dict:
    """Load prompt template from registry or use draft (sync — run via to_thread)."""
    if request.draft_template is not None:
        system_prompt, user_template = parse_system_user(request.draft_template)
        return {
            "template": user_template,
            "system_prompt": system_prompt,
            "variables": parse_template_variables(request.draft_template),
        }
    return get_prompt_template(request.prompt_name, request.prompt_version)


async def _load_prompt_data(request: RunRequest) -> dict:
    try:
        return await asyncio.to_thread(_load_prompt_data_sync, request)
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


async def _run_with_mlflow_logging(
    request: RunRequest,
    rendered: str,
    rendered_system: str | None,
) -> tuple[dict, str | None, str | None]:
    """Run model call inside an MLflow run; blocking MLflow work runs in a thread."""
    configure_mlflow()
    exp_id = await asyncio.to_thread(get_experiment_id, request.experiment_name)
    run_name = f"{request.prompt_name.split('.')[-1]}-v{request.prompt_version}"
    run_id = None
    exp_url = experiment_url(exp_id) if exp_id else None
    result = None

    try:
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
            await asyncio.to_thread(
                lambda: (
                    mlflow.set_tags({
                        "mlflow.runName": run_name,
                        "prompt_name": request.prompt_name,
                        "prompt_version": request.prompt_version,
                        "model": request.model_name,
                        "is_draft": str(request.draft_template is not None).lower(),
                    }),
                    mlflow.log_params({k: v[:250] for k, v in request.variables.items()}),
                    mlflow.log_param("model_name", request.model_name),
                    mlflow.log_param("prompt_version", request.prompt_version),
                )
            )

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

            trace_id = mlflow.get_last_active_trace_id()
            if trace_id:
                await asyncio.to_thread(mlflow.flush_trace_async_logging, False)

            if trace_id:
                await asyncio.to_thread(
                    _set_trace_previews,
                    trace_id,
                    rendered,
                    result["content"],
                )

            if request.draft_template is None and trace_id:
                await asyncio.to_thread(
                    _link_prompt_to_trace,
                    request,
                    trace_id,
                )

            await asyncio.to_thread(
                _log_run_artifacts,
                run.info.run_id,
                rendered,
                rendered_system,
                result,
                request,
            )
            run_id = run.info.run_id

    except HTTPException:
        raise
    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal): %s", e)
        if result is None:
            result = await call_model(
                endpoint_name=request.model_name,
                prompt=rendered,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system_prompt=rendered_system,
            )
        return result, run_id, exp_url

    return result, run_id, exp_url


def _set_trace_previews(trace_id: str, rendered: str, response_content: str) -> None:
    try:
        client = get_mlflow_client()
        client.set_trace_tag(trace_id, "mlflow.traceRequestPreview", rendered[:200])
        client.set_trace_tag(trace_id, "mlflow.traceResponsePreview", response_content[:200])
    except Exception:
        pass


def _link_prompt_to_trace(request: RunRequest, trace_id: str) -> None:
    try:
        client = get_mlflow_client()
        pv = client.get_prompt_version(
            name=request.prompt_name,
            version=request.prompt_version,
        )
        client.link_prompt_versions_to_trace(prompt_versions=[pv], trace_id=trace_id)
    except Exception as e:
        logger.warning("link_prompt_versions_to_trace failed (non-fatal): %s", e)
    try:
        prompt_link = json.dumps([{
            "name": request.prompt_name,
            "version": request.prompt_version,
        }])
        get_mlflow_client().set_trace_tag(trace_id, "mlflow.linkedPrompts", prompt_link)
    except Exception as e:
        logger.warning("set_trace_tag mlflow.linkedPrompts failed (non-fatal): %s", e)


@router.post("/run", response_model=RunResponse)
async def api_run_prompt(request: RunRequest):
    """Run a prompt with variable substitution against a selected model."""
    _validate_variables(request.variables)
    prompt_data = await _load_prompt_data(request)
    rendered = render_template(prompt_data["template"], request.variables)
    system_prompt_raw = prompt_data.get("system_prompt")
    rendered_system = render_template(system_prompt_raw, request.variables) if system_prompt_raw else None

    result, run_id, exp_url = await _run_with_mlflow_logging(request, rendered, rendered_system)

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
        prompt_data = await asyncio.to_thread(
            get_prompt_template, request.prompt_name, request.prompt_version
        )
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
