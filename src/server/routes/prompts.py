"""API routes for MLflow Prompt Registry operations.

Prompt names in Unity Catalog are fully qualified (catalog.schema.name),
so we use query parameters instead of path parameters to avoid conflicts
with dots in names.
"""

import asyncio
import logging
import os
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import httpx
from server.mlflow_client import (
    list_prompts,
    get_prompt_versions,
    get_prompt_template,
    create_prompt,
    create_prompt_version,
    get_mlflow_client,
)
from server.mlflow_helpers import configure_mlflow

logger = logging.getLogger(__name__)


def _is_permission_error(detail: str) -> bool:
    low = detail.lower()
    return "permission_denied" in low or (
        "does not have" in low and "privilege" in low
    )


def _permission_error_detail(schema_ref: str, needs_manage: bool = False) -> str:
    if needs_manage:
        return (
            f"Permission denied: the user or service principal lacks MANAGE privilege on schema '{schema_ref}'. "
            f"MANAGE is required to create or update prompts and is NOT included in ALL PRIVILEGES. "
            f"Grant it with:\n\n"
            f"GRANT USE SCHEMA, CREATE FUNCTION, EXECUTE, MANAGE ON SCHEMA {schema_ref} "
            f"TO `<service-principal-client-id>`;"
        )
    return (
        f"Permission denied accessing schema '{schema_ref}'. "
        f"Ensure the user or service principal has USE CATALOG and USE SCHEMA privileges:\n\n"
        f"GRANT USE CATALOG ON CATALOG {schema_ref.split('.')[0]} TO `<service-principal-client-id>`;\n"
        f"GRANT USE SCHEMA ON SCHEMA {schema_ref} TO `<service-principal-client-id>`;"
    )

async def _register_prompt_in_experiment(prompt_name: str, experiment_name: str) -> None:
    """Tag the prompt with the experiment ID so it shows in the experiment-filtered view."""
    configure_mlflow()
    client = get_mlflow_client()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return
    exp_id = experiment.experiment_id

    # Read existing experiment IDs tag (comma-delimited, e.g. ",123,456,")
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not host or not token:
        # Fall back to mlflow config
        import mlflow
        host = mlflow.get_tracking_uri().replace("databricks", os.environ.get("DATABRICKS_HOST", ""))
        token = os.environ.get("DATABRICKS_TOKEN", "")
    if not host or not token:
        logger.warning("Cannot set experiment tag: no DATABRICKS_HOST/TOKEN")
        return

    headers = {"Authorization": f"Bearer {token}"}
    url = f"{host}/api/2.0/mlflow/unity-catalog/prompts/{prompt_name}/tags"
    async with httpx.AsyncClient() as http:
        resp = await http.post(url, headers=headers, json={
            "key": "_mlflow_experiment_ids",
            "value": f",{exp_id},",
        })
        resp.raise_for_status()


router = APIRouter(prefix="/api/prompts", tags=["prompts"])


@router.get("")
async def api_list_prompts(
    catalog: str = Query(..., min_length=1, description="Unity Catalog name"),
    schema: str = Query(..., min_length=1, description="Schema name"),
):
    """List all registered prompts in the given catalog.schema."""
    try:
        prompts = await asyncio.to_thread(list_prompts, catalog=catalog, schema=schema)
        return {"prompts": prompts, "catalog": catalog, "schema": schema}
    except Exception as e:
        detail = str(e)
        if _is_permission_error(detail):
            raise HTTPException(
                status_code=403,
                detail=_permission_error_detail(f"{catalog}.{schema}"),
            )
        raise HTTPException(status_code=500, detail=detail)


@router.get("/versions")
async def api_get_prompt_versions(
    name: str = Query(description="Fully qualified prompt name (catalog.schema.name)"),
):
    """Get all versions and aliases for a specific prompt."""
    try:
        versions = await asyncio.to_thread(get_prompt_versions, name)
        return {"name": name, "versions": versions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/template")
async def api_get_prompt_template(
    name: str = Query(description="Fully qualified prompt name"),
    version: str = Query(description="Version number or alias"),
):
    """Load a prompt template by name and version or alias."""
    try:
        template = await asyncio.to_thread(get_prompt_template, name, version)
        return template
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Prompt creation/versioning ---

class CreatePromptRequest(BaseModel):
    name: str
    template: str
    description: str = ""
    experiment_name: str = ""


class CreateVersionRequest(BaseModel):
    name: str
    template: str
    description: str = ""


@router.post("")
async def api_create_prompt(request: CreatePromptRequest):
    """Create a brand new prompt with its initial version."""
    if not request.name or not request.name.strip():
        raise HTTPException(status_code=400, detail="Prompt name is required")
    if not request.template or not request.template.strip():
        raise HTTPException(status_code=400, detail="Template cannot be empty")
    try:
        result = create_prompt(
            name=request.name.strip(),
            template=request.template,
            description=request.description,
        )
        # Register prompt in the experiment so it shows in the filtered list
        if request.experiment_name:
            try:
                await _register_prompt_in_experiment(
                    request.name.strip(), request.experiment_name
                )
            except Exception as e:
                logger.warning("Failed to register prompt in experiment (non-fatal): %s", e)
        return result
    except Exception as e:
        detail = str(e)
        if "ALREADY_EXISTS" in detail or "already exists" in detail.lower():
            raise HTTPException(
                status_code=409, detail=f"Prompt '{request.name}' already exists"
            )
        if _is_permission_error(detail):
            schema_ref = ".".join(request.name.strip().split(".")[:2])
            raise HTTPException(
                status_code=403,
                detail=_permission_error_detail(schema_ref, needs_manage=True),
            )
        raise HTTPException(status_code=500, detail=detail)


@router.post("/versions")
async def api_create_prompt_version(request: CreateVersionRequest):
    """Create a new version of an existing prompt."""
    if not request.name or not request.name.strip():
        raise HTTPException(status_code=400, detail="Prompt name is required")
    if not request.template or not request.template.strip():
        raise HTTPException(status_code=400, detail="Template cannot be empty")
    try:
        result = create_prompt_version(
            name=request.name.strip(),
            template=request.template,
            description=request.description,
        )
        return result
    except Exception as e:
        detail = str(e)
        if "NOT_FOUND" in detail or "not found" in detail.lower():
            raise HTTPException(
                status_code=404, detail=f"Prompt '{request.name}' not found"
            )
        if _is_permission_error(detail):
            schema_ref = ".".join(request.name.strip().split(".")[:2])
            raise HTTPException(
                status_code=403,
                detail=_permission_error_detail(schema_ref, needs_manage=True),
            )
        raise HTTPException(status_code=500, detail=detail)
