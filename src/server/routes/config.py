"""Config route — exposes and persists app configuration."""

import asyncio
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from server.settings import get_effective_config, save_settings
from server.config import get_workspace_client

router = APIRouter(prefix="/api/config")


async def _resolve_and_cache_warehouse_name(wh_id: str) -> None:
    """Resolve warehouse name from Databricks and persist it for future requests."""
    try:
        w = get_workspace_client()
        wh = await asyncio.to_thread(lambda: w.warehouses.get(wh_id))
        wh_name = wh.name or ""
        if wh_name:
            save_settings({"sql_warehouse_name": wh_name})
    except Exception:
        pass


@router.get("")
async def get_config(background_tasks: BackgroundTasks):
    """Return effective config: env var defaults merged with persisted settings."""
    cfg = get_effective_config()
    wh_id = cfg["sql_warehouse_id"]
    wh_name = cfg.get("sql_warehouse_name", "")

    # If we have a warehouse ID but no saved name, resolve it in the background
    # so this request returns immediately instead of waiting on Databricks.
    if wh_id and not wh_name:
        background_tasks.add_task(_resolve_and_cache_warehouse_name, wh_id)

    return {
        "prompt_catalog": cfg["prompt_catalog"],
        "prompt_schema": cfg["prompt_schema"],
        "eval_catalog": cfg.get("eval_catalog") or cfg["prompt_catalog"],
        "eval_schema": cfg["eval_schema"],
        "mlflow_experiment_name": cfg["mlflow_experiment_name"],
        "sql_warehouse_id": wh_id,
        "sql_warehouse_name": wh_name,
        "evaluate_tab_enabled": bool(cfg.get("evaluate_tab_enabled", False)),
    }


class ConfigUpdate(BaseModel):
    prompt_catalog: str | None = None
    prompt_schema: str | None = None
    eval_catalog: str | None = None
    eval_schema: str | None = None
    mlflow_experiment_name: str | None = None
    sql_warehouse_id: str | None = None
    sql_warehouse_name: str | None = None
    evaluate_tab_enabled: bool | None = None


@router.post("")
async def update_config(body: ConfigUpdate, background_tasks: BackgroundTasks):
    """Persist user-configured settings to disk."""
    save_settings(body.model_dump(exclude_none=True))
    return await get_config(background_tasks)
