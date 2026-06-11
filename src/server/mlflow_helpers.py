"""Shared MLflow configuration and experiment helpers.

Centralises the MLflow tracking/registry setup that was previously duplicated
across mlflow_client.py, routes/run.py, and routes/evaluate.py.
"""

import os
import logging
import mlflow
from mlflow import MlflowClient
from server.config import get_workspace_host, get_workspace_client, IS_DATABRICKS_APP

logger = logging.getLogger(__name__)


def _sync_databricks_env_from_sdk() -> None:
    """Align MLflow's env-based auth with the SDK workspace client (local dev)."""
    if IS_DATABRICKS_APP:
        return
    if os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"):
        return
    try:
        w = get_workspace_client()
        if w.config.host:
            os.environ.setdefault("DATABRICKS_HOST", w.config.host)
        if w.config.token:
            os.environ.setdefault("DATABRICKS_TOKEN", w.config.token)
    except Exception as e:
        logger.debug("Could not sync Databricks env from SDK: %s", e)


def configured_mlflow_experiment_name() -> str:
    """MLflow experiment name from settings / env only (no hardcoded path in code)."""
    from server.settings import get_effective_config

    return (get_effective_config().get("mlflow_experiment_name") or "").strip()


def configure_mlflow():
    """Set up MLflow tracking and registry URIs for Databricks."""
    if IS_DATABRICKS_APP:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
    else:
        _sync_databricks_env_from_sdk()
        profile = os.environ.get("DATABRICKS_PROFILE")
        mlflow.set_tracking_uri(f"databricks://{profile}" if profile else "databricks")
        mlflow.set_registry_uri("databricks-uc")


def get_mlflow_client() -> MlflowClient:
    """Return a configured MlflowClient instance."""
    configure_mlflow()
    return MlflowClient()


def get_experiment_id(experiment_name: str | None = None) -> str | None:
    """Look up experiment ID by name, else the name from app settings / MLFLOW_EXPERIMENT_NAME env."""
    try:
        configure_mlflow()
        name = (experiment_name or "").strip() or configured_mlflow_experiment_name()
        if not name:
            return None
        exp = mlflow.get_experiment_by_name(name)
        return exp.experiment_id if exp else None
    except Exception as e:
        logger.warning("Could not get experiment: %s", e)
        return None


def experiment_url(experiment_id: str) -> str:
    """Build a workspace URL for an MLflow experiment."""
    return f"{get_workspace_host().rstrip('/')}/ml/experiments/{experiment_id}?searchFilter="
