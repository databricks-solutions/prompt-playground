"""Tests for the /api/config endpoint."""

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def no_settings_file(monkeypatch):
    """Prevent a local pp_settings.json from contaminating config tests."""
    monkeypatch.setattr("server.settings.load_settings", lambda: {})


@pytest.fixture
def client():
    """FastAPI test client with a clean environment."""
    from fastapi.testclient import TestClient
    from server.routes.config import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_config_returns_defaults(client, monkeypatch):
    monkeypatch.delenv("PROMPT_CATALOG", raising=False)
    monkeypatch.delenv("PROMPT_SCHEMA", raising=False)
    monkeypatch.delenv("EVAL_CATALOG", raising=False)
    monkeypatch.delenv("EVAL_SCHEMA", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("SQL_WAREHOUSE_ID", raising=False)

    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["prompt_catalog"] == ""
    assert data["prompt_schema"] == ""
    assert data["eval_catalog"] == ""
    assert data["eval_schema"] == ""
    assert data["sql_warehouse_id"] == ""
    assert data["evaluate_tab_enabled"] is False
    assert data["is_configured"] is False


def test_config_reads_env_vars(client, monkeypatch):
    monkeypatch.setenv("PROMPT_CATALOG", "my_catalog")
    monkeypatch.setenv("PROMPT_SCHEMA", "my_prompts")
    monkeypatch.setenv("EVAL_SCHEMA", "my_eval")

    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["prompt_catalog"] == "my_catalog"
    assert data["prompt_schema"] == "my_prompts"
    assert data["eval_catalog"] == "my_catalog"   # same catalog as prompt
    assert data["eval_schema"] == "my_eval"


def test_eval_catalog_matches_prompt_catalog(client, monkeypatch):
    monkeypatch.setenv("PROMPT_CATALOG", "shared_catalog")
    monkeypatch.delenv("PROMPT_SCHEMA", raising=False)
    monkeypatch.delenv("EVAL_SCHEMA", raising=False)

    response = client.get("/api/config")
    data = response.json()
    assert data["eval_catalog"] == data["prompt_catalog"] == "shared_catalog"


def test_config_response_has_required_keys(client, monkeypatch):
    monkeypatch.delenv("PROMPT_CATALOG", raising=False)
    monkeypatch.delenv("PROMPT_SCHEMA", raising=False)
    monkeypatch.delenv("EVAL_SCHEMA", raising=False)

    response = client.get("/api/config")
    data = response.json()
    assert set(data.keys()) == {
        "prompt_catalog", "prompt_schema", "eval_catalog", "eval_schema",
        "mlflow_experiment_name", "sql_warehouse_id", "sql_warehouse_name",
        "evaluate_tab_enabled", "is_configured",
    }


def test_config_is_configured_prompts_only(client, monkeypatch):
    monkeypatch.setenv("PROMPT_CATALOG", "cat")
    monkeypatch.setenv("PROMPT_SCHEMA", "prompts")
    monkeypatch.delenv("EVALUATE_TAB_ENABLED", raising=False)

    response = client.get("/api/config")
    assert response.json()["is_configured"] is True


def test_config_not_configured_without_schema(client, monkeypatch):
    monkeypatch.setenv("PROMPT_CATALOG", "cat")
    monkeypatch.delenv("PROMPT_SCHEMA", raising=False)

    response = client.get("/api/config")
    assert response.json()["is_configured"] is False


def test_config_requires_eval_fields_when_evaluate_enabled(client, monkeypatch):
    monkeypatch.setenv("PROMPT_CATALOG", "cat")
    monkeypatch.setenv("PROMPT_SCHEMA", "prompts")
    monkeypatch.setenv("EVALUATE_TAB_ENABLED", "true")
    monkeypatch.delenv("EVAL_SCHEMA", raising=False)
    monkeypatch.delenv("SQL_WAREHOUSE_ID", raising=False)

    response = client.get("/api/config")
    assert response.json()["is_configured"] is False


def test_config_warehouse_name_empty_by_default(client, monkeypatch):
    monkeypatch.delenv("SQL_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("SQL_WAREHOUSE_NAME", raising=False)

    response = client.get("/api/config")
    assert response.status_code == 200
    assert response.json()["sql_warehouse_name"] == ""


def test_config_warehouse_name_from_env(client, monkeypatch):
    monkeypatch.setenv("SQL_WAREHOUSE_NAME", "My Warehouse")

    response = client.get("/api/config")
    assert response.status_code == 200
    assert response.json()["sql_warehouse_name"] == "My Warehouse"


# ---------------------------------------------------------------------------
# POST /api/config — update persisted settings
# ---------------------------------------------------------------------------

def test_post_config_calls_save_settings(client, monkeypatch):
    """POST /api/config should call save_settings with the provided fields."""
    monkeypatch.delenv("SQL_WAREHOUSE_ID", raising=False)

    with patch("server.routes.config.save_settings") as mock_save:
        response = client.post("/api/config", json={"prompt_catalog": "foo_catalog"})

    assert response.status_code == 200
    mock_save.assert_called_once()
    saved = mock_save.call_args[0][0]
    assert saved.get("prompt_catalog") == "foo_catalog"


def test_post_config_excludes_none_fields(client, monkeypatch):
    """Fields not provided in the request body should not be passed to save_settings."""
    monkeypatch.delenv("SQL_WAREHOUSE_ID", raising=False)

    with patch("server.routes.config.save_settings") as mock_save:
        client.post("/api/config", json={"prompt_catalog": "cat"})

    saved = mock_save.call_args[0][0]
    # Only the field we provided should appear (None fields excluded)
    assert "eval_catalog" not in saved


def test_post_config_returns_200(client, monkeypatch):
    """POST /api/config returns HTTP 200 with a config body."""
    monkeypatch.delenv("SQL_WAREHOUSE_ID", raising=False)

    with patch("server.routes.config.save_settings"):
        response = client.post("/api/config", json={"eval_schema": "my_eval"})

    assert response.status_code == 200
    assert "eval_schema" in response.json()


def test_post_config_accepts_warehouse_id(client, monkeypatch):
    """Posting a warehouse_id persists it via save_settings."""
    monkeypatch.delenv("SQL_WAREHOUSE_ID", raising=False)

    with patch("server.routes.config.save_settings") as mock_save, \
         patch("server.routes.config._resolve_and_cache_warehouse_name"):
        client.post("/api/config", json={"sql_warehouse_id": "abc-123"})

    saved = mock_save.call_args[0][0]
    assert saved.get("sql_warehouse_id") == "abc-123"
