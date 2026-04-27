"""Tests for eval route pre-flight validation, dataset table preview, and eval run response shape.

Covers:
- POST /api/eval/run returns 400 when a mapped column is absent from the dataset
- POST /api/eval/run error message names the missing column(s)
- POST /api/eval/run error message lists available columns
- Multiple missing columns are all reported
- All columns present → pre-flight passes (request continues)
- GET /api/eval/table-preview returns columns, rows, and total_rows
- GET /api/eval/table-preview uses limit=20 by default
- GET /api/eval/table-preview forwards a custom limit
- GET /api/eval/table-preview returns 500 on warehouse error
- POST /api/eval/run rendered_system_prompt is None when prompt has no system prompt
- POST /api/eval/run rendered_system_prompt is populated when prompt has a system prompt
- POST /api/eval/run system prompt variables are correctly substituted
"""

import pytest
from contextlib import contextmanager, ExitStack
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes.evaluate import router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


_BASE_EVAL_PAYLOAD = {
    "prompt_name": "main.prompts.test",
    "prompt_version": "1",
    "model_name": "databricks-test-model",
    "dataset_catalog": "main",
    "dataset_schema": "eval_data",
    "dataset_table": "test_table",
    "column_mapping": {"topic": "topic_col"},
    "max_rows": 1,
}

_PROMPT_DATA = {"template": "{{topic}}", "variables": ["topic"]}

_MODEL_RESULT = {"content": "response", "model": "m", "usage": {}}


@contextmanager
def _eval_patches(rows, extra_patches=()):
    """Context manager providing all patches needed for POST /api/eval/run without workspace calls."""
    patches = [
        patch("server.routes.evaluate._get_warehouse_id", return_value="test-warehouse-id"),
        patch("server.routes.evaluate.get_prompt_template", return_value=_PROMPT_DATA),
        patch("server.routes.evaluate.read_table_rows", return_value=rows),
        patch("server.routes.evaluate.call_model", new=AsyncMock(return_value=_MODEL_RESULT)),
        patch("server.routes.evaluate.mlflow_genai_evaluate", return_value=("run-id", {0: (None, None, None)})),
        patch("server.routes.evaluate.configure_mlflow"),
        patch("server.routes.evaluate.get_experiment_id", return_value=None),
        patch("server.routes.evaluate.make_experiment_url", return_value=None),
        *extra_patches,
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


# ---------------------------------------------------------------------------
# Column mapping pre-flight
# ---------------------------------------------------------------------------

class TestEvalColumnPreflight:

    def test_missing_column_returns_400(self, client):
        """Mapped column not in dataset rows → 400 before any model calls."""
        rows = [{"other_col": "val"}]  # 'topic_col' is absent
        payload = {**_BASE_EVAL_PAYLOAD, "column_mapping": {"topic": "topic_col"}}

        with _eval_patches(rows):
            resp = client.post("/api/eval/run", json=payload)

        assert resp.status_code == 400

    def test_error_message_names_missing_column(self, client):
        rows = [{"other_col": "val"}]
        payload = {**_BASE_EVAL_PAYLOAD, "column_mapping": {"topic": "topic_col"}}

        with _eval_patches(rows):
            resp = client.post("/api/eval/run", json=payload)

        detail = resp.json()["detail"]
        assert "topic_col" in detail

    def test_error_message_lists_available_columns(self, client):
        rows = [{"actual_col": "v1", "another_col": "v2"}]
        payload = {**_BASE_EVAL_PAYLOAD, "column_mapping": {"topic": "nonexistent_col"}}

        with _eval_patches(rows):
            resp = client.post("/api/eval/run", json=payload)

        detail = resp.json()["detail"]
        assert "actual_col" in detail or "another_col" in detail

    def test_multiple_missing_columns_all_reported(self, client):
        rows = [{"unrelated": "v"}]
        payload = {**_BASE_EVAL_PAYLOAD, "column_mapping": {"topic": "col_a", "name": "col_b"}}

        with _eval_patches(rows):
            resp = client.post("/api/eval/run", json=payload)

        assert resp.status_code == 400
        detail = resp.json()["detail"]
        # Both missing columns should be mentioned
        assert "col_a" in detail and "col_b" in detail

    def test_missing_column_blocks_call_model(self, client):
        """call_model must NOT be called when pre-flight fails."""
        rows = [{"wrong_col": "val"}]
        mock_call = AsyncMock(return_value=_MODEL_RESULT)
        payload = {**_BASE_EVAL_PAYLOAD, "column_mapping": {"topic": "missing_col"}}

        with (
            patch("server.routes.evaluate._get_warehouse_id", return_value="test-warehouse-id"),
            patch("server.routes.evaluate.get_prompt_template", return_value=_PROMPT_DATA),
            patch("server.routes.evaluate.read_table_rows", return_value=rows),
            patch("server.routes.evaluate.call_model", new=mock_call),
            patch("server.routes.evaluate.configure_mlflow"),
            patch("server.routes.evaluate.get_experiment_id", return_value=None),
        ):
            resp = client.post("/api/eval/run", json=payload)

        assert resp.status_code == 400
        mock_call.assert_not_called()

    def test_all_columns_present_passes_preflight(self, client):
        """When all mapped columns exist, pre-flight passes and eval proceeds."""
        rows = [{"topic_col": "Python"}]
        payload = {**_BASE_EVAL_PAYLOAD, "column_mapping": {"topic": "topic_col"}}

        with _eval_patches(rows):
            resp = client.post("/api/eval/run", json=payload)

        # Pre-flight passed — may succeed or fail later for other reasons, but not 400 column error
        assert resp.status_code != 400 or "not found in dataset" not in resp.json().get("detail", "")

    def test_column_name_with_different_case_is_flagged(self, client):
        """Column names are case-sensitive — 'Topic_Col' != 'topic_col'."""
        rows = [{"Topic_Col": "val"}]  # capital T
        payload = {**_BASE_EVAL_PAYLOAD, "column_mapping": {"topic": "topic_col"}}

        with _eval_patches(rows):
            resp = client.post("/api/eval/run", json=payload)

        assert resp.status_code == 400

    def test_empty_dataset_still_returns_400(self, client):
        """Empty dataset → 'Dataset is empty' 400 (different error, same status)."""
        payload = {**_BASE_EVAL_PAYLOAD}

        with (
            patch("server.routes.evaluate._get_warehouse_id", return_value="test-warehouse-id"),
            patch("server.routes.evaluate.get_prompt_template", return_value=_PROMPT_DATA),
            patch("server.routes.evaluate.read_table_rows", return_value=[]),
            patch("server.routes.evaluate.configure_mlflow"),
            patch("server.routes.evaluate.get_experiment_id", return_value=None),
        ):
            resp = client.post("/api/eval/run", json=payload)

        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Table preview endpoint
# ---------------------------------------------------------------------------

class TestTablePreviewEndpoint:

    def test_returns_columns_and_rows(self, client):
        cols = ["col_a", "col_b"]
        rows = [{"col_a": "v1", "col_b": "v2"}, {"col_a": "v3", "col_b": "v4"}]

        with (
            patch("server.routes.evaluate._get_warehouse_id", return_value="wh-id"),
            patch("server.routes.evaluate.get_table_columns", return_value=cols),
            patch("server.routes.evaluate.read_table_rows", return_value=rows),
            patch("server.routes.evaluate.count_table_rows", return_value=2),
        ):
            resp = client.get("/api/eval/table-preview?catalog=main&schema=eval&table=t")

        assert resp.status_code == 200
        data = resp.json()
        assert data["columns"] == cols
        assert data["rows"] == rows
        assert data["total_rows"] == 2

    def test_default_limit_is_20(self, client):
        mock_rows = MagicMock(return_value=[])

        with (
            patch("server.routes.evaluate._get_warehouse_id", return_value="wh-id"),
            patch("server.routes.evaluate.get_table_columns", return_value=[]),
            patch("server.routes.evaluate.read_table_rows", mock_rows),
            patch("server.routes.evaluate.count_table_rows", return_value=0),
        ):
            client.get("/api/eval/table-preview?catalog=main&schema=eval&table=t")

        mock_rows.assert_called_once_with("main", "eval", "t", "wh-id", limit=20)

    def test_custom_limit_passed_through(self, client):
        mock_rows = MagicMock(return_value=[])

        with (
            patch("server.routes.evaluate._get_warehouse_id", return_value="wh-id"),
            patch("server.routes.evaluate.get_table_columns", return_value=[]),
            patch("server.routes.evaluate.read_table_rows", mock_rows),
            patch("server.routes.evaluate.count_table_rows", return_value=0),
        ):
            client.get("/api/eval/table-preview?catalog=main&schema=eval&table=t&limit=5")

        mock_rows.assert_called_once_with("main", "eval", "t", "wh-id", limit=5)

    def test_warehouse_error_returns_500(self, client):
        with (
            patch("server.routes.evaluate._get_warehouse_id", return_value="wh-id"),
            patch(
                "server.routes.evaluate.get_table_columns",
                side_effect=RuntimeError("Warehouse timeout"),
            ),
        ):
            resp = client.get("/api/eval/table-preview?catalog=main&schema=eval&table=t")

        assert resp.status_code == 500

    def test_missing_required_params_returns_422(self, client):
        resp = client.get("/api/eval/table-preview?catalog=main&schema=eval")
        assert resp.status_code == 422

    def test_empty_table_returns_empty_rows(self, client):
        with (
            patch("server.routes.evaluate._get_warehouse_id", return_value="wh-id"),
            patch("server.routes.evaluate.get_table_columns", return_value=["col"]),
            patch("server.routes.evaluate.read_table_rows", return_value=[]),
            patch("server.routes.evaluate.count_table_rows", return_value=0),
        ):
            resp = client.get("/api/eval/table-preview?catalog=main&schema=eval&table=t")

        assert resp.status_code == 200
        assert resp.json()["rows"] == []
        assert resp.json()["columns"] == ["col"]
        assert resp.json()["total_rows"] == 0

    def test_column_order_preserved(self, client):
        """Column order from DESCRIBE TABLE must be maintained for the preview table."""
        cols = ["z_col", "a_col", "m_col"]  # non-alphabetical order

        with (
            patch("server.routes.evaluate._get_warehouse_id", return_value="wh-id"),
            patch("server.routes.evaluate.get_table_columns", return_value=cols),
            patch("server.routes.evaluate.read_table_rows", return_value=[]),
            patch("server.routes.evaluate.count_table_rows", return_value=0),
        ):
            resp = client.get("/api/eval/table-preview?catalog=main&schema=eval&table=t")

        assert resp.json()["columns"] == cols


# ---------------------------------------------------------------------------
# Eval run response shape — rendered_system_prompt
# ---------------------------------------------------------------------------

_PROMPT_WITH_SYSTEM = {
    "template": "{{topic}}",
    "variables": ["topic"],
    "system_prompt": "You are an expert on {{topic}}.",
}

_PROMPT_NO_SYSTEM = {
    "template": "Tell me about {{topic}}.",
    "variables": ["topic"],
}


class TestEvalRunResponse:

    def test_rendered_system_prompt_none_when_no_system_prompt(self, client):
        """Prompts without a system_prompt field produce rendered_system_prompt=None."""
        rows = [{"topic_col": "Python"}]
        payload = {**_BASE_EVAL_PAYLOAD, "column_mapping": {"topic": "topic_col"}}

        with _eval_patches(rows, extra_patches=[
            patch("server.routes.evaluate.get_prompt_template", return_value=_PROMPT_NO_SYSTEM),
        ]):
            resp = client.post("/api/eval/run", json=payload)

        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["rendered_system_prompt"] is None

    def test_rendered_system_prompt_populated_when_system_prompt_present(self, client):
        """Prompts with a system_prompt field produce a non-None rendered_system_prompt."""
        rows = [{"topic_col": "Python"}]
        payload = {**_BASE_EVAL_PAYLOAD, "column_mapping": {"topic": "topic_col"}}

        with _eval_patches(rows, extra_patches=[
            patch("server.routes.evaluate.get_prompt_template", return_value=_PROMPT_WITH_SYSTEM),
        ]):
            resp = client.post("/api/eval/run", json=payload)

        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["rendered_system_prompt"] is not None

    def test_rendered_system_prompt_variables_substituted(self, client):
        """Variables in the system prompt are substituted with the row's column values."""
        rows = [{"topic_col": "Python"}]
        payload = {**_BASE_EVAL_PAYLOAD, "column_mapping": {"topic": "topic_col"}}

        with _eval_patches(rows, extra_patches=[
            patch("server.routes.evaluate.get_prompt_template", return_value=_PROMPT_WITH_SYSTEM),
        ]):
            resp = client.post("/api/eval/run", json=payload)

        result = resp.json()["results"][0]
        assert result["rendered_system_prompt"] == "You are an expert on Python."

    def test_rendered_prompt_still_populated(self, client):
        """The main rendered_prompt field is unaffected by system prompt changes."""
        rows = [{"topic_col": "Spark"}]
        payload = {**_BASE_EVAL_PAYLOAD, "column_mapping": {"topic": "topic_col"}}

        with _eval_patches(rows, extra_patches=[
            patch("server.routes.evaluate.get_prompt_template", return_value=_PROMPT_WITH_SYSTEM),
        ]):
            resp = client.post("/api/eval/run", json=payload)

        result = resp.json()["results"][0]
        assert result["rendered_prompt"] == "Spark"

    def test_multiple_rows_each_get_correct_system_prompt(self, client):
        """Each row gets its own variables substituted into the system prompt."""
        rows = [{"topic_col": "Python"}, {"topic_col": "Scala"}]
        payload = {
            **_BASE_EVAL_PAYLOAD,
            "column_mapping": {"topic": "topic_col"},
            "max_rows": 2,
        }

        with _eval_patches(rows, extra_patches=[
            patch("server.routes.evaluate.get_prompt_template", return_value=_PROMPT_WITH_SYSTEM),
            patch("server.routes.evaluate.mlflow_genai_evaluate",
                  return_value=("run-id", {0: (None, None, None), 1: (None, None, None)})),
        ]):
            resp = client.post("/api/eval/run", json=payload)

        assert resp.status_code == 200
        results_by_index = {r["row_index"]: r for r in resp.json()["results"]}
        assert results_by_index[0]["rendered_system_prompt"] == "You are an expert on Python."
        assert results_by_index[1]["rendered_system_prompt"] == "You are an expert on Scala."

    def test_system_prompt_none_when_key_absent_vs_empty(self, client):
        """system_prompt key entirely absent → None (not empty string)."""
        rows = [{"topic_col": "test"}]
        payload = {**_BASE_EVAL_PAYLOAD}

        # Prompt data with no system_prompt key at all
        with _eval_patches(rows, extra_patches=[
            patch("server.routes.evaluate.get_prompt_template",
                  return_value={"template": "{{topic}}", "variables": ["topic"]}),
        ]):
            resp = client.post("/api/eval/run", json=payload)

        assert resp.json()["results"][0]["rendered_system_prompt"] is None
