"""Tests for eval run abort behavior on model errors.

Covers:
- TokenLimitError from call_model → 422 with token-limit message
- RateLimitError from call_model → 422 with rate-limit message
- Token limit message includes eval-specific hint about Max Rows
- EvalAbortError aborts immediately (mlflow_genai_evaluate not called)
- All-error responses (non-abort errors) → mlflow_genai_evaluate skipped, run_id None
- Partial errors (some rows ok) → mlflow_genai_evaluate still called
- Successful eval → mlflow_genai_evaluate called and run_id returned
"""

import pytest
from contextlib import ExitStack
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes.evaluate import router
from server.llm import TokenLimitError, RateLimitError


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


_BASE_PAYLOAD = {
    "prompt_name": "main.prompts.test",
    "prompt_version": "1",
    "model_name": "databricks-test-model",
    "dataset_catalog": "main",
    "dataset_schema": "eval_data",
    "dataset_table": "test_table",
    "column_mapping": {"topic": "topic_col"},
    "max_rows": 3,
}

_PROMPT_DATA = {"template": "{{topic}}", "variables": ["topic"]}
_ROWS = [{"topic_col": "A"}, {"topic_col": "B"}, {"topic_col": "C"}]
_MODEL_OK = {"content": "good response", "model": "m", "usage": {}}


def _base_patches(model_side_effect=None, rows=None):
    """Return the standard list of patches for eval run tests."""
    model_mock = AsyncMock(
        side_effect=model_side_effect,
        return_value=None if model_side_effect else _MODEL_OK,
    )
    if model_side_effect is None:
        model_mock = AsyncMock(return_value=_MODEL_OK)

    return [
        patch("server.routes.evaluate._get_warehouse_id", return_value="wh"),
        patch("server.routes.evaluate.get_prompt_template", return_value=_PROMPT_DATA),
        patch("server.routes.evaluate.read_table_rows", return_value=rows or _ROWS),
        patch("server.routes.evaluate.call_model", new=model_mock),
        patch("server.routes.evaluate.configure_mlflow"),
        patch("server.routes.evaluate.get_experiment_id", return_value=None),
        patch("server.routes.evaluate.make_experiment_url", return_value=None),
    ]


# ---------------------------------------------------------------------------
# TokenLimitError fast-fail
# ---------------------------------------------------------------------------

class TestTokenLimitFastFail:

    def test_token_limit_returns_422(self, client):
        """TokenLimitError from call_model → 422, not 200 or 500."""
        patches = _base_patches(model_side_effect=TokenLimitError("context exceeded"))
        with ExitStack() as stack:
            mocks = [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        assert resp.status_code == 422

    def test_token_limit_message_in_response(self, client):
        patches = _base_patches(model_side_effect=TokenLimitError("context window exceeded"))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        detail = resp.json()["detail"]
        assert "context window" in detail.lower()

    def test_token_limit_message_includes_eval_hint(self, client):
        """The 422 message for eval runs should mention Max Rows."""
        patches = _base_patches(model_side_effect=TokenLimitError("context exceeded"))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        detail = resp.json()["detail"]
        assert "max rows" in detail.lower() or "reducing" in detail.lower()

    def test_token_limit_does_not_call_mlflow_evaluate(self, client):
        """mlflow_genai_evaluate must NOT be called when all rows abort."""
        mock_evaluate = MagicMock()
        patches = _base_patches(model_side_effect=TokenLimitError("context exceeded"))
        patches.append(patch("server.routes.evaluate.mlflow_genai_evaluate", mock_evaluate))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            client.post("/api/eval/run", json=_BASE_PAYLOAD)
        mock_evaluate.assert_not_called()


# ---------------------------------------------------------------------------
# RateLimitError fast-fail
# ---------------------------------------------------------------------------

class TestRateLimitFastFail:

    def test_rate_limit_returns_422(self, client):
        """RateLimitError from call_model → 422."""
        patches = _base_patches(model_side_effect=RateLimitError("rate limit exceeded"))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        assert resp.status_code == 422

    def test_rate_limit_message_in_response(self, client):
        patches = _base_patches(model_side_effect=RateLimitError("rate limit exceeded for this model"))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        detail = resp.json()["detail"]
        assert "rate limit" in detail.lower()

    def test_rate_limit_does_not_call_mlflow_evaluate(self, client):
        mock_evaluate = MagicMock()
        patches = _base_patches(model_side_effect=RateLimitError("rate exceeded"))
        patches.append(patch("server.routes.evaluate.mlflow_genai_evaluate", mock_evaluate))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            client.post("/api/eval/run", json=_BASE_PAYLOAD)
        mock_evaluate.assert_not_called()


# ---------------------------------------------------------------------------
# All-error responses (non-abort, generic errors swallowed per row)
# ---------------------------------------------------------------------------

class TestAllErrorResponses:

    def _all_error_patches(self):
        """call_model raises a generic (non-abort) Exception for every row."""
        return [
            patch("server.routes.evaluate._get_warehouse_id", return_value="wh"),
            patch("server.routes.evaluate.get_prompt_template", return_value=_PROMPT_DATA),
            patch("server.routes.evaluate.read_table_rows", return_value=_ROWS),
            patch("server.routes.evaluate.call_model",
                  new=AsyncMock(side_effect=RuntimeError("generic model error"))),
            patch("server.routes.evaluate.configure_mlflow"),
            patch("server.routes.evaluate.get_experiment_id", return_value=None),
            patch("server.routes.evaluate.make_experiment_url", return_value=None),
        ]

    def test_all_errors_returns_200(self, client):
        """Generic per-row errors are swallowed; eval still returns results to the user."""
        with ExitStack() as stack:
            [stack.enter_context(p) for p in self._all_error_patches()]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        assert resp.status_code == 200

    def test_all_errors_run_id_is_none(self, client):
        """When all rows error, no MLflow run should be logged → run_id is None."""
        with ExitStack() as stack:
            [stack.enter_context(p) for p in self._all_error_patches()]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        assert resp.json()["run_id"] is None

    def test_all_errors_skips_mlflow_evaluate(self, client):
        mock_evaluate = MagicMock()
        patches = self._all_error_patches()
        patches.append(patch("server.routes.evaluate.mlflow_genai_evaluate", mock_evaluate))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            client.post("/api/eval/run", json=_BASE_PAYLOAD)
        mock_evaluate.assert_not_called()

    def test_all_errors_responses_contain_error_text(self, client):
        """Each row's response should show the [ERROR: ...] text."""
        with ExitStack() as stack:
            [stack.enter_context(p) for p in self._all_error_patches()]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        for row in resp.json()["results"]:
            assert row["response"].startswith("[ERROR:")


# ---------------------------------------------------------------------------
# Partial errors — mlflow_genai_evaluate should still run
# ---------------------------------------------------------------------------

class TestPartialErrors:

    def test_partial_errors_calls_mlflow_evaluate(self, client):
        """If at least one row succeeds, mlflow_genai_evaluate should be called."""
        call_count = 0

        async def _flaky_model(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return _MODEL_OK

        mock_evaluate = MagicMock(return_value=("run-123", {}))
        patches = [
            patch("server.routes.evaluate._get_warehouse_id", return_value="wh"),
            patch("server.routes.evaluate.get_prompt_template", return_value=_PROMPT_DATA),
            patch("server.routes.evaluate.read_table_rows",
                  return_value=[{"topic_col": "A"}, {"topic_col": "B"}]),
            patch("server.routes.evaluate.call_model", new=_flaky_model),
            patch("server.routes.evaluate.mlflow_genai_evaluate", mock_evaluate),
            patch("server.routes.evaluate.configure_mlflow"),
            patch("server.routes.evaluate.get_experiment_id", return_value=None),
            patch("server.routes.evaluate.make_experiment_url", return_value=None),
        ]
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json={**_BASE_PAYLOAD, "max_rows": 2})
        assert resp.status_code == 200
        mock_evaluate.assert_called_once()


# ---------------------------------------------------------------------------
# Successful eval — run_id is returned
# ---------------------------------------------------------------------------

class TestSuccessfulEval:

    def test_successful_eval_returns_run_id(self, client):
        mock_evaluate = MagicMock(return_value=("run-abc", {0: (4.5, "good", None)}))
        patches = _base_patches()
        patches.append(patch("server.routes.evaluate.mlflow_genai_evaluate", mock_evaluate))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json={**_BASE_PAYLOAD, "max_rows": 1})
        assert resp.status_code == 200
        assert resp.json()["run_id"] == "run-abc"

    def test_successful_eval_returns_avg_score(self, client):
        mock_evaluate = MagicMock(return_value=("run-abc", {
            0: (4.0, "ok", None), 1: (5.0, "great", None), 2: (3.0, "fair", None),
        }))
        patches = _base_patches()
        patches.append(patch("server.routes.evaluate.mlflow_genai_evaluate", mock_evaluate))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        assert resp.status_code == 200
        assert resp.json()["avg_score"] == pytest.approx(4.0, abs=0.01)
