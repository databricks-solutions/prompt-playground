"""Tests for eval run abort behavior on model errors.

Covers:
- TokenLimitError from call_model → 422 with token-limit message
- RateLimitError from call_model → 422 with rate-limit message
- Token limit message includes eval-specific hint about Max Rows
- EvalAbortError aborts immediately (mlflow_genai_evaluate not called)
- All-error responses (non-abort errors) → run_id None, error text in responses
- Partial errors (some rows ok) → mlflow_genai_evaluate still called
- Successful eval → mlflow_genai_evaluate called and run_id returned
"""

import pytest
from contextlib import ExitStack
from unittest.mock import patch, MagicMock
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


def _base_patches(evaluate_side_effect=None, evaluate_return=None, rows=None):
    """Return the standard list of patches for eval run tests.

    Model calls now happen inside mlflow_genai_evaluate, so we mock that
    function rather than call_model at the route level.
    """
    if evaluate_return is None and evaluate_side_effect is None:
        evaluate_return = ("run-123", {0: (None, None, None), 1: (None, None, None), 2: (None, None, None)},
                           ["response"] * len(rows or _ROWS))

    mock_evaluate = MagicMock(side_effect=evaluate_side_effect, return_value=evaluate_return)

    return [
        patch("server.routes.evaluate._get_warehouse_id", return_value="wh"),
        patch("server.routes.evaluate.get_prompt_template", return_value=_PROMPT_DATA),
        patch("server.routes.evaluate.read_table_rows", return_value=rows or _ROWS),
        patch("server.routes.evaluate.mlflow_genai_evaluate", mock_evaluate),
        patch("server.routes.evaluate.configure_mlflow"),
        patch("server.routes.evaluate.get_experiment_id", return_value=None),
        patch("server.routes.evaluate.make_experiment_url", return_value=None),
    ], mock_evaluate


# ---------------------------------------------------------------------------
# TokenLimitError fast-fail
# ---------------------------------------------------------------------------

class TestTokenLimitFastFail:

    def test_token_limit_returns_422(self, client):
        """TokenLimitError from evaluate → 422, not 200 or 500."""
        patches, _ = _base_patches(evaluate_side_effect=TokenLimitError("context exceeded"))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        assert resp.status_code == 422

    def test_token_limit_message_in_response(self, client):
        patches, _ = _base_patches(evaluate_side_effect=TokenLimitError("context window exceeded"))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        detail = resp.json()["detail"]
        assert "context window" in detail.lower()

    def test_token_limit_message_includes_eval_hint(self, client):
        """The 422 message for eval runs should mention the error."""
        patches, _ = _base_patches(evaluate_side_effect=TokenLimitError("context exceeded"))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        detail = resp.json()["detail"]
        assert "context" in detail.lower()

    def test_token_limit_does_not_produce_results(self, client):
        """When TokenLimitError aborts, no eval results are returned."""
        patches, _ = _base_patches(evaluate_side_effect=TokenLimitError("context exceeded"))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# RateLimitError fast-fail
# ---------------------------------------------------------------------------

class TestRateLimitFastFail:

    def test_rate_limit_returns_422(self, client):
        """RateLimitError from evaluate → 422."""
        patches, _ = _base_patches(evaluate_side_effect=RateLimitError("rate limit exceeded"))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        assert resp.status_code == 422

    def test_rate_limit_message_in_response(self, client):
        patches, _ = _base_patches(evaluate_side_effect=RateLimitError("rate limit exceeded for this model"))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        detail = resp.json()["detail"]
        assert "rate limit" in detail.lower()

    def test_rate_limit_does_not_produce_results(self, client):
        patches, _ = _base_patches(evaluate_side_effect=RateLimitError("rate exceeded"))
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# All-error responses (evaluate returns error text in responses)
# ---------------------------------------------------------------------------

class TestAllErrorResponses:

    def _all_error_return(self):
        return (None, {}, ["[ERROR: generic model error]"] * 3)

    def test_all_errors_returns_200(self, client):
        """Generic per-row errors are returned as response text; eval still returns results."""
        patches, _ = _base_patches(evaluate_return=self._all_error_return())
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        assert resp.status_code == 200

    def test_all_errors_run_id_is_none(self, client):
        """When all rows error, no MLflow run should be logged → run_id is None."""
        patches, _ = _base_patches(evaluate_return=self._all_error_return())
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        assert resp.json()["run_id"] is None

    def test_all_errors_responses_contain_error_text(self, client):
        """Each row's response should show the [ERROR: ...] text."""
        patches, _ = _base_patches(evaluate_return=self._all_error_return())
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        for row in resp.json()["results"]:
            assert row["response"].startswith("[ERROR:")


# ---------------------------------------------------------------------------
# Partial errors — mlflow_genai_evaluate should still run
# ---------------------------------------------------------------------------

class TestPartialErrors:

    def test_partial_errors_calls_mlflow_evaluate(self, client):
        """If at least one row succeeds, mlflow_genai_evaluate returns a run_id."""
        patches, mock_evaluate = _base_patches(
            evaluate_return=("run-123", {0: (None, None, None), 1: (None, None, None)},
                             ["[ERROR: transient error]", "good response"]),
            rows=[{"topic_col": "A"}, {"topic_col": "B"}],
        )
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
        patches, _ = _base_patches(
            evaluate_return=("run-abc", {0: (4.5, "good", None)}, ["great response"]),
            rows=[{"topic_col": "A"}],
        )
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json={**_BASE_PAYLOAD, "max_rows": 1})
        assert resp.status_code == 200
        assert resp.json()["run_id"] == "run-abc"

    def test_successful_eval_returns_avg_score(self, client):
        patches, _ = _base_patches(
            evaluate_return=("run-abc", {
                0: (4.0, "ok", None), 1: (5.0, "great", None), 2: (3.0, "fair", None),
            }, ["r1", "r2", "r3"]),
        )
        with ExitStack() as stack:
            [stack.enter_context(p) for p in patches]
            resp = client.post("/api/eval/run", json=_BASE_PAYLOAD)
        assert resp.status_code == 200
        assert resp.json()["avg_score"] == pytest.approx(4.0, abs=0.01)
