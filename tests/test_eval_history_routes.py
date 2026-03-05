"""Tests for /api/eval/history and /api/eval/run-traces endpoints.

Covers:
- GET /api/eval/history   — return past eval runs for a prompt, all versions or filtered
- GET /api/eval/run-traces — return aggregated per-row scores from MLflow traces for a run
"""

import pytest
from unittest.mock import patch, MagicMock, call
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes.evaluate import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# --- Helpers ---

def _make_run(
    run_id="run-001",
    run_name="eval-my-prompt-v1",
    start_time=1700000000000,
    metrics=None,
    tags=None,
):
    """Build a mock MLflow run with sensible defaults."""
    run = MagicMock()
    run.info.run_id = run_id
    run.info.run_name = run_name
    run.info.start_time = start_time
    run.data.metrics = metrics if metrics is not None else {}
    run.data.tags = {
        "prompt_name": "catalog.schema.my_prompt",
        "prompt_version": "1",
        "eval_type": "batch",
        "model": "claude-3-5-sonnet",
        "dataset": "catalog.eval.test_data",
        "scorer": "response_quality",
        "total_rows": "50",
        **(tags or {}),
    }
    return run


def _make_experiment(exp_id="exp-123", name="/Shared/eval"):
    exp = MagicMock()
    exp.experiment_id = exp_id
    exp.name = name
    return exp


def _mock_mlflow(mock_client, runs=None, experiment=None):
    """Return the standard patch context for history endpoint tests."""
    mock_exp = experiment or _make_experiment()
    mock_client.get_experiment_by_name.return_value = mock_exp
    mock_client.search_runs.return_value = runs if runs is not None else []
    return [
        patch("server.routes.evaluate.configure_mlflow"),
        patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client),
        patch("server.routes.evaluate.make_experiment_url", return_value="https://ex.com/123"),
    ]


# ---------------------------------------------------------------------------
# GET /api/eval/history
# ---------------------------------------------------------------------------

class TestEvalHistory:

    def test_returns_runs_list(self, client):
        mock_client = MagicMock()
        run = _make_run(metrics={"response_quality/mean": 4.25})
        patches = _mock_mlflow(mock_client, runs=[run])
        with patches[0], patches[1], patches[2]:
            resp = client.get("/api/eval/history", params={"prompt_name": "catalog.schema.my_prompt"})
        assert resp.status_code == 200
        assert len(resp.json()["runs"]) == 1

    def test_run_has_all_required_fields(self, client):
        mock_client = MagicMock()
        patches = _mock_mlflow(mock_client, runs=[_make_run()])
        with patches[0], patches[1], patches[2]:
            resp = client.get("/api/eval/history", params={"prompt_name": "p"})
        run = resp.json()["runs"][0]
        for field in ("run_id", "run_name", "created_at", "avg_score", "model",
                      "dataset", "scorer", "prompt_version", "total_rows", "run_url"):
            assert field in run, f"missing field: {field}"

    def test_avg_score_from_scorer_mean_metric(self, client):
        mock_client = MagicMock()
        patches = _mock_mlflow(mock_client, runs=[_make_run(metrics={"response_quality/mean": 4.25})])
        with patches[0], patches[1], patches[2]:
            resp = client.get("/api/eval/history", params={"prompt_name": "p"})
        assert resp.json()["runs"][0]["avg_score"] == 4.25

    def test_avg_score_falls_back_to_direct_scorer_metric(self, client):
        mock_client = MagicMock()
        # No /mean key — should fall back to plain scorer key
        patches = _mock_mlflow(mock_client, runs=[_make_run(metrics={"response_quality": 3.5})])
        with patches[0], patches[1], patches[2]:
            resp = client.get("/api/eval/history", params={"prompt_name": "p"})
        assert resp.json()["runs"][0]["avg_score"] == 3.5

    def test_avg_score_none_when_no_matching_metric(self, client):
        mock_client = MagicMock()
        patches = _mock_mlflow(mock_client, runs=[_make_run(metrics={"unrelated_metric": 99.0})])
        with patches[0], patches[1], patches[2]:
            resp = client.get("/api/eval/history", params={"prompt_name": "p"})
        assert resp.json()["runs"][0]["avg_score"] is None

    def test_empty_list_when_experiment_not_found(self, client):
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None
        with patch("server.routes.evaluate.configure_mlflow"), \
             patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client):
            resp = client.get("/api/eval/history", params={"prompt_name": "p"})
        assert resp.status_code == 200
        assert resp.json()["runs"] == []

    def test_version_filter_added_to_query_when_provided(self, client):
        mock_client = MagicMock()
        patches = _mock_mlflow(mock_client, runs=[])
        with patches[0], patches[1], patches[2]:
            client.get("/api/eval/history", params={"prompt_name": "p", "prompt_version": "3"})
        filter_arg = mock_client.search_runs.call_args[0][1]
        assert "prompt_version = '3'" in filter_arg

    def test_no_version_filter_when_omitted(self, client):
        mock_client = MagicMock()
        patches = _mock_mlflow(mock_client, runs=[])
        with patches[0], patches[1], patches[2]:
            client.get("/api/eval/history", params={"prompt_name": "p"})
        filter_arg = mock_client.search_runs.call_args[0][1]
        assert "prompt_version" not in filter_arg

    def test_limit_bumped_to_50_when_no_version_filter(self, client):
        """When no version is specified we want more runs to cover all versions."""
        mock_client = MagicMock()
        patches = _mock_mlflow(mock_client, runs=[])
        with patches[0], patches[1], patches[2]:
            client.get("/api/eval/history", params={"prompt_name": "p", "limit": "5"})
        _, kwargs = mock_client.search_runs.call_args
        assert kwargs["max_results"] == 50

    def test_limit_respected_when_version_filter_provided(self, client):
        mock_client = MagicMock()
        patches = _mock_mlflow(mock_client, runs=[])
        with patches[0], patches[1], patches[2]:
            client.get("/api/eval/history", params={
                "prompt_name": "p", "prompt_version": "1", "limit": "5"
            })
        _, kwargs = mock_client.search_runs.call_args
        assert kwargs["max_results"] == 5

    def test_prompt_version_tag_returned_in_run(self, client):
        mock_client = MagicMock()
        run = _make_run(tags={"prompt_version": "7"})
        patches = _mock_mlflow(mock_client, runs=[run])
        with patches[0], patches[1], patches[2]:
            resp = client.get("/api/eval/history", params={"prompt_name": "p"})
        assert resp.json()["runs"][0]["prompt_version"] == "7"

    def test_total_rows_parsed_as_integer(self, client):
        mock_client = MagicMock()
        run = _make_run(tags={"total_rows": "100"})
        patches = _mock_mlflow(mock_client, runs=[run])
        with patches[0], patches[1], patches[2]:
            resp = client.get("/api/eval/history", params={"prompt_name": "p"})
        assert resp.json()["runs"][0]["total_rows"] == 100

    def test_total_rows_none_when_tag_missing(self, client):
        mock_client = MagicMock()
        run = _make_run()
        run.data.tags.pop("total_rows")
        patches = _mock_mlflow(mock_client, runs=[run])
        with patches[0], patches[1], patches[2]:
            resp = client.get("/api/eval/history", params={"prompt_name": "p"})
        assert resp.json()["runs"][0]["total_rows"] is None

    def test_experiment_name_param_forwarded_to_mlflow(self, client):
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None
        with patch("server.routes.evaluate.configure_mlflow"), \
             patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client):
            client.get("/api/eval/history", params={
                "prompt_name": "p", "experiment_name": "my_exp"
            })
        mock_client.get_experiment_by_name.assert_called_once_with("my_exp")

    def test_run_url_contains_run_id(self, client):
        mock_client = MagicMock()
        run = _make_run(run_id="run-xyz")
        patches = _mock_mlflow(mock_client, runs=[run])
        with patches[0], patches[1], patches[2]:
            resp = client.get("/api/eval/history", params={"prompt_name": "p"})
        assert "run-xyz" in resp.json()["runs"][0]["run_url"]

    def test_prompt_name_required(self, client):
        resp = client.get("/api/eval/history")
        assert resp.status_code == 422

    def test_mlflow_error_returns_500(self, client):
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.side_effect = RuntimeError("MLflow unavailable")
        with patch("server.routes.evaluate.configure_mlflow"), \
             patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client):
            resp = client.get("/api/eval/history", params={"prompt_name": "p"})
        assert resp.status_code == 500
        assert "MLflow unavailable" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/eval/run-traces
# ---------------------------------------------------------------------------

class TestRunTraces:

    def _get_traces(self, client, run_id="run-abc", scorer_tag="response_quality", row_scores=None):
        mock_run = MagicMock()
        mock_run.data.tags = {"scorer": scorer_tag}
        mock_client = MagicMock()
        mock_client.get_run.return_value = mock_run
        if row_scores is None:
            row_scores = {0: (4.0, "good", None), 1: (3.0, "ok", None)}
        with patch("server.routes.evaluate.configure_mlflow"), \
             patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate._extract_row_scores", return_value=row_scores):
            return client.get("/api/eval/run-traces", params={"run_id": run_id})

    def test_returns_scorer_and_rows(self, client):
        resp = self._get_traces(client)
        assert resp.status_code == 200
        assert resp.json()["scorer"] == "response_quality"
        assert len(resp.json()["rows"]) == 2

    def test_row_has_required_fields(self, client):
        resp = self._get_traces(client, row_scores={0: (4.0, "good", None)})
        row = resp.json()["rows"][0]
        for field in ("row_index", "score", "rationale", "details"):
            assert field in row, f"missing field: {field}"

    def test_rows_sorted_by_index(self, client):
        # Return keys out of order — endpoint must sort them
        row_scores = {2: (5.0, None, None), 0: (3.0, None, None), 1: (4.0, None, None)}
        resp = self._get_traces(client, row_scores=row_scores)
        indices = [r["row_index"] for r in resp.json()["rows"]]
        assert indices == [0, 1, 2]

    def test_scorer_tag_used_when_present(self, client):
        mock_run = MagicMock()
        mock_run.data.tags = {"scorer": "my_custom_judge"}
        mock_client = MagicMock()
        mock_client.get_run.return_value = mock_run
        with patch("server.routes.evaluate.configure_mlflow"), \
             patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate._extract_row_scores", return_value={}) as mock_extract:
            resp = client.get("/api/eval/run-traces", params={"run_id": "run-abc"})
        mock_extract.assert_called_once_with("run-abc", "my_custom_judge")
        assert resp.json()["scorer"] == "my_custom_judge"

    def test_falls_back_to_response_quality_when_scorer_tag_missing(self, client):
        mock_run = MagicMock()
        mock_run.data.tags = {}  # no scorer tag
        mock_client = MagicMock()
        mock_client.get_run.return_value = mock_run
        with patch("server.routes.evaluate.configure_mlflow"), \
             patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate._extract_row_scores", return_value={}) as mock_extract:
            resp = client.get("/api/eval/run-traces", params={"run_id": "run-abc"})
        mock_extract.assert_called_once_with("run-abc", "response_quality")
        assert resp.json()["scorer"] == "response_quality"

    def test_empty_rows_when_no_trace_data(self, client):
        resp = self._get_traces(client, row_scores={})
        assert resp.status_code == 200
        assert resp.json()["rows"] == []

    def test_guidelines_details_passed_through(self, client):
        details = [
            {"name": "judge/rule_a", "value": True, "rationale": None},
            {"name": "judge/rule_b", "value": False, "rationale": None},
        ]
        resp = self._get_traces(
            client,
            scorer_tag="my_guidelines",
            row_scores={0: ("1/2", None, details)},
        )
        row = resp.json()["rows"][0]
        assert row["score"] == "1/2"
        assert row["details"] == details

    def test_numeric_score_and_rationale_passed_through(self, client):
        resp = self._get_traces(
            client, row_scores={0: (4.5, "well structured", None)}
        )
        row = resp.json()["rows"][0]
        assert row["score"] == 4.5
        assert row["rationale"] == "well structured"
        assert row["details"] is None

    def test_run_id_required(self, client):
        resp = client.get("/api/eval/run-traces")
        assert resp.status_code == 422

    def test_mlflow_error_returns_500(self, client):
        mock_client = MagicMock()
        mock_client.get_run.side_effect = RuntimeError("Run not found")
        with patch("server.routes.evaluate.configure_mlflow"), \
             patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client):
            resp = client.get("/api/eval/run-traces", params={"run_id": "bad-id"})
        assert resp.status_code == 500
        assert "Run not found" in resp.json()["detail"]
