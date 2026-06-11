"""Tests for experiment discovery endpoints in /api/eval.

Covers:
- GET /api/eval/experiments — all experiments when no catalog/schema provided
- GET /api/eval/experiments — filtered by catalog.schema using prompt_name tags
- GET /api/eval/experiments — fallback to all when no matches found
- GET /api/eval/experiments — regex sanitization rejects bad catalog/schema names
- GET /api/eval/experiments — only active (not deleted) experiments returned
- GET /api/eval/experiments — error returns 500
- GET /api/eval/experiments/prompts — returns sorted distinct prompt names
- GET /api/eval/experiments/prompts — empty list when experiment not found
- GET /api/eval/experiments/prompts — error returns 500
"""

import types
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes.evaluate import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_experiment_browse_cache():
    from server.routes import evaluate as evaluate_module

    evaluate_module._EXPERIMENT_BROWSE_CACHE.invalidate()
    yield
    evaluate_module._EXPERIMENT_BROWSE_CACHE.invalidate()


def _make_experiment(name, experiment_id, lifecycle_stage="active"):
    e = MagicMock()
    e.name = name
    e.experiment_id = experiment_id
    e.lifecycle_stage = lifecycle_stage
    return e


def _make_run(experiment_id, prompt_name=None):
    r = MagicMock()
    r.info.experiment_id = experiment_id
    r.data.tags = {}
    if prompt_name:
        r.data.tags["prompt_name"] = prompt_name
    return r


# ---------------------------------------------------------------------------
# GET /api/eval/experiments — no catalog/schema filter
# ---------------------------------------------------------------------------

class TestListExperimentsNoFilter:

    def test_returns_capped_active_experiments(self, client):
        exps = [_make_experiment("exp1", "1"), _make_experiment("exp2", "2")]
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = exps

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value=""), \
             patch("server.routes.evaluate.make_experiment_url", return_value=None):
            resp = client.get("/api/eval/experiments")

        assert resp.status_code == 200
        names = [e["name"] for e in resp.json()["experiments"]]
        assert "exp1" in names and "exp2" in names
        mock_client.search_experiments.assert_called_once()
        assert mock_client.search_experiments.call_args.kwargs.get("max_results") == 50

    def test_returns_configured_experiment_only(self, client):
        configured = _make_experiment("/Shared/my-exp", "cfg-1")
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = configured

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value="/Shared/my-exp"), \
             patch("server.routes.evaluate.make_experiment_url", return_value="https://example.com/exp"):
            resp = client.get("/api/eval/experiments")

        assert resp.status_code == 200
        data = resp.json()["experiments"]
        assert len(data) == 1
        assert data[0]["name"] == "/Shared/my-exp"
        mock_client.search_experiments.assert_not_called()

    def test_configured_only_skips_workspace_scan(self, client):
        configured = _make_experiment("/Shared/my-exp", "cfg-1")
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = configured

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value="/Shared/my-exp"), \
             patch("server.routes.evaluate.make_experiment_url", return_value="https://example.com/exp"):
            resp = client.get("/api/eval/experiments?configured_only=true")

        assert resp.status_code == 200
        assert resp.json()["experiments"][0]["name"] == "/Shared/my-exp"
        mock_client.search_experiments.assert_not_called()

    def test_browse_skips_configured_only_fast_path(self, client):
        configured = _make_experiment("/Shared/my-exp", "cfg-1")
        browsed = [
            _make_experiment("/Shared/my-exp", "cfg-1"),
            _make_experiment("/Shared/other-exp", "2"),
        ]
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = configured
        mock_client.search_experiments.return_value = browsed

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value="/Shared/my-exp"), \
             patch("server.routes.evaluate.make_experiment_url", return_value=None):
            resp = client.get("/api/eval/experiments?browse=true")

        assert resp.status_code == 200
        names = [e["name"] for e in resp.json()["experiments"]]
        assert "/Shared/other-exp" in names
        mock_client.search_experiments.assert_called_once()

    def test_search_filters_browsed_experiments(self, client):
        exps = [
            _make_experiment("/Shared/hinge-prompt-playground", "1"),
            _make_experiment("/Shared/other-exp", "2"),
        ]
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = exps

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value=""), \
             patch("server.routes.evaluate.make_experiment_url", return_value=None):
            resp = client.get("/api/eval/experiments?q=hinge")

        assert resp.status_code == 200
        names = [e["name"] for e in resp.json()["experiments"]]
        assert names == ["/Shared/hinge-prompt-playground"]

    def test_filters_out_deleted_experiments(self, client):
        exps = [
            _make_experiment("active_exp", "1", lifecycle_stage="active"),
            _make_experiment("deleted_exp", "2", lifecycle_stage="deleted"),
        ]
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = exps

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value=""), \
             patch("server.routes.evaluate.make_experiment_url", return_value=None):
            resp = client.get("/api/eval/experiments")

        names = [e["name"] for e in resp.json()["experiments"]]
        assert "active_exp" in names
        assert "deleted_exp" not in names

    def test_response_includes_experiment_id_and_url(self, client):
        exps = [_make_experiment("exp1", "exp-id-1")]
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = exps

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value=""), \
             patch("server.routes.evaluate.make_experiment_url", return_value="https://example.com/exp"):
            resp = client.get("/api/eval/experiments")

        item = resp.json()["experiments"][0]
        assert item["experiment_id"] == "exp-id-1"
        assert item["url"] == "https://example.com/exp"

    def test_empty_experiments_list(self, client):
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = []

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value=""), \
             patch("server.routes.evaluate.make_experiment_url", return_value=None):
            resp = client.get("/api/eval/experiments")

        assert resp.json()["experiments"] == []

    def test_error_returns_500(self, client):
        mock_client = MagicMock()
        mock_client.search_experiments.side_effect = RuntimeError("MLflow unavailable")

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client):
            resp = client.get("/api/eval/experiments")

        assert resp.status_code == 500
        assert "MLflow unavailable" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/eval/experiments — with catalog/schema filter
# ---------------------------------------------------------------------------

class TestListExperimentsWithFilter:

    def _setup_client_with_runs(self, experiments, runs_by_chunk):
        """Build a mock mlflow client where search_runs returns runs matching the chunk."""
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = experiments
        mock_client.get_experiment_by_name.return_value = None

        call_count = [0]

        def search_runs_side_effect(chunk, filter_string, max_results):
            idx = call_count[0]
            call_count[0] += 1
            return runs_by_chunk[idx] if idx < len(runs_by_chunk) else []

        mock_client.search_runs.side_effect = search_runs_side_effect
        return mock_client

    def test_filters_to_matching_experiments(self, client):
        exps = [
            _make_experiment("exp_matches", "1"),
            _make_experiment("exp_no_match", "2"),
        ]
        runs = [_make_run("1", "main.prompts.my_prompt")]
        mock_client = self._setup_client_with_runs(exps, [runs])

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value=""), \
             patch("server.routes.evaluate.make_experiment_url", return_value=None):
            resp = client.get("/api/eval/experiments?catalog=main&schema=prompts")

        names = [e["name"] for e in resp.json()["experiments"]]
        assert "exp_matches" in names
        assert "exp_no_match" not in names

    def test_falls_back_to_candidates_when_no_matches(self, client):
        """When no experiments match the filter, return the capped candidate list."""
        exps = [
            _make_experiment("exp_a", "1"),
            _make_experiment("exp_b", "2"),
        ]
        mock_client = self._setup_client_with_runs(exps, [[]])

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value=""), \
             patch("server.routes.evaluate.make_experiment_url", return_value=None):
            resp = client.get("/api/eval/experiments?catalog=main&schema=prompts")

        names = [e["name"] for e in resp.json()["experiments"]]
        assert "exp_a" in names and "exp_b" in names
        mock_client.search_experiments.assert_called_once()
        assert mock_client.search_experiments.call_args.kwargs.get("max_results") == 100

    def test_invalid_catalog_returns_all(self, client):
        """Catalog name with special chars (not matching \\w-) bypasses filter and returns all."""
        exps = [_make_experiment("exp1", "1"), _make_experiment("exp2", "2")]
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = exps

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value=""), \
             patch("server.routes.evaluate.make_experiment_url", return_value=None):
            resp = client.get("/api/eval/experiments?catalog=bad.cat!&schema=prompts")

        assert resp.status_code == 200
        assert len(resp.json()["experiments"]) == 2
        mock_client.search_runs.assert_not_called()

    def test_invalid_schema_returns_browse_list(self, client):
        """Schema name with dots returns capped browse list without filtering."""
        exps = [_make_experiment("exp1", "1")]
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = exps

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value=""), \
             patch("server.routes.evaluate.make_experiment_url", return_value=None):
            resp = client.get("/api/eval/experiments?catalog=main&schema=bad.schema")

        assert resp.status_code == 200
        mock_client.search_runs.assert_not_called()

    def test_empty_catalog_returns_browse_without_filtering(self, client):
        """Empty catalog string skips filtering entirely."""
        exps = [_make_experiment("exp1", "1")]
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = exps

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value=""), \
             patch("server.routes.evaluate.make_experiment_url", return_value=None):
            resp = client.get("/api/eval/experiments?catalog=&schema=prompts")

        assert resp.status_code == 200
        mock_client.search_runs.assert_not_called()

    def test_search_runs_uses_catalog_schema_prefix(self, client):
        """search_runs filter string should contain the catalog.schema prefix."""
        exps = [_make_experiment("exp1", "1")]
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = exps
        mock_client.search_runs.return_value = []

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configured_mlflow_experiment_name", return_value=""), \
             patch("server.routes.evaluate.make_experiment_url", return_value=None):
            client.get("/api/eval/experiments?catalog=my_catalog&schema=my_schema")

        call_args = mock_client.search_runs.call_args
        filter_string = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("filter_string", "")
        assert "my_catalog.my_schema." in filter_string


# ---------------------------------------------------------------------------
# GET /api/eval/experiments/prompts
# ---------------------------------------------------------------------------

class TestGetExperimentPrompts:

    def test_returns_sorted_prompt_names_from_tags(self, client):
        """Primary path: filter prompts by _mlflow_experiment_ids tag."""
        experiment = MagicMock()
        experiment.experiment_id = "exp-1"

        prompts = [
            {"name": "main.prompts.z_prompt", "tags": {"_mlflow_experiment_ids": ",exp-1,"}},
            {"name": "main.prompts.a_prompt", "tags": {"_mlflow_experiment_ids": ",exp-1,"}},
            {"name": "main.prompts.m_prompt", "tags": {"_mlflow_experiment_ids": ",exp-1,"}},
        ]
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = experiment

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.list_prompts", return_value=prompts), \
             patch("server.routes.evaluate.configure_mlflow"):
            resp = client.get("/api/eval/experiments/prompts?experiment_name=my_exp&catalog=main&schema=prompts")

        assert resp.status_code == 200
        names = resp.json()["prompt_names"]
        assert names == sorted(names)
        assert len(names) == 3

    def test_filters_by_experiment_id(self, client):
        """Only prompts tagged with the correct experiment ID are returned."""
        experiment = MagicMock()
        experiment.experiment_id = "exp-1"

        prompts = [
            {"name": "main.prompts.my_prompt", "tags": {"_mlflow_experiment_ids": ",exp-1,"}},
            {"name": "main.prompts.other_exp", "tags": {"_mlflow_experiment_ids": ",exp-999,"}},
            {"name": "main.prompts.no_tag", "tags": {}},
        ]
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = experiment

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.list_prompts", return_value=prompts), \
             patch("server.routes.evaluate.configure_mlflow"):
            resp = client.get("/api/eval/experiments/prompts?experiment_name=my_exp&catalog=main&schema=prompts")

        names = resp.json()["prompt_names"]
        assert len(names) == 1
        assert names == ["main.prompts.my_prompt"]

    def test_returns_empty_when_experiment_not_found(self, client):
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configure_mlflow"):
            resp = client.get(
                "/api/eval/experiments/prompts?experiment_name=nonexistent&catalog=main&schema=prompts"
            )

        assert resp.status_code == 200
        assert resp.json()["prompt_names"] == []

    def test_falls_back_to_runs_when_no_tags(self, client):
        """When no prompts have experiment tags, fall back to searching runs."""
        experiment = MagicMock()
        experiment.experiment_id = "exp-1"

        # No prompts have the experiment tag
        prompts = [
            {"name": "main.prompts.has_tag", "tags": {}},
        ]
        runs = [
            _make_run("exp-1", "main.prompts.has_tag"),
            _make_run("exp-1", None),  # no prompt_name tag — should be skipped
        ]
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = experiment
        mock_client.search_runs.return_value = runs

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.list_prompts", return_value=prompts), \
             patch("server.routes.evaluate.configure_mlflow"):
            resp = client.get("/api/eval/experiments/prompts?experiment_name=my_exp&catalog=main&schema=prompts")

        assert resp.json()["prompt_names"] == ["main.prompts.has_tag"]

    def test_requires_experiment_name_param(self, client):
        resp = client.get("/api/eval/experiments/prompts")
        assert resp.status_code == 422

    def test_error_returns_500(self, client):
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.side_effect = RuntimeError("MLflow error")

        with patch("server.routes.evaluate.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.evaluate.configure_mlflow"):
            resp = client.get(
                "/api/eval/experiments/prompts?experiment_name=exp&catalog=main&schema=prompts"
            )

        assert resp.status_code == 500
        assert "MLflow error" in resp.json()["detail"]
