"""Tests for prompt creation with experiment registration.

Covers:
- Creating a prompt with experiment_name tags the prompt with the experiment ID
- Creating a prompt without experiment_name skips tagging
- Registration failure is non-fatal (prompt still created)
- Experiment not found → tag skipped silently
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes.prompts import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _mock_create_prompt(**kwargs):
    """Simulate a successful prompt creation."""
    return {
        "name": kwargs.get("name", "cat.sch.test"),
        "version": "1",
        "template": kwargs.get("template", "hello"),
        "variables": [],
    }


class TestExperimentRegistrationOnCreate:

    def test_create_with_experiment_name_tags_prompt(self, client):
        """When experiment_name is provided, the prompt gets tagged with the experiment ID."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-42"
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = mock_experiment

        mock_http = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        mock_http.post = AsyncMock(return_value=mock_response)

        with patch("server.routes.prompts.create_prompt", side_effect=_mock_create_prompt), \
             patch("server.routes.prompts.configure_mlflow"), \
             patch("server.routes.prompts.get_mlflow_client", return_value=mock_client), \
             patch("server.routes.prompts.get_workspace_host", return_value="https://test.databricks.com"), \
             patch("server.routes.prompts.get_oauth_token", return_value="test-token"), \
             patch("httpx.AsyncClient", return_value=mock_http):
            resp = client.post("/api/prompts", json={
                "name": "cat.sch.test",
                "template": "hello {{x}}",
                "experiment_name": "/Shared/my-experiment",
            })

        assert resp.status_code == 200
        # Verify the tag POST was made
        mock_http.post.assert_called_once()
        call_kwargs = mock_http.post.call_args
        assert "_mlflow_experiment_ids" in str(call_kwargs)

    def test_create_without_experiment_name_skips_registration(self, client):
        """When experiment_name is empty, no tagging is attempted."""
        with patch("server.routes.prompts.create_prompt", side_effect=_mock_create_prompt):
            resp = client.post("/api/prompts", json={
                "name": "cat.sch.test",
                "template": "hello {{x}}",
            })

        assert resp.status_code == 200

    def test_registration_failure_is_non_fatal(self, client):
        """If tagging fails, the prompt is still created successfully."""
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.side_effect = RuntimeError("mlflow down")

        with patch("server.routes.prompts.create_prompt", side_effect=_mock_create_prompt), \
             patch("server.routes.prompts.configure_mlflow"), \
             patch("server.routes.prompts.get_mlflow_client", return_value=mock_client):
            resp = client.post("/api/prompts", json={
                "name": "cat.sch.test",
                "template": "hello {{x}}",
                "experiment_name": "/Shared/my-experiment",
            })

        assert resp.status_code == 200
        assert resp.json()["name"] == "cat.sch.test"

    def test_experiment_not_found_skips_silently(self, client):
        """If the experiment doesn't exist, no tag is set."""
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None

        with patch("server.routes.prompts.create_prompt", side_effect=_mock_create_prompt), \
             patch("server.routes.prompts.configure_mlflow"), \
             patch("server.routes.prompts.get_mlflow_client", return_value=mock_client):
            resp = client.post("/api/prompts", json={
                "name": "cat.sch.test",
                "template": "hello {{x}}",
                "experiment_name": "/Shared/nonexistent",
            })

        assert resp.status_code == 200
