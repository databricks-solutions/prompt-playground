"""Tests for prompt API routes.

Covers:
- GET  /api/prompts           — list prompts in catalog.schema
- GET  /api/prompts/versions  — get versions for a prompt
- GET  /api/prompts/template  — load a prompt template by name+version
- POST /api/prompts           — create a new prompt
- POST /api/prompts/versions  — save a new version of an existing prompt
"""

from unittest.mock import patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.routes.prompts import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /api/prompts — create a new prompt
# ---------------------------------------------------------------------------

def test_create_prompt_success(client):
    with patch("server.routes.prompts.create_prompt") as mock_create:
        mock_create.return_value = {
            "name": "main.prompts.my_prompt",
            "version": "1",
        }
        response = client.post("/api/prompts", json={
            "name": "main.prompts.my_prompt",
            "template": "Answer: {{question}}",
        })

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "main.prompts.my_prompt"
    assert data["version"] == "1"


def test_create_prompt_with_description(client):
    with patch("server.routes.prompts.create_prompt") as mock_create:
        mock_create.return_value = {
            "name": "main.prompts.my_prompt",
            "version": "1",
        }
        response = client.post("/api/prompts", json={
            "name": "main.prompts.my_prompt",
            "template": "Answer: {{question}}",
            "description": "A helpful Q&A prompt",
        })
        _, kwargs = mock_create.call_args
        assert kwargs.get("description") == "A helpful Q&A prompt"

    assert response.status_code == 200


def test_create_prompt_missing_name_returns_400(client):
    response = client.post("/api/prompts", json={
        "name": "",
        "template": "Answer: {{question}}",
    })
    assert response.status_code == 400
    assert "name" in response.json()["detail"].lower()


def test_create_prompt_whitespace_name_returns_400(client):
    response = client.post("/api/prompts", json={
        "name": "   ",
        "template": "Answer: {{question}}",
    })
    assert response.status_code == 400


def test_create_prompt_empty_template_returns_400(client):
    response = client.post("/api/prompts", json={
        "name": "main.prompts.my_prompt",
        "template": "",
    })
    assert response.status_code == 400
    assert "template" in response.json()["detail"].lower()


def test_create_prompt_already_exists_returns_409(client):
    with patch("server.routes.prompts.create_prompt") as mock_create:
        mock_create.side_effect = Exception("ALREADY_EXISTS: prompt exists")
        response = client.post("/api/prompts", json={
            "name": "main.prompts.existing",
            "template": "Hello {{name}}",
        })

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"].lower()


def test_create_prompt_other_error_returns_500(client):
    with patch("server.routes.prompts.create_prompt") as mock_create:
        mock_create.side_effect = Exception("Unexpected MLflow error")
        response = client.post("/api/prompts", json={
            "name": "main.prompts.my_prompt",
            "template": "Hello {{name}}",
        })

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# POST /api/prompts/versions — save a new version of an existing prompt
# ---------------------------------------------------------------------------

def test_save_version_success(client):
    with patch("server.routes.prompts.create_prompt_version") as mock_ver:
        mock_ver.return_value = {
            "name": "main.prompts.my_prompt",
            "version": "2",
        }
        response = client.post("/api/prompts/versions", json={
            "name": "main.prompts.my_prompt",
            "template": "Updated answer: {{question}}",
        })

    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "2"


def test_save_version_missing_name_returns_400(client):
    response = client.post("/api/prompts/versions", json={
        "name": "",
        "template": "Some template",
    })
    assert response.status_code == 400


def test_save_version_empty_template_returns_400(client):
    response = client.post("/api/prompts/versions", json={
        "name": "main.prompts.my_prompt",
        "template": "",
    })
    assert response.status_code == 400


def test_save_version_prompt_not_found_returns_404(client):
    with patch("server.routes.prompts.create_prompt_version") as mock_ver:
        mock_ver.side_effect = Exception("NOT_FOUND: prompt not found")
        response = client.post("/api/prompts/versions", json={
            "name": "main.prompts.nonexistent",
            "template": "Hello {{name}}",
        })

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_save_version_other_error_returns_500(client):
    with patch("server.routes.prompts.create_prompt_version") as mock_ver:
        mock_ver.side_effect = Exception("Internal MLflow error")
        response = client.post("/api/prompts/versions", json={
            "name": "main.prompts.my_prompt",
            "template": "Hello {{name}}",
        })

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# GET /api/prompts — list prompts
# ---------------------------------------------------------------------------

def test_list_prompts_success(client):
    with patch("server.routes.prompts.list_prompts") as mock_list:
        mock_list.return_value = [
            {"name": "main.prompts.prompt_a"},
            {"name": "main.prompts.prompt_b"},
        ]
        response = client.get("/api/prompts?catalog=main&schema=prompts")

    assert response.status_code == 200
    data = response.json()
    assert len(data["prompts"]) == 2
    assert data["catalog"] == "main"
    assert data["schema"] == "prompts"


def test_list_prompts_requires_catalog_and_schema(client):
    response = client.get("/api/prompts")
    assert response.status_code == 422


def test_list_prompts_custom_catalog_schema(client):
    with patch("server.routes.prompts.list_prompts") as mock_list:
        mock_list.return_value = []
        client.get("/api/prompts?catalog=my_cat&schema=my_schema")

    mock_list.assert_called_once_with(catalog="my_cat", schema="my_schema")


def test_list_prompts_permission_error_returns_403(client):
    with patch("server.routes.prompts.list_prompts") as mock_list:
        mock_list.side_effect = Exception("PERMISSION_DENIED: does not have USE SCHEMA privilege")
        response = client.get("/api/prompts?catalog=main&schema=prompts")

    assert response.status_code == 403
    assert "Permission denied" in response.json()["detail"]


def test_list_prompts_other_error_returns_500(client):
    with patch("server.routes.prompts.list_prompts") as mock_list:
        mock_list.side_effect = Exception("MLflow connection failed")
        response = client.get("/api/prompts?catalog=main&schema=prompts")

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# GET /api/prompts/versions — get versions for a prompt
# ---------------------------------------------------------------------------

def test_get_versions_success(client):
    with patch("server.routes.prompts.get_prompt_versions") as mock_ver:
        mock_ver.return_value = [
            {"version": "1", "template": "v1 template"},
            {"version": "2", "template": "v2 template"},
        ]
        response = client.get("/api/prompts/versions?name=main.prompts.my_prompt")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "main.prompts.my_prompt"
    assert len(data["versions"]) == 2


def test_get_versions_requires_name_param(client):
    response = client.get("/api/prompts/versions")
    assert response.status_code == 422


def test_get_versions_error_returns_500(client):
    with patch("server.routes.prompts.get_prompt_versions") as mock_ver:
        mock_ver.side_effect = Exception("MLflow error")
        response = client.get("/api/prompts/versions?name=main.prompts.bad_prompt")

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# GET /api/prompts/template — load a template by name+version
# ---------------------------------------------------------------------------

def test_get_template_success(client):
    with patch("server.routes.prompts.get_prompt_template") as mock_tpl:
        mock_tpl.return_value = {
            "template": "Answer: {{question}}",
            "variables": ["question"],
            "system_prompt": None,
        }
        response = client.get("/api/prompts/template?name=main.prompts.my_prompt&version=1")

    assert response.status_code == 200
    data = response.json()
    assert data["template"] == "Answer: {{question}}"
    assert data["variables"] == ["question"]


def test_get_template_not_found_returns_404(client):
    with patch("server.routes.prompts.get_prompt_template") as mock_tpl:
        mock_tpl.side_effect = ValueError("Prompt version not found")
        response = client.get("/api/prompts/template?name=main.prompts.gone&version=99")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_template_other_error_returns_500(client):
    with patch("server.routes.prompts.get_prompt_template") as mock_tpl:
        mock_tpl.side_effect = Exception("MLflow internal error")
        response = client.get("/api/prompts/template?name=main.prompts.my_prompt&version=1")

    assert response.status_code == 500


def test_get_template_requires_name_and_version(client):
    # Missing version
    response = client.get("/api/prompts/template?name=main.prompts.my_prompt")
    assert response.status_code == 422

    # Missing name
    response = client.get("/api/prompts/template?version=1")
    assert response.status_code == 422
