"""Tests for /api/setup/* discovery routes (settings panel dropdowns).

Covers:
- GET /api/setup/catalogs — list Unity Catalog catalogs, sorted
- GET /api/setup/schemas  — list schemas within a catalog, sorted
- GET /api/setup/warehouses — list SQL warehouses, sorted by name, id+name required
- 401 on Databricks unauthenticated / bad credential errors
- Other workspace errors return 500
"""

import pytest
from databricks.sdk.errors import Unauthenticated
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes.setup import router


@pytest.fixture(autouse=True)
def clear_setup_caches():
    import server.routes.setup as setup_module
    setup_module._catalog_cache = None
    setup_module._schema_cache = {}
    setup_module._warehouse_cache = None
    yield
    setup_module._catalog_cache = None
    setup_module._schema_cache = {}
    setup_module._warehouse_cache = None


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _mock_workspace(catalogs=None, schemas=None, warehouses=None):
    """Build a mock WorkspaceClient with configurable return values."""
    w = MagicMock()
    if catalogs is not None:
        mocks = []
        for c in catalogs:
            m = MagicMock()
            m.name = c
            mocks.append(m)
        w.catalogs.list.return_value = mocks
    if schemas is not None:
        mocks = []
        for s in schemas:
            m = MagicMock()
            m.name = s
            mocks.append(m)
        w.schemas.list.return_value = mocks
    if warehouses is not None:
        wh_mocks = []
        for wh in warehouses:
            m = MagicMock()
            m.id = wh.get("id")
            m.name = wh.get("name")
            wh_mocks.append(m)
        w.warehouses.list.return_value = wh_mocks
    return w


# ---------------------------------------------------------------------------
# GET /api/setup/catalogs
# ---------------------------------------------------------------------------

class TestListCatalogs:

    def test_search_returns_matching_catalogs(self, client):
        w = _mock_workspace(catalogs=["main", "sandbox", "main_staging"])
        with patch("server.routes.setup.get_workspace_client", return_value=w), \
             patch("server.routes.setup._catalog_cache", None):
            resp = client.get("/api/setup/catalogs?q=main")
        assert resp.status_code == 200
        assert resp.json()["catalogs"] == ["main", "main_staging"]

    def test_search_caps_results_at_50(self, client):
        names = [f"catalog_{i:03d}" for i in range(60)]
        w = _mock_workspace(catalogs=names)
        with patch("server.routes.setup.get_workspace_client", return_value=w), \
             patch("server.routes.setup._catalog_cache", None):
            resp = client.get("/api/setup/catalogs?q=catalog")
        assert resp.status_code == 200
        assert len(resp.json()["catalogs"]) == 50

    def test_no_query_returns_configured_only_not_full_list(self, client):
        w = _mock_workspace(catalogs=["main", "sandbox"])
        with patch("server.routes.setup.get_workspace_client", return_value=w), \
             patch("server.routes.setup.get_effective_config", return_value={"prompt_catalog": "", "eval_catalog": ""}):
            resp = client.get("/api/setup/catalogs")
        assert resp.status_code == 200
        assert resp.json()["catalogs"] == []
        w.catalogs.list.assert_not_called()

    def test_short_query_skips_workspace_scan(self, client):
        w = _mock_workspace(catalogs=["main"])
        with patch("server.routes.setup.get_workspace_client", return_value=w), \
             patch("server.routes.setup.get_effective_config", return_value={"prompt_catalog": "", "eval_catalog": ""}):
            resp = client.get("/api/setup/catalogs?q=a")
        assert resp.status_code == 200
        assert resp.json()["catalogs"] == []
        w.catalogs.list.assert_not_called()

    def test_catalogs_q_filter(self, client):
        w = _mock_workspace(catalogs=["demo_prompt_app", "main", "sandbox"])
        with patch("server.routes.setup.get_workspace_client", return_value=w), \
             patch("server.routes.setup._catalog_cache", None):
            resp = client.get("/api/setup/catalogs?q=demo")
        assert resp.status_code == 200
        assert resp.json()["catalogs"] == ["demo_prompt_app"]

    def test_configured_only_skips_workspace_list(self, client):
        with patch("server.routes.setup.get_effective_config", return_value={
            "prompt_catalog": "my_catalog",
            "eval_catalog": "my_catalog",
        }), patch("server.routes.setup.get_workspace_client") as mock_ws:
            resp = client.get("/api/setup/catalogs?configured_only=true")
        assert resp.status_code == 200
        assert resp.json()["catalogs"] == ["my_catalog"]
        mock_ws.assert_not_called()

    def test_workspace_error_returns_500(self, client):
        with patch(
            "server.routes.setup.get_workspace_client",
            side_effect=RuntimeError("Auth failed"),
        ), patch("server.routes.setup._catalog_cache", None):
            resp = client.get("/api/setup/catalogs?q=ab")
        assert resp.status_code == 500
        assert "Auth failed" in resp.json()["detail"]

    def test_catalogs_list_error_returns_500(self, client):
        w = MagicMock()
        w.catalogs.list.side_effect = RuntimeError("Permission denied")
        with patch("server.routes.setup.get_workspace_client", return_value=w), \
             patch("server.routes.setup._catalog_cache", None):
            resp = client.get("/api/setup/catalogs?q=ab")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /api/setup/schemas
# ---------------------------------------------------------------------------

class TestListSchemas:

    def test_search_returns_matching_schemas(self, client):
        w = _mock_workspace(schemas=["prompts", "eval_data", "prompts_staging"])
        with patch("server.routes.setup.get_workspace_client", return_value=w), \
             patch("server.routes.setup._schema_cache", {}):
            resp = client.get("/api/setup/schemas?catalog=main&q=prompt")
        assert resp.status_code == 200
        assert resp.json()["schemas"] == ["prompts", "prompts_staging"]

    def test_no_query_returns_configured_only(self, client):
        w = _mock_workspace(schemas=["prompts", "eval_data"])
        with patch("server.routes.setup.get_workspace_client", return_value=w), \
             patch("server.routes.setup.get_effective_config", return_value={
                 "prompt_catalog": "main",
                 "prompt_schema": "prompts",
                 "eval_catalog": "",
                 "eval_schema": "",
             }):
            resp = client.get("/api/setup/schemas?catalog=main")
        assert resp.status_code == 200
        assert resp.json()["schemas"] == ["prompts"]
        w.schemas.list.assert_not_called()

    def test_requires_catalog_param(self, client):
        resp = client.get("/api/setup/schemas")
        assert resp.status_code == 422

    def test_catalog_passed_to_workspace_on_search(self, client):
        w = _mock_workspace(schemas=["s"])
        with patch("server.routes.setup.get_workspace_client", return_value=w), \
             patch("server.routes.setup._schema_cache", {}):
            client.get("/api/setup/schemas?catalog=my_catalog&q=sc")
        w.schemas.list.assert_called_once_with(catalog_name="my_catalog")

    def test_empty_schema_list_on_search(self, client):
        w = _mock_workspace(schemas=[])
        with patch("server.routes.setup.get_workspace_client", return_value=w), \
             patch("server.routes.setup._schema_cache", {}):
            resp = client.get("/api/setup/schemas?catalog=main&q=ab")
        assert resp.json()["schemas"] == []

    def test_workspace_error_returns_500(self, client):
        w = MagicMock()
        w.schemas.list.side_effect = RuntimeError("Catalog not found")
        with patch("server.routes.setup.get_workspace_client", return_value=w), \
             patch("server.routes.setup._schema_cache", {}):
            resp = client.get("/api/setup/schemas?catalog=nonexistent&q=ab")
        assert resp.status_code == 500
        assert "Catalog not found" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/setup/warehouses
# ---------------------------------------------------------------------------

class TestListWarehouses:

    def test_returns_warehouse_id_and_name(self, client):
        w = _mock_workspace(warehouses=[
            {"id": "wh-1", "name": "Shared Warehouse"},
            {"id": "wh-2", "name": "Dev Warehouse"},
        ])
        with patch("server.routes.setup.get_workspace_client", return_value=w):
            resp = client.get("/api/setup/warehouses")
        assert resp.status_code == 200
        whs = resp.json()["warehouses"]
        assert len(whs) == 2
        assert all("id" in wh and "name" in wh for wh in whs)

    def test_warehouses_sorted_by_name(self, client):
        w = _mock_workspace(warehouses=[
            {"id": "wh-z", "name": "Zebra Warehouse"},
            {"id": "wh-a", "name": "Alpha Warehouse"},
            {"id": "wh-m", "name": "Main Warehouse"},
        ])
        with patch("server.routes.setup.get_workspace_client", return_value=w):
            resp = client.get("/api/setup/warehouses")
        names = [wh["name"] for wh in resp.json()["warehouses"]]
        assert names == ["Alpha Warehouse", "Main Warehouse", "Zebra Warehouse"]

    def test_warehouses_without_id_excluded(self, client):
        """Warehouses missing id are skipped; missing name get a display label."""
        w = _mock_workspace(warehouses=[
            {"id": "wh-1", "name": "Good"},
            {"id": None, "name": "No ID"},
            {"id": "wh-3", "name": None},
        ])
        with patch("server.routes.setup.get_workspace_client", return_value=w):
            resp = client.get("/api/setup/warehouses")
        whs = resp.json()["warehouses"]
        assert len(whs) == 2
        assert whs[0]["id"] == "wh-1"
        assert whs[1]["id"] == "wh-3"
        assert whs[1]["name"] == "Warehouse wh-3"

    def test_empty_warehouse_list(self, client):
        w = _mock_workspace(warehouses=[])
        with patch("server.routes.setup.get_workspace_client", return_value=w):
            resp = client.get("/api/setup/warehouses")
        assert resp.json()["warehouses"] == []

    def test_workspace_error_returns_500(self, client):
        w = MagicMock()
        w.warehouses.list.side_effect = RuntimeError("Workspace unreachable")
        with patch("server.routes.setup.get_workspace_client", return_value=w):
            resp = client.get("/api/setup/warehouses")
        assert resp.status_code == 500
        assert "Workspace unreachable" in resp.json()["detail"]

    def test_unauthenticated_returns_401_with_hint(self, client):
        w = MagicMock()
        w.warehouses.list.side_effect = Unauthenticated(
            "401: Credential was not sent or was of an unsupported type"
        )
        with patch("server.routes.setup.get_workspace_client", return_value=w):
            resp = client.get("/api/setup/warehouses")
        assert resp.status_code == 401
        body = resp.json()["detail"]
        assert "DATABRICKS_HOST" in body

    def test_message_containing_401_returns_401(self, client):
        w = MagicMock()
        w.warehouses.list.side_effect = RuntimeError(
            "401: Credential was not sent or was of an unsupported type [ReqId: x]"
        )
        with patch("server.routes.setup.get_workspace_client", return_value=w):
            resp = client.get("/api/setup/warehouses")
        assert resp.status_code == 401
        assert "DATABRICKS_TOKEN" in resp.json()["detail"]
