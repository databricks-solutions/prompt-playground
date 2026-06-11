"""Setup/discovery routes — used by the settings panel to populate dropdowns."""

import asyncio
import logging

from databricks.sdk.errors import DatabricksError, Unauthenticated
from fastapi import APIRouter, HTTPException

from server.config import IS_DATABRICKS_APP, get_workspace_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/setup", tags=["setup"])

# Shown on 401 so local dev and misconfigured env are actionable without digging in logs.
_SETUP_AUTH_HINT_LOCAL = (
    "For local dev, authenticate for the same workspace the backend calls: "
    "set DATABRICKS_HOST (e.g. https://xxx.cloud.databricks.com) and DATABRICKS_TOKEN "
    "(User Settings → Developer → Access tokens), or run `databricks auth login` and set "
    "DATABRICKS_PROFILE if you use a non-default CLI profile. Host and token must match."
)
_SETUP_AUTH_HINT_APP = (
    "Deployed apps use the app’s identity. If this persists after redeploy, confirm the app "
    "is running in Databricks Apps (not a mismatched local proxy) and that SQL Warehouse APIs "
    "are enabled for the workspace."
)


def _is_unauthenticated(exc: BaseException) -> bool:
    if isinstance(exc, Unauthenticated):
        return True
    msg = str(exc).lower()
    if "credential was not sent" in msg or "unsupported type" in msg:
        return True
    if "401" in msg or msg.startswith("401:"):
        return True
    return False


def _raise_from_setup_error(exc: BaseException) -> None:
    """Map workspace SDK errors to HTTP responses; always raises."""
    if _is_unauthenticated(exc):
        hint = _SETUP_AUTH_HINT_APP if IS_DATABRICKS_APP else _SETUP_AUTH_HINT_LOCAL
        detail = f"{exc} {hint}"
        raise HTTPException(status_code=401, detail=detail) from exc
    if isinstance(exc, DatabricksError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/catalogs")
async def list_catalogs():
    """List Unity Catalog catalogs the service principal can see."""
    try:
        w = get_workspace_client()
        catalogs = await asyncio.to_thread(lambda: [c.name for c in w.catalogs.list()])
        return {"catalogs": sorted(catalogs)}
    except Exception as e:
        _raise_from_setup_error(e)


@router.get("/schemas")
async def list_schemas(catalog: str):
    """List schemas within a catalog."""
    try:
        w = get_workspace_client()
        schemas = await asyncio.to_thread(
            lambda: [s.name for s in w.schemas.list(catalog_name=catalog)]
        )
        return {"schemas": sorted(schemas)}
    except Exception as e:
        _raise_from_setup_error(e)


@router.get("/warehouses")
async def list_warehouses():
    """List SQL warehouses available in the workspace."""
    try:
        w = get_workspace_client()

        def _collect():
            rows = []
            for wh in w.warehouses.list():
                wid = getattr(wh, "id", None)
                if wid is None:
                    continue
                name = getattr(wh, "name", None)
                if not (name and str(name).strip()):
                    name = f"Warehouse {wid}"
                rows.append({"id": str(wid), "name": str(name).strip()})
            return rows

        warehouses = await asyncio.to_thread(_collect)
        return {"warehouses": sorted(warehouses, key=lambda x: x["name"].lower())}
    except Exception as e:
        logger.warning("list_warehouses failed: %s", e)
        _raise_from_setup_error(e)
