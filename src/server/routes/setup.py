"""Setup/discovery routes — used by the settings panel to populate dropdowns."""

import asyncio
import logging
import time

from databricks.sdk.errors import DatabricksError, Unauthenticated
from fastapi import APIRouter, HTTPException, Query

from server.config import IS_DATABRICKS_APP, get_workspace_client
from server.settings import get_effective_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/setup", tags=["setup"])

_CATALOG_CACHE_TTL_SEC = 300
_SCHEMA_CACHE_TTL_SEC = 180
_WAREHOUSE_CACHE_TTL_SEC = 180
_CATALOG_SEARCH_MIN_LEN = 2
_CATALOG_SEARCH_MAX_RESULTS = 50
_SCHEMA_SEARCH_MAX_RESULTS = 50
_catalog_cache: tuple[float, list[str]] | None = None
_schema_cache: dict[str, tuple[float, list[str]]] = {}
_warehouse_cache: tuple[float, list[dict]] | None = None

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


def _configured_catalog_names() -> list[str]:
    cfg = get_effective_config()
    names = {
        (cfg.get("prompt_catalog") or "").strip(),
        (cfg.get("eval_catalog") or "").strip(),
    }
    return sorted(n for n in names if n)


def _configured_schema_names(catalog: str) -> list[str]:
    cfg = get_effective_config()
    names: set[str] = set()
    if (cfg.get("prompt_catalog") or "").strip() == catalog:
        ps = (cfg.get("prompt_schema") or "").strip()
        if ps:
            names.add(ps)
    eval_cat = (cfg.get("eval_catalog") or cfg.get("prompt_catalog") or "").strip()
    if eval_cat == catalog:
        es = (cfg.get("eval_schema") or "").strip()
        if es:
            names.add(es)
    return sorted(names)


def _list_all_catalog_names() -> list[str]:
    global _catalog_cache
    now = time.time()
    if _catalog_cache and now - _catalog_cache[0] < _CATALOG_CACHE_TTL_SEC:
        return list(_catalog_cache[1])

    w = get_workspace_client()
    catalogs = [c.name for c in w.catalogs.list()]
    sorted_catalogs = sorted(catalogs)
    _catalog_cache = (now, sorted_catalogs)
    return sorted_catalogs


@router.post("/catalogs/warm")
async def warm_catalog_cache():
    """Populate the catalog name cache in the background (Settings open). Returns immediately."""
    async def _warm() -> None:
        try:
            await asyncio.to_thread(_list_all_catalog_names)
        except Exception as e:
            logger.debug("catalog cache warm failed (non-fatal): %s", e)

    asyncio.create_task(_warm())
    return {"status": "warming"}


@router.get("/catalogs")
async def list_catalogs(
    q: str | None = Query(None, description="Case-insensitive substring filter (min 2 chars)"),
    configured_only: bool = Query(
        False,
        description="Return only catalogs already saved in app settings (instant)",
    ),
):
    """List Unity Catalog catalogs for the Settings dropdown.

    Avoids scanning the full workspace on every open (shared workspaces can have
    thousands of catalogs). Search with ``q`` (2+ characters) returns up to 50
    matches from a short-lived server cache. With no ``q``, returns only catalogs
  already configured via Settings — never the full list.
    """
    try:
        if configured_only:
            return {"catalogs": _configured_catalog_names()}

        query = (q or "").strip()
        if len(query) < _CATALOG_SEARCH_MIN_LEN:
            return {"catalogs": _configured_catalog_names()}

        catalogs = await asyncio.to_thread(_list_all_catalog_names)
        needle = query.lower()
        matches = [c for c in catalogs if needle in c.lower()]
        return {"catalogs": matches[:_CATALOG_SEARCH_MAX_RESULTS]}
    except Exception as e:
        _raise_from_setup_error(e)


def _list_all_schema_names(catalog: str) -> list[str]:
    global _schema_cache
    now = time.time()
    cached = _schema_cache.get(catalog)
    if cached and now - cached[0] < _SCHEMA_CACHE_TTL_SEC:
        return list(cached[1])

    w = get_workspace_client()
    schemas = sorted(s.name for s in w.schemas.list(catalog_name=catalog))
    _schema_cache[catalog] = (now, schemas)
    return schemas


@router.get("/schemas")
async def list_schemas(
    catalog: str,
    q: str | None = Query(None, description="Case-insensitive substring filter (min 2 chars)"),
    configured_only: bool = Query(
        False,
        description="Return only schemas already saved in app settings (instant)",
    ),
):
    """List schemas within a catalog for the Settings dropdown.

    With no ``q``, returns only configured schemas (no full catalog scan).
    Search with ``q`` (2+ characters) returns up to 50 matches from a short-lived cache.
    """
    try:
        if configured_only:
            return {"schemas": _configured_schema_names(catalog)}

        query = (q or "").strip()
        if len(query) < _CATALOG_SEARCH_MIN_LEN:
            return {"schemas": _configured_schema_names(catalog)}

        schemas = await asyncio.to_thread(_list_all_schema_names, catalog)
        needle = query.lower()
        matches = [s for s in schemas if needle in s.lower()]
        return {"schemas": matches[:_CATALOG_SEARCH_MAX_RESULTS]}
    except Exception as e:
        _raise_from_setup_error(e)


def _list_all_warehouses() -> list[dict]:
    global _warehouse_cache
    now = time.time()
    if _warehouse_cache and now - _warehouse_cache[0] < _WAREHOUSE_CACHE_TTL_SEC:
        return list(_warehouse_cache[1])

    w = get_workspace_client()
    rows = []
    for wh in w.warehouses.list():
        wid = getattr(wh, "id", None)
        if wid is None:
            continue
        name = getattr(wh, "name", None)
        if not (name and str(name).strip()):
            name = f"Warehouse {wid}"
        rows.append({"id": str(wid), "name": str(name).strip()})
    sorted_rows = sorted(rows, key=lambda x: x["name"].lower())
    _warehouse_cache = (now, sorted_rows)
    return sorted_rows


@router.get("/warehouses")
async def list_warehouses():
    """List SQL warehouses available in the workspace."""
    try:
        warehouses = await asyncio.to_thread(_list_all_warehouses)
        return {"warehouses": warehouses}
    except Exception as e:
        logger.warning("list_warehouses failed: %s", e)
        _raise_from_setup_error(e)
