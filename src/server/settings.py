"""Persistent app settings stored in a JSON file alongside app.py.

Settings in the file override env var defaults, allowing marketplace installs
to be configured via the in-app settings panel without editing YAML files.
"""

import json
import os
from pathlib import Path

# Stored next to app.py (i.e. src/pp_settings.json)
_SETTINGS_FILE = Path(__file__).parent.parent / "pp_settings.json"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def load_settings() -> dict:
    """Load persisted settings. Returns empty dict if file doesn't exist."""
    try:
        if _SETTINGS_FILE.exists():
            return json.loads(_SETTINGS_FILE.read_text())
    except Exception:
        pass
    return {}


def save_settings(data: dict) -> None:
    """Persist settings to disk. Merges with existing settings."""
    current = load_settings()
    current.update({k: v for k, v in data.items() if v is not None})
    _SETTINGS_FILE.write_text(json.dumps(current, indent=2))


def get_effective_config() -> dict:
    """Merge env vars (defaults) with persisted settings (overrides).

    Persisted settings take precedence over env vars so that marketplace
    installs configured via the UI work correctly even when app.yaml has
    placeholder values.
    """
    # All empty unless set via environment (e.g. app bundle) or pp_settings.json / UI.
    env_defaults = {
        "prompt_catalog": os.environ.get("PROMPT_CATALOG", ""),
        "prompt_schema": os.environ.get("PROMPT_SCHEMA", ""),
        "eval_catalog": os.environ.get("EVAL_CATALOG", ""),
        "eval_schema": os.environ.get("EVAL_SCHEMA", ""),
        "mlflow_experiment_name": os.environ.get("MLFLOW_EXPERIMENT_NAME", ""),
        "sql_warehouse_id": os.environ.get("SQL_WAREHOUSE_ID", ""),
        "sql_warehouse_name": os.environ.get("SQL_WAREHOUSE_NAME", ""),
        "evaluate_tab_enabled": _env_bool("EVALUATE_TAB_ENABLED", False),
    }
    persisted = load_settings()
    merged = {**env_defaults, **persisted}
    merged["evaluate_tab_enabled"] = _coerce_bool(merged.get("evaluate_tab_enabled", False))
    return merged
