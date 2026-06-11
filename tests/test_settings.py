"""Unit tests for server/settings.py — load, save, merge logic.

Covers:
- load_settings() returns empty dict when file doesn't exist
- save_settings() creates the file and persists data
- save_settings() merges new values into existing settings
- save_settings() does not overwrite existing keys with None
- load_settings() returns empty dict on malformed JSON (silent error)
- get_effective_config() returns env var defaults when no persisted settings
- get_effective_config() returns persisted settings overriding env vars
- get_effective_config() partial override (only some keys persisted)
"""

import json
import pytest

import server.settings as settings_module
from server.settings import load_settings, save_settings, get_effective_config


@pytest.fixture(autouse=True)
def isolated_settings_file(tmp_path, monkeypatch):
    """Redirect _SETTINGS_FILE to a temp path for every test in this module."""
    settings_path = tmp_path / "pp_settings.json"
    monkeypatch.setattr(settings_module, "_SETTINGS_FILE", settings_path)
    return settings_path


# ---------------------------------------------------------------------------
# load_settings
# ---------------------------------------------------------------------------

def test_load_returns_empty_when_file_missing():
    result = load_settings()
    assert result == {}


def test_load_returns_saved_data(isolated_settings_file):
    isolated_settings_file.write_text(json.dumps({"prompt_catalog": "my_cat"}))
    result = load_settings()
    assert result == {"prompt_catalog": "my_cat"}


def test_load_returns_empty_on_malformed_json(isolated_settings_file):
    isolated_settings_file.write_text("{not valid json")
    result = load_settings()
    assert result == {}


def test_load_returns_empty_on_empty_file(isolated_settings_file):
    isolated_settings_file.write_text("")
    result = load_settings()
    assert result == {}


# ---------------------------------------------------------------------------
# save_settings
# ---------------------------------------------------------------------------

def test_save_creates_file(isolated_settings_file):
    save_settings({"prompt_catalog": "new_cat"})
    assert isolated_settings_file.exists()


def test_save_persists_data(isolated_settings_file):
    save_settings({"prompt_schema": "my_prompts"})
    data = json.loads(isolated_settings_file.read_text())
    assert data["prompt_schema"] == "my_prompts"


def test_save_merges_with_existing(isolated_settings_file):
    isolated_settings_file.write_text(json.dumps({"prompt_catalog": "cat1", "eval_schema": "eval"}))
    save_settings({"prompt_schema": "new_schema"})
    data = json.loads(isolated_settings_file.read_text())
    # Existing keys preserved
    assert data["prompt_catalog"] == "cat1"
    assert data["eval_schema"] == "eval"
    # New key added
    assert data["prompt_schema"] == "new_schema"


def test_save_overwrites_existing_key(isolated_settings_file):
    isolated_settings_file.write_text(json.dumps({"prompt_catalog": "old_cat"}))
    save_settings({"prompt_catalog": "new_cat"})
    data = json.loads(isolated_settings_file.read_text())
    assert data["prompt_catalog"] == "new_cat"


def test_save_skips_none_values(isolated_settings_file):
    isolated_settings_file.write_text(json.dumps({"prompt_catalog": "existing"}))
    save_settings({"prompt_catalog": None, "eval_schema": None})
    data = json.loads(isolated_settings_file.read_text())
    # None values don't overwrite existing keys
    assert data["prompt_catalog"] == "existing"
    # None values not written
    assert "eval_schema" not in data


def test_save_writes_valid_json(isolated_settings_file):
    save_settings({"k": "v"})
    # Should be parseable JSON
    result = json.loads(isolated_settings_file.read_text())
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# get_effective_config — env var defaults
# ---------------------------------------------------------------------------

def test_effective_config_env_defaults(monkeypatch):
    monkeypatch.setenv("PROMPT_CATALOG", "env_catalog")
    monkeypatch.setenv("PROMPT_SCHEMA", "env_schema")
    monkeypatch.setenv("EVAL_SCHEMA", "env_eval")
    monkeypatch.setenv("SQL_WAREHOUSE_ID", "wh-env")
    monkeypatch.delenv("SQL_WAREHOUSE_NAME", raising=False)

    cfg = get_effective_config()
    assert cfg["prompt_catalog"] == "env_catalog"
    assert cfg["prompt_schema"] == "env_schema"
    assert cfg["eval_schema"] == "env_eval"
    assert cfg["sql_warehouse_id"] == "wh-env"


def test_effective_config_defaults_when_no_env(monkeypatch):
    monkeypatch.delenv("PROMPT_CATALOG", raising=False)
    monkeypatch.delenv("PROMPT_SCHEMA", raising=False)
    monkeypatch.delenv("EVAL_CATALOG", raising=False)
    monkeypatch.delenv("EVAL_SCHEMA", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("SQL_WAREHOUSE_ID", raising=False)

    cfg = get_effective_config()
    assert cfg["prompt_catalog"] == ""
    assert cfg["prompt_schema"] == ""
    assert cfg["eval_schema"] == ""
    assert cfg["sql_warehouse_id"] == ""


def test_effective_config_includes_all_expected_keys(monkeypatch):
    monkeypatch.delenv("PROMPT_CATALOG", raising=False)
    cfg = get_effective_config()
    expected = {"prompt_catalog", "prompt_schema", "eval_catalog", "eval_schema",
                "mlflow_experiment_name", "sql_warehouse_id", "sql_warehouse_name"}
    assert expected.issubset(set(cfg.keys()))


# ---------------------------------------------------------------------------
# get_effective_config — persisted settings override env vars
# ---------------------------------------------------------------------------

def test_persisted_setting_overrides_env(isolated_settings_file, monkeypatch):
    monkeypatch.setenv("PROMPT_CATALOG", "env_catalog")
    isolated_settings_file.write_text(json.dumps({"prompt_catalog": "persisted_catalog"}))

    cfg = get_effective_config()
    assert cfg["prompt_catalog"] == "persisted_catalog"


def test_persisted_partial_override(isolated_settings_file, monkeypatch):
    """Only the persisted keys override; others still come from env vars."""
    monkeypatch.setenv("PROMPT_CATALOG", "env_catalog")
    monkeypatch.setenv("PROMPT_SCHEMA", "env_schema")
    isolated_settings_file.write_text(json.dumps({"prompt_schema": "persisted_schema"}))

    cfg = get_effective_config()
    # env var not overridden
    assert cfg["prompt_catalog"] == "env_catalog"
    # persisted setting wins
    assert cfg["prompt_schema"] == "persisted_schema"


def test_persisted_warehouse_name_returned(isolated_settings_file, monkeypatch):
    monkeypatch.delenv("SQL_WAREHOUSE_NAME", raising=False)
    isolated_settings_file.write_text(json.dumps({"sql_warehouse_name": "My Serverless"}))

    cfg = get_effective_config()
    assert cfg["sql_warehouse_name"] == "My Serverless"
