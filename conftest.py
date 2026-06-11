import sys
from pathlib import Path

import pytest

# Make src/ importable so tests can do `from server.xxx import ...`
# without needing the package to be installed.
sys.path.insert(0, str(Path(__file__).parent / "src"))


@pytest.fixture(autouse=True)
def clear_route_caches():
    """Prevent TTL caches from leaking state between tests."""
    import server.routes.setup as setup_module
    import server.routes.prompts as prompts_module
    import server.routes.evaluate as evaluate_module

    setup_module._catalog_cache = None
    setup_module._schema_cache = {}
    setup_module._warehouse_cache = None
    prompts_module._PROMPTS_CACHE.invalidate()
    evaluate_module._EXPERIMENT_PROMPTS_CACHE.invalidate()
    yield
    setup_module._catalog_cache = None
    setup_module._schema_cache = {}
    setup_module._warehouse_cache = None
    prompts_module._PROMPTS_CACHE.invalidate()
    evaluate_module._EXPERIMENT_PROMPTS_CACHE.invalidate()


def run_eval_job(client, payload: dict) -> dict:
    """POST /api/eval/run and return final job status (TestClient runs background tasks)."""
    resp = client.post("/api/eval/run", json=payload)
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    return client.get(f"/api/eval/run/status?job_id={job_id}").json()
