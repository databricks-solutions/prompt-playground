"""In-memory background evaluation job store (single app instance)."""

import logging
import threading
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def _default_job() -> dict[str, Any]:
    return {
        "status": "pending",
        "result": None,
        "error": None,
        "progress": 0,
        "total": 0,
        "message": "",
    }


async def create_job() -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = _default_job()
    return job_id


async def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        row = _jobs.get(job_id)
        return dict(row) if row else None


async def update_job(job_id: str, **updates: Any) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)


async def run_job(
    job_id: str,
    runner: Callable[[str], Awaitable[dict]],
) -> None:
    """Execute an async eval runner and store the result on the job."""
    await update_job(job_id, status="running", message="Starting…")
    try:
        result = await runner(job_id)
        await update_job(
            job_id,
            status="completed",
            result=result,
            progress=100,
            message="Complete",
        )
    except Exception as e:
        logger.exception("Eval job %s failed", job_id)
        await update_job(job_id, status="failed", error=str(e), message="Failed")


def prune_old_jobs(max_jobs: int = 50) -> None:
    with _lock:
        if len(_jobs) <= max_jobs:
            return
        excess = len(_jobs) - max_jobs
        for key in list(_jobs.keys())[:excess]:
            _jobs.pop(key, None)
