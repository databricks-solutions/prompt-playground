"""Prompt Playground - FastAPI entry point."""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add src to path so server module is importable
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

from server.routes.prompts import router as prompts_router
from server.routes.models import router as models_router
from server.routes.run import router as run_router
from server.routes.evaluate import router as evaluate_router
from server.routes.config import router as config_router
from server.routes.setup import router as setup_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Prompt Playground starting up...")
    # Enable OpenAI autologging so every call_model() invocation automatically
    # produces MLflow traces with token usage, latencies, and structured spans.
    import mlflow
    mlflow.openai.autolog()
    yield
    logger.info("Prompt Playground shutting down...")


app = FastAPI(
    title="Prompt Playground",
    description="Testing playground for MLflow Prompt Registry",
    lifespan=lifespan,
)

# Register API routers
app.include_router(prompts_router)
app.include_router(models_router)
app.include_router(run_router)
app.include_router(evaluate_router)
app.include_router(config_router)
app.include_router(setup_router)


# Health check
@app.get("/api/health")
async def health():
    return {"status": "healthy", "app": "prompt-playground"}


# Serve React frontend
frontend_dist = Path(__file__).parent / "frontend" / "dist"

if frontend_dist.exists():
    # Serve static assets (JS, CSS, images)
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    # Serve other static files at root level (favicon, etc.)
    @app.get("/vite.svg")
    async def serve_vite_svg():
        svg_path = frontend_dist / "vite.svg"
        if svg_path.exists():
            return FileResponse(str(svg_path))
        return {"error": "not found"}

    # SPA fallback - serve index.html for all non-API routes
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            return {"error": "Not found"}, 404
        return FileResponse(str(frontend_dist / "index.html"))
else:
    @app.get("/")
    async def no_frontend():
        return {
            "message": "Prompt Playground API is running. Frontend not built yet.",
            "hint": "Run 'cd src/frontend && npm run build' to build the UI.",
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
