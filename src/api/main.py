"""FastAPI application entry point.

Run with:  uvicorn src.api.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.agent.graph import get_models
from src.agent.memory import DuplicateMemoryError, MemoryNotFoundError
from src.api.routes import chat, memory, monitoring
from src.api.schemas import HealthResponse
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging import get_logger, setup_logging

CONFIG = load_config()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging(CONFIG["logging"]["level"])
    get_models()  # fail fast if the model / API key is misconfigured
    logger.info("Agent API ready (%s / %s)", CONFIG["model"]["provider"], CONFIG["model"]["name"])
    yield


def create_app() -> FastAPI:
    api = FastAPI(
        title="Agent API",
        version="1.0.0",
        description="HTTP API for the LangGraph tool-using agent.",
        lifespan=lifespan,
    )

    api.add_middleware(
        CORSMiddleware,
        allow_origins=CONFIG["api"]["cors_origins"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api.include_router(chat.router, prefix="/api")
    api.include_router(memory.router, prefix="/api")
    api.include_router(monitoring.observability_router, prefix="/api")
    api.include_router(monitoring.evaluation_router, prefix="/api")

    @api.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            provider=CONFIG["model"]["provider"],
            model=CONFIG["model"]["name"],
        )

    @api.exception_handler(MemoryNotFoundError)
    async def _not_found(_: Request, exc: MemoryNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @api.exception_handler(DuplicateMemoryError)
    async def _duplicate(_: Request, exc: DuplicateMemoryError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    # Serve the chat UI (frontend/) at "/" — mounted last so API routes win.
    frontend_dir = PROJECT_ROOT / "frontend"
    if frontend_dir.is_dir():
        api.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return api


app = create_app()
