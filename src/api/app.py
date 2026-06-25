"""FastAPI application — REST API + WebSocket for Personal Agent."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import setup_logging
from src.api.routes import tasks, ws

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging("personal_agent_api")

    app = FastAPI(
        title="Personal Agent API",
        description="Multi-Agent Workflow Automation — v0.2.0",
        version="0.2.0",
    )

    # CORS: allow local dev origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",   # Streamlit
            "http://localhost:3000",   # React/Vue dev
            "http://127.0.0.1:8501",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(tasks.router, prefix="/api", tags=["tasks"])
    app.include_router(ws.router, tags=["websocket"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.2.0"}

    logger.info("FastAPI application created")
    return app


app = create_app()
