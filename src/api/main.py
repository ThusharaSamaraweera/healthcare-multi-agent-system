import asyncio
import json
import time
import sys
import os
import re

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from api.schemas import (
    ChatRequest, ChatResponse,
    HealthResponse, ToolStatus,
)

# ── Lifespan — build agent once at startup ────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Nawaloka Health Assistant API...")

    from agents.orchestrator import build_agent

    agent = await asyncio.to_thread(
        build_agent
    )

    app.state.agent = agent
    logger.success("Agent ready — API is live")

    yield  # app runs here

    logger.info("Shutting down...")


# ── App ───────────────────────────────────────────────────────

app = FastAPI(
    title="Nawaloka Health Assistant API",
    description=(
        "Multi-Agent LangGraph system for Nawaloka Hospitals.\n\n"
        "- **POST /chat** — send a message, get an AI response\n"
        "- **GET /health** — liveness check\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Helper ────────────────────────────────────────────────────

def _get_agent():
    """Retrieve the agent from app state, or raise 503."""
    agent = getattr(app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet")
    return agent

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness and readiness check — reports which tools are enabled."""
    agent = getattr(app.state, "agent", None)

    if agent is None:
        return HealthResponse(status="starting", tools=ToolStatus())

    return HealthResponse(
        status="ok"
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    agent = _get_agent()

    try:
        resp = await agent.achat(
            user_message=request.user_message,
            user_id=request.user_id,
            session_id=request.session_id,
        )

        return ChatResponse(
            answer=resp.answer,
            route=resp.route,
            routes=resp.routes,
            action=resp.action,
            tool_output=resp.tool_output,
            memory_context=resp.memory_context,
            latency_ms=resp.latency_ms,
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

