"""
Pydantic request / response models for the Health Assistant API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────

class ChatRequest(BaseModel):
    """POST /chat — send a message to the agent."""
    user_message: str = Field(..., min_length=1, description="The user's message")
    user_id: str = Field(..., min_length=1, description="Unique user identifier (e.g. phone number)")
    session_id: str = Field(default="default", description="Conversation session identifier")


# ── Response Models ───────────────────────────────────────────

class ChatResponse(BaseModel):
    """Response from the agent after processing a message."""
    answer: str = Field(..., description="The agent's response to the user")
    route: str = Field(..., description="Primary route taken (crm, rag, web_search, direct)")
    routes: List[str] = Field(default_factory=list, description="All routes taken (multi-route)")
    action: Optional[str] = Field(None, description="CRM sub-action if route is crm")
    tool_output: str = Field("", description="Raw tool output for debugging")
    memory_context: str = Field("", description="Memory context used by the agent")
    latency_ms: int = Field(0, description="End-to-end processing time in milliseconds")

    
class ToolStatus(BaseModel):
    """Status of each optional tool."""
    crm: bool = False
    rag: bool = False
    web_search: bool = False


class HealthResponse(BaseModel):
    """GET /health — system liveness and readiness."""
    status: str = "ok"
    tools: ToolStatus = Field(default_factory=ToolStatus)