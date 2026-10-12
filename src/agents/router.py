from loguru import logger
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from infrastructure.observability import observe, update_current_observation
from agents.prompts.agent_prompts import build_router_prompt

# Valid routes
VALID_ROUTES = {"direct", "crm"}

# Maximum routes per query (safety cap)
MAX_ROUTES = 3


@dataclass
class RouteDecision:
    """
    A single routing decision for one intent.
    """

    route: str = "direct"
    confidence: float = 0.0
    reasoning: str = ""
    action: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiRouteDecision:
    """
    Container for one or more RouteDecision objects.
    """

    decisions: List[RouteDecision] = field(default_factory=list)

    @property
    def is_multi_route(self) -> bool:
        return len(self.decisions) > 1

    @property
    def primary(self) -> RouteDecision:
        """First (or only) decision — backward compatibility."""
        return self.decisions[0] if self.decisions else RouteDecision()


class QueryRouter:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    @observe(name="router", as_type="generation")
    def route(self, user_message: str, memory_context: str = "") -> MultiRouteDecision:

        system_prompt, user_prompt = build_router_prompt(
            user_message=user_message,
            memory_context=memory_context,
        )

        update_current_observation(
            input=user_prompt[:1000],
            model=self._model_name(),
        )

        try:
            response = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Extract token usage if available
            usage = {}
            if hasattr(response, "response_metadata"):
                meta = response.response_metadata or {}
                token_usage = meta.get("token_usage") or meta.get("usage", {})
                if token_usage:
                    usage = {
                        "input": token_usage.get("prompt_tokens", 0),
                        "output": token_usage.get("completion_tokens", 0),
                        "total": token_usage.get("total_tokens", 0),
                    }

            update_current_observation(
                output=content[:500],
                usage=usage if usage else None,
            )

        except Exception as exc:
            logger.error("Router LLM call failed: {}", exc)
            return MultiRouteDecision(
                decisions=[
                    RouteDecision(
                        route="direct",
                        confidence=0.0,
                        reasoning=f"Router LLM error: {exc}",
                    )
                ]
            )

        return self._parse_response(content)
    
    def _model_name(self) -> str:
        if hasattr(self.llm, "model_name"):
            return self.llm.model_name
        if hasattr(self.llm, "model"):
            return self.llm.model
        return "unknown"


    def _parse_response(self, raw: str) -> MultiRouteDecision:
        
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        # Locate JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            logger.warning("Router output is not JSON; falling back to direct.")
            return MultiRouteDecision(decisions=[
                RouteDecision(route="direct", confidence=0.0,
                                reasoning="Failed to parse router output as JSON.")
            ])

        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            logger.warning("Router JSON parse error: {}", exc)
            return MultiRouteDecision(decisions=[
                RouteDecision(route="direct", confidence=0.0,
                                reasoning=f"JSON parse error: {exc}")
            ])

        # ── Normalise to a list of route dicts ──────────────────
        if "routes" in data and isinstance(data["routes"], list):
            # New multi-route format
            route_dicts = data["routes"][:MAX_ROUTES]
        else:
            # Old single-route format — wrap in list
            route_dicts = [data]

        # ── Build RouteDecision objects ──────────────────────────
        decisions: List[RouteDecision] = []
        seen_routes: set = set()
        
        for rd in route_dicts:
            route = rd.get("route", "direct")
            if route not in VALID_ROUTES:
                logger.warning("Invalid route '{}'; skipping.", route)
                continue
            # Deduplicate (same route appearing twice)
            if route in seen_routes:
                continue
            seen_routes.add(route)

            action = rd.get("action")
            decisions.append(RouteDecision(
                route=route,
                confidence=float(rd.get("confidence", 0.5)),
                reasoning=rd.get("reasoning", ""),
                action=action if route == "crm" else None,
                params=rd.get("params", {}),
            ))

        # Fallback if nothing valid was parsed
        if not decisions:
            decisions = [RouteDecision(route="direct", confidence=0.0,
                                        reasoning="No valid routes parsed.")]

        return MultiRouteDecision(decisions=decisions)
