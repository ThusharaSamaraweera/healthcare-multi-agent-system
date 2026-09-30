import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from agents.state import AgentState
from infrastructure.observability import observe, update_current_trace
from agents.router import QueryRouter
from agents.prompts.agent_prompts import build_direct_agent_prompt
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

@dataclass
class AgentResponse:
    """
    Complete agent response with metadata for the UI/Notebooks.
    """
    answer: str
    route: str = "direct"
    routes: List[str] = field(default_factory=list)   # all routes taken (multi-route)
    action: Optional[str] = None
    tool_output: str = ""
    memory_context: str = ""
    latency_ms: int = 0


class AgentOrchestrator:
    def __init__(self, llm_chat: Any, llm_router: Any) -> None:
        self.llm_chat = llm_chat
        self.llm_router = llm_router
        self.router = QueryRouter(llm=llm_router)

        # Build the graph
        self.graph = self._build_graph()

    
    # ── Node Implementations ────────────────────────────────────────

    @observe(name="node_recall")
    def recall_node(self, state: AgentState) -> Dict:
        pass

    @observe(name="node_supervisor")
    def supervisor_node(self, state: AgentState) -> Dict:

        user_message = state["messages"][-1].content
        
        # Router now returns MultiRouteDecision
        multi_decision = self.router.route(user_message)

        # Serialise all decisions for the state
        route_decisions = [
            {
                "route": d.route,
                "action": d.action,
                "params": d.params or {},
                "reasoning": d.reasoning,
            }
            for d in multi_decision.decisions
        ]

        if not route_decisions:
            route_decisions = [
                {
                    "route": "direct",
                    "action": None,
                    "params": {},
                    "reasoning": "No valid route returned; using direct.",
                }
            ]

        return {
            # Full list of decisions (multi-route)
            "route_decisions": route_decisions,
            # Primary decision (backward compat)
            "route_decision": route_decisions[0],
        }

    @observe(name="node_admin_agent")
    def admin_agent_node(self, state: AgentState) -> Dict:
        pass

    @observe(name="node_clinical_agent")
    def clinical_agent_node(self, state: AgentState) -> Dict:
        pass

    @observe(name="node_direct_agent")
    def direct_agent_node(self, state: AgentState) -> Dict:
        """Specialized Agent for greetings and general inquiries."""
        system_prompt = build_direct_agent_prompt()

        route_label = "direct"

        answer = self._generate_agent_response(state, system_prompt, "")

        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "tool_output": "",
            "agent_outputs": [{"route": route_label,"tool_output": "", "answer": answer}],
        }

    @observe(name="node_merge_responses")
    def merge_responses_node(self, state: AgentState) -> Dict:
        pass

    @observe(name="node_save_memory")
    def save_memory_node(self, state: AgentState) -> Dict:
        pass

    def _generate_agent_response(self, state: AgentState, system_prompt: str, tool_output: str, extra_context: str = "") -> str:

        user_message = state["messages"][-1].content
        memory_context = state.get("memory_context", "") + extra_context

        system_content = (
            f"{system_prompt}\n\n"
            f"=== MEMORY CONTEXT ===\n{memory_context}\n\n"
            f"=== TOOL OUTPUT ===\n{tool_output}"
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_message),
        ]

        response = self.llm_chat.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def supervisor_routing(self, state: AgentState) -> Union[str, List[str]]:
        route_map = {
            "crm": "admin",
            "rag": "clinical",
            "web_search": "direct",
            "direct": "direct",
        }

        decisions = state.get("route_decisions", [])
        if not decisions:
            return "direct"

        # Map routes to node names, deduplicate, preserve order
        node_names = []
        seen = set()
        for d in decisions:
            node = route_map.get(d.get("route", "direct"), "direct")
            if node not in seen:
                node_names.append(node)
                seen.add(node)

        # Single route → return string (no fan-out)
        # Multiple routes → return list (LangGraph fan-out)
        if len(node_names) == 1:
            return node_names[0]
        return node_names


    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("recall", self.recall_node)
        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("direct_agent", self.direct_agent_node)
        workflow.add_node("admin_agent", self.admin_agent_node)
        workflow.add_node("clinical_agent", self.clinical_agent_node)
        workflow.add_node("merge_responses", self.merge_responses_node)
        workflow.add_node("save_memory", self.save_memory_node)

        workflow.set_entry_point("supervisor")

        workflow.add_conditional_edges(
            "supervisor",
            self.supervisor_routing,
            {
                "direct": "direct_agent",
                "admin": "admin_agent",
                "clinical": "clinical_agent",
            },
        )

        workflow.add_edge("direct_agent", END)

        return workflow.compile()
    
    @observe(name="agent_chat")
    async def achat(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        operational_context: str = "",
    ) -> AgentResponse:
        t0 = time.time()

        update_current_trace(
            user_id=user_id,
            session_id=session_id,
            metadata={"entrypoint": "achat"},
        )

        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "user_id": user_id,
            "session_id": session_id,
            "operational_context": operational_context,
            "agent_outputs": [],
        }

        # Non-blocking graph execution
        final_state = await self.graph.ainvoke(initial_state)

        latency = int((time.time() - t0) * 1000)

        route_decisions = final_state.get("route_decisions", [])
        all_routes = [d.get("route", "direct") for d in route_decisions]
        primary = route_decisions[0] if route_decisions else {"route": "direct"}

        return AgentResponse(
            answer=final_state["final_answer"],
            route=primary.get("route", "direct"),
            routes=all_routes,
            action=primary.get("action"),
            tool_output=final_state.get("tool_output", ""),
            memory_context=final_state.get("memory_context", ""),
            latency_ms=latency
        )

def build_agent(enable_crm: bool = True, enable_rag: bool = True, enable_web: bool = True) -> AgentOrchestrator:
    """Builds the Multi-Agent Orchestrator."""
    from infrastructure.llm import get_chat_llm, get_router_llm

    llm_chat = get_chat_llm(temperature=0)
    llm_router = get_router_llm(temperature=0)


    return AgentOrchestrator(
        llm_chat=llm_chat,
        llm_router=llm_router,
    )
