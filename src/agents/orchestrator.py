import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from loguru import logger

from agents.state import AgentState
from infrastructure.observability import observe, update_current_trace
from agents.router import QueryRouter
from agents.prompts.agent_prompts import (build_direct_agent_prompt, build_admin_agent_prompt, build_clinical_agent_prompt)
from agents.tools.crm_tool import CRMTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from memory.schemas import ConversationTurn
@dataclass
class AgentResponse:
    """
    Complete agent response with metadata for the UI.
    """
    answer: str
    route: str = "direct"
    routes: List[str] = field(default_factory=list)   # all routes taken (multi-route)
    action: Optional[str] = None
    tool_output: str = ""
    memory_context: str = ""
    latency_ms: int = 0


class AgentOrchestrator:
    def __init__(
        self,
        llm_chat: Any,
        llm_router: Any,
        st_store: Optional[Any] = None,
        recaller: Optional[Any] = None,
        crm_tool: Optional[Any] = None,
        rag_tool: Optional[Any] = None,
    ) -> None:
        self.llm_chat = llm_chat
        self.llm_router = llm_router

        self.crm_tool = crm_tool
        self.rag_tool = rag_tool

        self.recaller = recaller
        self.st_store = st_store

        self.router = QueryRouter(llm=llm_router)
        # Build the graph
        self.graph = self._build_graph()

    
    # ── Node Implementations ────────────────────────────────────────

    @observe(name="node_recall")
    def recall_node(self, state: AgentState) -> Dict:
        """Reads conversation history and long-term facts into the state."""

        user_id = state["user_id"]
        session_id = state["session_id"]
        user_message = state["messages"][-1].content

        try:
            st_turns, lt_facts = self.recaller.recall(
                user_id=user_id,
                session_id=session_id,
                query=user_message
            )
            memory_context = self.recaller.format_context(st_turns)
            semantic_facts = [f.to_dict() if hasattr(f, 'to_dict') else vars(f) for f in lt_facts]
 
            return {
                "memory_context": memory_context,
                "semantic_facts": semantic_facts
            }
        except Exception as e:
            logger.warning(f"Recall node failed: {e}")
            return {"memory_context": "(memory offline)"}
        
    @observe(name="node_supervisor")
    def supervisor_node(self, state: AgentState) -> Dict:

        user_message = state["messages"][-1].content
        memory_context = state.get("memory_context", "")
        
        # Augment context with LT facts for the Router
        facts = state.get("semantic_facts", [])
        if facts:
            memory_context += "\n=== LONG-TERM FACTS ===\n"
            for f in facts:
                memory_context += f"- {f.get('text', '')}\n"

        # Router now returns MultiRouteDecision
        multi_decision = self.router.route(user_message, memory_context)

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
        """Specialized Agent for CRM and Scheduling."""
        # Find the CRM-specific decision from route_decisions
        decisions = state.get("route_decisions", [])
        crm_decision = next(
            (d for d in decisions if d.get("route") == "crm"),
            state.get("route_decision", {})
        )
        action = crm_decision.get("action", "lookup_patient")
        params = dict(crm_decision.get("params", {}))

        system_prompt = build_admin_agent_prompt()

        if not self.crm_tool:
            tool_output = "CRM Tool unavailable."
        else:
            if action in {"create_booking"}:
                patient_id = self.crm_tool.patient_id_for_external_user(state["user_id"])
                if patient_id is None:
                    tool_output = "No patient record found for the current user."
                else:
                    params["patient_id"] = patient_id
                    tool_output = self.crm_tool.dispatch(action, params)
            else:
                tool_output = self.crm_tool.dispatch(action, params)

        answer = self._generate_agent_response(state, system_prompt, tool_output)

        return {
            "messages": [AIMessage(content=answer)],
            "tool_output": tool_output,
            "final_answer": answer,
            "agent_outputs": [{"route": "crm", "tool_output": tool_output, "answer": answer}],
        }

    @observe(name="node_clinical_agent")
    def clinical_agent_node(self, state: AgentState) -> Dict:
        """Specialized Agent for Medical Info and Patient History."""
        # Find the RAG-specific decision from route_decisions
        decisions = state.get("route_decisions", [])
        rag_decision = next(
            (d for d in decisions if d.get("route") == "rag"),
            state.get("route_decision", {})
        )
        params = rag_decision.get("params", {})
        query = params.get("query", state["messages"][-1].content)

        system_prompt = build_clinical_agent_prompt()

        # Inject semantic facts for clinical context
        facts = state.get("semantic_facts", [])
        kb_context = ""
        if facts:
            kb_context += "\n=== PATIENT CLINICAL HISTORY ===\n"
            for f in facts:
                kb_context += f"- {f.get('text', '')}\n"

        if not self.rag_tool:
            tool_output = "RAG Tool unavailable."
        else:
            tool_output = self.rag_tool.dispatch("search", {"query": query})

        answer = self._generate_agent_response(state, system_prompt, tool_output, extra_context=kb_context)

        return {
            "messages": [AIMessage(content=answer)],
            "tool_output": tool_output,
            "final_answer": answer,
            "agent_outputs": [{"route": "rag", "tool_output": tool_output, "answer": answer}],
        }

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
    def save_memory(self, state: AgentState) -> Dict:
        """Saves messages to short-term memory."""
        user_message = state["messages"][0].content
        answer = state["final_answer"]
        user_id = state["user_id"]
        session_id = state["session_id"]

        # Store ST turns
        now = time.time()
        self.st_store.add(user_id, session_id, ConversationTurn(user_id=user_id, session_id=session_id, role="user", content=user_message, ts=now))
        self.st_store.add(user_id, session_id, ConversationTurn(user_id=user_id, session_id=session_id, role="assistant", content=answer, ts=now))

        return {"should_distill": False}


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
            "direct": "direct",
            "rag": "clinical"
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
        workflow.add_node("save_memory", self.save_memory)

        workflow.set_entry_point("recall")
        workflow.add_edge("recall", "supervisor")

        workflow.add_conditional_edges(
            "supervisor",
            self.supervisor_routing,
            {
                "direct": "direct_agent",
                "admin": "admin_agent",
                "clinical": "clinical_agent",
            },
        )

        workflow.add_edge("direct_agent", "save_memory")
        workflow.add_edge("admin_agent", "save_memory")
        workflow.add_edge("clinical_agent", "save_memory")

        workflow.add_edge("save_memory", END)

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

        normalized_user_id = CRMTool._normalize_external_user_id(user_id)
        canonical_user_id = normalized_user_id or user_id.strip()

        update_current_trace(
            user_id=canonical_user_id,
            session_id=session_id,
            metadata={"entrypoint": "achat"},
        )

        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "user_id": canonical_user_id,
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
    from infrastructure.llm import get_chat_llm, get_router_llm, get_default_embeddings
    from memory.st_store import ShortTermMemoryStore
    from memory.lt_store import LongTermMemoryStore
    from memory.memory_ops import MemoryRecaller

    llm_chat = get_chat_llm(temperature=0)
    llm_router = get_router_llm(temperature=0)
    embedder = get_default_embeddings()

    st_store = ShortTermMemoryStore()
    lt_store = LongTermMemoryStore(embedder)
    recaller = MemoryRecaller(st_store=st_store, lt_store=lt_store)

    crm_tool = None
    if enable_crm:
        try:
            from agents.tools import CRMTool
            crm_tool = CRMTool()
            logger.info("CRM tool initialised")
        except Exception as e:
            logger.warning(f"CRM tool unavailable: {e}")

    rag_tool = None
    if enable_rag:
        try:
            from agents.tools import RAGTool
            rag_tool = RAGTool(embedder=embedder, llm=llm_chat)
            logger.info("RAG tool initialised")
        except Exception as e:
            logger.warning(f"RAG tool unavailable: {e}")

    return AgentOrchestrator(
        llm_chat=llm_chat,
        llm_router=llm_router,
        crm_tool=crm_tool,
        rag_tool=rag_tool,
        st_store=st_store,
        recaller=recaller
    )
