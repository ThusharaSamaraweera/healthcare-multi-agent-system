import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    user_id: str
    session_id: str

    memory_context: Optional[str]
    operational_context: Optional[str]
    semantic_facts: Optional[list[dict]]

    route_decision: Optional[dict]
    route_decisions: Optional[list[dict]]

    tool_output: Optional[str]    # Raw tool output 
    final_answer: Optional[str]   # User-facing response

    agent_outputs: Annotated[list[dict], operator.add]

    should_distill: Optional[bool]  # True → trigger LT fact extraction
