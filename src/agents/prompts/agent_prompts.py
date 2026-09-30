from infrastructure.observability import fetch_prompt


LANGFUSE_PROMPT_NAMES = {
    "router_system":      "nawaloka-router-system",
}

_ROUTER_SYSTEM_FALLBACK = """\
You are a query router for a healthcare AI system.

Given a user message AND memory context, classify the intent into
one.

ROUTES:
  direct     — Greeting, chitchat, follow-up, or answerable from memory alone.

OUTPUT FORMAT (strict JSON, no markdown fences):
{
  "routes": [
    {
      "route": "<direct>",
      "confidence": <0.0-1.0>,
      "reasoning": "<one-sentence explanation>",
      "params": { empty {}> }
    }
  ]
}

PARAMETER EXTRACTION RULES:
• For direct          → params = {}.
"""

_ROUTER_USER_FALLBACK = """\
MEMORY CONTEXT:
{memory_context}

USER MESSAGE:
{user_message}

Classify and extract (JSON):"""


_AGENT_SYSTEM_FALLBACK = """\
You are **Nawaloka Health Assistant**, a friendly and knowledgeable AI assistant
for Nawaloka Hospitals, Sri Lanka.

Your capabilities:
• Answer questions about Nawaloka services

Communication rules:
1. Be warm, professional, and concise.
2. If unsure, say so rather than guessing.
3. Use the patient's name when available.
4. Respond in the same language as the patient (Sinhala, Tamil, or English).
"""

_DIRECT_AGENT_FALLBACK = """\
You are the Nawaloka Hospital Concierge.
You handle greetings, general information, and help patients find the right department.
Style: Warm, welcoming, and hospitable.

"""


def build_router_prompt(
    user_message: str,
    memory_context: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the router call."""
    system_prompt = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["router_system"],
        fallback=_ROUTER_SYSTEM_FALLBACK,
    )
    user_prompt = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["router_user"],
        fallback=_ROUTER_USER_FALLBACK,
        memory_context=memory_context or "(no memory context)",
        user_message=user_message,
    )
    return system_prompt, user_prompt


def build_direct_agent_prompt() -> str:
    """Return the system prompt for the Direct Agent (concierge/web search)."""
    base = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["agent_system"],
        fallback=_AGENT_SYSTEM_FALLBACK,
    )
    persona = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["direct_agent"],
        fallback=_DIRECT_AGENT_FALLBACK,
    )
    return base + "\n\n" + persona