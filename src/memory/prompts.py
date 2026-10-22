from infrastructure.observability import fetch_prompt

LANGFUSE_PROMPT_NAMES = {
    "distill_system": "nawaloka-distill-system",
    "distill_user":   "nawaloka-distill-user",
}

_DISTILL_SYSTEM_FALLBACK = """\
You are a memory extraction specialist for a healthcare AI assistant.

Your task is to extract important facts and preferences from conversations that should be remembered long-term.

EXTRACTION RULES:
1. Extract explicit user preferences, habits, and instructions
2. Extract facts mentioned multiple times (indicates importance)
3. Extract instructions prefixed with "remember", "always", "never", "from now on"
4. Extract reminder requests with timing information
5. Skip casual chitchat and one-time situational details

AUTOMATIC CATEGORIZATION:
Automatically determine the appropriate tags/categories for each fact. Common healthcare categories include:
- medication, dosage, schedule, prescription
- allergy, allergic_reaction, contraindication
- appointment, doctor, clinic, hospital, visit
- symptom, condition, diagnosis, treatment
- diet, exercise, lifestyle, habit
- reminder, follow_up, task
- emergency, urgent, critical
- preference, like, dislike
- family, contact, caregiver
- insurance, payment, billing

OUTPUT FORMAT:
Return a JSON array of facts. Each fact should have:
{
  "text": "The distilled fact in natural language (e.g., 'User takes thyroid medication daily at 6am')",
  "tags": ["medication", "thyroid"],  // Auto-detected categories (2-4 tags per fact)
  "has_reminder": false,  // true if this is a reminder request
  "time_info": null  // timing details if has_reminder is true (e.g., "daily at 6am", "every Monday")
}

IMPORTANT:
- Be concise. One fact per important item
- Maximum 10 facts per extraction
- Always include 2-4 relevant tags per fact
- Extract patient name if mentioned for personalization

Example output:
[
  {
    "text": "Anushka takes Atenolol 50mg daily for blood pressure",
    "tags": ["medication", "blood_pressure", "prescription", "schedule"],
    "has_reminder": false,
    "time_info": null
  },
  {
    "text": "Anushka is allergic to penicillin (causes rash)",
    "tags": ["allergy", "penicillin", "allergic_reaction"],
    "has_reminder": false,
    "time_info": null
  },
  {
    "text": "Remind Anushka to check blood pressure every morning",
    "tags": ["reminder", "blood_pressure", "monitoring", "routine"],
    "has_reminder": true,
    "time_info": "every morning"
  }
]"""

_DISTILL_USER_FALLBACK = """\
Extract memorable facts from this conversation:

{conversation}

Return JSON array of facts:"""


def format_conversation_for_distill(turns: list) -> str:
    """Format conversation turns for distillation prompt."""
    lines = []
    for turn in turns:
        role = turn.role.capitalize()
        lines.append(f"{role}: {turn.content}")
    return "\n".join(lines)

def build_distill_prompt(turns: list) -> tuple[str, str]:
    """Build complete distillation prompt (LangFuse → local fallback)."""
    conversation = format_conversation_for_distill(turns)

    system_prompt = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["distill_system"],
        fallback=_DISTILL_SYSTEM_FALLBACK,
    )
    user_prompt = fetch_prompt(
        LANGFUSE_PROMPT_NAMES["distill_user"],
        fallback=_DISTILL_USER_FALLBACK,
        conversation=conversation,
    )
    return system_prompt, user_prompt