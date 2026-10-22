from typing import List
from memory.schemas import ConversationTurn

class MemoryRecaller:
    def __init__(self, st_store):
        self.st_store = st_store

    def format_context(self, st_turns: List[ConversationTurn]) -> str:
        """Format recalled memories as context string."""
        lines = []
        if st_turns:
            lines.append("=== RECENT CONVERSATION ===")
            for turn in st_turns:
                role = turn.role.capitalize()
                lines.append(f"{role}: {turn.content}")
            lines.append("")
        return "\n".join(lines)