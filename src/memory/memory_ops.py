from typing import List, Tuple
from loguru import logger
import tiktoken

from memory.schemas import ConversationTurn, MemoryFact
from infrastructure.observability import observe, update_current_observation
class MemoryRecaller:
    def __init__(self, st_store, lt_store):
        self.st_store = st_store
        self.lt_store = lt_store
        self.tokenizer = tiktoken.get_encoding("cl100k_base")


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
    
    @observe(name="memory_recall_inner")
    def recall(
        self, user_id: str, session_id: str, query: str,
        k_st: int = 6, k_lt: int = 5, max_tokens: int = 500,
    ) -> Tuple[List[ConversationTurn], List[MemoryFact]]:
        """Recall relevant memories (traced via LangFuse)."""
        from infrastructure.config import LT_SIM_THRESHOLD

        st_turns = self.st_store.recent(user_id, session_id, k_st)
        lt_facts = self.lt_store.query(
            user_id=user_id, query_text=query, k=k_lt, threshold=LT_SIM_THRESHOLD,
        )

        st_turns_filtered, lt_facts_filtered = self._budget_tokens(st_turns, lt_facts, max_tokens)

        update_current_observation(
            input=query,
            metadata={
                "st_raw": len(st_turns),
                "st_filtered": len(st_turns_filtered),
                "lt_raw": len(lt_facts),
                "lt_filtered": len(lt_facts_filtered),
            },
        )

        logger.info(
            f"Recalled {len(st_turns_filtered)} ST turns, "
            f"{len(lt_facts_filtered)} LT facts for user {user_id}"
        )
        return st_turns_filtered, lt_facts_filtered

    def _budget_tokens(
        self, st_turns: List[ConversationTurn], lt_facts: List[MemoryFact], max_tokens: int,
    ) -> Tuple[List[ConversationTurn], List[MemoryFact]]:
        """Apply token budget to memories."""
        st_budget = int(max_tokens * 0.6)
        lt_budget = int(max_tokens * 0.4)

        st_filtered = []
        st_tokens = 0
        for turn in reversed(st_turns):
            turn_tokens = self.count_tokens(turn.content)
            if st_tokens + turn_tokens <= st_budget:
                st_filtered.insert(0, turn)
                st_tokens += turn_tokens
            if len(st_filtered) >= 4:
                break

        lt_filtered = []
        lt_tokens = 0
        for fact in lt_facts:
            fact_tokens = self.count_tokens(fact.text)
            if lt_tokens + fact_tokens <= lt_budget:
                lt_filtered.append(fact)
                lt_tokens += fact_tokens
            if len(lt_filtered) >= 3:
                break

        logger.debug(f"Token budget: ST={st_tokens}/{st_budget}, LT={lt_tokens}/{lt_budget}")
        return st_filtered, lt_filtered

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
