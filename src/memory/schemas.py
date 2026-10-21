from dataclasses import dataclass
from typing import Any, Dict, Literal

@dataclass
class ConversationTurn:
    """
    A single turn in a conversation (user or assistant message).
    """
    user_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    ts: float  # epoch seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "ts": self.ts,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            session_id=data["session_id"],
            role=data["role"],
            content=data["content"],
            ts=data["ts"],
        )