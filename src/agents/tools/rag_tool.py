from typing import Any, Optional, Dict
from loguru import logger

from services.chat_service.rag_service import QdrantRetriever
from infrastructure.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD
from infrastructure.observability import observe
from infrastructure.utils import format_docs


class RAGTool:
    def __init__(self, embedder: Any, llm: Optional[Any] = None): 
        self.embedder = embedder
        self.llm = llm

        self.retriever = QdrantRetriever(
            embedder=embedder,
            top_k=TOP_K_RESULTS,
            score_threshold=SIMILARITY_THRESHOLD,
        )



    @observe(name="rag_search")
    def search(self, query: str) -> str:
        documents = self.retriever.invoke(query)
        return format_docs(documents)
    
    
    def dispatch(self, action: str, params: Dict[str, Any]) -> str:
        if action == "search":
            return self.search(**params)
        return f"Unknown RAG action: {action}"