from typing import Any, Optional, Dict
from loguru import logger

from services.chat_service.crag_service import CRAGService
from services.chat_service.rag_service import QdrantRetriever
from infrastructure.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD, CRAG_EXPANDED_K
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

        if llm is not None:
            self._crag_service = CRAGService(
                retriever=self.retriever,
                llm=llm,
                initial_k=TOP_K_RESULTS,
                expanded_k=CRAG_EXPANDED_K,
            )
        else:
            self._crag_service = None
            logger.warning("LLM is not provided. CRAGService will not be initialized.")


    @observe(name="rag_search")
    def search(self, query: str) -> str:
        if self._crag_service is not None:
            try:
                result = self._crag_service.generate(query)
                return result.get('answer', 'No relevant information found in the internal knowledge base.')
            except Exception as e:
                logger.error(f"Error occurred while generating answer: {e}")

        # Fallback to simple retrieval if CRAGService is not available or fails
        logger.info("CRAGService is not available or failed. Falling back to simple retrieval.")
        documents = self.retriever.invoke(query)
        return format_docs(documents)
    
    
    def dispatch(self, action: str, params: Dict[str, Any]) -> str:
        if action == "search":
            return self.search(**params)
        return f"Unknown RAG action: {action}"