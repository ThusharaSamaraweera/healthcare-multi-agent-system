from typing import Any, Optional, Dict
from loguru import logger

from infrastructure.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD, CRAG_EXPANDED_K, CRAG_CONFIDENCE_THRESHOLD
from infrastructure.observability import observe
from infrastructure.utils import format_docs


class RAGTool:
    def __init__(self, embedder: Any, llm: Optional[Any] = None): 
        from services.chat_service.cag_service import CAGService
        from services.chat_service.crag_service import CRAGService
        from services.chat_service.rag_service import QdrantRetriever
        from services.chat_service.cag_cache import CAGCache

        self.embedder = embedder
        self.llm = llm

        self._cag_service: Optional[CAGService] = None
        self._cache = CAGCache(embedder=embedder)

        self.retriever = QdrantRetriever(
            embedder=embedder,
            top_k=TOP_K_RESULTS,
            score_threshold=SIMILARITY_THRESHOLD,
        )

        if llm is not None:
            crag_service = CRAGService(
                retriever=self.retriever,
                llm=llm,
                initial_k=TOP_K_RESULTS,
                expanded_k=CRAG_EXPANDED_K,
            )

            self._cag_service = CAGService(
                crag_service=crag_service,
                cache=self._cache,
            )
            logger.info(
                "RAGTool initialised: CAG cache ({}) -> CRAG (k={}, expanded_k={}, threshold={:.2f})",
                self._cache,
                TOP_K_RESULTS,
                CRAG_EXPANDED_K,
                CRAG_CONFIDENCE_THRESHOLD,
            )
        else:
            self._crag_service = None
            logger.warning("LLM is not provided. CRAGService and CAGService will not be initialized.")


    @observe(name="rag_search")
    def search(self, query: str) -> str:
        if self._cag_service is not None:
            try:
                result = self._cag_service.generate(query)
                return result.get("answer", "")  or "No relevant information found in the internal knowledge base."
            except Exception as e:
                logger.error(f"Error occurred while generating answer: {e}")

        # Fallback to simple retrieval if CRAGService is not available or fails
        logger.info("CRAGService is not available or failed. Falling back to simple retrieval.")
        documents = self.retriever.invoke(query)
        return format_docs(documents)
    
    
    def dispatch(self, action: str, params: Dict[str, Any]) -> str:
        if action == "search":
            return self.search(**params)
        if action == "cache_stats":
            return str(self.cache_stats())
        if action == "clear_cache":
            self.clear_cache()
            return "CAG cache cleared."
        return f"Unknown RAG action: {action}"

    def warm_cache(self, queries: list[Any]) -> int:
        if self._cag_service is None:
            return 0
        return self._cag_service.warm_cache(queries)

    def cache_stats(self) -> Dict[str, Any]:
        return self._cache.stats()

    def clear_cache(self) -> None:
        self._cache.clear()
