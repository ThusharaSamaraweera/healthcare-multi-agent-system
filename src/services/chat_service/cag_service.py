from loguru import logger
from typing import Any, Dict, List
import time

from services.chat_service.cag_cache import CAGCache
from services.chat_service.crag_service import CRAGService
from infrastructure.observability import observe, update_current_observation


class CAGService:
    """
    Cache-Augmented Generation backed by Corrective RAG.
    """

    def __init__(self, crag_service: CRAGService, cache: CAGCache):
        self.crag_service = crag_service
        self.cache = cache

    @observe(name="cag_generate")
    def generate(
        self,
        query: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate answer with CAG + CRAG pipeline.
        """
        start = time.time()

        if use_cache:
            cached = self.cache.get(query)
            if cached:
                latency_ms = int((time.time() - start) * 1000)
                logger.info(
                    "CAG cache HIT (score={:.3f}) for: {}",
                    cached.get("score", 0),
                    query[:60],
                )
                update_current_observation(
                    input=query,
                    output=cached["answer"][:500],
                    metadata={
                        "cache_hit": True,
                        "cache_score": cached.get("score", 0),
                        "latency_ms": latency_ms,
                    },
                )
                return {
                    "answer": cached["answer"],
                    "evidence_urls": cached.get("evidence_urls", []),
                    "cache_hit": True,
                    "cache_score": cached.get("score", 0),
                    "generation_time": 0.0,
                }

        # Cache miss -- run CRAG
        crag_result = self.crag_service.generate(query, verbose=False)

        answer = crag_result.get("answer", "")
        evidence_urls = crag_result.get("evidence_urls", [])

        result: Dict[str, Any] = {
            "answer": answer,
            "evidence_urls": evidence_urls,
            "cache_hit": False,
            "confidence_initial": crag_result.get("confidence_initial", 0),
            "confidence_final": crag_result.get("confidence_final", 0),
            "correction_applied": crag_result.get("correction_applied", False),
            "generation_time": crag_result.get("generation_time", 0),
            "num_docs": crag_result.get("docs_used", 0),
        }

        # Cache only grounded answers. A non-empty refusal or fallback message
        # without evidence must not become a future semantic-cache hit.
        if use_cache and answer.strip() and evidence_urls:
            self.cache.set(query, {"answer": answer, "evidence_urls": evidence_urls})
            logger.info("CAG cache MISS -> cached for: {}", query[:60])

        latency_ms = int((time.time() - start) * 1000)
        update_current_observation(
            input=query,
            output=answer[:500] if answer else "No results",
            metadata={
                "cache_hit": False,
                "latency_ms": latency_ms,
                "correction_applied": result["correction_applied"],
                "confidence_final": result["confidence_final"],
            },
        )

        return result

    def warm_cache(self, queries: List[Any]) -> int:
        """Pre-populate the cache from FAQ records or plain query strings."""
        cached_count = 0
        for item in queries:
            if isinstance(item, dict) and item.get("query") and item.get("answer"):
                self.cache.set(
                    item["query"],
                    {"answer": item["answer"], "evidence_urls": item.get("evidence_urls", [])},
                )
                cached_count += 1
            elif isinstance(item, str) and item.strip():
                result = self.generate(item, use_cache=True)
                if result.get("answer", "").strip() and result.get("evidence_urls"):
                    cached_count += 1
        return cached_count

    def cache_stats(self) -> Dict[str, Any]:
        return self.cache.stats()

    def clear_cache(self) -> None:
        self.cache.clear()
