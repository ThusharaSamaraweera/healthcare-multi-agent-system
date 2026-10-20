from typing import Any, Optional, Dict
from loguru import logger
import time
import json
import uuid

class CAGCache:
    """
    Qdrant-backed semantic cache for pre-computed RAG responses.
    """

    def __init__(
        self,
        embedder: Any,
        collection_name: Optional[str] = None,
        dim: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Initialise the semantic CAG cache.
        """

        from infrastructure.config import (
            CAG_SIMILARITY_THRESHOLD,
            CAG_CACHE_TTL,
            EMBEDDING_DIM,
            CAG_COLLECTION_NAME,
        )

        self.embedder = embedder
        self.collection_name = collection_name or CAG_COLLECTION_NAME
        self.dim = dim or EMBEDDING_DIM
        self.similarity_threshold = similarity_threshold or CAG_SIMILARITY_THRESHOLD
        self.ttl_seconds = ttl_seconds or CAG_CACHE_TTL

        # Ensure the Qdrant collection exists
        self._available = False
        try:
            from infrastructure.db.qdrant_client import (
                get_qdrant_client,
                collection_exists,
            )

            self._client = get_qdrant_client()
            if not collection_exists(self.collection_name):
                self._create_collection()
            self._available = True
            logger.info(
                "✓ CAG cache ready (Qdrant collection='{}', dim={}, threshold={:.2f})",
                self.collection_name,
                self.dim,
                self.similarity_threshold,
            )
        except Exception as exc:
            logger.warning(
                "CAG cache DISABLED — Qdrant unavailable: {}. "
                "All lookups will miss; every query runs full RAG.",
                exc,
            )

    def _create_collection(self) -> None:
        """Create the Qdrant collection for CAG cache."""
        from qdrant_client.http.models import Distance, VectorParams

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dim,
                distance=Distance.COSINE,
                on_disk=False,  # keep in-memory for speed
            ),
        )
        logger.info(
            "Created CAG cache collection '{}' (dim={}, COSINE)",
            self.collection_name,
            self.dim,
        )


    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Semantic cache lookup via KNN-1 search.
        """
        if not self._available:
            return None

        # Embed query
        try:
            query_vec = self.embedder.embed_query(query)
        except Exception as exc:
            logger.warning("CAG embed failed: {}", exc)
            return None

        # KNN-1 search
        try:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=1,
                score_threshold=self.similarity_threshold,
            )
        except Exception as exc:
            logger.warning("CAG cache GET error: {}", exc)
            return None

        if not response.points:
            return None

        hit = response.points[0]
        similarity = hit.score
        payload = hit.payload or {}

        # TTL filtering: skip entries older than ttl_seconds
        if self.ttl_seconds and self.ttl_seconds > 0:
            entry_ts = payload.get("ts", 0)
            if entry_ts and (time.time() - float(entry_ts)) > self.ttl_seconds:
                logger.debug("CAG cache expired: '{}' → deleting point {}", query[:50], hit.id)
                try:
                    self._client.delete(
                        collection_name=self.collection_name,
                        points_selector=[hit.id],
                    )
                except Exception as exc:
                    logger.warning("Failed to delete expired CAG entry: {}", exc)
                return None

        cached_query = payload.get("query", "")
        logger.info(
            "CAG cache HIT (sim={:.3f}): '{}' → matched '{}'",
            similarity,
            query[:50],
            cached_query[:50],
        )

        evidence_raw = payload.get("evidence_urls", "[]")
        try:
            evidence_urls = json.loads(evidence_raw) if isinstance(evidence_raw, str) else evidence_raw
        except (json.JSONDecodeError, TypeError):
            evidence_urls = []

        return {
            "query": cached_query,
            "answer": payload.get("answer", ""),
            "evidence_urls": evidence_urls,
            "ts": float(payload.get("ts", 0)),
            "score": similarity,
        }

    def set(self, query: str, response: Dict[str, Any]) -> None:
        """
        Cache a response, indexed by the query's embedding.

        """
        if not self._available:
            return

        # Embed query
        try:
            query_vec = self.embedder.embed_query(query)
        except Exception as exc:
            logger.warning("CAG embed failed on SET: {}", exc)
            return

        from qdrant_client.http.models import PointStruct

        # First, search and destroy any existing entries for this exact same semantic query
        # This prevents the cache from filling up with duplicates over time when TTL expires
        try:
            existing = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=10, # Check for a few in case duplicates already exist
                score_threshold=0.99, # practically identical queries
            )
            if existing.points:
                existing_ids = [p.id for p in existing.points]
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=existing_ids,
                )
                logger.debug("CAG cache replaced {} existing duplicates for '{}'", len(existing_ids), query[:50])
        except Exception as exc:
            logger.warning("CAG cache failed to clean existing duplicates: {}", exc)

        point_id = str(uuid.uuid4())
        payload = {
            "query": query,
            "answer": response.get("answer", ""),
            "evidence_urls": json.dumps(response.get("evidence_urls", [])),
            "ts": time.time(),
        }

        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=point_id, vector=query_vec, payload=payload)],
            )
            logger.debug("CAG cache SET: '{}' → point={}", query[:60], point_id)
        except Exception as exc:
            logger.warning("CAG cache SET error: {}", exc)

    def clear(self) -> None:
        """Delete all cached FAQ/answer points while keeping the collection."""
        if not self._available:
            return
        try:
            from qdrant_client.http.models import Filter

            self._client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(),
            )
            logger.info("Cleared CAG cache collection '{}'", self.collection_name)
        except Exception as exc:
            logger.warning("CAG cache CLEAR error: {}", exc)

    def stats(self) -> Dict[str, Any]:
        """Return basic cache collection information."""
        if not self._available:
            return {"available": False, "collection": self.collection_name}
        try:
            info = self._client.get_collection(self.collection_name)
            return {
                "available": True,
                "collection": self.collection_name,
                "points_count": info.points_count,
            }
        except Exception as exc:
            logger.warning("CAG cache STATS error: {}", exc)
            return {"available": False, "collection": self.collection_name}
