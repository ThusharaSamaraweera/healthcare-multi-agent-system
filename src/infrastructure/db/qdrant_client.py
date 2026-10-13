from typing import Optional, Any, Dict
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from infrastructure.config import (QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, EMBEDDING_DIM)

_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """
    Return a singleton QdrantClient connected to Qdrant Cloud.

    Requires QDRANT_URL and QDRANT_API_KEY in .env.
    """
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    if not QDRANT_URL:
        raise RuntimeError(
            "QDRANT_URL is not set.  Add it to your .env file.\n"
            "Example: QDRANT_URL=https://xxxxx.us-east.aws.cloud.qdrant.io"
        )
    if not QDRANT_API_KEY:
        raise RuntimeError(
            "QDRANT_API_KEY is not set.  Add it to your .env file."
        )

    _qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30,
    )
    logger.info("Connected to Qdrant Cloud at {}", QDRANT_URL)
    return _qdrant_client



def ensure_collection(
    collection_name: str = QDRANT_COLLECTION_NAME,
    vector_size: int = EMBEDDING_DIM,
    distance: Distance = Distance.COSINE,
    on_disk: bool = True,
) -> None:
    """
    Create the Qdrant collection if it does not exist.
    """
    client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        logger.info("Collection '{}' already exists — skipping creation.", collection_name)
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance,
            on_disk=on_disk,
        ),
    )
    logger.info(
        "Created Qdrant collection '{}' (dim={}, distance={})",
        collection_name,
        vector_size,
        distance.name,
    )


def collection_exists(collection_name: str = QDRANT_COLLECTION_NAME) -> bool:
    """Check whether *collection_name* exists in Qdrant."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    return collection_name in existing

def collection_info(collection_name: str = QDRANT_COLLECTION_NAME) -> Dict[str, Any]:
    """Return collection stats (point count, vector size, etc.)."""
    client = get_qdrant_client()
    info = client.get_collection(collection_name)
    return {
        "name": collection_name,
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "vector_size": info.config.params.vectors.size,  # type: ignore[union-attr]
        "distance": info.config.params.vectors.distance.name,  # type: ignore[union-attr]
        "status": info.status.name,
    }

