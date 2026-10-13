from typing import Any, Dict, List
from pathlib import Path
from loguru import logger
import sys
import time

from infrastructure.config import KB_DIR, EMBEDDING_BATCH_SIZE
from infrastructure.llm import get_default_embeddings
from services.ingest_service import parent_child_chunk
from infrastructure.db.qdrant_client import (
    ensure_collection, upsert_chunks, collection_info)


def load_kb_docs(kb_dir: Path | None = None) -> List[Dict[str, Any]]:
    """Load internal knowledge-base markdown documents."""
    kb_dir = Path(kb_dir or KB_DIR)
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge-base directory not found: {kb_dir}")

    docs: List[Dict[str, Any]] = []
    for md_file in sorted(kb_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        title = content.split("\n", 1)[0].lstrip("# ").strip() or md_file.stem
        doc_slug = md_file.stem.lstrip("0123456789_")
        url = f"internal://nawaloka/{doc_slug}"
        docs.append({"url": url, "title": title, "content": content})

    return docs

def embed_texts(
    texts: List[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> List[List[float]]:
    """Embed a list of texts using the configured embedding model."""
    embedder = get_default_embeddings(batch_size=batch_size)
    all_embeddings: List[List[float]] = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(
            "Embedding batch {}/{} ({} texts)...",
            batch_num,
            total_batches,
            len(batch),
        )
        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

def run_ingest(
) -> int:
    """
    End-to-end ingestion pipeline.
    """
    logger.info("=" * 70)
    logger.info("🚀 QDRANT INGESTION PIPELINE")
    logger.info("=" * 70)

    source = "kb"
    strategy = "parent_child"

    # ── 1. Load documents ────────────────────────────────────
    loader = load_kb_docs
    if loader is None:
        raise ValueError(
            f"Unknown loader: {loader}"
        )

    logger.info(f"\n📂 Loading documents (source={source})...")
    docs = loader()
    if not docs:
        logger.error("❌ No documents loaded. Nothing to ingest.")
        sys.exit(1)

    # ── 2. Chunk ─────────────────────────────────────────────
    logger.info(f"\n✂️  Chunking (strategy={strategy})...")
    chunk_fn = STRATEGY_MAP.get(strategy)
    if chunk_fn is None:
        raise ValueError(
            f"Unknown strategy: {strategy}. Choose from {list(STRATEGY_MAP.keys())}"
        )

    children, parents = chunk_fn(docs)
    logger.info(f"   → {len(children)} child chunks, {len(parents)} parent chunks")
    parent_lookup = _build_parent_lookup(parents)
    chunks = _enrich_children_with_parent_text(children, parent_lookup)
    logger.info("   → Each child enriched with parent_text for richer LLM context")

    if not chunks:
        logger.error("❌ No chunks produced. Check your documents.")
        sys.exit(1)

    # ── 3. Embed ─────────────────────────────────────────────
    logger.info(f"\n🔢 Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    embeddings = embed_texts(texts)
    embed_secs = time.time() - t0
    logger.success(f"   → Embedding done in {embed_secs:.1f}s")

    # ── 4. Create collection ──────────────────────
    ensure_collection()

    # ── 5. Upsert ────────────────────────────────────────────
    logger.info(f"\n⬆️  Upserting {len(chunks)} points into Qdrant...")
    t0 = time.time()
    n = upsert_chunks(chunks, embeddings)
    upsert_secs = time.time() - t0
    logger.info(f"   → Upserted {n} points in {upsert_secs:.1f}s")

    # ── 6. Verify ────────────────────────────────────────────
    logger.info("\n📊 Collection info:")
    info = collection_info()
    for k, v in info.items():
        logger.info(f"   {k}: {v}")

    logger.info("\n" + "=" * 70)
    logger.success("✅ INGESTION COMPLETE")
    logger.info(f"   Source: {source}")
    logger.info(f"   Strategy: {strategy}")
    logger.info(f"   Chunks indexed: {n}")
    if strategy == "parent_child":
        logger.info("   Parent context: Stored in payload for richer LLM generation")
    logger.info("=" * 70)

    return n

STRATEGY_MAP = {
    "parent_child": parent_child_chunk,
}


def _build_parent_lookup(parents: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build a mapping from parent_id → parent text."""
    return {p["parent_id"]: p["text"] for p in parents}

def _enrich_children_with_parent_text(
    children: List[Dict[str, Any]],
    parent_lookup: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Attach ``parent_text`` to each child chunk for richer LLM context."""
    for child in children:
        pid = child.get("parent_id", "")
        child["parent_text"] = parent_lookup.get(pid, child["text"])
    return children

