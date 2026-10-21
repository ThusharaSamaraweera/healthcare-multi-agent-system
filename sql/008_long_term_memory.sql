-- ============================================================================
-- LONG-TERM SEMANTIC MEMORY (Replaces ChromaDB for facts)
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS mem_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1536),
    score REAL NOT NULL CHECK (score >= 0 AND score <= 1),
    tags JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ DEFAULT NOW(),
    ttl_at TIMESTAMPTZ,
    pin BOOLEAN DEFAULT FALSE,
    deleted BOOLEAN DEFAULT FALSE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_mem_facts_user_id ON mem_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_mem_facts_score ON mem_facts(score DESC);
CREATE INDEX IF NOT EXISTS idx_mem_facts_deleted ON mem_facts(deleted) WHERE deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_mem_facts_ttl ON mem_facts(ttl_at) WHERE ttl_at IS NOT NULL;

-- pgvector index (IVFFlat supports higher dimensions, HNSW limited to 2000)
CREATE INDEX IF NOT EXISTS idx_mem_facts_embedding 
ON mem_facts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);