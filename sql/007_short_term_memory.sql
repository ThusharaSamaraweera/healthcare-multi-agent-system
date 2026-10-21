CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS st_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ttl_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_st_turns_user_session
ON st_turns (user_id, session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_st_turns_ttl
ON st_turns (ttl_at)
WHERE ttl_at IS NOT NULL;