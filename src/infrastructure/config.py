"""
Application configuration - loads from YAML param files.

CONFIGURATION POLICY:
====================
Configuration is loaded from config/param.yaml and config/models.yaml.
Secrets (API keys) live ONLY in .env and are loaded via os.getenv().
"""



from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml

# ========================================
# Project Paths
# ========================================

# Get project root (parent of src/infrastructure/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
KB_DIR = _PROJECT_ROOT / "data" / "knowledge_base"

# ========================================
# YAML Config Loading
# ========================================

def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    filepath = _CONFIG_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_nested(d: Dict, *keys, default=None):
    """Get nested dictionary value safely."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


# Load configs
_PARAMS = _load_yaml("param.yaml")
_MODELS = _load_yaml("models.yaml")

# ========================================
# YAML Config Loading
# ========================================

def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    filepath = _CONFIG_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
    
def _get_nested(d: Dict, *keys, default=None):
    """Get nested dictionary value safely."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


# Load configs
_PARAMS = _load_yaml("param.yaml")
_MODELS = _load_yaml("models.yaml")

# ========================================
# Provider Configuration
# ========================================

PROVIDER = _get_nested(_PARAMS, "provider", "default", default="openrouter")
MODEL_TIER = _get_nested(_PARAMS, "provider", "tier", default="general")
OPENROUTER_BASE_URL = _get_nested(_PARAMS, "provider", "openrouter_base_url",
                                   default="https://openrouter.ai/api/v1")


# ========================================
# Model Names (from models.yaml)
# ========================================

EMBEDDING_TIER = _get_nested(_PARAMS, "embedding", "tier", default="default")


def get_embedding_model(provider: Optional[str] = None, tier: Optional[str] = None) -> str:
    """Get embedding model name for specified provider and tier."""
    provider = provider or PROVIDER
    tier = tier or EMBEDDING_TIER

    # Handle provider name mapping
    if provider == "google" or provider == "gemini":
        provider = "google"

    return _get_nested(_MODELS, provider, "embedding", tier, default="openai/text-embedding-3-small")


ROUTER_MODEL = "openai/gpt-4o-mini"
ROUTER_PROVIDER = "openrouter"

EXTRACTOR_MODEL = "meta-llama/llama-3.1-8b-instruct"
EXTRACTOR_PROVIDER = "openrouter"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

CHAT_MODEL = "google/gemini-2.5-flash"
CHAT_PROVIDER = "openrouter"

EMBEDDING_MODEL = get_embedding_model()

# ========================================
# Embedding Dimensions
# ========================================


# ⚠️ IMPORTANT: EMBEDDING_DIM must match the model's output dimensions
#
# Supported models:
#   - "text-embedding-3-small"  → 1536 dims (recommended for Qdrant)
#   - "text-embedding-3-large"  → 3072 dims
#   - "text-embedding-ada-002"  → 1536 dims (legacy)
#
# Qdrant supports any dimension size (no limit like Supabase)
EMBEDDING_DIM = 1536  # Default for text-embedding-3-small

# ========================================
# Embedding Defaults
# ========================================

EMBEDDING_BATCH_SIZE = _get_nested(_PARAMS, "embedding", "batch_size", default=100)
EMBEDDING_SHOW_PROGRESS = _get_nested(_PARAMS, "embedding", "show_progress", default=False)

# Timezone used when formatting CRM appointment times.
TIMEZONE = "Asia/Colombo"

# Qdrant Cloud configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = _get_nested(
    _PARAMS, "qdrant", "collection_name", default="nawaloka"
)


# Parent-child chunking settings (token targets).
PARENT_CHUNK_SIZE = _get_nested(
    _PARAMS, "chunking", "parent_child", "parent_size", default=1200
)
CHILD_CHUNK_SIZE = _get_nested(
    _PARAMS, "chunking", "parent_child", "child_size", default=250
)
CHILD_OVERLAP = _get_nested(
    _PARAMS, "chunking", "parent_child", "child_overlap", default=50
)

# ========================================
# Retrieval Configuration
# ========================================

TOP_K_RESULTS = _get_nested(_PARAMS, "retrieval", "top_k", default=4)
SIMILARITY_THRESHOLD = _get_nested(_PARAMS, "retrieval", "similarity_threshold", default=0.7)

# ========================================
# CRAG Configuration
# ========================================

CRAG_CONFIDENCE_THRESHOLD = _get_nested(_PARAMS, "crag", "confidence_threshold", default=0.6)
CRAG_EXPANDED_K = _get_nested(_PARAMS, "crag", "expanded_k", default=8)

# ========================================
# CAG Configuration (Qdrant Semantic Cache)
# ========================================

CAG_COLLECTION_NAME = _get_nested(_PARAMS, "cag", "collection_name", default="cag_cache")
CAG_SIMILARITY_THRESHOLD = _get_nested(_PARAMS, "cag", "similarity_threshold", default=0.90)
CAG_CACHE_TTL = _get_nested(_PARAMS, "cag", "cache_ttl", default=86400)  # 24h
CAG_CACHE_MAX_SIZE = _get_nested(_PARAMS, "cag", "max_cache_size", default=1000)

# ========================================
# Memory Configuration (NOT in YAML - kept as constants)
# ========================================

# Short-term memory (Supabase st_turns table)
ST_MAX_TURNS = 30
ST_TTL_SECONDS = 60 * 60 * 24  # 24 hours

LT_TOP_K = 5
LT_SIM_THRESHOLD = 0.30
LT_TTL_SECONDS = 60 * 60 * 24 * 90
LT_DECAY_HALF_LIFE_DAYS = 30

# ========================================
# Helper Functions
# ========================================



def load_faqs() -> list:
    """Flatten configured FAQ categories into query/answer records."""
    faqs_config = _load_yaml("faqs.yaml")
    if not faqs_config:
        return []

    faqs = []
    for items in faqs_config.values():
        if isinstance(items, list):
            faqs.extend(item for item in items if isinstance(item, dict))
    return faqs


KNOWN_FAQS = load_faqs()


# ========================================
# Helper Functions
# ========================================

def get_api_key(provider: Optional[str] = None) -> Optional[str]:
    """Get API key for the specified provider."""
    provider = provider or PROVIDER
    key_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "tavily": "TAVILY_API_KEY",
    }
    env_var = key_map.get(provider, f"{provider.upper()}_API_KEY")
    return os.getenv(env_var)
