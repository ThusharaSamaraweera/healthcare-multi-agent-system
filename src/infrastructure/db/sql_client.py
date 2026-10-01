import os
from sqlalchemy import (create_engine, text)
from sqlalchemy.orm import sessionmaker
from loguru import logger
from typing import Optional

# Singleton engine
_engine: Optional[object] = None
_SessionLocal: Optional[object] = None

def get_sql_engine():
    global _engine
    if _engine is None:
        db_url = os.getenv("SUPABASE_DB_URL")
        
        if not db_url:
            raise ValueError(
                "SUPABASE_DB_URL must be set in .env file. "
                "Format: postgresql://postgres:[password]@db.xxxxx.supabase.co:5432/postgres"
            )
        
        _engine = create_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,  # Set to True for SQL debugging
        )
        logger.info(f"✓ Supabase SQL engine created")
    return _engine

def get_session():
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_sql_engine()
        _SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return _SessionLocal()

def test_connection():
    try:
        engine = get_sql_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        logger.info("✅ Supabase connection test: SUCCESS")
        return True
    except Exception as e:
        logger.error(f"❌ Supabase connection test: FAILED - {e}")
        return False
