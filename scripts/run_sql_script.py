from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_FILE = PROJECT_ROOT / "sql" / "007_short_term_memory.sql"

# Make the src-layout imports work when this file is run from the repository
# root, as required by the project scripts.
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _sql_statements(sql_text: str) -> list[str]:

    lines = [
        line
        for line in sql_text.splitlines()
        if line.strip() and not line.lstrip().startswith("--")
    ]
    return [statement.strip() for statement in "\n".join(lines).split(";") if statement.strip()]


def initialize_crm_schema() -> int:
    if not SQL_FILE.exists():
        raise FileNotFoundError(f"CRM migration not found: {SQL_FILE}")

    from infrastructure.db.sql_client import get_sql_engine

    statements = _sql_statements(SQL_FILE.read_text(encoding="utf-8"))
    if not statements:
        raise ValueError(f"CRM migration is empty: {SQL_FILE}")

    engine = get_sql_engine()
    with engine.begin() as connection:
        for statement in statements:
            connection.exec_driver_sql(statement)

    logger.success("CRM schema initialized from {} ({} statements)", SQL_FILE, len(statements))
    return 0


def main() -> int:

    load_dotenv(PROJECT_ROOT / ".env")

    try:
        from infrastructure.log import setup_logging

        setup_logging()
        return initialize_crm_schema()
    except Exception as exc:
        logger.error("CRM schema initialization failed: {}", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
