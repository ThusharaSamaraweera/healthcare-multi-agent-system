from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_FILES = [
    PROJECT_ROOT / "sql" / "002_specialties.sql",
    PROJECT_ROOT / "sql" / "003_locations.sql",
    PROJECT_ROOT / "sql" / "004_doctors.sql",
    PROJECT_ROOT / "sql" / "005_patients.sql",
    PROJECT_ROOT / "sql" / "006_bookings.sql",
]

EXPECTED_COUNTS = {
    "specialties": 10,
    "locations": 4,
    "doctors": 50,
    "patients": 200,
    "bookings": 1500,
}

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _sql_statements(sql_text: str) -> list[str]:
    lines = [
        line
        for line in sql_text.splitlines()
        if line.strip() and not line.lstrip().startswith("--")
    ]
    return [
        statement.strip()
        for statement in "\n".join(lines).split(";")
        if statement.strip()
    ]


def _validate_seed_files() -> None:
    missing = [str(path) for path in SEED_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing CRM seed file(s):\n" + "\n".join(missing)
        )


def seed_crm() -> dict[str, int]:
    """Load all CRM seed files in dependency order in one transaction."""
    _validate_seed_files()

    from infrastructure.db.sql_client import get_sql_engine

    engine = get_sql_engine()
    statement_count = 0

    with engine.begin() as connection:
        for seed_file in SEED_FILES:
            statements = _sql_statements(seed_file.read_text(encoding="utf-8"))
            logger.info("Loading {} ({} statements)", seed_file.name, len(statements))

            for statement in statements:
                connection.exec_driver_sql(statement)
                statement_count += 1

        counts: dict[str, int] = {}
        for table_name in EXPECTED_COUNTS:
            result = connection.exec_driver_sql(
                f"SELECT COUNT(*) FROM {table_name}"
            )
            counts[table_name] = int(result.scalar_one())

    logger.success(
        "CRM seed complete: {} statements committed",
        statement_count,
    )
    for table_name, expected in EXPECTED_COUNTS.items():
        actual = counts[table_name]
        if actual != expected:
            raise RuntimeError(
                f"Unexpected {table_name} count: expected {expected}, got {actual}"
            )
        logger.info("{}: {} rows", table_name, actual)

    return counts


def main() -> int:
    """CLI entry point."""
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        from infrastructure.log import setup_logging

        setup_logging()
        seed_crm()
        return 0
    except Exception as exc:
        logger.error("CRM seed failed: {}", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
