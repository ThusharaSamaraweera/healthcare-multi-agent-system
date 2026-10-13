#!/usr/bin/env python3
from dotenv import load_dotenv

load_dotenv()

from services.ingest_service.pipeline import run_ingest


def main() -> None:
    run_ingest()


if __name__ == "__main__":
    main()
