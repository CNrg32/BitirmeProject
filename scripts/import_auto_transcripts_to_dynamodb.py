from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from services.transcript_store import (  # noqa: E402
    DEFAULT_AUTO_TRANSCRIPTS_CSV,
    DynamoDBTranscriptStore,
    load_auto_transcripts_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import auto_transcripts.csv into AWS DynamoDB."
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_AUTO_TRANSCRIPTS_CSV),
        help="CSV path to import.",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="DynamoDB table name. Defaults to TRANSCRIPTS_DYNAMODB_TABLE.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region. Defaults to AWS_REGION/AWS_DEFAULT_REGION.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help="Optional DynamoDB endpoint URL for local testing.",
    )
    args = parser.parse_args()

    records = load_auto_transcripts_csv(args.csv)
    store = DynamoDBTranscriptStore(
        table_name=args.table,
        region_name=args.region,
        endpoint_url=args.endpoint_url,
    )
    imported_count = store.put_many(records)
    print(f"Imported {imported_count} transcripts into {store.table_name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
