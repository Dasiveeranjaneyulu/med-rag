
"""
Standalone PubMed Ingestion Script
===================================

Ingests the slinusc/PubMedAbstractsSubsetEmbedded dataset into a dedicated
ChromaDB collection (pubmed_medcpt_docs) using its precomputed 768-dim
MedCPT article embeddings.

Features:
  * Streams the dataset from Hugging Face (never loads all 1.61M rows into
    RAM at once).
  * Validates every embedding has exactly 768 finite float dimensions.
  * Uses PMID as the deterministic ChromaDB document ID.
  * Stores PMID, title, source="pubmed", dataset name as metadata.
  * Restartable: skips PMIDs already present in the target collection, so
    you can Ctrl-C and resume at any time.
  * Supports --limit for a small test subset before full ingestion.
  * --peek-schema prints the dataset schema and exits.

Usage examples (from project root c:\medical_rag_system):

  # Activate venv first
  venv\Scripts\Activate.ps1

  # Step 1: Verify schema (fast, safe)
  python scripts\ingest_pubmed.py --peek-schema

  # Step 2: Ingest a 1000-record test subset
  python scripts\ingest_pubmed.py --limit 1000

  # Step 3: Full ingestion (1.61M records, takes time)
  python scripts\ingest_pubmed.py

  # Custom batch size / resume
  python scripts\ingest_pubmed.py --batch-size 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from backend.app.core.config import settings
from backend.app.stores.chroma_store import ChromaStore
from backend.app.services.pubmed_dataset_service import (
    peek_dataset_schema,
    ingest_pubmed_into_chroma,
)


def _configure_logging():
    log_file = settings.LOGS_DIR / f"pubmed_ingest_{settings.BASE_DIR.name}.log"
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level="INFO", format=fmt)
    logger.add(log_file, level="DEBUG", format=fmt, rotation="50 MB", retention="30 days")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest slinusc/PubMedAbstractsSubsetEmbedded into ChromaDB (MedCPT collection)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max PubMed records to process. Omit for full dataset (≈1.61M rows).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.PUBMED_BATCH_SIZE,
        help=f"ChromaDB batch add size (default: {settings.PUBMED_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--peek-schema",
        action="store_true",
        help="Print dataset schema/feature info and exit (no ingestion performed).",
    )
    parser.add_argument(
        "--print-stats-json",
        action="store_true",
        help="Print final stats as JSON to stdout at end.",
    )
    args = parser.parse_args()

    _configure_logging()

    if args.peek_schema:
        info = peek_dataset_schema()
        logger.info("Schema info: " + json.dumps(info, indent=2, default=str))
        print(json.dumps(info, indent=2, default=str))
        return

    logger.info(
        "Starting PubMed ingestion | limit=%s | batch_size=%d | "
        "collection=%s | vectorstore=%s",
        args.limit if args.limit is not None else "FULL",
        args.batch_size,
        settings.PUBMED_COLLECTION_NAME,
        settings.VECTORSTORE_DIR,
    )

    store = ChromaStore()
    pubmed_collection = store.get_pubmed_collection()

    pre_count = store.pubmed_count()
    logger.info(f"Pre-ingestion PubMed collection count: {pre_count}")

    stats = ingest_pubmed_into_chroma(
        pubmed_collection=pubmed_collection,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    post_count = store.pubmed_count()
    stats["pre_count"] = pre_count
    stats["post_count"] = post_count
    stats["net_new_in_this_run"] = post_count - pre_count

    logger.info(
        f"Ingestion complete. Net new={stats['net_new_in_this_run']}, "
        f"final_count={post_count}"
    )

    if args.print_stats_json:
        print(json.dumps(stats, indent=2, default=str))
    else:
        print()
        print("=== PubMed Ingestion Report ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
