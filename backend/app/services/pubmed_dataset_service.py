
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from backend.app.core.config import settings
from loguru import logger
import numpy as np

DATASET_NAME = settings.PUBMED_DATASET_NAME
EXPECTED_DIM = settings.PUBMED_EMBEDDING_DIM


@dataclass
class PubMedRecord:
    pmid: str
    title: str
    abstract: str
    embedding: list[float]

    @property
    def document(self) -> str:
        title = (self.title or "").strip()
        abstract = (self.abstract or "").strip()
        if title and abstract:
            return f"{title}\n\n{abstract}"
        return title or abstract

    @property
    def metadata(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "source": "pubmed",
            "dataset": DATASET_NAME,
        }


def _validate_embedding(vec, pmid) -> Optional[list[float]]:
    if vec is None:
        logger.warning(f"PMID {pmid}: embedding is None — skipping")
        return None

    if isinstance(vec, list):
        arr = np.asarray(vec, dtype=np.float32)
    elif isinstance(vec, np.ndarray):
        arr = vec.astype(np.float32)
    else:
        try:
            arr = np.asarray(list(vec), dtype=np.float32)
        except Exception as e:
            logger.warning(f"PMID {pmid}: cannot cast embedding to array ({e}) — skipping")
            return None

    if arr.ndim != 1:
        logger.warning(f"PMID {pmid}: embedding has {arr.ndim} dims (expected 1) — skipping")
        return None

    if arr.shape[0] != EXPECTED_DIM:
        logger.warning(
            f"PMID {pmid}: embedding dim={arr.shape[0]} (expected {EXPECTED_DIM}) — skipping"
        )
        return None

    if not np.all(np.isfinite(arr)):
        logger.warning(f"PMID {pmid}: embedding contains NaN/Inf — skipping")
        return None

    return arr.tolist()


def stream_pubmed_records(
    limit: Optional[int] = None,
    streaming: bool = True,
) -> Iterator[PubMedRecord]:
    """
    Stream PubMed records from the Hugging Face dataset without loading
    the entire dataset into RAM. Each record's MedCPT embedding is
    validated to be exactly 768-dimensional with finite values.
    """
    from datasets import load_dataset

    logger.info(
        f"Loading PubMed dataset '{DATASET_NAME}' "
        f"(streaming={streaming}, limit={limit if limit else 'full'})"
    )

    ds = load_dataset(DATASET_NAME, split="train", streaming=streaming)

    yielded = 0
    for idx, row in enumerate(ds):
        try:
            pmid_raw = row.get("PMID")
            pmid = str(int(pmid_raw)) if pmid_raw is not None else None
            if not pmid:
                logger.warning(f"Row {idx}: missing PMID — skipping")
                continue

            title = row.get("title") or ""
            abstract = row.get("abstract") or ""
            embedding_raw = row.get("embedding")

            embedding = _validate_embedding(embedding_raw, pmid)
            if embedding is None:
                continue

            yield PubMedRecord(
                pmid=pmid,
                title=str(title),
                abstract=str(abstract),
                embedding=embedding,
            )
            yielded += 1

            if limit and yielded >= limit:
                logger.info(f"Reached requested limit of {limit} records")
                return

        except Exception as e:
            logger.error(f"Error processing PubMed row {idx}: {e}")
            continue

    logger.info(f"Finished streaming. Total valid records yielded: {yielded}")


def peek_dataset_schema() -> dict:
    """
    Load a tiny slice of the dataset and return its inferred schema / stats.
    Useful for verifying the dataset structure before running a full ingestion.
    """
    from datasets import load_dataset

    logger.info(f"Peeking schema of {DATASET_NAME}")
    ds_small = load_dataset(DATASET_NAME, split="train", streaming=True)
    sample_count = 0
    features_repr = None
    dim_seen = set()
    for i, row in enumerate(ds_small):
        if features_repr is None:
            features_repr = {k: str(type(v).__name__) for k, v in row.items()}
        emb = row.get("embedding")
        if emb is not None:
            if isinstance(emb, (list, np.ndarray)):
                dim_seen.add(len(emb))
        sample_count += 1
        if sample_count >= 3:
            break

    return {
        "dataset": DATASET_NAME,
        "features": features_repr,
        "samples_peeked": sample_count,
        "embedding_dims_seen": sorted(dim_seen),
    }


def batch_iterate(
    records: Iterable[PubMedRecord],
    batch_size: int,
) -> Iterator[list[PubMedRecord]]:
    """Group PubMed records into fixed-size batches for efficient ingestion."""
    batch = []
    for r in records:
        batch.append(r)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ingest_pubmed_into_chroma(
    pubmed_collection,
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> dict:
    """
    Stream the dataset, validate records, and ingest into the provided
    ChromaDB collection in batches. Ensures restartable ingestion by
    skipping PMIDs that already exist in the target collection.

    Returns a report dict with counts.
    """
    if batch_size is None:
        batch_size = settings.PUBMED_BATCH_SIZE

    stats = {
        "batches": 0,
        "seen": 0,
        "skipped_existing": 0,
        "skipped_invalid": 0,
        "ingested": 0,
        "pmid_max_seen": None,
        "pmid_min_seen": None,
    }

    logger.info(
        f"Starting PubMed ingestion: batch_size={batch_size}, "
        f"limit={limit if limit else 'full'}, collection={pubmed_collection.name}"
    )

    for batch in batch_iterate(stream_pubmed_records(limit=limit), batch_size):
        stats["batches"] += 1
        batch_ids: list[str] = []
        batch_docs: list[str] = []
        batch_meta: list[dict] = []
        batch_embs: list[list[float]] = []

        for rec in batch:
            stats["seen"] += 1
            try:
                pmid_int = int(rec.pmid)
                if stats["pmid_min_seen"] is None or pmid_int < stats["pmid_min_seen"]:
                    stats["pmid_min_seen"] = pmid_int
                if stats["pmid_max_seen"] is None or pmid_int > stats["pmid_max_seen"]:
                    stats["pmid_max_seen"] = pmid_int
            except Exception:
                pass

        try:
            existing = pubmed_collection.get(ids=[r.pmid for r in batch], include=[])
            existing_ids = set(existing.get("ids", []) or [])
        except Exception as e:
            logger.error(f"Failed to check existing PMIDs in batch {stats['batches']}: {e}")
            existing_ids = set()

        for rec in batch:
            if rec.pmid in existing_ids:
                stats["skipped_existing"] += 1
                continue

            doc = rec.document.strip()
            if not doc:
                stats["skipped_invalid"] += 1
                continue

            batch_ids.append(rec.pmid)
            batch_docs.append(doc)
            batch_meta.append(rec.metadata)
            batch_embs.append(rec.embedding)

        if batch_ids:
            try:
                pubmed_collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta,
                    embeddings=batch_embs,
                )
                stats["ingested"] += len(batch_ids)
            except Exception as e:
                logger.error(
                    f"Batch {stats['batches']}: ChromaDB add failed ({e}). "
                    f"Attempting record-by-record fallback."
                )
                for rec_id, rec_doc, rec_meta, rec_emb in zip(
                    batch_ids, batch_docs, batch_meta, batch_embs
                ):
                    try:
                        pubmed_collection.add(
                            ids=[rec_id],
                            documents=[rec_doc],
                            metadatas=[rec_meta],
                            embeddings=[rec_emb],
                        )
                        stats["ingested"] += 1
                    except Exception as ie:
                        logger.error(f"  Could not ingest PMID {rec_id}: {ie}")
                        stats["skipped_invalid"] += 1

        if stats["batches"] % 10 == 0 or stats["batches"] == 1:
            logger.info(
                f"Batch {stats['batches']}: ingested_total={stats['ingested']}, "
                f"skipped_existing={stats['skipped_existing']}, "
                f"skipped_invalid={stats['skipped_invalid']}"
            )

    final_count = None
    try:
        final_count = pubmed_collection.count()
    except Exception:
        pass
    stats["final_collection_count"] = final_count

    logger.info(f"PubMed ingestion complete: {stats}")
    return stats
