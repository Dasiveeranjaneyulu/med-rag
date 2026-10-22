
import chromadb
from backend.app.core.config import settings
from loguru import logger
import numpy as np


def _chroma_cos_to_cos_sim(chroma_distance: float) -> float:
    """
    ChromaDB cosine distance = 1 - cosine_similarity.
    Returns raw cosine similarity in [-1, 1].
    """
    return 1.0 - float(chroma_distance)


class ChromaStore:
    """
    ChromaDB wrapper managing TWO separate collections with incompatible
    embedding spaces:

      * pdf_collection    - uploaded medical PDFs, embedded with the general
                            sentence encoder (all-MiniLM-L6-v2 / 384-dim or
                            similar; Gemini embeddings).
      * pubmed_collection - PubMed abstracts with precomputed MedCPT
                            article embeddings (768-dim).

    These two collections MUST remain separate because their vector
    representations live in different embedding spaces; comparing a
    MedCPT query embedding against a MiniLM vector is mathematically
    meaningless and produces garbage similarity scores.
    """

    PDF_ADD_BATCH = 5000
    PUBMED_ADD_BATCH = 5000

    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(settings.VECTORSTORE_DIR))

        self.pdf_collection = self.client.get_or_create_collection(
            name=settings.PDF_COLLECTION_NAME,
        )

        self.pubmed_collection = self.client.get_or_create_collection(
            name=settings.PUBMED_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "ChromaStore initialized at %s | pdf_collection=%s (count=%d) | "
            "pubmed_collection=%s (count=%d, space=cosine)",
            settings.VECTORSTORE_DIR,
            settings.PDF_COLLECTION_NAME,
            self.pdf_collection.count(),
            settings.PUBMED_COLLECTION_NAME,
            self.pubmed_collection.count(),
        )

        self.collection = self.pdf_collection

    # ------------------------------------------------------------------
    # Generic / legacy helpers (delegate to the PDF collection)
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self.pdf_collection.count()

    def get_all_documents(self) -> list:
        if self.count() == 0:
            return []
        return self.pdf_collection.get()["documents"] or []

    def add_documents(self, ids, documents, metadatas=None, embeddings=None):
        self.add_pdf_documents(ids, documents, metadatas, embeddings)

    def query(self, query_embeddings, n_results=5):
        return self.query_pdf(query_embeddings, n_results)

    # ------------------------------------------------------------------
    # PDF collection operations
    # ------------------------------------------------------------------

    def add_pdf_documents(
        self,
        ids: list,
        documents: list,
        metadatas: list | None = None,
        embeddings: list | None = None,
    ):
        for i in range(0, len(ids), self.PDF_ADD_BATCH):
            end = min(i + self.PDF_ADD_BATCH, len(ids))
            self.pdf_collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end] if metadatas else None,
                embeddings=embeddings[i:end] if embeddings is not None else None,
            )
        logger.info(f"PDF collection: added {len(documents)} chunks")

    def query_pdf(
        self,
        query_embeddings: list,
        n_results: int = 5,
    ):
        return self.pdf_collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
        )

    # ------------------------------------------------------------------
    # PubMed collection operations (MedCPT embedding space)
    # ------------------------------------------------------------------

    def pubmed_count(self) -> int:
        return self.pubmed_collection.count()

    def get_pubmed_collection(self):
        return self.pubmed_collection

    def add_pubmed_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add precomputed MedCPT embeddings to the PubMed collection.
        Validates dimension before insert.
        """
        if not ids:
            return 0

        dim_ok = all(
            isinstance(v, list) and len(v) == settings.PUBMED_EMBEDDING_DIM
            for v in embeddings
        )
        if not dim_ok:
            dims_seen = {
                len(v) if isinstance(v, list) else -1 for v in embeddings
            }
            raise ValueError(
                f"PubMed embedding dim mismatch: expected {settings.PUBMED_EMBEDDING_DIM}, "
                f"got {dims_seen}"
            )

        added = 0
        for i in range(0, len(ids), self.PUBMED_ADD_BATCH):
            end = min(i + self.PUBMED_ADD_BATCH, len(ids))
            self.pubmed_collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end],
            )
            added += (end - i)

        logger.info(f"PubMed collection: added {added} records")
        return added

    def query_pubmed(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> tuple[list[str], list[dict], list[float]]:
        """
        Retrieve Top-K PubMed documents using a MedCPT query embedding.

        Returns
        -------
        (documents, metadatas, cosine_similarities)
            Documents and metadata are parallel lists of length M <= top_k.
            cosine_similarities is a list of floats in [-1, 1]; higher = more
            similar.
        """
        if len(query_embedding) != settings.PUBMED_EMBEDDING_DIM:
            raise ValueError(
                f"PubMed query embedding dim mismatch: got {len(query_embedding)}, "
                f"expected {settings.PUBMED_EMBEDDING_DIM}"
            )

        if top_k is None:
            top_k = settings.PUBMED_TOP_K

        if self.pubmed_count() == 0:
            logger.warning("PubMed collection is empty — returning 0 hits")
            return [], [], []

        result = self.pubmed_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.pubmed_count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        sims = [_chroma_cos_to_cos_sim(d) for d in dists]

        logger.info(
            f"PubMed query returned {len(docs)} hits. "
            f"Raw cosine similarities: {[round(s, 4) for s in sims]}"
        )
        return docs, metas, sims

    def pubmed_pmids_exist(self, pmids: list[str]) -> set[str]:
        """Check which PMIDs already exist in the collection (for restartable ingest)."""
        if not pmids:
            return set()
        try:
            res = self.pubmed_collection.get(ids=list(pmids), include=[])
            return set(res.get("ids") or [])
        except Exception as e:
            logger.error(f"pubmed_pmids_exist failed: {e}")
            return set()

