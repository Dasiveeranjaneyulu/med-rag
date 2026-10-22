
from backend.app.models.embeddings import embedding_model
from backend.app.models.medcpt_embeddings import medcpt_encoder
from backend.app.stores.chroma_store import ChromaStore
from backend.app.stores.bm25_store import BM25Retriever
from backend.app.models.reranker_model import reranker
from backend.app.services.llm_router import generate_general_response
from backend.app.core.config import settings
from backend.app.core.prompts import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI
from groq import Groq
from loguru import logger
import os

rag_llm = None
groq_rag_client = None

try:
    rag_llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL_NAME,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.2,
    )
except Exception as e:
    logger.error(f"Failed to initialize RAG LLM (Gemini): {e}")

try:
    if settings.GROQ_API_KEY:
        groq_rag_client = Groq(api_key=settings.GROQ_API_KEY)
        logger.info(f"Successfully initialized RAG Groq model: {settings.GROQ_MODEL_NAME}")
    else:
        logger.warning("GROQ_API_KEY not set for RAG fallback")
except Exception as e:
    logger.error(f"Failed to initialize RAG LLM (Groq): {e}")


class MedicalRAGPipeline:
    """
    Two-pipeline Medical RAG.

    * PDF path: Gemini/SentenceTransformer embeddings → Chroma (pdf_collection)
                + BM25 sparse retrieval → reranker (cross-encoder).
    * PubMed path: precomputed MedCPT article embeddings in Chroma
                   (pubmed_collection) ← MedCPT query encoder.

    The two paths use SEPARATE vector collections because MedCPT and
    all-MiniLM / Gemini-style embeddings do not share the same latent
    space — mixing them in a single collection produces meaningless
    nearest-neighbour results.
    """

    def __init__(self):
        self.chroma = ChromaStore()
        self.bm25 = BM25Retriever()
        self._sync_bm25_from_chroma()
        logger.info(
            "Medical RAG Pipeline initialized. "
            "PDF collection count=%d, PubMed collection count=%d. "
            "MedCPT encoder loaded=%s",
            self.chroma.count(),
            self.chroma.pubmed_count(),
            medcpt_encoder.is_loaded,
        )

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def _sync_bm25_from_chroma(self):
        try:
            chroma_docs = self.chroma.get_all_documents()
            if not chroma_docs:
                return
            if len(self.bm25.documents) != len(chroma_docs):
                self.bm25.rebuild(chroma_docs)
                logger.info(f"Rebuilt BM25 from Chroma ({len(chroma_docs)} docs)")
        except Exception as e:
            logger.error(f"Failed to sync BM25 from Chroma: {e}")

    # ------------------------------------------------------------------
    # Ingestion (PDF pipeline) - unchanged
    # ------------------------------------------------------------------

    def ingest_documents(
        self,
        documents: list[str],
        id_prefix: str = "doc",
        metadatas: list[dict] | None = None,
    ) -> dict:
        if not documents:
            return {"chunks_added": 0, "total_in_db": self.chroma.count()}

        embeddings = embedding_model.embed_documents(documents)
        ids = [f"{id_prefix}_{i}_{os.urandom(4).hex()}" for i in range(len(documents))]
        self.chroma.add_pdf_documents(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        self.bm25.rebuild(self.chroma.get_all_documents())

        total = self.chroma.count()
        logger.info(f"Ingested {len(documents)} chunks; total in PDF collection: {total}")
        return {"chunks_added": len(documents), "total_in_db": total}

    def ingest_pubmed_legacy(self, num_samples: int = 500) -> dict:
        """
        Legacy PubMed ingestion path (re-embeds through the generic
        sentence encoder into the PDF collection). Left for backwards
        compatibility. New ingestion should use the dedicated script.
        """
        from datasets import load_dataset
        from backend.app.services.chunking_service import MedicalChunker

        logger.warning(
            "ingest_pubmed_legacy: re-embedding PubMed abstracts using the "
            "general sentence encoder — this ignores the precomputed MedCPT "
            "embeddings. For proper MedCPT-based ingestion use scripts/ingest_pubmed.py"
        )

        stream = load_dataset(
            "slinusc/PubMedAbstractsSubsetEmbedded",
            split="train",
            streaming=True,
        )

        chunker = MedicalChunker(chunk_size=800, chunk_overlap=100)
        all_chunks: list[str] = []
        metadatas: list[dict] = []
        abstracts_loaded = 0

        for item in stream:
            title = (item.get("title") or "").strip()
            abstract = (item.get("abstract") or "").strip()
            text = f"{title}\n{abstract}".strip() if title and abstract else (title or abstract)
            if not text:
                continue

            abstracts_loaded += 1
            chunks = chunker.chunk_text(text) if len(text) > 900 else [text]
            for chunk in chunks:
                all_chunks.append(chunk)
                metadatas.append({
                    "source": "pubmed_legacy",
                    "pmid": str(item.get("PMID", "")),
                    "pubmed_index": abstracts_loaded - 1,
                })
            if abstracts_loaded >= num_samples:
                break

        logger.info(f"Prepared {len(all_chunks)} PubMed chunks from {abstracts_loaded} abstracts")
        result = self.ingest_documents(all_chunks, id_prefix="pubmed_legacy", metadatas=metadatas)
        result["abstracts_loaded"] = abstracts_loaded
        return result

    # Back-compat alias, but prefer ingest_pubmed_legacy to avoid confusion
    ingest_pubmed = ingest_pubmed_legacy

    def ingest_pdfs(self, pdf_dir: str) -> dict:
        from backend.app.services.pdf_service import PDFProcessor
        from backend.app.services.chunking_service import MedicalChunker

        pdf_processor = PDFProcessor()
        chunker = MedicalChunker()

        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
            return {"files_seen": 0, "chunks_added": 0, "skipped_files": []}

        all_chunks = []
        skipped_files = []
        for filename in os.listdir(pdf_dir):
            if not filename.lower().endswith(".pdf"):
                skipped_files.append(filename)
                continue
            file_path = os.path.join(pdf_dir, filename)
            text = pdf_processor.process_pdf(file_path)
            if not text or not text.strip():
                logger.warning(f"No extractable text from {filename}")
                skipped_files.append({"file": filename, "reason": "no extractable text"})
                continue
            chunks = chunker.chunk_text(text)
            if not chunks:
                logger.warning(f"0 chunks for {filename}")
                skipped_files.append({"file": filename, "reason": "0 chunks"})
                continue
            all_chunks.extend(chunks)

        chunks_added = 0
        if all_chunks:
            metadatas = [{"source": "pdf", "chunk_index": i} for i in range(len(all_chunks))]
            report = self.ingest_documents(all_chunks, id_prefix="pdf", metadatas=metadatas)
            chunks_added = report["chunks_added"]

        total_in_db = self.chroma.count()
        return {
            "files_seen": len([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]),
            "chunks_added": chunks_added,
            "total_in_db": total_in_db,
            "skipped_files": skipped_files,
        }

    # ------------------------------------------------------------------
    # Retrieval paths
    # ------------------------------------------------------------------

    def hybrid_retrieval(self, query: str, query_embedding: list) -> tuple[list[str], list[float]]:
        """Original PDF pipeline hybrid retrieval (BM25 + Chroma) + reranker."""
        bm25_docs, _ = self.bm25.query(query, n_results=settings.TOP_K)
        chroma_results = self.chroma.query_pdf(
            query_embeddings=[query_embedding], n_results=settings.TOP_K
        )
        chroma_docs = chroma_results.get("documents", [[]])[0]

        all_candidates = []
        seen = set()
        for doc in bm25_docs + chroma_docs:
            if doc not in seen:
                seen.add(doc)
                all_candidates.append(doc)

        logger.info(f"PDF pipeline: combined {len(all_candidates)} unique candidate chunks")

        if len(all_candidates) > 0:
            reranked = reranker.rerank(query, all_candidates, top_k=settings.RERANK_TOP_K)
            top_docs = [doc for doc, score in reranked]
            top_scores = [score for doc, score in reranked]
            logger.info(
                f"PDF pipeline: reranked to top {len(top_docs)} chunks. "
                f"reranker_scores={[round(s, 4) for s in top_scores]}"
            )
            return top_docs, top_scores

        return [], []

    def pubmed_retrieval(self, query: str) -> tuple[list[str], list[dict], list[float]]:
        """
        PubMed MedCPT dense retrieval path.
        Returns (docs, metadatas, cosine_similarities) filtered by
        PUBMED_SIMILARITY_THRESHOLD.
        """
        if not medcpt_encoder.is_loaded:
            logger.warning("MedCPT query encoder not available — skipping PubMed retrieval")
            return [], [], []

        if self.chroma.pubmed_count() == 0:
            logger.info("PubMed collection empty — skipping PubMed retrieval")
            return [], [], []

        try:
            q_emb = medcpt_encoder.embed_query(query)
        except Exception as e:
            logger.error(f"MedCPT query embedding failed: {e}")
            return [], [], []

        docs, metas, sims = self.chroma.query_pubmed(q_emb, top_k=settings.PUBMED_TOP_K)

        threshold = settings.PUBMED_SIMILARITY_THRESHOLD
        filtered_docs: list[str] = []
        filtered_metas: list[dict] = []
        filtered_sims: list[float] = []
        for d, m, s in zip(docs, metas, sims):
            if s >= threshold:
                filtered_docs.append(d)
                filtered_metas.append(m if m is not None else {})
                filtered_sims.append(s)

        if len(filtered_docs) != len(docs):
            logger.info(
                f"PubMed threshold {threshold} kept {len(filtered_docs)}/{len(docs)} "
                f"hits. Kept sims={[round(s, 4) for s in filtered_sims]}"
            )
        else:
            logger.info(
                f"All {len(docs)} PubMed hits pass threshold {threshold}. "
                f"sims={[round(s, 4) for s in filtered_sims]}"
            )

        return filtered_docs, filtered_metas, filtered_sims

    # ------------------------------------------------------------------
    # LLM calling
    # ------------------------------------------------------------------

    def _call_gemini_rag(self, full_prompt: str) -> str:
        if not rag_llm:
            raise RuntimeError("Gemini RAG LLM not initialized")
        response = rag_llm.invoke(full_prompt)
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            return "".join(text_parts)
        return str(content)

    def _call_groq_rag(self, full_prompt: str) -> str:
        if not groq_rag_client:
            raise RuntimeError("Groq RAG client not initialized")
        completion = groq_rag_client.chat.completions.create(
            model=settings.GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt.replace(f"{SYSTEM_PROMPT}\n\n", "")},
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content

    def generate_rag_response(self, query: str, context: str) -> str:
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        providers = [
            ("gemini", self._call_gemini_rag),
            ("groq", self._call_groq_rag),
        ]
        for name, fn in providers:
            try:
                logger.info(f"Generating RAG response using {name}...")
                return fn(full_prompt)
            except Exception as e:
                logger.error(f"{name} RAG error: {e}")

        return self.generate_extractive_response(query, context)

    def generate_extractive_response(self, query: str, context: str) -> str:
        intro = (
            f"Based on retrieved medical literature and textbook content for: **{query}**\n\n"
            "*(LLM API unavailable — answer built from retrieved passages)*\n\n"
        )
        passages = [p.strip() for p in context.split("\n\n---\n\n") if p.strip()]
        body_parts = []
        for i, passage in enumerate(passages, 1):
            snippet = passage[:1200] + ("..." if len(passage) > 1200 else "")
            body_parts.append(f"**Passage {i}:**\n{snippet}")
        return intro + "\n\n".join(body_parts)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, query: str, show_sources: bool = False) -> str:
        result = self.run_with_details(query)
        answer = result["answer"]
        if show_sources and result["sources"]:
            answer += "\n\n---\n**Sources used:**\n"
            for i, src in enumerate(result["sources"], 1):
                preview = src[:300] + ("..." if len(src) > 300 else "")
                answer += f"\n{i}. {preview}"
        return answer

    def run_with_details(self, query: str) -> dict:
        """
        Unified run path:

        1. PDF retrieval (hybrid + reranker) → PDF_RERANKER_RELEVANCE_THRESHOLD
        2. PubMed retrieval (MedCPT cosine similarity) → PUBMED_SIMILARITY_THRESHOLD
        3. Combine relevant contexts from BOTH paths.
        4. If combined context non-empty → Gemini RAG.
        5. Else → general Gemini/Groq fallback.
        """
        try:
            query_embedding = embedding_model.embed_query(query)
            pdf_docs, pdf_scores = self.hybrid_retrieval(query, query_embedding)

            pdf_relevant_mask = [
                s >= settings.RERANKER_RELEVANCE_THRESHOLD for s in pdf_scores
            ]
            pdf_relevant_docs = [
                d for d, keep in zip(pdf_docs, pdf_relevant_mask) if keep
            ]
            pdf_relevant_scores = [
                s for s, keep in zip(pdf_scores, pdf_relevant_mask) if keep
            ]

            pubmed_docs, pubmed_metas, pubmed_sims = self.pubmed_retrieval(query)

            combined_docs: list[str] = []
            combined_sources_meta: list[dict] = []
            combined_scores: list[float] = []

            for d, s in zip(pdf_relevant_docs, pdf_relevant_scores):
                combined_docs.append(d)
                combined_sources_meta.append({"retrieval": "pdf_rerank", "score": s})
                combined_scores.append(s)

            for d, m, s in zip(pubmed_docs, pubmed_metas, pubmed_sims):
                combined_docs.append(d)
                meta = dict(m) if isinstance(m, dict) else {}
                meta["retrieval"] = "pubmed_medcpt"
                meta["score"] = s
                combined_sources_meta.append(meta)
                combined_scores.append(s)

            has_context = len(combined_docs) > 0

            if not has_context:
                logger.info(
                    f"No relevant context found. "
                    f"PDF kept {len(pdf_relevant_docs)}/{len(pdf_docs)} "
                    f"(threshold={settings.RERANKER_RELEVANCE_THRESHOLD}); "
                    f"PubMed kept {len(pubmed_docs)}/{settings.PUBMED_TOP_K} "
                    f"(threshold={settings.PUBMED_SIMILARITY_THRESHOLD}). "
                    f"→ Falling back to general LLM"
                )
                return {
                    "answer": generate_general_response(query),
                    "sources": [],
                    "source_metas": [],
                    "mode": "general_llm_fallback",
                }

            logger.info(
                f"RAG mode: using {len(combined_docs)} total context chunks "
                f"(PDF={len(pdf_relevant_docs)}, PubMed={len(pubmed_docs)}). "
                f"Scores={[round(s, 4) for s in combined_scores]}"
            )

            context = "\n\n---\n\n".join(combined_docs)
            answer = self.generate_rag_response(query, context)
            if answer.startswith("⚠️") or answer.startswith("👋"):
                answer = self.generate_extractive_response(query, context)

            return {
                "answer": answer,
                "sources": combined_docs,
                "source_metas": combined_sources_meta,
                "mode": "rag",
                "score_stats": {
                    "pdf_relevant_count": len(pdf_relevant_docs),
                    "pubmed_relevant_count": len(pubmed_docs),
                    "pubmed_sims": pubmed_sims,
                    "pdf_rerank_scores": pdf_scores,
                },
            }

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return {
                "answer": generate_general_response(query),
                "sources": [],
                "source_metas": [],
                "mode": f"error_fallback ({type(e).__name__})",
            }


rag_pipeline = MedicalRAGPipeline()
