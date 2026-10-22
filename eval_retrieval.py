"""
Retrieval-level evaluation -> Table II.

Run from your project root (the directory containing backend/), so that
`from backend.app... import ...` resolves the same way your app does:

    python eval_retrieval.py --questions questions_template.csv --top_k 10

Requires your real .env / GOOGLE_API_KEY and an already-ingested ChromaDB
store (i.e. you have already run ingest_pdfs.py or uploaded PDFs through the
running app) for these numbers to mean anything.
"""
import argparse
import csv
import sys
from pathlib import Path

# Make sure project root is importable, same pattern your own scripts use.
sys.path.insert(0, str(Path(__file__).parent))


def load_questions(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            snippets = [s.strip() for s in (r.get("gold_snippets") or "").split(";;") if s.strip()]
            rows.append({"id": r["id"], "question": r["question"], "gold_snippets": snippets})
    return rows


def is_relevant(doc_text, gold_snippets):
    if not gold_snippets:
        return False
    low = doc_text.lower()
    return any(s.lower() in low for s in gold_snippets)


def snippets_covered(doc_texts, gold_snippets):
    """How many distinct gold snippets are found across the retrieved docs."""
    low_docs = [d.lower() for d in doc_texts]
    covered = 0
    for s in gold_snippets:
        if any(s.lower() in d for d in low_docs):
            covered += 1
    return covered


def score_method(name, doc_lists_by_q, questions):
    """doc_lists_by_q: list of list[str] retrieved doc texts, aligned to questions."""
    hits = 0          # for hit-rate "accuracy"
    precisions, recalls, f1s = [], [], []
    per_query_rows = []

    for q, docs in zip(questions, doc_lists_by_q):
        gold = q["gold_snippets"]
        if not gold:
            # unanswerable-by-design question: skip from retrieval P/R, note it
            per_query_rows.append([name, q["id"], "n/a (no gold snippets)", "", "", ""])
            continue

        relevant_flags = [is_relevant(d, gold) for d in docs]
        retrieved_relevant = sum(relevant_flags)
        precision = retrieved_relevant / len(docs) if docs else 0.0
        covered = snippets_covered(docs, gold)
        recall = covered / len(gold) if gold else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if retrieved_relevant > 0:
            hits += 1

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_query_rows.append([name, q["id"], round(precision, 3), round(recall, 3), round(f1, 3), retrieved_relevant])

    n_scored = len(precisions)
    hit_rate = hits / n_scored if n_scored else 0.0
    avg_p = sum(precisions) / n_scored if n_scored else 0.0
    avg_r = sum(recalls) / n_scored if n_scored else 0.0
    avg_f1 = sum(f1s) / n_scored if n_scored else 0.0

    return {
        "method": name,
        "accuracy_hit_rate": hit_rate,
        "precision": avg_p,
        "recall": avg_r,
        "f1": avg_f1,
        "n_scored": n_scored,
    }, per_query_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="questions_template.csv")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if all(not q["gold_snippets"] for q in questions):
        print("ERROR: no gold_snippets found in any row. Edit questions_template.csv "
              "with real questions and real snippets before running this.")
        sys.exit(1)

    # Import the real app pieces. This will fail loudly if run outside your
    # project root or without your real .env — that is intentional; it means
    # you are not accidentally scoring against nothing.
    from backend.app.services.rag_service import MedicalRAGPipeline
    from backend.app.models.embeddings import embedding_model
    from backend.app.core import config as _config

    pipeline = MedicalRAGPipeline()

    bm25_docs_by_q, chroma_docs_by_q, prererank_docs_by_q, hybrid_rerank_docs_by_q = [], [], [], []

    for q in questions:
        query = q["question"]
        emb = embedding_model.embed_query(query)

        bm25_docs, _ = pipeline.bm25.query(query, n_results=args.top_k)
        chroma_res = pipeline.chroma.query(query_embeddings=[emb], n_results=args.top_k)
        chroma_docs = chroma_res.get("documents", [[]])[0]

        seen, prererank = set(), []
        for d in bm25_docs + chroma_docs:
            if d not in seen:
                seen.add(d)
                prererank.append(d)
        prererank = prererank[: args.top_k]

        hybrid_docs, hybrid_scores = pipeline.hybrid_retrieval(query, emb)

        bm25_docs_by_q.append(bm25_docs)
        chroma_docs_by_q.append(chroma_docs)
        prererank_docs_by_q.append(prererank)
        hybrid_rerank_docs_by_q.append(hybrid_docs)

    results = []
    all_rows = [["method", "question_id", "precision", "recall", "f1", "n_relevant_retrieved"]]
    for name, doc_lists in [
        ("BM25 only", bm25_docs_by_q),
        ("Semantic (ChromaDB) only", chroma_docs_by_q),
        ("Hybrid (BM25 + ChromaDB)", prererank_docs_by_q),
        ("Hybrid + Cross-Encoder Reranking", hybrid_rerank_docs_by_q),
    ]:
        summary, rows = score_method(name, doc_lists, questions)
        results.append(summary)
        all_rows.extend(rows)

    with open("retrieval_results.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(all_rows)

    n_answerable = sum(1 for q in questions if q["gold_snippets"])
    n_total = len(questions)
    print(f"\nScored {n_answerable} of {n_total} questions (rows with no gold_snippets "
          f"are excluded from retrieval scoring; use them for Table III fallback testing).\n")

    print("TABLE II. RETRIEVAL PERFORMANCE BY METHOD")
    print("| Retrieval Method | Accuracy (hit rate) | Recall | Precision | F1 Score |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['method']} | {r['accuracy_hit_rate']:.3f} | {r['recall']:.3f} | "
              f"{r['precision']:.3f} | {r['f1']:.3f} |")
    print("\nPer-query breakdown written to retrieval_results.csv")


if __name__ == "__main__":
    main()