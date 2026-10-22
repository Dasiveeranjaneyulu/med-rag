"""
End-to-end evaluation -> feeds Table III.

Run from your project root:

    python eval_endtoend.py --questions questions_template.csv

This calls your real pipeline under three configurations and writes
endtoend_graded.csv with an empty `human_correct` column for you to fill in
by hand (see README.md — free-text medical answer correctness needs a human
reader, not a keyword match).
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_questions(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            snippets = [s.strip() for s in (r.get("gold_snippets") or "").split(";;") if s.strip()]
            keywords = [k.strip() for k in (r.get("expected_answer_keywords") or "").split(";;") if k.strip()]
            rows.append({"id": r["id"], "question": r["question"], "gold_snippets": snippets, "keywords": keywords})
    return rows


def is_relevant(doc_text, gold_snippets):
    if not gold_snippets:
        return False
    low = doc_text.lower()
    return any(s.lower() in low for s in gold_snippets)


def snippets_covered(doc_texts, gold_snippets):
    low_docs = [d.lower() for d in doc_texts]
    return sum(1 for s in gold_snippets if any(s.lower() in d for d in low_docs))


def retrieval_scores(docs, gold_snippets):
    if not gold_snippets:
        return "", "", ""  # unanswerable-by-design question, retrieval scoring not meaningful
    if not docs:
        return 0.0, 0.0, 0.0  # e.g. fallback config that performed no retrieval
    relevant = sum(is_relevant(d, gold_snippets) for d in docs)
    precision = relevant / len(docs)
    covered = snippets_covered(docs, gold_snippets)
    recall = covered / len(gold_snippets)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


def keyword_hit(answer, keywords):
    if not keywords:
        return ""
    low = answer.lower()
    return int(any(k.lower() in low for k in keywords))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="questions_template.csv")
    args = ap.parse_args()

    questions = load_questions(args.questions)

    from backend.app.services.rag_service import MedicalRAGPipeline
    from backend.app.services.llm_router import generate_general_response
    from backend.app.models.embeddings import embedding_model

    pipeline = MedicalRAGPipeline()

    out_rows = [[
        "config", "question_id", "question", "retrieved_precision", "retrieved_recall",
        "retrieved_f1", "keyword_hit", "answer", "human_correct"
    ]]

    for q in questions:
        query, gold, kws = q["question"], q["gold_snippets"], q["keywords"]
        emb = embedding_model.embed_query(query)
        top_docs, top_scores = pipeline.hybrid_retrieval(query, emb)
        context = "\n\n---\n\n".join(top_docs)

        # Config 1: RAG-only, forced -- always generate from retrieved context
        # regardless of the relevance threshold, to isolate grounded-generation
        # quality from the threshold-gating decision.
        try:
            rag_answer = pipeline.generate_rag_response(query, context)
        except Exception as e:
            rag_answer = f"[error: {e}]"
        p, r, f1 = retrieval_scores(top_docs, gold)
        out_rows.append(["RAG mode only (grounded)", q["id"], query, p, r, f1,
                          keyword_hit(rag_answer, kws), rag_answer, ""])

        # Config 2: Fallback-only, forced -- always use the general LLM path,
        # ignoring retrieval entirely, regardless of what was retrieved.
        try:
            fallback_answer = generate_general_response(query)
        except Exception as e:
            fallback_answer = f"[error: {e}]"
        # No retrieval used in this config by definition.
        p2, r2, f12 = retrieval_scores([], gold)
        out_rows.append(["Fallback mode only (ungrounded)", q["id"], query, p2, r2, f12,
                          keyword_hit(fallback_answer, kws), fallback_answer, ""])

        # Config 3: Full system, adaptive routing -- exactly what the deployed
        # app does today (your real run_with_details, threshold and all).
        try:
            result = pipeline.run_with_details(query)
            full_answer = result["answer"]
            full_docs = result["sources"]
        except Exception as e:
            full_answer, full_docs = f"[error: {e}]", []
        p3, r3, f13 = retrieval_scores(full_docs, gold)
        out_rows.append(["Full system (adaptive routing)", q["id"], query, p3, r3, f13,
                          keyword_hit(full_answer, kws), full_answer, ""])

    with open("endtoend_graded.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)

    print(f"Wrote endtoend_graded.csv with {len(out_rows) - 1} rows "
          f"({len(questions)} questions x 3 configurations).")
    print("Next: open endtoend_graded.csv and fill in the human_correct column "
          "(1 = correct, 0 = wrong, 0.5 = partially correct) by reading each answer.")
    print("Then run: python aggregate_endtoend.py --graded endtoend_graded.csv")


if __name__ == "__main__":
    main()