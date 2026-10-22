"""
Aggregate a hand-graded endtoend_graded.csv into Table III.

Run after you have filled in the human_correct column:

    python aggregate_endtoend.py --graded endtoend_graded.csv
"""
import argparse
import csv


def to_float(x):
    x = (x or "").strip()
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", default="endtoend_graded.csv")
    args = ap.parse_args()

    with open(args.graded, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ungraded = [r for r in rows if to_float(r.get("human_correct")) is None]
    if ungraded:
        print(f"WARNING: {len(ungraded)} of {len(rows)} rows still have an empty "
              f"human_correct column. Fill those in for an accurate Table III -- "
              f"accuracy below is computed only over the rows that ARE graded.")

    configs = ["RAG mode only (grounded)", "Fallback mode only (ungrounded)", "Full system (adaptive routing)"]
    print("\nTABLE III. END-TO-END ANSWER QUALITY")
    print("| Configuration | Accuracy | Recall | Precision | F1 Score |")
    print("|---|---|---|---|---|")

    for cfg in configs:
        cfg_rows = [r for r in rows if r["config"] == cfg]
        graded = [r for r in cfg_rows if to_float(r["human_correct"]) is not None]
        acc = (sum(to_float(r["human_correct"]) for r in graded) / len(graded)) if graded else None

        p_vals = [to_float(r["retrieved_precision"]) for r in cfg_rows if to_float(r["retrieved_precision"]) is not None]
        r_vals = [to_float(r["retrieved_recall"]) for r in cfg_rows if to_float(r["retrieved_recall"]) is not None]
        f_vals = [to_float(r["retrieved_f1"]) for r in cfg_rows if to_float(r["retrieved_f1"]) is not None]

        avg_p = (sum(p_vals) / len(p_vals)) if p_vals else None
        avg_r = (sum(r_vals) / len(r_vals)) if r_vals else None
        avg_f = (sum(f_vals) / len(f_vals)) if f_vals else None

        def fmt(v):
            return f"{v:.3f}" if v is not None else "n/a"

        note = "" if graded else "  (no graded rows yet)"
        print(f"| {cfg}{note} | {fmt(acc)} | {fmt(avg_r)} | {fmt(avg_p)} | {fmt(avg_f)} |")

    print("\nNote: 'Fallback mode only' performs no retrieval by design, so its "
          "Recall/Precision/F1 are computed against an empty retrieved set and "
          "will read as 0.000 rather than n/a for answerable questions -- that is "
          "expected and itself a meaningful data point (it quantifies how much "
          "worse ungrounded generation is when relevant context does exist).")
    print(f"\nTotal questions in set: {len(set(r['question_id'] for r in rows))} "
          f"-- report this sample size alongside the table; this is a pilot-scale "
          f"evaluation, not a large benchmark.")


if __name__ == "__main__":
    main()