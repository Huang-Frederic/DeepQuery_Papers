# retrieval eval on the test set: MRR, hit@1, hit@3
# run from root, after building the index: python src/evaluate.py <run_name> ["notes"]
# 1 line per run in reports/runs.csv, per-query details in reports/details/

import csv
import json
import os
import sys
from datetime import datetime

from sentence_transformers import SentenceTransformer

from retrieval.index import MODEL_NAME, QUERY_INSTRUCTION, load_index, search

TEST_FILE = "eval/test_set_v0.jsonl"
RUNS_FILE = "reports/runs.csv"
DETAILS_DIR = "reports/details"

GROUPS = [
    ("intent", "problem"),
    ("intent", "topic"),
    ("intent", "similar_paper"),
    ("difficulty", "direct"),
    ("difficulty", "paraphrase"),
    ("difficulty", "acronym"),
    ("difficulty", "confusable"),
]


def print_scores(name, rows):
    n = len(rows)
    mrr = sum(1 / row["rank"] for row in rows) / n
    hit1 = sum(1 for row in rows if row["rank"] == 1) / n
    hit3 = sum(1 for row in rows if row["rank"] <= 3) / n
    print(f"{name:<24} n={n:<3} MRR={mrr:.3f}  hit@1={hit1:.2f}  hit@3={hit3:.2f}")
    return round(mrr, 3), round(hit1, 2), round(hit3, 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/evaluate.py <run_name> [notes]")
        sys.exit(1)
    run_name = sys.argv[1]
    notes = sys.argv[2] if len(sys.argv) > 2 else ""

    queries = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))

    model = SentenceTransformer(MODEL_NAME)
    index, ids = load_index()

    rows = []
    for q in queries:
        if not q["expected_ids"]:  # out_of_corpus, no rank
            continue

        results = search(q["query"], model, index, ids, k=len(ids))  # full ranking
        ranked_ids = [paper_id for paper_id, score in results]

        # rank of the first expected paper
        rank = min(ranked_ids.index(paper_id) for paper_id in q["expected_ids"]) + 1

        rows.append({
            "id": q["id"],
            "intent": q["intent"],
            "difficulty": q["difficulty"],
            "rank": rank,
            "expected": " ".join(q["expected_ids"]),
            "top3": " ".join(ranked_ids[:3]),
            "top1_score": round(results[0][1], 3),
        })

    run = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run": run_name,
        "model": MODEL_NAME,
        "instruction": QUERY_INSTRUCTION or "none",
        "test_set": TEST_FILE,
        "notes": notes,
        "n": len(rows),
    }
    run["mrr"], run["hit1"], run["hit3"] = print_scores("all", rows)
    for field, value in GROUPS:
        mrr, hit1, hit3 = print_scores(field + "=" + value, [row for row in rows if row[field] == value])
        run["mrr_" + value] = mrr

    os.makedirs(DETAILS_DIR, exist_ok=True)
    with open(DETAILS_DIR + "/" + run_name + ".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    new_file = not os.path.exists(RUNS_FILE)
    with open(RUNS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(run)

    print("run saved in", RUNS_FILE)
