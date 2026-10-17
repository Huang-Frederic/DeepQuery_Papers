# retrieval eval on the test set: MRR, hit@1, hit@3
# run from root, after building the index: python src/evaluate.py

import csv
import json
import os

from sentence_transformers import SentenceTransformer

from retrieval.index import MODEL_NAME, load_index, search

TEST_FILE = "eval/test_set_v0.jsonl"
REPORT_FILE = "reports/retrieval_v1.csv"


def print_scores(name, rows):
    n = len(rows)
    mrr = sum(1 / row["rank"] for row in rows) / n
    hit1 = sum(1 for row in rows if row["rank"] == 1) / n
    hit3 = sum(1 for row in rows if row["rank"] <= 3) / n
    print(f"{name:<24} n={n:<3} MRR={mrr:.3f}  hit@1={hit1:.2f}  hit@3={hit3:.2f}")


if __name__ == "__main__":
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

    print_scores("all", rows)
    for intent in ["problem", "topic", "similar_paper"]:
        print_scores("intent=" + intent, [row for row in rows if row["intent"] == intent])
    for difficulty in ["direct", "paraphrase", "acronym", "confusable"]:
        print_scores("difficulty=" + difficulty, [row for row in rows if row["difficulty"] == difficulty])

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("details saved in", REPORT_FILE)
