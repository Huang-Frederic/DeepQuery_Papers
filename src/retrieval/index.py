# retrieval v1: bge embeddings + faiss
# build: python src/retrieval/index.py (from root)
# rebuild if model or corpus changes

import json
import os

import faiss
from sentence_transformers import SentenceTransformer

PAPERS_FILE = "data/raw/papers.jsonl"
INDEX_DIR = "data/index"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = ""  # bge query prefix, empty = none


def load_papers():
    papers = []
    with open(PAPERS_FILE, encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def build_index():
    papers = load_papers()
    texts = []
    for paper in papers:
        texts.append(paper["title"] + ". " + paper["abstract"])

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(texts, normalize_embeddings=True)

    # normalized vectors: inner product = cosine
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_DIR + "/papers.faiss")
    with open(INDEX_DIR + "/ids.json", "w") as f:
        json.dump([paper["arxiv_id"] for paper in papers], f)

    print(len(papers), "papers indexed")


def load_index():
    index = faiss.read_index(INDEX_DIR + "/papers.faiss")
    with open(INDEX_DIR + "/ids.json") as f:
        ids = json.load(f)
    return index, ids


def search(query, model, index, ids, k=3):
    vector = model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)
    scores, positions = index.search(vector, k)

    results = []
    for score, pos in zip(scores[0], positions[0]):
        results.append((ids[pos], float(score)))
    return results


if __name__ == "__main__":
    build_index()
