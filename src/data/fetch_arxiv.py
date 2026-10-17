# corpus v0: 5 papers x 4 topics
# run from root

import json
import os

import arxiv

OUTPUT_FILE = "data/raw/papers.jsonl"

# own labels, arxiv categories don't fit
CORPUS_V0 = {
    "nlp": [
        "1810.04805",  # BERT
        "1802.05365",  # ELMo
        "2005.14165",  # GPT-3
        "2201.11903",  # Chain-of-Thought
        "2106.09685",  # LoRA
    ],
    "rag": [
        "2005.11401",  # RAG (Lewis et al.)
        "2407.01219",  # Best Practices in RAG
        "2004.04906",  # DPR
        "2004.12832",  # ColBERT
        "2212.10496",  # HyDE
    ],
    "computer_vision": [
        "1512.03385",  # ResNet
        "2010.11929",  # ViT
        "1506.02640",  # YOLO
        "1505.04597",  # U-Net
        "2103.00020",  # CLIP
    ],
    "reinforcement_learning": [
        "1312.5602",  # DQN
        "1712.01815",  # AlphaZero
        "1707.06347",  # PPO
        "1602.01783",  # A3C
        "1801.01290",  # SAC
    ],
}


def fetch_papers(corpus):
    all_ids = []
    for topic in corpus:
        all_ids += corpus[topic]

    client = arxiv.Client()  # 3s delay + retries
    search = arxiv.Search(id_list=all_ids)

    # arxiv order != ours
    results = {}
    for result in client.results(search):
        paper_id = result.get_short_id().split("v")[0]  # drop version
        results[paper_id] = result

    papers = []
    for topic in corpus:
        for paper_id in corpus[topic]:
            if paper_id not in results:
                print("WARNING: paper not found on arXiv:", paper_id)
                continue
            result = results[paper_id]
            papers.append({
                "arxiv_id": paper_id,
                "topic": topic,
                "title": result.title,
                "abstract": result.summary,
                "authors": [author.name for author in result.authors],
                "published": str(result.published.date()),
                "primary_category": result.primary_category,
                "categories": result.categories,
                "url": result.entry_id,
            })
    return papers


if __name__ == "__main__":
    papers = fetch_papers(CORPUS_V0)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print(len(papers), "papers saved in", OUTPUT_FILE)
