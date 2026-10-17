# DeepQuery Papers

NLP 2026 course project (ESGI, A. Fouchet): a RAG over scientific papers.

## Goal

Given a natural-language question, find the most relevant paper (or the top 3) in a corpus of AI papers from arXiv, then generate an answer that cites those papers.

Long term, the corpus should cover the main domains an AI engineer needs to know.

## Approach

We work in iterations and measure every version on the same test set:

1. Corpus v0: 20 hand-picked papers (title + abstract), 5 per topic: NLP, RAG, computer vision, reinforcement learning
2. Test set v0: questions, each with the paper(s) expected to be retrieved
3. Retrieval v1: embeddings + FAISS, measured with MRR and hit@3
4. Generation v1: an LLM answers and cites the retrieved papers
5. Failure analysis, then improvements (BM25, reranker, full-text chunking)
6. Scale up to more data

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python src/data/fetch_arxiv.py  # corpus v0 -> data/raw/papers.jsonl
python src/retrieval/index.py   # embeddings + FAISS index -> data/index/
python src/evaluate.py <run_name> ["notes"]  # MRR, hit@1, hit@3 -> reports/runs.csv
```

Run every command from the project root. Each evaluation adds one line to `reports/runs.csv` (parameters + scores) and writes per-query details in `reports/details/<run_name>.csv`. Progress and failures are logged in `reports/journal.md`.

## Data

arXiv metadata (titles, abstracts, authors) is released under CC0. The `data/` folder is not versioned: the scripts regenerate it.
