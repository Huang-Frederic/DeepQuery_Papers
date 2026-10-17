# DeepQuery Papers

NLP 2026 course project (ESGI, A. Fouchet): a RAG over scientific papers.

## Goal

Given a natural-language question, find the most relevant paper (or the top 3) in a corpus of NLP papers from arXiv, then generate an answer that cites those papers.

## Approach

We work in iterations and measure every version on the same test set:

1. Corpus v0: about twenty papers (title + abstract) fetched from the arXiv API
2. Test set v0: questions, each with the paper(s) expected to be retrieved
3. Retrieval v1: embeddings + FAISS, measured with MRR and hit@3
4. Generation v1: an LLM answers and cites the retrieved papers
5. Failure analysis, then improvements (BM25, reranker, full-text chunking)
6. Scale up to more data

## Data

arXiv metadata (titles, abstracts, authors) is released under CC0. The `data/` folder is not versioned: the scripts regenerate it.
