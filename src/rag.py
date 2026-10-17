# rag v1: top k papers with bge + faiss, answer with a local llm (ollama)
# run from root: python src/rag.py "your question"

import sys

import ollama
from sentence_transformers import SentenceTransformer

from retrieval.index import MODEL_NAME, load_index, load_papers, search

LLM_NAME = "qwen2.5:7b"
TOP_K = 3

SYSTEM_PROMPT = """You are a research assistant for AI papers.
Answer the question using only the papers given below.
Cite every paper you use with its arXiv id in square brackets, like [1810.04805].
If the papers don't answer the question, reply exactly: I don't know.
Keep the answer short (5 sentences max)."""


def build_prompt(query, papers):
    text = "Papers:\n\n"
    for paper in papers:
        abstract = " ".join(paper["abstract"].split())
        text += "[" + paper["arxiv_id"] + "] " + paper["title"] + "\n" + abstract + "\n\n"
    text += "Question: " + query
    return text


def answer(query, model, index, ids, papers_by_id, llm=LLM_NAME, k=TOP_K):
    results = search(query, model, index, ids, k=k)
    papers = [papers_by_id[paper_id] for paper_id, score in results]

    response = ollama.chat(
        model=llm,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(query, papers)},
        ],
        options={"temperature": 0, "num_ctx": 4096},  # default ctx may cut the prompt
    )
    return response["message"]["content"], results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: python src/rag.py "your question"')
        sys.exit(1)
    query = sys.argv[1]

    model = SentenceTransformer(MODEL_NAME)
    index, ids = load_index()
    papers_by_id = {paper["arxiv_id"]: paper for paper in load_papers()}

    text, results = answer(query, model, index, ids, papers_by_id)

    print("retrieved:")
    for paper_id, score in results:
        print(" ", paper_id, round(score, 3), papers_by_id[paper_id]["title"])
    print()
    print(text)
