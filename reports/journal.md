# Journal

What we did, why, results and failures. Newest entries at the bottom.
Every evaluation run is also logged in `reports/runs.csv`.

## 2026-10-17 - Corpus v0

- 20 hand-picked arXiv papers, 5 per topic: nlp, rag, computer_vision, reinforcement_learning.
- Title + abstract only, fetched by arXiv id with `src/data/fetch_arxiv.py`.
- Topic is our own label: arXiv categories don't map to AI domains (RL papers are in cs.LG or cs.AI).

Issues:
- First raw API call refused by arXiv (HTTP 429). Switched to the `arxiv` lib (3 s delay + retries).
- arXiv returns papers in a different order than asked. We re-sort, so every run writes the same file.
- The `.gitignore` rule `data/` also hid `src/data/`. Fixed with `/data/`.
- Common names missing from the abstracts: ELMo, DQN, A3C, DPR. CLIP only appears in a GitHub URL.

## 2026-10-17 - Test set v0

- 30 queries in `eval/test_set_v0.jsonl`.
- intent: problem 12, topic 12, similar_paper 6.
- difficulty: direct 4, paraphrase 7, acronym 5, confusable 5, out_of_corpus 3.
- 24 queries written by an LLM agent with no access to the abstracts, to limit wording bias. Expected papers checked by hand against the abstracts.
- 6 similar_paper queries: title + abstract of papers outside the corpus (RoBERTa, DistilBERT, TRPO, Double DQN, Self-RAG, Faster R-CNN).

Issues:
- Without seeing the abstracts, the agent still wrote q19 almost like the YOLO abstract (famous papers are memorized). To rephrase.
- PPO is a right answer for q04, but its abstract never mentions actor-critic. Hard to find from abstracts only.

TODO: group review and rephrasing of the queries.

## 2026-10-17 - Retrieval v1 (run v1_baseline)

- Model BAAI/bge-base-en-v1.5, one vector per paper (title + abstract), normalized.
- FAISS inner product index (= cosine similarity), no query instruction.
- Evaluated on the 27 queries that have an expected paper.

| run | MRR | hit@1 | hit@3 |
|---|---|---|---|
| random ranking (computed) | 0.199 | 0.06 | 0.18 |
| v1_baseline | 0.795 | 0.70 | 0.85 |

By difficulty: direct 1.00, paraphrase 1.00, confusable 0.65, acronym 0.34 (MRR).

Failures:
- Acronyms: ELMo rank 13, CLIP rank 10, DPR rank 5, A3C rank 3. Only DQN found first.
- q13 (SAC) rank 4, behind PPO: the query names PPO.
- q03 (ColBERT) 2nd behind HyDE. q17 (RAG best practices) 2nd behind RAG.
- q26 (DistilBERT abstract) puts GPT-3 before BERT.

Observation: top-1 score is low on failed acronym queries (0.42 to 0.53), at least 0.64 on successes. A threshold could help to answer "I don't know".

Ideas to test: BGE query instruction, add common names to the indexed text, HyDE, BM25 as a second baseline.

## 2026-10-17 - Generation v1, first manual tries

- Local LLM to keep it free: qwen2.5:7b with Ollama on an RTX 4060 (8 GB). Claude Haiku 5.5 was considered (about 2 cents per run) but dropped for cost.
- `src/rag.py`: top 3 papers from retrieval v1, then the LLM answers with arXiv id citations, or "I don't know". Temperature 0, context 4096 tokens.

First tries:
- q06 (LoRA): right paper first, correct answer, cited [2106.09685].
- q12 ("What's CLIP?"): retrieval fails (3 RAG papers), the LLM answers "I don't know". No hallucination, but no answer: the error comes from retrieval.
- q18 (diffusion, out of corpus): "I don't know", followed by an explanation. The prompt asked for exactly "I don't know": the check will have to look at the start of the answer.

## 2026-10-17 - Decision: no formal evaluation of the generated answers

- The single job of our RAG is to find the right papers. Retrieval is what we evaluate objectively (MRR, hit@1, hit@3 on the test set, runs logged in `reports/runs.csv`).
- The generated answer is a presentation layer on top of the retrieved papers. We show it with commented examples (see the first tries above) instead of scoring it.
- Considered and not kept for now: reference answers + ROUGE / semantic similarity (a lot of writing for a weak metric), LLM as a judge.
- Known risks we accept: the LLM could cite a paper it was not given, say "I don't know" when the right paper was retrieved, or answer an out-of-corpus question. Possible follow-up: automatic citation checks (cited ids must be among the given papers).
