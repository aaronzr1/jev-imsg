# iMessage search

Search exported conversations with local query planning, dense/BM25 retrieval,
rank fusion, and Jev verification.

## Setup and run

```sh
./install.sh
# Put exports in raw/ and set JEV_KEY in .env.
.venv/bin/python main.py -q 'when did I want to work at a startup' -k 10
```

The installer sets up uv, a Python virtual environment, Ollama, and the default
models. After a reboot, start Ollama (`ollama serve`) before searching.
Planning and embeddings run locally with two CPU threads; Jev receives the query
and shortlisted conversation text, with up to four concurrent requests.

Existing chunks in `segments/chunk_data/` are reused. Pass `--resegment` after
changing exports; use a fresh `--chunks-dir` when removing or replacing a corpus.
Exports, chunks, credentials, and caches are ignored by Git.

Each run saves `plan.json`, `retrieve.json`, `fuse.json`, `rerank.json`, and
`timings.json` under a new `.cache/run-<timestamp>/` directory. Completed stages
survive later failures. `--out-dir` must name a new directory. Exit codes:
0 complete, 1 failed, 2 partial reranking results.

## Debug individual stages

Run from the repository root:

```sh
.venv/bin/python -m segments.example
.venv/bin/python -m chunk_search.example --method sparse
.venv/bin/python -m agent_query_operator.example
.venv/bin/python -m jev_reranker.example --input .cache/run-<timestamp>/fuse.json
```

Chunk search supports `--method dense`, `sparse`, or `both` (default).
The query operator accepts `--plan` to reuse a saved plan.

`main.py --stage` also supports `segment`, `plan`, `retrieve`, `fuse`, and `rerank`.
Use `--plan <plan.json>` for retrieval, or `--input <retrieve.json|fuse.json>`
for fusion/reranking. See `main.py --help` for paths, models, aliases, and limits.

## Code

- `segments/`: export parsing and [conversation segmentation](segments/SEGMENTATION.md).
- `chunk_search/`: dense and BM25 indexes.
- `agent_query_operator/`: query planning, retrieval, and reciprocal-rank fusion.
- `jev_reranker/`: Jev client and match-probability ranking.
- `main.py`: pipeline orchestration and saved stage outputs.

Results retain complete messages and source citations. Retrieval cannot guarantee
first-ever or last-ever occurrences. Provider failures and oversized candidates
are reported as errors rather than nonmatches.
