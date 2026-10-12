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

## Web interface

```sh
.venv/bin/python app.py
```

Open http://127.0.0.1:8000, enter a query, and click a result to read the
conversation. Matching passages are outlined; load earlier or later messages
to explore the surrounding context. On small screens, use the back arrow to
return to results. The interface uses system fonts and needs no extra packages
or frontend build step. Use `--port 8001` if port 8000 is already occupied.

The server listens only on your computer and runs `main.py` with its existing
defaults, one search at a time. Keep Ollama running and JEV_KEY configured as
above. A single persistent worker shares the loaded corpus, keyword index,
cached embeddings, and embedding model across queries. The first search loads
them; subsequent searches reuse them. Changes to chunk files automatically
rebuild the indices on the next search. Restarting the server resets this cache.
Reloading the page recovers the most recent search in that tab while the
server is running. Pipeline outputs and logs are saved in `.cache/web-<id>/`.
Restarting the server clears the in-memory search; run your query again.

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

Generated lexical keywords are filtered to words in the original query. If the
planner supplies only related terms, the query's own distinctive words are used.
Queries without lexical terms use semantic retrieval alone. Saved plans remain
strictly validated, including keywords and explicitly named participants.

`main.py --stage` also supports `segment`, `plan`, `retrieve`, `fuse`, and `rerank`.
Use `--plan <plan.json>` for retrieval, or `--input <retrieve.json|fuse.json>`
for fusion/reranking. See `main.py --help` for paths, models, aliases, and limits.

## Optional chunk-summary experiment

Original messages are the default retrieval text. Summaries are a separate,
removable experiment in [chunk_summaries/](chunk_summaries/README.md); they never
replace source messages or change the chunk schema.

```sh
# Generate summaries once, locally and sequentially; rerunning resumes the cache.
.venv/bin/python main.py --stage summarize

# Search summaries, or blend them with message retrieval using weighted ranks.
.venv/bin/python main.py -q 'when did I want to work at a startup' --retrieval-text summaries
.venv/bin/python main.py -q 'when did I want to work at a startup' --retrieval-text both --summary-weight 0.5

# Compare all three modes with the same query plan; Jev verification is opt-in.
.venv/bin/python -m chunk_summaries.compare --cases .cache/summary-cases.json -k 10
```

The web interface has the same three modes and a summary-weight control.
Generate summaries before selecting summary retrieval. `--summary-limit 3`
only generates three chunks for a smoke test; searching the whole corpus still
requires coverage of all eligible chunks. See the experiment's README for
small-corpus tests, labeled cases, and clean removal instructions.

## Code

- `segments/`: export parsing and [conversation segmentation](segments/SEGMENTATION.md).
- `chunk_search/`: dense and BM25 indexes.
- `agent_query_operator/`: query planning, retrieval, and reciprocal-rank fusion.
- `jev_reranker/`: Jev client and match-probability ranking.
- `main.py`: pipeline orchestration and saved stage outputs.

Results retain complete messages and source citations. Retrieval cannot guarantee
first-ever or last-ever occurrences. Provider failures and oversized candidates
are reported as errors rather than nonmatches.
