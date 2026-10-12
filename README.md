# iMessage search

Search exported conversations with local query planning, dense/BM25 retrieval,
rank fusion, and Jev verification. The phase 1 checkpoint includes a local web
interface and optional chunk-summary retrieval.

## Run

```sh
./install.sh
# Put text exports in raw/ and set JEV_KEY in .env.
.venv/bin/python app.py
```

Open http://127.0.0.1:8000 to search and read conversations around each match.
The frontend uses system fonts and plain HTML/CSS/JavaScript, with no build step.
Use `--port 8001` to change the port. After rebooting, start Ollama or run
`ollama serve` before searching.

For the command line:

```sh
.venv/bin/python main.py -q 'when did I want to work at a startup' -k 10
.venv/bin/python -m imsg --help
```

The installer uses uv and a virtual environment. Local models use two CPU
threads. The web server runs one search at a time, reusing its corpus, indexes,
and embedding model; Jev verification uses up to four concurrent requests.
Jev receives the query and shortlisted original conversation text.

## Pipeline and data

`segment → plan → retrieve → fuse → rerank`

Existing chunks in `segments/chunk_data/` are reused. Pass `--resegment` after
changing exports; choose a fresh `--chunks-dir` when replacing a corpus. Chunk
changes refresh the web worker's indexes on the next search.

Each CLI run saves stage outputs and `timings.json` under
`.cache/run-<timestamp>/`. Completed stages survive later failures. `--out-dir`
must name a new directory. Exit codes are 0 for complete, 1 for failure, and 2
for partial results. Web outputs and logs are under `.cache/web-<id>/`.

Use the same CLI for individual stages:

```sh
.venv/bin/python main.py --stage segment
.venv/bin/python main.py --stage plan -q 'plans to eat together'
.venv/bin/python main.py --stage retrieve --plan .cache/run-<id>/plan.json
.venv/bin/python main.py --stage fuse --input .cache/run-<id>/retrieve.json
.venv/bin/python main.py --stage rerank --input .cache/run-<id>/fuse.json
```

Original messages are the default retrieval text. Generate local summaries with
`--stage summarize`, then select `--retrieval-text summaries` or
`--retrieval-text both --summary-weight 0.5`. The web interface exposes these
modes too. See [summary retrieval](docs/summaries.md) for generation and cache
requirements and [segmentation](docs/segmentation.md) for the chunk schema.

Source messages and citations remain authoritative. Retrieval cannot guarantee
first-ever or last-ever occurrences; provider failures are recorded as errors.
Restarting the server clears its in-memory search and index cache. Reloading a
page recovers the latest search in that tab while the server is running.

## Repository

- `main.py`, `app.py`: CLI and web launchers.
- `imsg/config.py`: shared paths, defaults, and argument types.
- `imsg/cli.py`, `imsg/pipeline.py`, `imsg/server.py`: interfaces and orchestration.
- `imsg/segments/`: export parsing and segmentation.
- `imsg/chunk_search/`: corpus loading and dense/BM25 indexes.
- `imsg/agent_query_operator/`: planning, participant scoping, and rank fusion.
- `imsg/jev_reranker/`: Jev client and match-probability ranking.
- `imsg/chunk_summaries/`: optional summary generation, storage, and retrieval.
- `web/`: frontend assets; `docs/`: references and [phase 2 handoff](docs/phase-2.md).

Exports, generated chunks, credentials, model caches, and logs stay local and
are ignored by Git. Tests, example runners, and development comparison/review
tools were removed for this checkpoint; their source is preserved in commit
`6b5b26a`.
