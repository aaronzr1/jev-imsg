# Intent-search service

The HTTP API and Python service run the same search as the web interface:
query planning, MiniLM/BM25 retrieval, rank fusion, and Jev reranking.
Paths and defaults are defined in `imsg/config.py`.

## Start and call the API

```sh
.venv/bin/python -m imsg.intent_search --port 8001
```

This starts an API-only server bound to `127.0.0.1`. It does not need the `web/`
assets. `app.py` also exposes these endpoints alongside the browser interface,
using one shared search worker. Run either launcher for the desired interface;
running both creates separate workers and model caches in memory.

```sh
curl -sS http://127.0.0.1:8001/api/intent-search \
  -H 'Content-Type: application/json' \
  -d '{"query":"when did I want to work at a startup","top_k":10}'
```

The response is HTTP 202 with a `Location` header and this JSON:

```json
{
  "id": "<job-id>",
  "status": "running",
  "status_url": "/api/intent-search/<job-id>",
  "schema_version": 1
}
```

Poll the URL until `status` is `done` or `error`:

```sh
curl -sS http://127.0.0.1:8001/api/intent-search/<job-id>
```

`POST /api/intent-search` accepts these JSON fields:

- `query` (required): nonblank string, at most 2,000 characters.
- `top_k`: integer from 1 to 100, default 100. Limits fused candidates sent for
  verification; the final match count can be lower.
- `retrieval_text`: `messages` (default), `summaries`, or `both`.
- `summary_weight`: number from 0 to 1, default 0.5; used with `both`.

Optional fields omitted or set to `null` use the service defaults. Summary modes
require current generated summaries; see [summary retrieval](summaries.md).
Unknown fields, malformed JSON, and invalid values return HTTP 400. The request
body must use `application/json` and be no larger than 8,192 bytes.

`GET /api/intent-search/<job-id>` returns HTTP 200 for a retained job, with:

- `schema_version`, `id`, `query`, `status`, and the request's retrieval settings.
- `completed_stages`: saved stages among `segment`, `plan`, `retrieve`, `fuse`,
  and `rerank`. Existing chunks usually skip `segment`.
- On `done`, `output`: the search result, including `plan`,
  `scope`, `results`, `decisions`, `counts`, `warnings`, `complete`, verification
  settings, and retrieval settings. Matches retain `chunk_id`, source hashes,
  original structured messages with source line ranges, and match probabilities.
- On `done`, `timings`: elapsed seconds per pipeline stage.
- On `error`, `error`: a safe explanation. Detailed logs stay on disk.

`status: done` means the pipeline finished. Check `output.complete` separately:
provider failures can yield partial results with `complete: false` and failed
candidate decisions. Search cannot confirm the first or last occurrence in the
archive.

`GET /api/intent-search/config` returns the default retrieval mode, summary
weight, and candidate limit. `GET /api/health` reports that the HTTP service is
up; it does not probe model providers.

One search runs at a time; overlapping submissions return HTTP 409. Jobs are
retained in memory until restart or until displaced by the latest 16 jobs.
Unknown/expired IDs return HTTP 404. A closing service rejects submissions with
HTTP 503. Outputs and logs stay in `.cache/intent-search-<job-id>/`, even after
in-memory expiration. Each worker reuses its corpus and embedding model, with
changed files triggering an index refresh. Requests must use a local Host and,
if supplied, a local Origin header. Command-line clients can omit Origin.

## Python integration

Within this repository, clients can use the same service without HTTP:

```python
import time
from imsg.intent_search import IntentSearchService

search = IntentSearchService()
try:
    job_id = search.start("when did I want to work at a startup", top_k=10)
    if job_id is None:
        raise RuntimeError("A search is already running")
    while True:
        job = search.status(job_id)
        if job["status"] == "error":
            raise RuntimeError(job["error"])
        if job["status"] == "done":
            output = job["output"]
            break
        time.sleep(1)
finally:
    search.close()
```

Reuse one service instance across queries to keep indexes warm. Server settings
can be passed to `IntentSearchService(settings={...})`, with filesystem settings
as `pathlib.Path` values. `imsg.intent_search.pipeline.run` runs individual
stages and is used by `main.py` and `python -m imsg`.

The browser uses `/api/search` for previews and conversation context.
Use `/api/intent-search` for the full search output.

## CLI outputs

Each CLI run saves stage results and `timings.json` in `.cache/run-<timestamp>/`.
`--out-dir` must name a new directory. Completed stages are saved even if a later
stage fails. Exit codes: 0 complete, 1 failed, 2 partial results.

Use `--stage plan`, `retrieve`, `fuse`, or `rerank` to run one stage.
`--plan <plan.json>` reuses a saved plan; `--input <results.json>` supplies
retrieval results for `fuse` or fused results for `rerank`. Run
`.venv/bin/python main.py --help` for all options.
