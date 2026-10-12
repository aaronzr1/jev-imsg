# Intent-search service

`imsg/intent_search/` owns the full search flow: planning, dense/BM25 retrieval,
rank fusion, Jev verification, optional summaries, saved outputs, and reusable
indexes. Segmentation remains in `imsg/segments/`; paths and defaults remain in
`imsg/config.py`. Existing exports, chunks, caches, and root `.env` are reused.

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
- On `done`, `output`: the full existing pipeline result, including `plan`,
  `scope`, `results`, `decisions`, `counts`, `warnings`, `complete`, verification
  settings, and retrieval settings. Matches retain `chunk_id`, source hashes,
  original structured messages with source line ranges, and match probabilities.
- On `done`, `timings`: elapsed seconds per pipeline stage.
- On `error`, `error`: a safe explanation. Detailed logs stay on disk.

`status: done` means the pipeline finished. Check `output.complete` separately:
provider failures can yield partial results with `complete: false` and failed
candidate decisions. The service preserves the existing search semantics;
earliest/latest queries still cannot establish corpus-wide first/last occurrences.

`GET /api/intent-search/config` returns the default retrieval mode, summary
weight, and candidate limit. `GET /api/health` reports that the HTTP service is
up; it does not probe model providers.

One search runs at a time; overlapping submissions return HTTP 409. Jobs are
retained in memory until restart or until displaced by the latest 16 jobs.
Unknown/expired IDs return HTTP 404. A closing service rejects submissions with
HTTP 503. Outputs and logs stay in `.cache/intent-search-<job-id>/`, even after
in-memory expiration. Each worker reuses its corpus and embedding model, with
file signatures refreshing changed indexes. Existing local-host/origin checks
apply; command-line and backend clients can call it without an Origin header.

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
as `pathlib.Path` values. `imsg.intent_search.pipeline.run` remains the direct
stage runner used by the CLI. `main.py`, `python -m imsg`, and the compatibility
imports in `imsg.pipeline` continue to work. Search component imports now live
under `imsg.intent_search`.

The browser's existing `/api/search` routes retain their preview/context format
and use this same service. New integrations should use `/api/intent-search` to
receive the full pipeline output.
