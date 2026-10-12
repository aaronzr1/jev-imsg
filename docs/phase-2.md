# Phase 2 handoff

Phase 1 is preserved in commit `6b5b26a` before cleanup. The cleaned baseline is
tagged `phase-1-checkpoint`. No phase 2 feature scope has been selected yet.

## Starting point

The CLI and local web interface share `imsg.pipeline.run`. A single web worker
owns `SearchIndexes`, which reuses the corpus and embedding model across message
and summary views. File signatures invalidate stale views. Saved stage outputs
make failures inspectable without rerunning completed stages.

The default path searches original messages: local Ollama planning, local
MiniLM/BM25 retrieval, reciprocal-rank fusion, then Jev verification. Generated
lexical terms are grounded in the query; saved plans are strictly validated.
Participant scoping uses sender metadata. Jev verifies original messages even
when retrieval uses summaries.

Summary generation is sequential, resumable, and stored in separate sidecars.
Summary retrieval requires current coverage of all eligible chunks. It remains
optional and defaults off.

## Working boundaries

- Keep shared defaults in `imsg/config.py` and orchestration in `imsg/pipeline.py`.
- Keep interfaces in `imsg/cli.py`, `imsg/server.py`, and `web/`.
- Keep retrieval and verification components independent of interface code.
- Preserve schema-version-1 chunks, source hashes, message IDs, and line citations
  unless phase 2 explicitly requires a migration.
- Install Python dependencies with uv in `.venv`; use bun if frontend dependencies
  become necessary. The current frontend needs no package installation.
- Budget for an eight-core CPU: two local inference threads, one web search
  worker, sequential summary generation, and four concurrent Jev requests.

The existing corpus remains in `raw/` and `segments/chunk_data/` to preserve
local state. The latter is a data directory; segmentation source now lives in
`imsg/segments/`. Caches and credentials also stay outside Git. A source
checkpoint does not back up this local data.

## Decisions for phase 2

Choose the next product outcome before adding features. In particular, decide
whether summary retrieval deserves continued investment, whether import/corpus
management needs a user interface, and whether earliest/latest queries need
broader retrieval or an explicit coverage measure. These are open decisions,
not promises of retrieval accuracy.

The previous 34 checks passed before the requested test removal. New automated
coverage should follow the chosen phase 2 behavior rather than restoring the
old suite wholesale. The old suite, comparison harness, stage examples, and
review converter remain available from the pre-cleanup commit.

The cleaned source passed 33 applicable archived checks using updated import
paths; the check for the removed comparison harness was excluded. No test files
were restored. All modules imported, CLI help and installer syntax passed, and
a temporary localhost server served the frontend assets and configuration while
rejecting invalid hosts and empty queries. Credentials still resolve from the
root `.env`. Verification did not make model or Jev API calls.

## Verify and start

```sh
.venv/bin/python -m imsg --help
.venv/bin/python app.py --help
.venv/bin/python -m imsg.segments.segment --help
bash -n install.sh
git status --short
```

For a live search, ensure Ollama is running and JEV_KEY is configured, then run
the documented CLI or web launcher. A full search makes Jev API calls.
