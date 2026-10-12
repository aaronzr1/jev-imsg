# Chunk-summary retrieval experiment

This package owns summary generation, sidecar storage, optional retrieval, and
comparison tooling. Segmentation and the chunk JSON schema stay unchanged.
DenseSearch/SparseSearch accept a generic text callback; rank fusion accepts
generic source labels and weights. Neither core component imports this package.

## Generate summaries

```sh
.venv/bin/python main.py --stage summarize
```

The default summarizer is local Ollama `qwen3:1.7b`, using two CPU threads and
one request at a time. Generate the corpus once before summary searches. This
can take a long time for thousands of chunks; use a small copied chunk corpus
for an initial experiment rather than generating everything immediately.

```sh
# Point all experiment commands at the same small corpus and summary settings.
.venv/bin/python main.py --stage summarize --chunks-dir .cache/experiment/chunks \
  --summaries-dir .cache/experiment/summaries
```

The model writes 1–3 short sentences (target: at most 80 words), preserving
sender labels, intentions, negation, uncertainty, concrete topics, and dates.
Attachment paths are removed; no attachment contents are inferred. Long chunks
are summarized in bounded parts and merged, preserving the ending of every
message rather than silently truncating it. Summaries remain model-generated
hints, with no guarantee of fidelity.

Each success is saved atomically under `.cache/summaries/<hash>.json` with the
chunk ID, source SHA-256, evidence SHA-256, requested/returned model, prompt
version, creation time, and summary. The hash includes evidence, provenance,
model, and prompt version. Changing them makes old artifacts ineligible.
Failures stop generation, preserve completed artifacts, and return a partial
report in the run directory. Rerunning resumes; corrupt sidecars are repaired.

Options: `--summary-model`, `--summary-url`, `--summary-timeout`,
`--summaries-dir`, `--summary-limit`, and `--force-summaries`. A model tag whose
underlying weights have changed requires `--force-summaries`; tag names alone
cannot detect that change. `--summary-limit` limits generation only and does
not silently reduce the search corpus. Existing chunk files must be available;
run `--stage segment` first if necessary.

## Retrieve and verify

```sh
.venv/bin/python main.py -q 'when did I want to work at a startup' --retrieval-text messages
.venv/bin/python main.py -q 'when did I want to work at a startup' --retrieval-text summaries
.venv/bin/python main.py -q 'when did I want to work at a startup' --retrieval-text both --summary-weight 0.7
```

Both dense and BM25 retrieval use the selected text. In `both` mode, each
summary ranking contributes `weight / (rrf_k + rank)` and each message ranking
contributes `(1 - weight) / (rrf_k + rank)`. The weights blend rank votes, avoiding
comparisons between incompatible cosine and BM25 scores. Identical queries are
deduplicated within a representation, while message and summary votes remain
distinct. Weight 0 is equivalent to messages; weight 1 is equivalent to summaries.

Participant scoping always uses original sender metadata. Missing, stale, or
invalid summaries cause an explicit error rather than falling back to messages
or silently excluding chunks. Retrieval settings, generated summaries, and
source contributions are retained in saved search outputs. Jev receives only
the original candidate messages for verification. Results open the original
conversation, and the web interface optionally shows the generated summary.

The persistent worker shares records and the embedding model across text views.
Chunk changes invalidate the corpus snapshot; summary-file changes invalidate
the summary view. The message view stays warm when switching modes.

## Compare on a few queries

Create an ignored JSON file, for example `.cache/summary-cases.json`:

```json
[
  {
    "query": "when did I want to work at a startup",
    "relevant_chunk_ids": ["replace-with-an-actual-chunk-id"]
  },
  { "query": "making plans to eat together" }
]
```

```sh
.venv/bin/python -m chunk_summaries.compare --cases .cache/summary-cases.json -k 10
# Optional: evaluate final Jev-filtered/ranked results too (makes API calls).
.venv/bin/python -m chunk_summaries.compare --cases .cache/summary-cases.json -k 10 --rerank
```

Each query is planned once and the exact saved plan is reused across messages,
summaries, and both. A case may supply an inline `plan` with the same `query` to
skip Ollama planning entirely. Use the same corpus/summaries/model flags as
generation. `--out-dir` must be a new directory.

The output folder contains `comparison.json`, a checkpoint for each completed
case, shared plans, and full per-mode stage outputs with original passages.
The report includes ranked chunk IDs, summaries, stage timings, result-set
overlap, and (when labeled) recall@k, precision@k, reciprocal rank, and macro
averages. Precision divides by the requested k; recall divides by the number
of unique labels. Unlabeled cases have no quality scores, only results/overlap.
Invalid labels are rejected rather than counted as misses. Partial Jev results
are explicitly marked incomplete.

For fair quality comparisons, manually label representative queries before
running, including negation, changing intentions, and speaker/date constraints.
Overlap alone does not establish which mode is better. Initial per-view timings
include index construction; `retrieve.json` records individual query-search
times separately. Keep labels/corpora fixed when sweeping `--summary-weight`.
No Jev calls occur unless `--rerank` is supplied.

## Disable or remove

For immediate rollback, use `--retrieval-text messages` (the default) or select
Original messages in the UI. Summary caches are ignored by Git and need no
migration or restoration of the original corpus.

For source-code removal, delete this package and `tests/test_chunk_summaries.py`,
remove its defaults/options and the summarize/retrieval hooks in `main.py`, and
remove mode/weight/summary-display fields from `app.py` and `web/`. The generic
text callbacks, multiple cached index views, and weighted fusion can remain:
their default behavior is the original message-only pipeline. Remove the
summary-setting test from `tests/test_app.py` and this section from the root
README. There is no dependency, segmentation change, or corpus migration to undo.
