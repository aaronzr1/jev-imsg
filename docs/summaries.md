# Chunk-summary retrieval experiment

`imsg/intent_search/chunk_summaries/` owns summary generation, sidecar storage, and optional
retrieval. Segmentation and the chunk JSON schema stay unchanged.
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

## Disable

Use `--retrieval-text messages` (the default), or select Original messages in
the web interface. Summary caches are ignored by Git; disabling the feature
requires no corpus migration.
