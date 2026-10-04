# Intent → retrieval → candidate shortlist

This is a bounded query-planning agent plus a deterministic answer-operator
baseline. One local LLM call produces a validated plan (at most one repair call).
Retrieval runs sequentially against the existing `chunk_search` indexes; the
operator merges their rankings using reciprocal-rank fusion (RRF).

## Default-loaded example

```sh
# No arguments or LLM needed: startup plan → retrieval → kept and pruned chunks.
.venv/bin/python agent_query_operator/examples/run_examples.py

# Save all retrieval results and operator decisions for inspection/replay.
.venv/bin/python agent_query_operator/examples/run_examples.py \
  --out .cache/startups-example.json

# Operator-only replay, with a different cutoff; no model or index loading.
.venv/bin/python agent_query_operator/examples/run_examples.py \
  --input .cache/startups-example.json -k 2
```

Defaults are editable at the top of `examples/run_examples.py`: the existing
`startups.plan.json`, five hits per retrieval query, three final results, and
`rrf_k=60`. Default paths resolve from the repository, even when invoked elsewhere.
The script also works as `python -m agent_query_operator.examples.run_examples`.

The output shows every retrieval list, repeated IDs combined across queries,
kept candidates with their RRF contributions, and every pruned candidate with
its global rank, score, contributions, excerpt, and cutoff reason. Duplicate
query lists and duplicate hits within a list are reported separately if present.
**Pruned means below the top-k cutoff, not verified irrelevant.**

Use `--plan` for another saved plan, `--per-query` to broaden retrieval, and
`--full` for complete transcripts. `--input` accepts saved example bundles,
regular search bundles, or the existing chunk-search list-of-runs format.
The JSON output contains `runs` plus an `operator` object with `kept`, `pruned`,
deduplication counts and details. No files are written unless `--out` is supplied;
existing output files are never overwritten. Keep private output in `.cache/`.

## Try it

Run from the repository root. The existing `.venv` and requirements suffice;
the new code uses only the Python standard library. For a fresh environment:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Ollama must be running (`ollama serve` if the application isn't already serving).
The default is `qwen3:1.7b`, with thinking disabled, two CPU threads, 4096 context
tokens and at most 1024 output tokens. The model download is already installed
on this machine; on another machine run `ollama pull qwen3:1.7b`.

```sh
# 1. Intent → readable queries + reusable JSON; does not load the corpus.
.venv/bin/python -m agent_query_operator plan \
  -q 'find me the first time I mentioned wanting to pivot into startups' \
  --out .cache/my-plan.json

# 2. Inspect/edit that plan, then run local retrieval and fusion without an LLM call.
HF_HUB_OFFLINE=1 .venv/bin/python -m agent_query_operator search \
  --plan .cache/my-plan.json --per-query 20 -k 10 --show-runs \
  --out .cache/my-search.json

# 3. Change fusion limits and inspect contributions without rerunning retrieval.
.venv/bin/python -m agent_query_operator merge .cache/my-search.json \
  -k 5 --rrf-k 60 --out .cache/my-merged.json

# Or perform all stages with one command:
.venv/bin/python -m agent_query_operator search \
  -q 'find when I wanted academic research instead of another internship' -k 5

# Model-free example: hand-authored, not a claimed LLM output.
HF_HUB_OFFLINE=1 .venv/bin/python -m agent_query_operator search \
  --plan agent_query_operator/examples/startups.plan.json -k 3
```

Output files must be new; existing files are never overwritten. Keep private
plans and results under ignored `.cache/`. `--full` shows complete transcripts;
the default excerpt shows only the first 600 characters, not necessarily the
matching passage. JSON retains complete messages and source-line citations.
`merge` also accepts the list-of-runs JSON exported by
`chunk_search/search_examples.py`. It does not need Ollama or dense dependencies.

`--model`, `--url`, and `--timeout` configure the planner. The URL is an Ollama
server base URL, not a chat-completions endpoint. A remote URL sends query text
there; the default is localhost. The planner receives only the query and fixed
instructions/examples, never transcripts or a contact catalog. No `.env` file
or API credentials are loaded. The small client in `retrieval/planner.py` can be
replaced with a provider returning the same validated plan.

## Plan and retrieval

The plan contains `query` (owned by the application), `rewrites` (2–4),
`keywords`, `participants`, `requirements`, and `answer_mode`
(`relevance`, `earliest`, or `latest`). The prompt encourages deliberately
different conversational angles and preserves direction, speaker, and negation.
Validation enforces shape, bounds, distinct strings, literal keywords, and named
participants present in the query. Keyword cleanup removes retrieval boilerplate
and duplicate tokens. These checks cannot establish semantic fidelity or diversity;
inspect plans, especially with this small model. Invalid plans get one repair
attempt, then a visible error. Network failures surface immediately.

Every execution searches:

1. Dense retrieval on the original query, unchanged.
2. Dense retrieval on each rewrite.
3. One BM25 query joining the extracted keywords. This is the existing local
   BM25 implementation, with OR word matching and no phrase/Boolean syntax.

Each list contains at most `--per-query` hits (default 20). The indexes retain
their existing two-thread embedding configuration and disk caches. Dense search
has no calibrated relevance threshold and can return unrelated candidates.

Participant names are scope hints for people spoken **to/with**, not people
merely discussed. Exact, case-insensitive sender labels resolve locally. When
exports use phone numbers, pass `--aliases .cache/contacts.json`, containing,
for example, `{"Alex": ["+15550000000"]}`. No identity is guessed. Unknown names
produce a warning and remain unfiltered; the next decision step must verify them.
Multiple resolved names require all of them in the conversation; multiple labels
for one alias are alternatives. Membership is the union of participants across
the source file, so the person needn't speak in the matching chunk itself.

When scoped, the current search APIs require ranking the whole snapshot, then
filtering by source **before** the per-query cutoff. This avoids losing relevant
threads behind unrelated global top-k hits, but costs more than a native filtered
index. Date, authorship and semantic evidence requirements are preserved for the
next stage, not enforced as hard filters. Aliases are trusted caller configuration.

## Operator and handoff

`operator/fusion.py` scores each chunk as `sum(1 / (rrf_k + rank))`, with
one-based ranks and `rrf_k=60` by default. Each distinct method/query list has
equal weight, so the 3–5 dense lists collectively get more influence than the
single sparse list. This is plain RRF, not calibrated probability or modality
balancing. Raw BM25/cosine scores are retained for inspection but never added.

Within each list, duplicate chunk IDs are removed before assigning ranks. Across
lists, the same ID accumulates votes. Repeated identical method/query lists count
once. Ties sort by chunk ID. Conflicting source hashes, source files or chunk
payloads for the same ID across contributing lists raise an error instead of
silently mixing corpus versions. Near-duplicate text with different IDs remains.

Fused results have `rrf_score` and `contributions` (query, method, rank, original
score and RRF contribution), plus unchanged citation metadata and the complete
chunk. The search bundle contains the plan, all individual runs, scope, warnings,
settings, and fused `results`. The `top_k` cutoff bounds what reaches the next
stage. Dropped candidates remain available in `runs` for replay.

**This stage returns candidates, not verified answers.** RRF does not determine
which results are irrelevant. `requirements` and `answer_mode` are explicit
handoff fields for a later evidence-verification step. In particular, ranking
or sorting a bounded shortlist cannot prove the first-ever mention, and a chunk's
start time is not necessarily the supporting message's timestamp. No automatic
chronological reordering or unsupported “first time” claim is made here.

Runtime references: [Ollama chat API](https://docs.ollama.com/api/chat),
[structured outputs](https://docs.ollama.com/capabilities/structured-outputs),
[Qwen3 1.7B download](https://ollama.com/library/qwen3:1.7b).
