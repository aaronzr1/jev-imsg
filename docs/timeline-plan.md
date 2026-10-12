# Person conversation timeline: implementation plan

Status: architecture accepted; implementation handoff to Codex Cloud.

## Outcome and scope

Build a local timeline of important events across conversations with a selected
person. Zooming out shows significant episodes and turning points; zooming in
reveals individual events and original-message evidence. Reuse the components
behind `/api/intent-search` without changing its current response contract.

Working defaults from the accepted proposal:

- Select direct messages and explicitly selected group chats containing the
  resolved person. Exclude mention-only chats from version one.
- Check person involvement per event; whole-chat membership alone is insufficient.
- Store relationship-impact and life/shared-event significance separately.
  Default the UI to relationship impact, with a switch to the other view.
- Persist atomic events and conservatively linked episodes. Defer automatic
  multi-year relationship narratives.
- Build events by scanning every selected conversation, not by bounded search.
- Keep evidence support, significance, and retrieval relevance separate.
- Keep the existing Python backend and plain JavaScript frontend initially.
  Use standard-library SQLite for durable local state.

## Instructions for the cloud implementation task

Read this plan, `README.md`, and `docs/phase-2.md`, then inspect the current
implementation before editing. Implement the milestones below in order, with
small reviewable changes. Maintain this document's progress log with completed
work, verification, remaining work, and justified deviations.

The user's repository instructions apply:

- Install requirements in a virtual environment. Use `uv` for Python and `bun`
  for TypeScript/JavaScript dependencies; do not use pip or npm.
- Budget for an eight-core local machine. Use two local inference threads, one
  bounded work scheduler, and at most four in-flight JEV requests across search
  and timeline work. Avoid heavy parallel processes.
- Before code changes, inspect unstaged changes and the staging area. If existing
  changes are unstaged and staging is empty, stage them before proceeding. If
  staging is already nonempty when beginning work, wait for user confirmation
  before proceeding. The handoff baseline should be clean.
- Before every commit, inspect the files for secrets and large data artifacts.
  Never commit `~/.Codex.json` or any `.Codex.json` file.
- Preserve private exports, generated chunks, credentials, and caches outside Git.

Cloud work should implement and verify the feature using synthetic fixtures and
injected providers. It does not have the local `raw/`, `segments/chunk_data/`,
`.cache/`, `.env`, Ollama service, or private corpus. Do not copy personal
messages into fixtures, fetch them from the local machine, or require a JEV key
for ordinary checks. Keep live provider integration runnable locally and report
which checks need local follow-up. Do not claim live extraction quality based on
mock-provider results.

Run appropriate checks after each milestone. Prepare the resulting changes for
review; do not merge, publish a website, or start full private-corpus processing.

## Existing components and reuse boundaries

- `imsg/segments/segment.py`: schema-version-1 source documents, chunks,
  message IDs, timestamps, reply links, source hashes and line occurrences.
- `imsg/intent_search/pipeline.py::SearchIndexes`: corpus snapshot loading and
  shared embedding model. Separate immutable corpus access from mutable search
  indexes only as required for reuse; avoid a broad search rewrite.
- `agent_query_operator/search.py::resolve_participants`: explicit aliases and
  whole-thread source selection. Timeline identities must resolve explicitly;
  unresolved identities must not silently broaden the timeline scope.
- `chunk_search/dense.py` and `sparse.py`: model and retrieval reuse for context
  search and shortlist generation, not the timeline inventory.
- `jev_reranker/client.py::JevJudge`: auth, HTTPS, retries, request packing,
  typed yes/no parsing, batch concurrency and actual provider usage.
- `chunk_summaries/store.py`: content-addressed artifacts with atomic writes,
  model/prompt versions and provenance checks. Reuse the pattern; existing prose
  summaries do not retain enough evidence for atomic-event extraction.
- `intent_search/service.py` and `api.py`: submit/poll conventions and local
  request checks. Current jobs are in memory and expire; timeline jobs must persist.
- `imsg/server.py::context`: original-message display and source-hash validation.
  Generalize the reader narrowly so event evidence does not depend on a search job.

JEV supports typed `noul`, `choice`, and `score` questions. The current client
only parses `noul`. Consult the official contract before implementing score
parsing: https://docs.typesafe.ai/api . Preserve requested and returned models,
the score legend/distribution, and actual token usage.

## Milestone 1: durable foundation and source scope

Create `imsg/timeline/` with validated models, local storage, a scope resolver,
and shared source-evidence access. Use a versioned SQLite schema and indexes on
person, snapshot, and time intervals. Generated stores live under ignored `.cache/`.

Persist these entities:

- Person: ID, display name, aliases scoped by source/chat where needed.
- Scope: selected source chats, alias-map revision, optional date range.
- Source snapshot: source file/hash, source revision and chunk membership.
- Event: ID, person, actors, type, atomic claims, discussion interval, optional
  mentioned event interval, temporal precision, planned/reported status.
- Evidence: snapshot-scoped message references and source line occurrences.
- Judgment: claim support, involvement support, separate significance dimensions,
  full distributions, rubric/model/prompt versions and failure status.
- Episode and links: event memberships, supporting continuation/contradiction/
  resolution links, interval and grounded title.
- Build and coverage: request, version, stage, per-work-unit progress, successful
  no-event outcomes, failures and resumable checkpoints.
- Overrides: pin/hide, corrected title/date, significance and split/merge choices.

Messages currently use filename/line-derived IDs; chunks use segmentation-order
IDs. Treat both as snapshot-scoped references. Preserve the chunk schema. Source
changes must produce explicit stale evidence or a new version, never point an
old claim at different text.

Timestamps have unknown timezone and preserve export wall time. Store and render
floating local times explicitly. Do not infer UTC, use the viewer's timezone, or
mistake a future plan for an event that already happened.

Acceptance: strict identity resolution; deterministic source selection; validated
round-trip persistence; source-hash checks; safe evidence lookup; restart retains
jobs/events; missing corpus produces an actionable state without provider calls.

## Milestone 2: resumable exhaustive extraction

Add an extractor provider interface and a structured local Ollama implementation.
Use the existing configured model as an initial integration, not a quality claim.
Allow a different configured model later without coupling storage to Ollama.

Each extraction returns zero or more atomic events with supported message IDs,
actors, concrete claims, event category, discussion times, optional mentioned
event times and uncertainty. Validate citations against the supplied evidence;
reject invented IDs, impossible dates and malformed output. Ask the model to
treat all conversation content as data, preserve negation and speaker identity,
and avoid inferring motives or attachment contents.

Scan all selected work units. Use existing chunks as starting units, bounded
overlapping message windows for long inputs, and reply context where applicable.
Keep each source chat's dialogue intact rather than interleaving different chats
before extraction. Assign a core window for event ownership and deduplicate
overlap. Oversized messages must retain endings via bounded fragments. Track
coverage by source/message ranges so overlap does not inflate progress.

Save each successful artifact atomically. Include the evidence-content digest,
person/scope inputs, extractor model and prompt version in the work key. Separate
content work keys from snapshot citation bindings; the current summary store's
whole-export hash would invalidate too much on append. Validate any rebinding and
restrict reuse to unambiguous unchanged evidence. Reprocess changed boundaries.

Distinguish successful scans with no events, pending work, and failures. Provider
outages must not become negative judgments or an apparently empty completed
timeline. Publish processing coverage without claiming complete semantic recall.

Acceptance: complete window coverage; event citations resolve; long endings and
cross-boundary replies survive; overlap is idempotent; restart and retries reuse
successful artifacts; new source revisions cannot silently reuse stale citations.

## Milestone 3: JEV support and significance

Extract a shared typed-question transport from `JevJudge` narrowly, preserving
search behavior. Add strict score response validation. Reuse retry, packing,
request-budget and usage patterns. Bound both individual question size and total
batch size; report any unprocessed item explicitly.

For each candidate, judge atomic claim support against original evidence with
sufficient surrounding context, and separately judge target-person involvement.
Use explicit rubric levels for two significance dimensions. Initial relationship
anchors are routine, notable, substantive, major, and turning point. Create a
separate life/shared-event rubric. Positive and negative events share an impact
scale; message count and emotional intensity alone do not establish significance.

Store distributions and model judgments. A score measures impact, not probability
of truth. A normalized display score is a documented transformation, not a new
confidence value. Reject unsupported claims from ordinary timeline display;
retain unknown/rejected outcomes for inspection. Low support and provider failure
remain distinguishable. Version the support and display policies.

Acceptance: mock API tests cover valid/malformed score distributions, unexpected
types, missing answers, retries, partial errors and oversized evidence. Existing
intent-search response semantics and original-message verification remain intact.
One shared request limiter caps concurrent JEV requests across both features.

## Milestone 4: cross-chat events and conservative episodes

Use time/entity/embedding blocking to shortlist possible duplicate events and
continuation links; avoid all-pairs comparisons across the corpus. Verify links
using original evidence. Merge duplicate descriptions of one incident while
preserving every distinct source citation. Do not deduplicate raw messages across
chats merely because sender/text/timestamps match.

Episodes group supported linked events and retain distinct stages: disagreement,
response, apology, explicit resolution or changed plans. A friendly follow-up is
insufficient evidence of resolution. Nearby unrelated conversations stay separate;
concurrent episodes can overlap. Do not force all events into one semantic tree.

Ground any generated episode title in its children. Episode prominence starts
from the strongest supported child, with only a bounded versioned adjustment for
independently evidenced developments. Repeated discussion must not accumulate
significance automatically. Preserve user decisions across deterministic rebuilds
and flag ambiguous reconciliation rather than guessing.

Acceptance: same incident across two chats merges with retained citations;
repeated discussion does not inflate impact; unrelated nearby events remain
separate; contradictions/resolutions keep explicit evidence; IDs and rebuilds
are deterministic within a version.

## Milestone 5: persistent jobs and timeline API

Add build orchestration with durable checkpoints and coherent published snapshots.
Keep model work off request threads. Use one bounded shared scheduler that yields
between timeline batches to serve interactive searches. Timeline reads operate on
published snapshots immediately; canceled/failed work does not corrupt the last
published version. Resume interrupted work and expose pending/partial status.

Proposed routes:

- `GET /api/people`: resolved identities and source availability.
- `POST /api/timeline-builds`: person, scope, optional time range; durable job ID.
- `GET /api/timeline-builds/{id}`: stages, coverage, failures, resumability/version.
- `GET /api/people/{id}/timeline`: from/to, detail level or viewport width,
  significance dimension, version and bounded event budget.
- `GET /api/timeline-events/{id}`: claims, scores, evidence refs and episode links.
- `GET /api/timeline-events/{id}/evidence`: original messages plus paginated context.
- Minimal explicit routes for identity/source selection and user overrides.

Return nodes, intervals, links, hidden counts and processing coverage. Avoid
dumping full transcripts in viewport responses. Bound query ranges/response
sizes, validate identifiers, and preserve the existing localhost/origin checks.
Support interval intersection so long episodes remain visible when their start
falls outside the viewport. Pin responses to a version during navigation.

Acceptance: submit/poll/restart/resume flow; strict HTTP validation; interval
queries; pagination; stale/unknown IDs; coherent publication; global resource
limits; existing API compatibility using injected providers and synthetic corpus.

## Milestone 6: semantic zoom UI and evidence drill-down

Extend the existing web app with a person selector, build/coverage state, a
relationship/life-event view switch, and a timeline pane. Use existing frontend
conventions; add dependencies only when their benefit is concrete.

Implement three deterministic detail bands derived from visible time span and
viewport width:

- Years/months: major landmarks and significant episode intervals.
- Weeks/days: episode stages and individual events with a lower cutoff.
- Hours/messages: all supported indexed events and exact-message evidence.

Fixed tiers and explicit promotion rules maintain landmark identity. Pan should
not rerank the viewport's top-N events. Zooming into a fixed interval reveals
more descendants while retaining a recognizable parent episode. Colliding labels
become expandable clusters with hidden counts, not silent dropped results. Keep
an activity/coverage strip separate from significance. Overlapping episodes get
separate lanes or explicit links.

An event opens cited messages in each original chat with surrounding context.
Provide raw-message access even where extraction found no events. Distinguish
discussion time from planned/reported event time in the product UI. Support basic
pin/hide and corrections, keyboard navigation and bounded rendering. Use text
rendering that cannot interpret message bodies as HTML.

Acceptance: synthetic multi-chat browser flow; zoom/pan identity stability;
cluster expansion; evidence navigation; persistence of overrides; readable dense
periods; safe message rendering; floating-time correctness; no model calls when
zooming/panning/reading an existing published timeline.

## Validation and rollout

Use meaningful standard-library Python tests where feasible, with deterministic
fake extractors/judges and small synthetic schema-version-1 fixtures. Test feature
invariants, provider boundaries and API compatibility rather than mirroring helper
implementations. Use bun for any required browser tooling; avoid heavy parallelism.
Do not restore the removed historical suite wholesale.

Cloud deliverable: working version-one pipeline/API/UI with synthetic end-to-end
verification, clear installation/start instructions, and an honest list of local
checks not run. Any demo fixtures must be unmistakably synthetic and must never
be mistaken for the user's actual events.

Local follow-up: choose a manageable contiguous multi-chat interval and a manual
evidence ledger including quiet changes, disagreements, positive experiences,
cross-chat continuation and unrelated group discussion. Measure event recall,
claim/citation precision, involvement errors, dates, merge precision, episode
coherence and zoom stability. Audit successful no-event windows for omissions.
Record wall time, JEV usage, candidates/chunk, cache reuse and failures. Estimate
full-build cost/time only after this pilot. Switch the extraction model if measured
quality is insufficient. Run the full private archive only after the pilot.

Deferred: mention-only chats, attachment understanding, corpus ingestion redesign,
automatic long relationship narratives, multi-user hosted deployment and new
database/server stacks.

## Progress log

- Architecture accepted; milestone plan prepared locally.
- Implementation has not started. Cloud repository/environment/task setup pending.
