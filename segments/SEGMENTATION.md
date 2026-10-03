# Conversation segmentation

`segments/segment.py` implements the approved segmentation stage of the pipeline. Full results are in one `segments/chunk_data/*.chunks.json` per input export; `segments/chunk_data/index.json` lists counts. Each run also writes `segments/chunk_data/review-sample.json`, a diagnostic sample of consecutive opening chunks and selected boundary examples with readable transcripts and optional review fields. It is not a random quality estimate.

## How boundaries work

- Parse messages and system events, match indented reply copies to timeline messages by timestamp, sender, and whitespace-normalized text, and record parent message IDs in `reply_to`. Sort the canonical messages by timestamp **before** forming bursts and chunks. Identical top-level messages remain distinct. Unmatched or ambiguous thread copies are retained. Resolving a reply copy removes a duplicate appearance; it does not guarantee that the reply and parent share a chunk.
- Embed short bursts of up to eight messages with `sentence-transformers/all-MiniLM-L6-v2`, locally on CPU with two threads. This keeps fragmentary chat messages together. Ignore attachment paths, URLs, reactions and system events when embedding; use the latest edited text, retaining the export body.
- Before adding a message, start a new burst if the existing burst has eight messages, at least 160 semantic words, a gap to the new message greater than 120 seconds, or a span from its first message to the new message greater than 90 seconds. The word limit is checked before adding the next message, so a burst can exceed 160 words.
- Compare cosine similarity between normalized mean embeddings of up to two bursts on each side of a potential boundary. For `A B C | D E`, compare **B+C** against **D+E**, not C against each neighbor individually. Include B only if B→C is at most ten minutes; include E only if D→E is at most ten minutes. These context gaps are separate from the C→D pause used to decide the split. Context can extend beyond the current chunk. Bursts without semantic text are omitted from the mean; if either side has no text, similarity is unknown. Reply metadata is preserved and used by boundary rules, but is not embedded.
- With a pause **under 5 minutes**, the time/topic rules do not split.
- With a pause **from 5 to under 45 minutes**, split if similarity is **below 0.35**. Unknown similarity does not cause this split.
- With a pause **from 45 minutes to under 4 hours**, split if similarity is **below 0.65** or unknown.
- Explicit replies from any message in the upcoming burst to a message in the current chunk suppress these two soft rules. Short acknowledgments within 45 minutes and likely question–answer pairs within 30 minutes also suppress them. These protections do not override size or long-pause splits.
- After **4 hours** of silence, split only if similarity is below **0.20** or unavailable. This weaker threshold replaces the normal pause rules to accommodate delayed replies. Tune it with `--long-gap-similarity`; `--hard-gap` retains its name but now sets this long-pause threshold, not an unconditional split. Reply heuristics do not override this rule.
- Always split before a burst would exceed **120 messages/events**. A size split is a retrieval constraint, not a claim of a new topic.

All pause measurements use the last message of the previous burst and the first message of the next. Similarity equal to a threshold permits continuation. The four-hour rule deliberately replaces the stricter 45-minute rule, so a four-hour pause can stay together at a similarity that would split a three-hour pause.

The question–answer heuristic requires a boundary gap of at most 30 minutes, different senders for the messages immediately across the boundary, and a `?` in any of the previous burst's last three messages. It does not verify that the response answers the question or save a question–answer relationship. The acknowledgment heuristic checks whether the upcoming burst's first message has at most 12 semantic words and starts with one of: `sure`, `yeah`, `yea`, `yes`, `yep`, `nah`, `no`, `okay`, `ok`, `lol`, `lmao`, `oh`, `thanks`, `thank`, `bet`, `sounds`, `got it`, or `unlucky` (case-insensitive). It does not require a sender change. These are English-oriented text heuristics, not an intent classifier.

Before embedding, a conservative local check skips threads where every non-event message is a short, text-only verification-code template: a 4–8 digit code plus wording such as “verification code,” “security code,” or “OTP.” Mixed conversations and uncertain cases remain eligible. This is a heuristic, not a general spam classifier; unfamiliar templates may be retained. No API key is needed. Skipped files still get an output document with `eligibility`, empty `chunks`, and all original structured `messages` and source references, so skipping segmentation never discards provenance. The index records eligibility for every file.

These are initial heuristic thresholds, not calibrated confidence probabilities. Fast topic changes within five minutes can remain merged. Interleaved group conversations remain chronological instead of being disentangled. Delayed replies can live in separate chunks, linked through `related_chunk_ids`. Images/audio are not inspected, and URLs are not fetched. Long bursts may exceed the embedding model's token limit and be truncated for similarity only; the full text is retained in JSON.

## JSON fields

There are three views of the same run:

- `index.json`: the catalog, with one entry per input export. It contains no transcripts or individual messages. Open the filename in each entry's `output` relative to the index directory to reach the full result.
- `<conversation>.chunks.json`: the complete segmentation result for one export, including messages and citation metadata. This is the authoritative output for downstream processing.
- `review-sample.json`: a small, readable selection across exports, intended for manually judging boundaries. It omits detailed message metadata; use `source_file` and `chunk_id` to look up the full chunk.

The top-level schema of a chunks document is:

```text
schema_version: integer (currently 1)
source: object
  file: string (export basename; locate it under raw/)
  sha256: string (hash of the original file bytes)
  line_count: integer
  exported_records: integer (including repeated thread appearances)
  unique_messages: integer (canonical messages/events after deduplication)
  thread_copies_removed: integer
  ambiguous_thread_matches: integer
eligibility: object
  eligible: boolean
  reason: string (keep_by_default or verification_codes_only)
  message_count: integer (non-event messages)
  verification_code_messages: integer
  method: string (local_template_heuristic)
timestamp_timezone: string (export wall time; timezone unknown)
method: object (eligible documents only)
  embedding_model: string
  settings: object (numeric similarity, gap and size parameters)
  inference: string
statistics: object
  chunks: integer
  median_messages: number
  single_message_chunks: integer
  boundary_reasons: object mapping reason strings to counts
chunks: array of chunk objects
messages: array of message objects (skipped documents only; chunks is empty)
```

Each element of `chunks` has this schema:

```text
id: string
start, end: ISO datetime strings
participants: array of sender strings
message_count: integer (includes events)
preview: string (short excerpt)
boundary_before: object
  reason: string
  gap_minutes: number or null
  cosine_similarity: number or null
  explicit_reply_crosses_boundary: boolean (absent for start_of_file)
transcript: array of strings (HH:MM:SS Sender: text)
messages: array of structured message objects
related_chunk_ids: array of chunk ID strings
```

The `transcript` and `messages` arrays represent the same messages: the former is for reading, the latter for programmatic use and citations. Each structured message has `id`, `timestamp`, `sender`, `text`, `record_type` (`message` or `event`), `attachments` and `reactions` (string arrays), `export_body`, nullable `header_metadata`, `reply_to` (message ID array), `reply_context_unresolved` (boolean), and `source_occurrences`. Each source occurrence is `{ "lines": [first, last], "kind": "timeline" }` or has kind `thread_copy`. Line ranges are inclusive and 1-based.

`index.json` has a single top-level key, `files`, whose value is an array of objects. Each object contains `source`, `eligibility`, and `statistics` with the same schemas above, plus `output`, a string naming the corresponding chunks JSON. It repeats summary metadata so consumers can find useful conversations without loading every message.

Moving an unchanged export into `raw/` does not invalidate its hash, line references, or IDs: `source.file` stores the basename, not its old directory. A citation consists of `raw/<source.file>` and the message's source line range; the SHA-256 identifies the exact source version.

### Convert full chunks to readable review format

`segments/chunk_to_human_readable.py` uses only Python's standard library and does not re-run segmentation. For example:

```sh
.venv/bin/python segments/chunk_to_human_readable.py 'segments/chunk_data/+16467502166.chunks.json'
.venv/bin/python segments/chunk_to_human_readable.py 'segments/chunk_data/+16467502166.chunks.json' --limit 20 -o segments/chunk_data/first-20.review.json
```

The first command writes `segments/chunk_data/+16467502166.review.json` with **all** chunks in chronological order. The second includes only the first 20. Both use the same `instructions` / `chunks` structure and per-chunk fields as `review-sample.json`, including blank `assessment` and `reviewer_notes`. They do not reproduce the segmenter's diagnostic sampling selection. Skipped conversations produce an empty review array. Existing files are never overwritten, preserving prior reviewer notes. Keep the full chunks JSON for detailed citations; review format deliberately omits structured messages.

Each full file includes its source filename, SHA-256, parse counts, eligibility, statistics, and `chunks`. Eligible files also include model/settings metadata under `method`; skipped files instead retain their messages at the document's top level.

Each chunk has an ID, start/end, participants, count, an excerpt (`preview`, not a generated topic label), readable `transcript`, structured `messages`, related chunk IDs, and `boundary_before` with the gap, cosine similarity, and rule that caused the boundary.

Each message includes sender, timestamp, text, attachment paths, reactions, original export body, header metadata (e.g. read receipt), reply IDs, and all source occurrences as **1-based inclusive line ranges**. `record_type` distinguishes system events from messages. Source files plus line ranges preserve exact provenance; bodies remove export indentation and trailing record separators. Timestamps preserve the export's wall time: its timezone is unknown, and no timezone is invented. Message IDs derive from filename and source line and are stable for unchanged exports; chunk IDs can change when thresholds change.

`reply_context_unresolved` means the exporter marked a reply but its parent could not be recovered. `thread_copies_removed` counts redundant nested appearances, not deleted conversations. Raw exports remain untouched.

Replies to old messages retain `reply_to` even when unrelated chunks intervene. The replying chunk's `related_chunk_ids` lists the chunks containing its reply targets; these links point toward the parent chunks and are not bidirectional. They preserve context without merging intervening conversation. They require a recoverable parent. `boundary_before.explicit_reply_crosses_boundary` only indicates a reply to the immediately preceding chunk at the split; it does not capture every older-chunk link. Use `reply_to` and `related_chunk_ids` for those relationships.

## Run / tune

Run the commands below from the repository root. Default input, output, and cache paths are anchored to that root even when running a script from another directory. Explicit relative arguments resolve against your working directory.

Dependencies live in `.venv`; cached models/embeddings live in `.cache`. No text is sent to an external embedding service. Network is needed only for installation and the initial model download.

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python segments/segment.py
```

Cached reruns can be forced offline:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python segments/segment.py --similarity 0.50 --topic-gap 3 --output-dir output-finer
```

Place source exports in `raw/`. Use `--input-dir` and `--output-dir` to select directories (defaults: repository `raw/` and `segments/chunk_data/`). Input discovery reads `*.txt` files directly inside the input directory, excluding `requirements.txt`; it does not recurse into subdirectories. Tune thresholds with `--similarity`, `--resume-similarity`, and `--long-gap-similarity`; tune pauses in minutes with `--topic-gap`, `--idle-gap`, and `--hard-gap`; tune chunk size with `--max-messages` (minimum eight). Because long pauses use a separate, weaker threshold, changing pause thresholds does not always change the number of splits monotonically.

Cached embeddings make threshold experiments fast. Output is indented UTF-8 JSON with the original Unicode characters. The `.gitignore` excludes private exports, output, model caches, and the environment. The pipeline retains inline assertions that check message coverage and uniqueness when forming chunks.
