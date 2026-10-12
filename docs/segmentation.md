# Conversation segmentation

Parse `raw/*.txt` exports into chronological conversation chunks using local
MiniLM embeddings (two CPU threads). Raw exports remain untouched.

```sh
.venv/bin/python main.py --stage segment
.venv/bin/python -m imsg.segments.segment --help
```

## Boundaries

Messages form bursts of up to eight messages, with word and time limits.
Boundary similarity compares mean embeddings of up to two bursts on each side.
Defaults:

- Under 5 minutes of silence: no topic split.
- 5–45 minutes: split below 0.35 similarity.
- 45 minutes–4 hours: split below 0.65 or when similarity is unknown.
- 4 hours or more: split below 0.20 or when similarity is unknown.
- Split before a burst would exceed 120 messages/events.

Explicit replies, short acknowledgments, and likely question–answer pairs can
suppress the two middle pause rules, but not size or long-pause splits. The weaker
four-hour threshold accommodates delayed replies; these are heuristic boundaries.
Verification-code-only conversations are skipped, with original messages retained.
Attachments, URLs, reactions, and events do not contribute semantic text.

Tune similarity with `--similarity`, `--resume-similarity`, and
`--long-gap-similarity`; pauses with `--topic-gap`, `--idle-gap`, and `--hard-gap`
(minutes); size with `--max-messages`. Embeddings are cached in `.cache/`.

## Outputs

Files are written under `segments/chunk_data/` by default:

- `<conversation>.chunks.json`: authoritative schema-version-1 document with
  source metadata, eligibility, statistics, and chunks. Skipped documents store
  their messages at the top level with an empty chunks array.
- `index.json`: per-file metadata and output filenames.
- `review-sample.json`: selected readable chunks for boundary inspection.

Chunks contain IDs, start/end times, participants, previews, boundary reasons,
structured `messages`, readable `transcript`, and `related_chunk_ids` for replies
to earlier chunks. Messages retain sender, timestamp, text, attachments, reactions,
reply IDs, original export body, and source occurrences (1-based inclusive lines).
Timestamps preserve export wall time with unknown timezone. Message IDs remain
stable for unchanged exports; chunk IDs can change with segmentation settings.

Indented reply copies are resolved against timeline messages when unambiguous;
identical top-level messages and unresolved copies are preserved. Source filenames,
SHA-256 hashes, and line ranges retain provenance.
