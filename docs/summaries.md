# Summary search

Search generated conversation summaries, original messages, or both. Jev always
ranks matches using the original messages. Summaries can omit or misread details.

## Generate

```sh
.venv/bin/python main.py --stage segment
.venv/bin/python main.py --stage summarize
```

Ollama `qwen3:1.7b` writes short summaries using two CPU threads, one chunk at a
time. Long chunks are summarized in parts and
merged; attachment contents are not read.

Summaries are cached in `.cache/summaries/`. Each completed summary is saved
immediately, so rerunning resumes after a failure. Changes to messages, source
hashes, model names, or prompts require new summaries. Use `--force-summaries`
if a model's weights change without its name changing.

Options: `--summary-model`, `--summary-url`, `--summary-timeout`,
`--summaries-dir`, `--summary-limit`, and `--force-summaries`.
`--summary-limit` caps generation, not the search corpus: summary search needs
current summaries for every eligible chunk.

## Search

```sh
.venv/bin/python main.py -q 'plans to work at a startup' --retrieval-text summaries
.venv/bin/python main.py -q 'plans to work at a startup' --retrieval-text both --summary-weight 0.7
```

The web interface offers the same choices under **Search in**. Select
**Original messages** or `--retrieval-text messages` to search without summaries.

Both MiniLM and BM25 search the selected text. In `both` mode, summary rankings
receive `summary_weight` and message rankings receive `1 - summary_weight`.
The default is 0.5; 0 searches messages only, and 1 searches summaries only.
Weights combine ranks, not raw similarity scores.

Sender filters use the original message metadata. Missing or outdated summaries
produce an error. Results open the original conversation, with the generated
summary available to expand.
