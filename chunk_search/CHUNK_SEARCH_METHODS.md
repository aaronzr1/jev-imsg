# Chunk search

Both endpoints read schema-v1 `*.chunks.json` directly from a directory (no recursion). The default is `segments/`, matching the current files and `segment.py`; pass `chunks_dir='output'` for that location. Ineligible conversations are skipped. Raw exports are never changed.

## Usage

The existing `fastembed==0.7.4` requirement supplies the dense model runtime and NumPy. Keyword search uses only the standard library.

```sh
uv venv .venv  # only if the environment does not already exist
uv pip install --python .venv/bin/python -r requirements.txt
```

```python
import dense
import sparse

semantic_hits = dense.search('making plans to eat together', top_k=5)
keyword_hits = sparse.search('pizza dinner', top_k=5)

# For an application, build once and reuse across requests:
semantic = dense.DenseSearch(chunks_dir='segments')
keywords = sparse.SparseSearch(chunks_dir='segments')
semantic_hits = semantic.search('travel arrangements', top_k=5)
keyword_hits = keywords.search('train tickets', top_k=5)
```

Each function returns a JSON-serializable list ordered by descending `score`, with `chunk_id`, `source_file`, `source_sha256`, `chunks_file`, and the complete original `chunk`. Its messages retain source line ranges for citations into `raw/<source_file>`. Scores from the two methods are not directly comparable. Ties follow sorted input filename and original chunk order.

## Try queries from the terminal

`search_examples.py` runs the sample queries used during exploration, loading each index once. It prints scores, source files, chunk IDs, dates, query timings, and transcript excerpts. Default paths resolve relative to the script.

```sh
# Default keyword and semantic queries, two results each:
.venv/bin/python search_examples.py

# Compare both methods on the same custom query:
.venv/bin/python search_examples.py -q "making plans to eat together"

# Multiple queries, one method, complete transcripts:
.venv/bin/python search_examples.py --method sparse -q flight -q internship -k 3 --full

# Save full results (including citation metadata) to a new JSON file:
.venv/bin/python search_examples.py --json-out .cache/search-results.json
```

Use `--excerpt-chars` to change the default 600-character excerpt length. Excerpts show the beginning of a chunk, which may precede the matching passage; `--full` reveals all messages. Edit `DEFAULT_QUERIES` in the script to change the samples. Dense scores and BM25 scores use different scales. JSON exports contain conversation text; existing output files are never overwritten.

## Semantic (`dense.py`)

Uses local `sentence-transformers/all-MiniLM-L6-v2` embeddings and cosine similarity. Message text is split into overlapping 80-word windows with a 60-word stride; the best window score ranks each chunk, which appears only once. This lets topics near the end of long chunks match. System events, attachment paths, reaction metadata, sender names, and URLs are excluded from embedding text.

Word windows approximate the model's token budget: unusual text can still be truncated, as can long queries. Best-window scoring can favor longer chunks; this is an initial retrieval baseline. Results have no relevance cutoff, so an unrelated query can still return matches. Scores are similarities, not confidence probabilities.

Embeddings are cached under `.cache/dense-<hash>.npy`, keyed by model, window recipe, and ordered text. Text changes generate a new cache; old files remain until removed. Models reuse `.cache/models`. Inference runs with two CPU threads and batches of 32, without worker processes. The first model download needs network access; conversation text stays local. With the model cached, use `HF_HUB_OFFLINE=1` to force offline operation.

## Keyword (`sparse.py`)

Uses BM25 (`k1=1.2`, `b=0.75`) with Unicode word/number tokens and case folding. Matches any query token; repeated query terms count once. No stemming, stopword filtering, phrase syntax, substring matching, or Boolean operators. URLs remain searchable as tokens. System events, attachment paths, and metadata are excluded. Only matching chunks are returned; scores are positive BM25 weights.

Both indexes are in-memory snapshots: construct a new instance after changing chunk files. Blank queries and empty eligible corpora return `[]`; keyword queries with no searchable tokens also return `[]`. Missing input files raise `FileNotFoundError`, malformed JSON/schema raises an error, and `top_k` must be a positive integer. Dense search scans window vectors, while keyword search uses inverted postings; neither needs a database or hosted API.
