# iMessage search

<img width="1089" height="756" alt="Image" src="https://github.com/user-attachments/assets/03e9983f-f519-49bd-b9fd-7106e6442432" />

Semantic search over exported iMessage conversations with a Jev reranker.
Local Ollama query planning → MiniLM/BM25 search → rank fusion → Jev ranking.
Browse matches with their original messages and surrounding conversation.

## Run locally

1. Manually export conversations as `.txt` with
   [imessage-exporter](https://github.com/ReagentX/imessage-exporter) and put them in `raw/`.
2. Run `./install.sh` (macOS/Linux; installs uv, a Python virtual environment, Ollama, and models).
3. Copy `.env.example` to `.env` and set `JEV_KEY`.
4. Run `.venv/bin/python app.py` and open http://127.0.0.1:8000.

Keep Ollama running (`ollama serve`). After changing exports, rebuild with
`.venv/bin/python main.py --stage segment`. When removing conversations,
clear `segments/chunk_data/` first.

CLI: `.venv/bin/python main.py -q 'plans to work at a startup' -k 10`.
[API](docs/intent-search.md) · [Segmentation](docs/segmentation.md) · [Optional summaries](docs/summaries.md)

Search can miss matches or rank them incorrectly; it cannot prove the first or
last occurrence. It reads exported text, not attachments, and does not sync with
Messages. Jev receives your query and shortlisted conversation text over the
network. Exports, indexes, and logs are stored locally and ignored by Git.
