#!/usr/bin/env bash
# Install local dependencies and models on macOS or Linux. Safe to rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

case "$(uname -s)" in
  Darwin|Linux) ;;
  *) echo 'This installer supports macOS and Linux.' >&2; exit 1 ;;
esac
command -v curl >/dev/null || { echo 'Install curl first.' >&2; exit 1; }
export PATH="$HOME/.local/bin:/usr/local/bin:/Applications/Ollama.app/Contents/Resources:$PATH"

if ! command -v uv >/dev/null; then
  echo 'Installing uv...'
  curl -fsSL https://astral.sh/uv/install.sh | sh
fi
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.11 .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt

if ! command -v ollama >/dev/null; then
  echo 'Installing Ollama (the official installer may request sudo)...'
  curl -fsSL https://ollama.com/install.sh | sh
fi
# The macOS application also contains the CLI if no PATH link was created.
export PATH="/Applications/Ollama.app/Contents/Resources:$PATH"
command -v ollama >/dev/null || { echo 'Ollama installation did not provide its CLI.' >&2; exit 1; }

mkdir -p .cache raw
export OLLAMA_HOST=127.0.0.1:11434
ready() { curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; }
if ! ready; then
  echo 'Starting Ollama on localhost:11434...'
  OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
    nohup ollama serve >.cache/ollama.log 2>&1 </dev/null &
  for ((attempt=0; attempt<60; attempt++)); do
    if ready; then break; fi
    sleep 1
  done
  if ! ready; then
    echo 'Ollama did not start; check .cache/ollama.log.' >&2
    exit 1
  fi
fi

# Read the pipeline defaults so the installer cannot drift to a different model.
planner_model="$(.venv/bin/python -c 'from imsg.config import DEFAULTS; print(DEFAULTS["planner_model"])')"
ollama pull "$planner_model"

echo 'Downloading and checking the local embedding model...'
.venv/bin/python - <<'PY'
from imsg.intent_search.chunk_search.dense import MODEL
from fastembed import TextEmbedding
from imsg.config import DEFAULTS

model = TextEmbedding(model_name=MODEL, cache_dir=str(DEFAULTS['cache_dir'] / 'models'), threads=2)
next(iter(model.embed(['installation check'], batch_size=1)))
PY

echo 'Installed. Put .txt exports from imessage-exporter in raw/ and set JEV_KEY in .env.'
echo 'Run: .venv/bin/python app.py'
echo 'After a reboot, start Ollama or run ollama serve before searching.'
