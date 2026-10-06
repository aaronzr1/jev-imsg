"""Local semantic search using FastEmbed and cosine similarity."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np

from .chunk_search import DEFAULT_CACHE_DIR, DEFAULT_CHUNKS_DIR, chunk_text, load_chunks, result, validate_query

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def windows(text):
    """Overlap short windows so a long chunk's ending is also searchable."""
    words = text.split()
    for start in range(0, len(words), 60):
        yield ' '.join(words[start:start + 80])
        if start + 80 >= len(words):
            break


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


class DenseSearch:
    """Snapshot of chunks with cached embeddings and lazy model loading."""

    def __init__(self, chunks_dir=DEFAULT_CHUNKS_DIR, cache_dir=DEFAULT_CACHE_DIR, use_cache=True):
        self.records = load_chunks(chunks_dir)
        self.cache_dir = Path(cache_dir)
        self.model = None
        texts, owners = [], []
        for index, record in enumerate(self.records):
            for text in windows(chunk_text(record, semantic=True)):
                texts.append(text)
                owners.append(index)
        self.owners = np.asarray(owners, dtype=np.int64)
        self.vectors = None
        if not texts:
            return
        key = hashlib.sha256(json.dumps([MODEL, 'windows-80-stride-60-v1', texts],
                                       ensure_ascii=False).encode()).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f'dense-{key}.npy'
        if use_cache and path.exists():
            self.vectors = np.load(path, allow_pickle=False)
        else:
            self.vectors = normalize(list(self._model().embed(texts, batch_size=32)))
            if not use_cache:
                return
            # Atomic replacement lets concurrent readers see only complete files.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix='.npy', delete=False) as file:
                    temporary = Path(file.name)
                    np.save(file, self.vectors, allow_pickle=False)
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def _model(self):
        if self.model is None:
            from fastembed import TextEmbedding
            self.model = TextEmbedding(model_name=MODEL, cache_dir=str(self.cache_dir / 'models'),
                                       threads=2)
        return self.model

    def search(self, query, top_k=10):
        """Rank chunks by their best window cosine; scores are not probabilities."""
        validate_query(query, top_k)
        if not query.strip() or self.vectors is None:
            return []
        vector = normalize(list(self._model().embed([query], batch_size=1)))[0]
        scores = np.full(len(self.records), -np.inf, dtype=np.float32)
        np.maximum.at(scores, self.owners, self.vectors @ vector)
        ranked = sorted(set(self.owners.tolist()), key=lambda index: (-float(scores[index]), index))[:top_k]
        return [result(self.records[index], scores[index]) for index in ranked]


def search(query, chunks_dir=DEFAULT_CHUNKS_DIR, top_k=10, cache_dir=DEFAULT_CACHE_DIR):
    """One-shot semantic endpoint; use DenseSearch for repeated queries."""
    validate_query(query, top_k)
    if not query.strip():
        return []
    return DenseSearch(chunks_dir, cache_dir).search(query, top_k)
