"""Local BM25 keyword search over segmented conversations."""
import math
import re
from collections import Counter, defaultdict

from .chunk_search import DEFAULT_CHUNKS_DIR, chunk_text, load_chunks, result, validate_query


def tokenize(text):
    """Unicode words/numbers, case-insensitive; no stemming or stopword removal."""
    return re.findall(r'[^\W_]+', text.casefold())


class SparseSearch:
    """Load once, then call search repeatedly. Rebuild to pick up changed files."""

    def __init__(self, chunks_dir=DEFAULT_CHUNKS_DIR, records=None, text=None):
        self.records = load_chunks(chunks_dir) if records is None else records
        self.text = text if text is not None else chunk_text
        self.postings = defaultdict(list)
        self.lengths = []
        for index, record in enumerate(self.records):
            terms = tokenize(self.text(record))
            self.lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                self.postings[term].append((index, frequency))
        self.count = sum(length > 0 for length in self.lengths)
        self.avg_length = sum(self.lengths) / max(self.count, 1)

    def search(self, query, top_k=10):
        """Return up to top_k chunks matching any query token, ranked by BM25."""
        validate_query(query, top_k)
        scores = defaultdict(float)
        for term in sorted(set(tokenize(query))):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            idf = math.log1p((self.count - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                norm = 1.2 * (1 - 0.75 + 0.75 * self.lengths[index] / self.avg_length)
                scores[index] += idf * frequency * 2.2 / (frequency + norm)
        ranked = sorted(scores, key=lambda index: (-scores[index], index))[:top_k]
        return [result(self.records[index], scores[index]) for index in ranked]


def search(query, chunks_dir=DEFAULT_CHUNKS_DIR, top_k=10):
    """One-shot keyword endpoint; use SparseSearch for repeated queries."""
    validate_query(query, top_k)
    return SparseSearch(chunks_dir).search(query, top_k)
