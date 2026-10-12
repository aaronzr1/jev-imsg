"""Shared paths, defaults, and argument types."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY = 'find me the first time I mentioned wanting to pivot into startups'
DEFAULTS = dict(
    raw_dir=ROOT / 'raw', chunks_dir=ROOT / 'segments' / 'chunk_data',
    cache_dir=ROOT / '.cache', planner_model='qwen3:1.7b',
    planner_url='http://localhost:11434', planner_timeout=120,
    per_query=100, top_k=100, rrf_k=60, threshold=0.5,
    model='jev-latest', timeout=60, retries=2, batch_size=100, concurrency=4,
    retrieval_text='messages', summary_weight=0.5, summaries_dir=ROOT / '.cache' / 'summaries',
    summary_model='qwen3:1.7b', summary_url='http://localhost:11434', summary_timeout=120,
    summary_limit=None, force_summaries=False,
)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number
