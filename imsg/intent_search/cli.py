"""Command-line interface for the message-search pipeline."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

from imsg.config import DEFAULTS, DEFAULT_QUERY, ROOT, positive
from .pipeline import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'segment', 'summarize', 'plan', 'retrieve', 'fuse', 'rerank'], default='all')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('-q', '--query', default=DEFAULT_QUERY)
    source.add_argument('--plan', type=Path, help='Saved plan; skip Ollama')
    source.add_argument('--input', type=Path, help='Saved retrieval/fusion JSON for fuse or rerank')
    parser.add_argument('--out-dir', type=Path, help='New run directory; default: .cache/run-<timestamp>')
    for name in ('raw-dir', 'chunks-dir', 'cache-dir'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--resegment', action='store_true', help='Rebuild chunks before a full run')
    parser.add_argument('--aliases', type=Path, help='JSON mapping names to sender labels')
    parser.add_argument('--no-cache', action='store_true', help='Recompute retrieval embeddings')
    parser.add_argument('--planner-model')
    parser.add_argument('--planner-url')
    parser.add_argument('--planner-timeout', type=positive)
    parser.add_argument('--per-query', type=positive)
    parser.add_argument('-k', '--top-k', type=positive)
    parser.add_argument('--rrf-k', type=positive)
    parser.add_argument('--model', help='Jev model')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--timeout', type=positive)
    parser.add_argument('--retries', type=int, choices=range(6))
    parser.add_argument('--batch-size', type=positive)
    parser.add_argument('--concurrency', type=int, choices=range(1, 9))
    from imsg.intent_search.chunk_summaries.options import add_options, validate_options
    add_options(parser)
    parser.set_defaults(**DEFAULTS)
    args = parser.parse_args(argv)
    try:
        validate_options(args.retrieval_text, args.summary_weight)
    except ValueError as exc:
        parser.error(str(exc))
    if (args.stage in ('fuse', 'rerank')) != bool(args.input):
        parser.error('--input is required only for --stage fuse or rerank')
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1')
    if args.plan and args.stage not in ('all', 'plan', 'retrieve'):
        parser.error('--plan requires --stage all, plan, or retrieve')
    args.out_dir = args.out_dir or ROOT / '.cache' / ('run-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    try:
        args.out_dir.mkdir(parents=True, exist_ok=False)
        print(f'Run directory: {args.out_dir}', flush=True)
        output = run(args)
        for hit in output.get('results', []):
            score = hit.get('match_probability', hit.get('rrf_score', 0))
            print(f"{score:.3f}  {hit['chunk_id']}  {hit['source_file']}")
        for warning in output.get('warnings', []):
            print('Note: ' + warning)
        return 0 if output.get('complete', True) else 2
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
