"""Run sample or custom chunk searches and print readable results."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_QUERIES = {
    'sparse': ['dinner', 'flight', 'internship'],
    'dense': [
        'making plans to eat together',
        'feeling stressed about school and upcoming tests',
        'traveling somewhere by plane',
        'looking for a summer job',
    ],
}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['both', 'dense', 'sparse'], default='both')
    parser.add_argument('-q', '--query', action='append', help='Custom query; repeat for multiple queries')
    parser.add_argument('-k', '--top-k', type=positive_int, default=2)
    parser.add_argument('--chunks-dir', type=Path, default=ROOT / 'segments')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / '.cache')
    parser.add_argument('--excerpt-chars', type=positive_int, default=600)
    parser.add_argument('--full', action='store_true', help='Print complete chunk transcripts')
    parser.add_argument('--json-out', type=Path, help='Save full results as JSON (new file only)')
    args = parser.parse_args()
    if args.query and any(not query.strip() for query in args.query):
        parser.error('queries must not be blank')
    if args.json_out and args.json_out.exists():
        parser.error(f'{args.json_out} already exists; choose a new output filename')

    methods = ['sparse', 'dense'] if args.method == 'both' else [args.method]
    runs = []
    for method in methods:
        print(f'\nLoading {method} index...', flush=True)
        started = time.perf_counter()
        if method == 'sparse':
            from sparse import SparseSearch
            index = SparseSearch(args.chunks_dir)
        else:
            from dense import DenseSearch
            print('The first embedding build can take several minutes; later runs reuse the cache.', flush=True)
            index = DenseSearch(args.chunks_dir, args.cache_dir)
        print(f'Loaded {len(index.records):,} chunks in {time.perf_counter() - started:.2f}s.', flush=True)
        for query in args.query or DEFAULT_QUERIES[method]:
            started = time.perf_counter()
            hits = index.search(query, top_k=args.top_k)
            elapsed = time.perf_counter() - started
            runs.append(dict(method=method, query=query, seconds=elapsed, results=hits))
            label = 'cosine' if method == 'dense' else 'BM25'
            print(f'\n[{method}] {query!r} ({elapsed:.2f}s)')
            if not hits:
                print('  No matches.')
            for rank, hit in enumerate(hits, 1):
                chunk = hit['chunk']
                print(f"  {rank}. {label}={hit['score']:.3f} | {hit['source_file']} | {hit['chunk_id']}")
                print(f"     {chunk.get('start', '?')} — {chunk.get('end', '?')}")
                transcript = '\n'.join(chunk['transcript'])
                if not args.full and len(transcript) > args.excerpt_chars:
                    transcript = transcript[:args.excerpt_chars] + '\n[… truncated; use --full]'
                for line in transcript.splitlines():
                    print(f'     {line}')
            sys.stdout.flush()
    if args.json_out:
        with args.json_out.open('x', encoding='utf-8') as file:
            json.dump(runs, file, ensure_ascii=False, indent=2)
            file.write('\n')
        print(f'\nSaved complete results to {args.json_out}')


if __name__ == '__main__':
    main()
