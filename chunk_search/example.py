"""Compare dense and sparse search on one query: python -m chunk_search.example."""
import argparse
import sys
from pathlib import Path

from .chunk_search import DEFAULT_CACHE_DIR, DEFAULT_CHUNKS_DIR
from main import positive, save


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-q', '--query', default='making plans to eat together')
    parser.add_argument('--method', choices=['both', 'dense', 'sparse'], default='both')
    parser.add_argument('-k', '--top-k', type=positive, default=3)
    parser.add_argument('--chunks-dir', type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument('--out', type=Path, help='Save full retrieval runs to a new JSON file')
    args = parser.parse_args(argv)
    if not args.query.strip():
        parser.error('query must not be blank')
    if args.out and args.out.exists():
        parser.error('--out already exists')
    runs = []
    for method in ['dense', 'sparse'] if args.method == 'both' else [args.method]:
        if method == 'dense':
            from .dense import DenseSearch
            index = DenseSearch(args.chunks_dir, args.cache_dir)
        else:
            from .sparse import SparseSearch
            index = SparseSearch(args.chunks_dir)
        hits = index.search(args.query, args.top_k)
        runs.append(dict(method=method, query=args.query, results=hits))
        for hit in hits:
            print(f"{method}: {hit['score']:.3f} | {hit['chunk_id']} | {hit['source_file']}")
            print('\n'.join(hit['chunk']['transcript'])[:600])
    if args.out:
        save(args.out, runs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
