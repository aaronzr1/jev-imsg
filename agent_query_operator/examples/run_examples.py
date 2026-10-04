"""Run the startup retrieval example and inspect every operator decision."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT))

from agent_query_operator.__main__ import load, positive, print_hits, print_plan
from agent_query_operator.operator import fuse
from agent_query_operator.retrieval.planner import validate_plan
from agent_query_operator.retrieval.search import retrieve
from chunk_search.chunk_search import DEFAULT_CACHE_DIR, DEFAULT_CHUNKS_DIR

DEFAULT_PLAN = Path(__file__).with_name('startups.plan.json')
DEFAULT_PER_QUERY = 5
DEFAULT_TOP_K = 3
DEFAULT_RRF_K = 60


def inspect_operator(runs, top_k=DEFAULT_TOP_K, rrf_k=DEFAULT_RRF_K):
    """Use the real operator, retaining its complete ranking before the cutoff."""
    occurrences = sum(len(run['results']) for run in runs)
    ranked = fuse(runs, top_k=max(occurrences, 1), rrf_k=rrf_k)
    repeated_queries = []
    repeated_hits = []
    seen_queries = set()
    for index, run in enumerate(runs, 1):
        run_id = run.get('run_id', f'run-{index}')
        key = (run['method'], ' '.join(run['query'].casefold().split()))
        if key in seen_queries:
            repeated_queries.append(dict(run_id=run_id, ignored_hits=len(run['results'])))
            continue
        seen_queries.add(key)
        seen_ids = set()
        for hit in run['results']:
            if hit['chunk_id'] in seen_ids:
                repeated_hits.append(dict(run_id=run_id, chunk_id=hit['chunk_id']))
            seen_ids.add(hit['chunk_id'])
    return dict(
        settings=dict(top_k=top_k, rrf_k=rrf_k), input_occurrences=occurrences,
        unique_candidates=len(ranked),
        ignored_duplicate_queries=repeated_queries, ignored_duplicate_hits=repeated_hits,
        combined_cross_query_occurrences=sum(len(hit['contributions']) - 1 for hit in ranked),
        kept=ranked[:top_k],
        pruned=[dict(rank=rank, reason=f'RRF rank {rank} exceeds top_k={top_k}', result=hit)
                for rank, hit in enumerate(ranked[top_k:], top_k + 1)],
    )


def print_operator(report, full=False):
    print('\nOPERATOR: reciprocal-rank fusion + chunk-ID deduplication')
    print(f"{report['input_occurrences']} retrieved occurrences → "
          f"{report['unique_candidates']} unique candidates → "
          f"{len(report['kept'])} kept / {len(report['pruned'])} pruned")
    print(f"Combined {report['combined_cross_query_occurrences']} repeated occurrences across queries.")
    print('Cross-query matches contribute votes; they are merged, not rejected as irrelevant.')
    for item in report['ignored_duplicate_queries']:
        print(f"Ignored repeated query list {item['run_id']} ({item['ignored_hits']} hits).")
    for item in report['ignored_duplicate_hits']:
        print(f"Ignored duplicate within {item['run_id']}: {item['chunk_id']}")
    print('\nKEPT (contributions explain the ranking):')
    print_hits(report['kept'], full)
    print('\nPRUNED BY TOP-K (not a relevance judgment):')
    if not report['pruned']:
        print('  None.')
    for item in report['pruned']:
        hit = item['result']
        print(f"  Rank {item['rank']}: {hit['chunk_id']} | {hit['source_file']} | "
              f"RRF={hit['rrf_score']:.6f}")
        print('    Reason: ' + item['reason'])
        for vote in hit['contributions']:
            print(f"    {vote['run_id']} rank {vote['rank']}: +{vote['contribution']:.6f}")
        transcript = '\n'.join(hit['chunk'].get('transcript', []))
        excerpt = transcript if full or len(transcript) <= 240 else transcript[:240] + ' […]'
        for line in excerpt.splitlines():
            print('    ' + line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--plan', type=Path, help='Saved plan; defaults to startups.plan.json')
    source.add_argument('--input', type=Path, help='Replay saved retrieval runs; skips retrieval')
    parser.add_argument('--per-query', type=positive, default=DEFAULT_PER_QUERY)
    parser.add_argument('-k', '--top-k', type=positive, default=DEFAULT_TOP_K)
    parser.add_argument('--rrf-k', type=positive, default=DEFAULT_RRF_K)
    parser.add_argument('--chunks-dir', type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--out', type=Path, help='Save retrieval + operator decisions to a new JSON file')
    args = parser.parse_args(argv)
    if args.out and args.out.exists():
        parser.error('--out already exists; choose a new filename')
    try:
        if args.input:
            saved = load(args.input)
            bundle = dict(saved) if isinstance(saved, dict) else dict(runs=saved)
            # A replay replaces previous operator output while preserving retrieval.
            for key in ('operator', 'results', 'settings'):
                bundle.pop(key, None)
            if 'plan' in bundle:
                print_plan(bundle['plan'])
        else:
            plan = validate_plan(load(args.plan or DEFAULT_PLAN))
            print_plan(plan)
            print('\nLoading local indexes (no LLM call)...', flush=True)
            from chunk_search.dense import DenseSearch
            from chunk_search.sparse import SparseSearch
            bundle = retrieve(plan, DenseSearch(args.chunks_dir, args.cache_dir),
                              SparseSearch(args.chunks_dir), per_query=args.per_query)
        print('\nRETRIEVAL: candidates before fusion')
        for index, run in enumerate(bundle['runs'], 1):
            print(f"\n[{run.get('run_id', f'run-{index}')}] {run['method']}: {run['query']}")
            print_hits(run['results'], args.full)
        for warning in bundle.get('warnings', []):
            print('Note: ' + warning)
        bundle['operator'] = inspect_operator(bundle['runs'], args.top_k, args.rrf_k)
        print_operator(bundle['operator'], args.full)
        if args.out:
            with args.out.open('x', encoding='utf-8') as file:
                json.dump(bundle, file, ensure_ascii=False, indent=2, allow_nan=False)
                file.write('\n')
            print(f'\nSaved {args.out}')
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
