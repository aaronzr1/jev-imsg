"""Inspect plans, execute retrieval, or replay fusion: python -m agent_query_operator."""
import argparse
import json
import sys
import time
from pathlib import Path

from chunk_search.chunk_search import DEFAULT_CACHE_DIR, DEFAULT_CHUNKS_DIR
from .operator import fuse
from .retrieval.planner import OllamaPlanner, validate_plan
from .retrieval.search import retrieve


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def print_plan(plan):
    print(f"Intent: {plan['query']}\nDense original: {plan['query']}")
    for number, query in enumerate(plan['rewrites'], 1):
        print(f'Dense rewrite {number}: {query}')
    print('BM25: ' + ' '.join(plan['keywords']))
    print('Participants to resolve: ' + (', '.join(plan['participants']) or '(none)'))
    print('Downstream answer mode: ' + plan['answer_mode'])
    for requirement in plan['requirements']:
        print('  Evidence requirement: ' + requirement)


def print_hits(hits, full=False):
    if not hits:
        print('  No candidates.')
    for rank, hit in enumerate(hits, 1):
        score_key = 'rrf_score' if 'rrf_score' in hit else 'score'
        chunk = hit['chunk']
        print(f"  {rank}. {hit['chunk_id']} | {hit['source_file']} | {score_key}={hit[score_key]:.6f}")
        print(f"     {chunk.get('start', '?')} — {chunk.get('end', '?')}")
        for vote in hit.get('contributions', []):
            print(f"     {vote['run_id']} rank {vote['rank']}: +{vote['contribution']:.6f} "
                  f"(raw {vote['score']:.4f})")
        transcript = '\n'.join(chunk.get('transcript', []))
        if not full and len(transcript) > 600:
            transcript = transcript[:600] + '\n[… use --full for complete transcript]'
        for line in transcript.splitlines():
            print('     ' + line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'search', 'merge'):
        sub = commands.add_parser(name)
        sub.add_argument('--out', type=Path, help='Save JSON to a new file')
        if name != 'merge':
            sub.add_argument('--query', '-q')
            sub.add_argument('--model', default='qwen3:1.7b')
            sub.add_argument('--url', default='http://localhost:11434')
            sub.add_argument('--timeout', type=positive, default=120)
        if name != 'plan':
            sub.add_argument('-k', '--top-k', type=positive, default=10)
            sub.add_argument('--rrf-k', type=positive, default=60)
            sub.add_argument('--full', action='store_true')
        if name == 'search':
            sub.add_argument('--plan', type=Path, help='Reuse/edit a saved plan; no LLM request')
            sub.add_argument('--aliases', type=Path, help='JSON mapping names to exact sender labels')
            sub.add_argument('--chunks-dir', type=Path, default=DEFAULT_CHUNKS_DIR)
            sub.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR)
            sub.add_argument('--per-query', type=positive, default=20)
            sub.add_argument('--show-runs', action='store_true')
        if name == 'merge':
            sub.add_argument('input', type=Path, help='Search bundle or chunk_search list of runs')
    args = parser.parse_args(argv)
    if args.out and args.out.exists():
        parser.error('--out already exists; choose a new filename')
    try:
        if args.command == 'merge':
            saved = load(args.input)
            runs = saved['runs'] if isinstance(saved, dict) else saved
            output = dict(runs=runs, results=fuse(runs, args.top_k, args.rrf_k),
                          settings=dict(top_k=args.top_k, rrf_k=args.rrf_k))
            if isinstance(saved, dict):
                for key in ('plan', 'warnings', 'scope'):
                    if key in saved:
                        output[key] = saved[key]
        else:
            saved_plan = getattr(args, 'plan', None)
            if bool(args.query) == bool(saved_plan):
                parser.error('provide exactly one of --query or --plan')
            started = time.perf_counter()
            plan = validate_plan(load(saved_plan)) if saved_plan else OllamaPlanner(
                args.model, args.url, args.timeout).plan(args.query)
            print_plan(plan)
            print(f'Plan ready in {time.perf_counter() - started:.2f}s.', flush=True)
            output = plan
            if args.command == 'search':
                from chunk_search.dense import DenseSearch
                from chunk_search.sparse import SparseSearch
                print('Loading local search indexes...', flush=True)
                dense = DenseSearch(args.chunks_dir, args.cache_dir)
                sparse = SparseSearch(args.chunks_dir)
                output = retrieve(plan, dense, sparse, args.per_query,
                                  load(args.aliases) if args.aliases else None)
                output['results'] = fuse(output['runs'], args.top_k, args.rrf_k)
                output['settings'] = dict(per_query=args.per_query, top_k=args.top_k, rrf_k=args.rrf_k)
                if args.show_runs:
                    for run in output['runs']:
                        print(f"\n[{run['run_id']}] {run['query']} ({run['seconds']:.3f}s)")
                        print_hits(run['results'], args.full)
        if args.command != 'plan':
            for warning in output.get('warnings', []):
                print('Note: ' + warning)
            print('\nFused candidates (not verified answers):')
            print_hits(output['results'], args.full)
        if args.out:
            with args.out.open('x', encoding='utf-8') as file:
                json.dump(output, file, ensure_ascii=False, indent=2, allow_nan=False)
                file.write('\n')
            print(f'Saved {args.out}')
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
