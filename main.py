"""Run the message-search pipeline, or one stage with saved inputs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
DEFAULT_QUERY = 'find me the first time I mentioned wanting to pivot into startups'
DEFAULTS = dict(
    raw_dir=ROOT / 'raw', chunks_dir=ROOT / 'segments' / 'chunk_data',
    cache_dir=ROOT / '.cache', planner_model='qwen3:1.7b',
    planner_url='http://localhost:11434', planner_timeout=120,
    per_query=100, top_k=100, rrf_k=60, threshold=0.5,
    model='jev-latest', timeout=60, retries=2, batch_size=100, concurrency=4,
)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run(args):
    """Persist each stage so failures can be retried without repeating earlier work."""
    timings = {}

    def stage(name, operation):
        started = time.perf_counter()
        output = operation()
        timings[name] = round(time.perf_counter() - started, 3)
        save(args.out_dir / f'{name}.json', output)
        print(f'{name}: {timings[name]:.3f}s → {args.out_dir / (name + ".json")}', flush=True)
        return output

    def segment():
        from segments.segment import main
        main(['--input-dir', str(args.raw_dir), '--output-dir', str(args.chunks_dir),
              '--cache-dir', str(args.cache_dir)])
        return load(args.chunks_dir / 'index.json')

    def plan():
        from agent_query_operator.planner import OllamaPlanner, validate_plan
        if args.plan:
            return validate_plan(load(args.plan))
        return OllamaPlanner(args.planner_model, args.planner_url, args.planner_timeout).plan(args.query)

    def retrieve(plan):
        from agent_query_operator.search import retrieve
        from chunk_search.dense import DenseSearch
        from chunk_search.sparse import SparseSearch
        return retrieve(plan, DenseSearch(args.chunks_dir, args.cache_dir, use_cache=not args.no_cache),
                        SparseSearch(args.chunks_dir), args.per_query,
                        load(args.aliases) if args.aliases else None)

    def fuse(bundle):
        from agent_query_operator.fusion import fuse
        bundle = dict(bundle) if isinstance(bundle, dict) else dict(runs=bundle)
        bundle.pop('operator', None)
        bundle['results'] = fuse(bundle['runs'], args.top_k, args.rrf_k)
        bundle['settings'] = dict(top_k=args.top_k, rrf_k=args.rrf_k)
        return bundle

    def rerank(bundle):
        from jev_reranker import JevJudge, load_key, rerank
        judge = JevJudge(load_key(), args.model, args.timeout, args.retries,
                         args.batch_size, args.concurrency)
        return rerank(bundle, judge, args.threshold)

    if args.stage == 'segment':
        output = stage('segment', segment)
    elif args.stage in ('fuse', 'rerank'):
        bundle = load(args.input)
        output = stage(args.stage, lambda: fuse(bundle) if args.stage == 'fuse' else rerank(bundle))
    else:
        if args.stage == 'all' and (args.resegment or not any(args.chunks_dir.glob('*.chunks.json'))):
            stage('segment', segment)
        output = stage('plan', plan)
        if args.stage != 'plan':
            output = stage('retrieve', lambda: retrieve(output))
        if args.stage == 'all':
            output = stage('fuse', lambda: fuse(output))
            output = stage('rerank', lambda: rerank(output))
    save(args.out_dir / 'timings.json', timings)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'segment', 'plan', 'retrieve', 'fuse', 'rerank'], default='all')
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
    parser.set_defaults(**DEFAULTS)
    args = parser.parse_args(argv)
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
