"""Search orchestration, saved stage outputs, and reusable indexes."""
import json
from pathlib import Path
import time

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


class SearchIndexes:
    """Reuse one corpus snapshot and its model on a single search worker."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.signature = None
        self.records = None
        self.views = {}

    def snapshot(self, args):
        paths = sorted(args.chunks_dir.resolve().glob('*.chunks.json'))
        files = []
        for path in paths:
            stat = path.stat()
            files.append((str(path), stat.st_mtime_ns, stat.st_size))
        signature = (str(args.chunks_dir.resolve()), str(args.cache_dir.resolve()), tuple(files))
        if args.no_cache or signature != self.signature:
            from imsg.chunk_search.chunk_search import load_chunks
            self.clear()
            self.records = load_chunks(args.chunks_dir)
            self.signature = signature
        return self.records

    def get(self, args, name='messages', text_factory=None, signature=None):
        records = self.snapshot(args)
        view = self.views.get(name)
        if view is None or view['signature'] != signature:
            from imsg.chunk_search.dense import DenseSearch
            from imsg.chunk_search.sparse import SparseSearch
            text = text_factory(records) if text_factory is not None else None
            model = next((v['pair'][0].model for v in self.views.values()
                          if v['pair'][0].model is not None), None)
            pair = (DenseSearch(args.chunks_dir, args.cache_dir, use_cache=not args.no_cache,
                                records=records, text=text, model=model),
                    SparseSearch(args.chunks_dir, records=records, text=text))
            view = self.views[name] = dict(signature=signature, pair=pair)
        return view['pair']


def run(args, indexes=None, log=None):
    """Persist each stage so failures can be retried without repeating earlier work."""
    timings = {}
    indexes = indexes if indexes is not None else SearchIndexes()
    report = log if log is not None else lambda message: print(message, flush=True)

    def stage(name, operation):
        started = time.perf_counter()
        output = operation()
        timings[name] = round(time.perf_counter() - started, 3)
        save(args.out_dir / f'{name}.json', output)
        report(f'{name}: {timings[name]:.3f}s → {args.out_dir / (name + ".json")}')
        return output

    def segment():
        indexes.clear()
        from imsg.segments.segment import main
        main(['--input-dir', str(args.raw_dir), '--output-dir', str(args.chunks_dir),
              '--cache-dir', str(args.cache_dir)])
        return load(args.chunks_dir / 'index.json')

    def plan():
        from imsg.agent_query_operator.planner import OllamaPlanner, validate_plan
        if args.plan:
            return validate_plan(load(args.plan))
        return OllamaPlanner(args.planner_model, args.planner_url, args.planner_timeout).plan(args.query)

    def retrieve(plan):
        from imsg.agent_query_operator.search import retrieve
        if args.retrieval_text != 'messages':
            from imsg.chunk_summaries.retrieval import retrieve_with_summaries
            return retrieve_with_summaries(plan, args, indexes,
                                          load(args.aliases) if args.aliases else None)
        dense, sparse = indexes.get(args)
        output = retrieve(plan, dense, sparse, args.per_query,
                          load(args.aliases) if args.aliases else None)
        output['settings'] = dict(retrieval_text='messages')
        return output

    def fuse(bundle):
        from imsg.agent_query_operator.fusion import fuse
        bundle = dict(bundle) if isinstance(bundle, dict) else dict(runs=bundle)
        bundle.pop('operator', None)
        bundle['results'] = fuse(bundle['runs'], args.top_k, args.rrf_k)
        bundle['settings'] = dict(bundle.get('settings', {}), top_k=args.top_k, rrf_k=args.rrf_k)
        return bundle

    def rerank(bundle):
        from imsg.jev_reranker import JevJudge, load_key, rerank
        judge = JevJudge(load_key(), args.model, args.timeout, args.retries,
                         args.batch_size, args.concurrency)
        output = rerank(bundle, judge, args.threshold)
        output['retrieval_settings'] = dict(bundle.get('settings', {}))
        return output

    if args.stage == 'summarize':
        from imsg.chunk_summaries.generate import generate
        from imsg.chunk_summaries.store import SummaryStore
        from imsg.chunk_summaries.summarizer import OllamaSummarizer
        output = stage('summarize', lambda: generate(
            indexes.snapshot(args), SummaryStore(args.summaries_dir, args.summary_model),
            OllamaSummarizer(args.summary_model, args.summary_url, args.summary_timeout),
            limit=args.summary_limit, force=args.force_summaries, log=report))
    elif args.stage == 'segment':
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
