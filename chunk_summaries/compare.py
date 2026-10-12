"""Compare retrieval modes with one shared plan per query; Jev is opt-in."""
import argparse
from datetime import datetime
from itertools import combinations
from pathlib import Path
import sys
import time

from agent_query_operator.fusion import fuse
from agent_query_operator.planner import OllamaPlanner, validate_plan
from main import DEFAULTS, ROOT, SearchIndexes, load, positive, run, save

from .options import add_options, validate_options

MODES = ('messages', 'summaries', 'both')


def metrics(ids, relevant, k):
    if not relevant:
        return None
    labels = set(relevant)
    found = len(set(ids[:k]) & labels)
    first = next((rank for rank, chunk_id in enumerate(ids[:k], 1) if chunk_id in labels), None)
    return dict(recall_at_k=found / len(labels), precision_at_k=found / k,
                reciprocal_rank=1 / first if first else 0.0)


def compare(cases, args, log=print):
    if not isinstance(cases, list) or not cases:
        raise ValueError('Cases must be a nonempty JSON list.')
    validate_options('both', args.summary_weight)
    indexes = SearchIndexes()
    corpus_ids = {r['chunk']['id'] for r in indexes.snapshot(args)}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get('query'), str) or not case['query'].strip():
            raise ValueError('Each case needs a nonblank query.')
        labels = case.get('relevant_chunk_ids', [])
        if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
            raise ValueError('relevant_chunk_ids must be a list of chunk IDs.')
        if set(labels) - corpus_ids:
            raise ValueError('Some labeled chunk IDs are absent from the configured corpus.')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, top_k=args.top_k, summary_weight=args.summary_weight,
                  summary_model=args.summary_model, reranked=args.rerank,
                  complete=True, cases=[], aggregates={})
    planner = None
    for number, case in enumerate(cases, 1):
        folder = args.out_dir / f'case-{number:03d}'
        folder.mkdir()
        started = time.perf_counter()
        if case.get('plan') is not None:
            plan = validate_plan(case['plan'])
            if plan['query'] != case['query']:
                raise ValueError('Saved plan query must exactly match the case query.')
        else:
            if planner is None:
                planner = OllamaPlanner(args.planner_model, args.planner_url, args.planner_timeout)
            plan = planner.plan(case['query'])
        planning_seconds = time.perf_counter() - started
        plan_path = folder / 'shared-plan.json'
        save(plan_path, plan)
        result = dict(query=case['query'], relevant_chunk_ids=case.get('relevant_chunk_ids', []),
                      planning_seconds=round(planning_seconds, 3), modes={}, overlaps={})
        for mode in MODES:
            mode_dir = folder / mode
            settings = dict(vars(args), query=case['query'], plan=plan_path, out_dir=mode_dir,
                            retrieval_text=mode, stage='all' if args.rerank else 'retrieve',
                            input=None, resegment=False)
            output = run(argparse.Namespace(**settings), indexes=indexes, log=lambda message: None)
            if not args.rerank:
                output = dict(output, results=fuse(output['runs'], args.top_k, args.rrf_k))
                output['settings'] = dict(output.get('settings', {}), top_k=args.top_k, rrf_k=args.rrf_k)
                save(mode_dir / 'fuse.json', output)
            hits = output['results']
            ids = [hit['chunk_id'] for hit in hits]
            complete = output.get('complete', True)
            timings = load(mode_dir / 'timings.json')
            result['modes'][mode] = dict(complete=complete, timings=timings,
                metrics=metrics(ids, result['relevant_chunk_ids'], args.top_k),
                results=[dict(chunk_id=h['chunk_id'], source_file=h['source_file'],
                              score=h.get('match_probability', h.get('rrf_score')),
                              summary=h.get('summary')) for h in hits])
            report['complete'] = report['complete'] and complete
            log(f'case {number}: {mode}, {len(hits)} results, retrieval {timings["retrieve"]:.3f}s')
        for a, b in combinations(MODES, 2):
            left = {h['chunk_id'] for h in result['modes'][a]['results']}
            right = {h['chunk_id'] for h in result['modes'][b]['results']}
            union = left | right
            result['overlaps'][a + '/' + b] = dict(common=len(left & right),
                                                  jaccard=len(left & right) / len(union) if union else 1.0)
        report['cases'].append(result)
        # A later failure still leaves the finished cases and all source passages reviewable.
        checkpoint = folder / 'comparison.json'
        save(checkpoint, result)
    for mode in MODES:
        values = [case['modes'][mode]['metrics'] for case in report['cases']
                  if case['modes'][mode]['metrics'] is not None]
        report['aggregates'][mode] = dict(labeled_cases=len(values), means={
            key: sum(value[key] for value in values) / len(values)
            for key in ('recall_at_k', 'precision_at_k', 'reciprocal_rank')} if values else None)
    save(args.out_dir / 'comparison.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--chunks-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--aliases', type=Path)
    parser.add_argument('-k', '--top-k', type=positive)
    parser.add_argument('--per-query', type=positive)
    parser.add_argument('--rrf-k', type=positive)
    parser.add_argument('--planner-model')
    parser.add_argument('--planner-url')
    parser.add_argument('--planner-timeout', type=positive)
    parser.add_argument('--rerank', action='store_true', help='Also compare the full pipeline with Jev verification')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--model')
    parser.add_argument('--no-cache', action='store_true')
    add_options(parser, modes=False, generation=False)
    parser.set_defaults(**DEFAULTS)
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir or ROOT / '.cache' / ('compare-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    try:
        if not 0 <= args.threshold <= 1:
            raise ValueError('Threshold must be between 0 and 1.')
        output = compare(load(args.cases), args)
        print(f'Saved comparison: {args.out_dir / "comparison.json"}')
        return 0 if output['complete'] else 2
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
