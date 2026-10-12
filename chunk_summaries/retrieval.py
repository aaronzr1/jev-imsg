"""The optional feature's single retrieval hook into the ordinary pipeline."""
from agent_query_operator.search import retrieve

from .options import validate_options
from .store import SummaryStore


def retrieve_with_summaries(plan, args, indexes, aliases=None):
    validate_options(args.retrieval_text, args.summary_weight)
    weight = 1.0 if args.retrieval_text == 'summaries' else args.summary_weight
    sources = [('messages', 1.0 - weight), ('summaries', weight)]
    runs, warnings, scope = [], [], None
    for name, share in sources:
        if share == 0:
            continue
        if name == 'summaries':
            store = SummaryStore(args.summaries_dir, args.summary_model)
            dense, sparse = indexes.get(args, name=name, text_factory=store.text_factory,
                                        signature=store.signature())
        else:
            dense, sparse = indexes.get(args)
        bundle = retrieve(plan, dense, sparse, args.per_query, aliases)
        for run in bundle['runs']:
            run.update(run_id=name + '-' + run['run_id'], text_source=name, weight=share)
            if name == 'summaries':
                for hit in run['results']:
                    hit['summary'] = dense.text(dict(chunk=hit['chunk']))
            runs.append(run)
        warnings.extend(bundle['warnings'])
        scope = bundle['scope']
    if weight:
        warnings.append('Summaries are generated retrieval hints. Verify matches against the original messages.')
    return dict(plan=plan, runs=runs, warnings=list(dict.fromkeys(warnings)), scope=scope,
                settings=dict(retrieval_text=args.retrieval_text, summary_weight=weight,
                              summary_model=args.summary_model if weight else None))
