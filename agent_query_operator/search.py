"""Sequential retrieval with conservative, locally resolved participant scope."""
import time

from .planner import validate_plan


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def resolve_participants(plan, records, aliases=None):
    """Resolve exact labels/explicit aliases, using whole-thread membership.

    A participant needn't speak in every chunk of their conversation. Unknown
    names remain evidence requirements rather than silently dropping all hits.
    """
    aliases = aliases or {}
    if not isinstance(aliases, dict) or any(
        not isinstance(k, str) or not isinstance(v, list) or not v or
        any(not isinstance(label, str) or not label.strip() for label in v)
        for k, v in aliases.items()
    ):
        raise ValueError('aliases must map names to nonempty lists of exact sender labels')
    aliases = {k.strip().casefold(): v for k, v in aliases.items()}
    by_source = {}
    for record in records:
        by_source.setdefault(record['source']['file'], set()).update(
            label.strip().casefold() for label in record['chunk'].get('participants', []))
    labels = set().union(*by_source.values()) if by_source else set()
    resolved, unresolved = [], []
    for name in plan['participants']:
        candidates = {label.strip().casefold() for label in aliases.get(name.strip().casefold(), [name])}
        matches = candidates & labels
        if matches:
            resolved.append(matches)
        else:
            unresolved.append(name)
    allowed = None if not resolved else {
        source for source, members in by_source.items()
        if all(members & candidates for candidates in resolved)
    }
    return allowed, unresolved


def retrieve(plan, dense, sparse, per_query=20, aliases=None):
    validate_plan(plan)
    positive_int(per_query, 'per_query')
    allowed, unresolved = resolve_participants(plan, sparse.records, aliases)
    runs = []
    queries = [('dense', plan['query'])] + [('dense', q) for q in plan['rewrites']]
    queries.append(('sparse', ' '.join(plan['keywords'])))
    for number, (method, query) in enumerate(queries, 1):
        index = dense if method == 'dense' else sparse
        started = time.perf_counter()
        # Existing indexes don't accept filters. Rank the complete snapshot when
        # scoped, then filter BEFORE truncating so other threads cannot crowd it out.
        if allowed == set():
            hits = []
        else:
            limit = max(len(index.records), 1) if allowed is not None else per_query
            hits = index.search(query, top_k=limit)
            if allowed is not None:
                hits = [hit for hit in hits if hit['source_file'] in allowed]
            hits = hits[:per_query]
        runs.append(dict(run_id=f'{method}-{number}', method=method, query=query,
                         seconds=round(time.perf_counter() - started, 4), results=hits))
    warnings = ['RRF ranks candidates; relevance and answer requirements remain unverified.']
    if unresolved:
        warnings.append('Unresolved participants (not filtered): ' + ', '.join(unresolved))
    if plan['answer_mode'] != 'relevance':
        warnings.append('Bounded retrieval cannot establish the first/last occurrence in the corpus.')
    return dict(plan=plan, runs=runs, warnings=warnings,
                scope=dict(source_files=None if allowed is None else sorted(allowed),
                           unresolved_participants=unresolved))
