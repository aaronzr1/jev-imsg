"""Reciprocal-rank fusion with one vote per chunk per distinct query list."""
import math

from .search import positive_int


def fuse(runs, top_k=10, rrf_k=60):
    positive_int(top_k, 'top_k')
    positive_int(rrf_k, 'rrf_k')
    if not isinstance(runs, list):
        raise ValueError('runs must be a list')
    merged, seen_queries = {}, set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or run.get('method') not in ('dense', 'sparse'):
            raise ValueError('each run needs method dense or sparse')
        query = run.get('query')
        if not isinstance(query, str) or not query.strip() or not isinstance(run.get('results'), list):
            raise ValueError('each run needs a nonblank query and results list')
        text_source = run.get('text_source', 'messages')
        weight = run.get('weight', 1.0)
        if not isinstance(text_source, str) or not text_source.strip():
            raise ValueError('text_source must be a nonblank string')
        if (isinstance(weight, bool) or not isinstance(weight, (int, float)) or
                not math.isfinite(weight) or weight < 0):
            raise ValueError('run weight must be a finite nonnegative number')
        if weight == 0:
            continue
        query_key = (text_source, run['method'], ' '.join(query.casefold().split()))
        if query_key in seen_queries:
            continue
        seen_queries.add(query_key)
        seen_ids = set()
        for hit in run['results']:
            if not isinstance(hit, dict):
                raise ValueError('each hit must be an object')
            chunk_id = hit.get('chunk_id')
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError('each hit needs a nonblank chunk_id')
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            rank = len(seen_ids)
            score = hit.get('score')
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError('each hit needs a finite numeric score')
            if not isinstance(hit.get('chunk'), dict) or hit['chunk'].get('id') != chunk_id:
                raise ValueError('chunk.id must match chunk_id')
            if any(not isinstance(hit.get(key), str) or not hit[key]
                   for key in ('source_file', 'source_sha256', 'chunks_file')):
                raise ValueError('each hit needs source_file, source_sha256, and chunks_file')
            if chunk_id not in merged:
                merged[chunk_id] = {key: hit[key] for key in
                                    ('chunk_id', 'source_file', 'source_sha256', 'chunks_file', 'chunk')}
                merged[chunk_id].update(rrf_score=0.0, contributions=[])
            candidate = merged[chunk_id]
            if any(candidate[key] != hit[key] for key in ('source_file', 'source_sha256', 'chunk')):
                raise ValueError(f'Conflicting corpus versions or ID collision for {chunk_id}')
            if 'summary' in hit:
                if not isinstance(hit['summary'], str) or not hit['summary'].strip():
                    raise ValueError('summary must be nonblank text')
                if 'summary' in candidate and candidate['summary'] != hit['summary']:
                    raise ValueError(f'Conflicting summaries for {chunk_id}')
                candidate['summary'] = hit['summary']
            contribution = weight / (rrf_k + rank)
            candidate['rrf_score'] += contribution
            candidate['contributions'].append(dict(run_id=run.get('run_id', f'run-{index + 1}'),
                method=run['method'], query=query, rank=rank, score=score, contribution=contribution))
            if 'text_source' in run or 'weight' in run:
                candidate['contributions'][-1].update(text_source=text_source, weight=weight)
    # Stable explicit tie-break, independent of input list ordering.
    ranked = sorted(merged.values(), key=lambda hit: (-hit['rrf_score'], hit['chunk_id']))
    return ranked[:top_k]
