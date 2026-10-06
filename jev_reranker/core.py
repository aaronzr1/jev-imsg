"""Candidate verification independent of retrieval and the model provider."""
from copy import deepcopy
from datetime import datetime
import math


STATUSES = ('match', 'no_match', 'error')


def candidates_from(bundle):
    if not isinstance(bundle, dict) or not isinstance(bundle.get('plan'), dict):
        raise ValueError('input must be an operator bundle with its original plan')
    plan = bundle['plan']
    if not isinstance(plan.get('query'), str) or not plan['query'].strip():
        raise ValueError('plan.query must contain the original query')
    if plan.get('answer_mode', 'relevance') not in ('relevance', 'earliest', 'latest'):
        raise ValueError('invalid answer_mode')
    requirements = plan.get('requirements', [])
    if not isinstance(requirements, list) or any(not isinstance(x, str) for x in requirements):
        raise ValueError('requirements must be a list of strings')
    operator = bundle.get('operator', {})
    candidates = bundle.get('results') if 'results' in bundle else (
        operator.get('kept') if isinstance(operator, dict) else None)
    if not isinstance(candidates, list):
        raise ValueError('input needs results or operator.kept')
    unique = {}
    for hit in candidates:
        if not isinstance(hit, dict) or not isinstance(hit.get('chunk_id'), str) or not hit['chunk_id']:
            raise ValueError('each candidate needs a chunk_id')
        if any(not isinstance(hit.get(k), str) or not hit[k] for k in ('source_file', 'source_sha256', 'chunks_file')):
            raise ValueError('each candidate needs source provenance')
        chunk = hit.get('chunk')
        if not isinstance(chunk, dict) or chunk.get('id') != hit['chunk_id']:
            raise ValueError('candidate chunk.id must equal chunk_id')
        messages = chunk.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError('each chunk needs complete structured messages')
        ids = set()
        for m in messages:
            if not isinstance(m, dict) or any(not isinstance(m.get(k), str) for k in ('id', 'text', 'sender', 'timestamp')):
                raise ValueError('each message needs id, text, sender, timestamp')
            if not m['id'] or m['id'] in ids:
                raise ValueError('message IDs must be nonblank and unique within a chunk')
            datetime.fromisoformat(m['timestamp'])
            occurrences = m.get('source_occurrences')
            if not isinstance(occurrences, list) or not occurrences:
                raise ValueError('each message needs source_occurrences')
            for occurrence in occurrences:
                lines = occurrence.get('lines') if isinstance(occurrence, dict) else None
                if (not isinstance(lines, list) or len(lines) != 2 or
                        any(type(n) is not int or n < 1 for n in lines) or lines[1] < lines[0]):
                    raise ValueError('invalid message source line range')
            ids.add(m['id'])
        previous = unique.get(hit['chunk_id'])
        if previous is not None and previous != hit:
            raise ValueError('conflicting duplicate chunk: ' + hit['chunk_id'])
        unique.setdefault(hit['chunk_id'], hit)
    return plan, list(unique.values()), len(candidates)



def probability(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('expected a finite probability between 0 and 1')
    return float(value)


def rerank(bundle, judge, threshold=0.5, progress=None):
    """Evaluate every unique chunk, retain matches, and rank by Jev probability.

    judge.evaluate(plan, hit, scope) returns {probability, model}.
    Judges with evaluate_many use native batching and return None for failed items.
    Provider failures remain visible and never become negative judgments.
    """
    threshold = probability(threshold)
    plan, candidates, input_count = candidates_from(bundle)
    scope = bundle.get('scope', {})
    warnings = bundle.get('warnings', [])
    if not isinstance(scope, dict) or not isinstance(warnings, list) or any(
            not isinstance(w, str) for w in warnings):
        raise ValueError('scope must be an object and warnings a list of strings')
    batch_answers = None
    if callable(getattr(type(judge), 'evaluate_many', None)):
        batch_answers = judge.evaluate_many(plan, candidates, scope, progress=progress)
        if not isinstance(batch_answers, list) or len(batch_answers) != len(candidates):
            raise ValueError('batch judge must return one answer per candidate')
    decisions = []
    for rank, hit in enumerate(candidates, 1):
        if progress and batch_answers is None:
            progress(rank, len(candidates), hit['chunk_id'])
        decision = dict(candidate_rank=rank, **deepcopy(hit))
        try:
            answer = (batch_answers[rank - 1] if batch_answers is not None
                      else judge.evaluate(plan, hit, scope))
            score = probability(answer['probability'])
            decision.update(status='match' if score >= threshold else 'no_match',
                            match_probability=score, model=answer['model'], error=None)
        except (ValueError, RuntimeError, OSError, TypeError, KeyError):
            # Do not serialize provider exceptions: they can include credentials or text.
            decision.update(status='error', match_probability=None, model=None,
                            error='Jev evaluation failed; check credentials, connectivity, and response schema.')
        decisions.append(decision)
    results = sorted((deepcopy(d) for d in decisions if d['status'] == 'match'),
                     key=lambda d: (-d['match_probability'], d['candidate_rank']))
    for rank, result in enumerate(results, 1):
        result['rank'] = rank
    warnings = list(warnings)
    warnings.append('Jev probabilities are model judgments; source metadata does not verify semantic correctness.')
    if plan.get('answer_mode', 'relevance') != 'relevance':
        warnings.append('Results are ranked by match probability. First/last occurrence is not established: '
                        'chunk start times are not matching-message times, and retrieval is bounded.')
    counts = {s: sum(d['status'] == s for d in decisions) for s in STATUSES}
    output = dict(schema_version=1, query=plan['query'], plan=deepcopy(plan),
                scope=deepcopy(scope), complete=counts['error'] == 0,
                results=results, decisions=decisions,
                counts=dict(input=input_count, unique=len(candidates), **counts),
                warnings=warnings, settings=dict(threshold=threshold,
                    order='match_probability_desc', judge=deepcopy(getattr(judge, 'settings', {}))))

    batch_stats = getattr(judge, 'last_batch_stats', None)
    if isinstance(batch_stats, list):
        output['jev_batches'] = deepcopy(batch_stats)
    return output
