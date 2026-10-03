"""Local, deterministic conversation segmentation for iMessage text exports."""
import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '2')
HEADER = re.compile(r'^( *)([A-Z][a-z]{2} \d{2}, \d{4}\s+\d{1,2}:\d{2}:\d{2} [AP]M)(.*)$')
REPLY = 'This message responded to an earlier message.'
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def parse(path):
    data = path.read_bytes()
    parts = data.decode('utf-8-sig').split('\n')
    lines = [s + '\n' for s in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
    starts = [(i, HEADER.match(line.rstrip('\r\n'))) for i, line in enumerate(lines)]
    starts = [(i, m) for i, m in starts if m and (not m[3].strip() or
              m[3].strip().startswith('(Read ') or
              re.fullmatch(r'.+ (?:unsent a message!|kept an audio message\.|changed the group photo\.|renamed the conversation to .+)', m[3].strip()))]
    if not starts or any(s.strip() for s in lines[:starts[0][0]]):
        raise ValueError(f'{path}: missing header or unexplained preamble')
    occurrences, stack = [], []
    for n, (start, match) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        indent = len(match[1])
        block = [s[indent:] if s.startswith(' ' * indent) else s for s in lines[start + 1:end]]
        original_body = ''.join(block[1:]).rstrip('\r\n')
        event = not block or not block[0].strip()
        if event:
            if not match[3].strip():
                raise ValueError(f'{path}:{start + 1}: missing sender and event')
            block = ['[system event]\n', match[3].strip()]
        body = ''.join(block[1:]).rstrip('\r\n')
        text_lines, reactions, attachments = [], [], []
        in_reactions = False
        for line in body.splitlines():
            stripped = line.strip()
            if stripped == REPLY:
                continue
            if stripped == 'Tapbacks:':
                in_reactions = True
                continue
            if in_reactions and not stripped:
                in_reactions = False
                continue
            if in_reactions:
                reactions.append(stripped)
            else:
                text_lines.append(line)
                if stripped.startswith('/Users/') and '/Library/Messages/Attachments/' in stripped:
                    attachments.append(stripped)
        text = '\n'.join(text_lines).strip()
        edits = list(re.finditer(r'^Edited [^\n]*? later: ?', text, re.MULTILINE))
        if edits:
            text = text[edits[-1].end():].strip()
        # Attachment paths are metadata, not semantic evidence.
        semantic = '\n'.join(s for s in text.splitlines() if s.strip() not in attachments)
        semantic = re.sub(r'https?://\S+', ' ', semantic).strip()
        while stack and occurrences[stack[-1]]['indent'] >= indent:
            stack.pop()
        occurrence = dict(timestamp=datetime.strptime(match[2], '%b %d, %Y %I:%M:%S %p').isoformat(),
                          sender=block[0].strip(), text=text, semantic='' if event else semantic,
                          record_type='event' if event else 'message',
                          attachments=attachments, reactions=reactions,
                          export_body=original_body, header_metadata=match[3].strip() or None,
                          is_reply=REPLY in body, source_lines=[start + 1, end],
                          indent=indent, parent=stack[-1] if stack else None)
        occurrences.append(occurrence)
        stack.append(len(occurrences) - 1)
    def key(m):
        return m['timestamp'], m['sender'], re.sub(r'\s+', ' ', m['text']).strip()
    canonical, mapping, top = [], {}, defaultdict(list)
    prefix = hashlib.sha256(path.name.encode()).hexdigest()[:8]
    def add(m):
        result = {k: m[k] for k in ('timestamp', 'sender', 'text', 'attachments', 'reactions', 'export_body', 'header_metadata', 'record_type')}
        result.update(id=f'{prefix}:m{m["source_lines"][0]}', reply_to=[],
                      reply_context_unresolved=m['is_reply'], source_occurrences=[], _semantic=m['semantic'])
        canonical.append(result)
        return len(canonical) - 1
    # Never collapse two top-level messages, even if their timestamp/text match.
    for i, m in enumerate(occurrences):
        if m['indent'] == 0:
            mapping[i] = add(m)
            top[key(m)].append(mapping[i])
    ambiguous = 0
    for i, m in enumerate(occurrences):
        if i not in mapping:
            matches = top[key(m)]
            if len(matches) == 1:
                mapping[i] = matches[0]
            else:
                mapping[i] = add(m)
                ambiguous += int(len(matches) > 1)
        c = canonical[mapping[i]]
        c['source_occurrences'].append(dict(lines=m['source_lines'], kind='thread_copy' if m['indent'] else 'timeline'))
        c['reactions'] = list(dict.fromkeys(c['reactions'] + m['reactions']))
    for i, m in enumerate(occurrences):
        if m['parent'] is not None:
            c, parent = canonical[mapping[i]], canonical[mapping[m['parent']]]
            if parent['id'] != c['id'] and parent['id'] not in c['reply_to']:
                c['reply_to'].append(parent['id'])
                c['reply_context_unresolved'] = False
    canonical.sort(key=lambda m: (m['timestamp'], m['source_occurrences'][0]['lines'][0]))
    return canonical, dict(file=path.name, sha256=hashlib.sha256(data).hexdigest(),
                          line_count=len(lines), exported_records=len(occurrences),
                          unique_messages=len(canonical), thread_copies_removed=len(occurrences)-len(canonical),
                          ambiguous_thread_matches=ambiguous)


def gap(a, b):
    return (datetime.fromisoformat(b['timestamp']) - datetime.fromisoformat(a['timestamp'])).total_seconds()


def eligibility(messages):
    """Skip only text-only threads made entirely of recognizable OTP templates."""
    texts = [m for m in messages if m['record_type'] == 'message']
    def is_code(m):
        text = m['text']
        return (not m['attachments'] and len(text) <= 600 and
                bool(re.search(r'\b\d{4,8}\b', text)) and
                bool(re.search(r'\b(?:(?:verification|security|authentication|login|sign[- ]in|one[- ]time|access)\s+(?:code|passcode|password)|OTP)\b', text, re.I)))
    code_count = sum(bool(is_code(m)) for m in texts)
    skip = bool(texts) and code_count == len(texts)
    return dict(eligible=not skip, reason='verification_codes_only' if skip else 'keep_by_default',
                message_count=len(texts), verification_code_messages=code_count,
                method='local_template_heuristic')


def bursts(messages):
    result, current = [], []
    for i, m in enumerate(messages):
        if current and (gap(messages[current[-1]], m) > 120 or len(current) >= 8 or
                        gap(messages[current[0]], m) > 90 or
                        sum(len(messages[j]['_semantic'].split()) for j in current) >= 160):
            result.append(current)
            current = []
        current.append(i)
    if current:
        result.append(current)
    return result


def segment(messages, units, vectors, args):
    import numpy as np
    def context(index, direction):
        picked = [index]
        other = index + direction
        if 0 <= other < len(units):
            left, right = sorted([index, other])
            if gap(messages[units[left][-1]], messages[units[right][0]]) <= 600:
                picked.append(other)
        valid = [j for j in picked if any(messages[k]['_semantic'] for k in units[j])]
        if not valid:
            return None
        v = vectors[valid].mean(axis=0)
        return v / max(float(np.linalg.norm(v)), 1e-9)
    starts = [(0, dict(reason='start_of_file', gap_minutes=None, cosine_similarity=None))]
    start_unit = 0
    count = len(units[0])
    for i in range(1, len(units)):
        seconds = gap(messages[units[i-1][-1]], messages[units[i][0]])
        a, b = context(i-1, -1), context(i, 1)
        score = float(a @ b) if a is not None and b is not None else None
        current_ids = {messages[k]['id'] for u in units[start_unit:i] for k in u}
        reply_bridge = any(set(messages[k]['reply_to']) & current_ids for k in units[i])
        previous = messages[units[i-1][-1]]
        upcoming = messages[units[i][0]]
        short_ack = (len(upcoming['_semantic'].split()) <= 12 and
                     bool(re.match(r"^(sure|yeah|yea|yes|yep|nah|no|okay|ok|lol|lmao|oh|thanks|thank|bet|sounds|got it|unlucky)\b", upcoming['_semantic'], re.I)))
        question_answer = (seconds <= 30 * 60 and previous['sender'] != upcoming['sender'] and
                           any('?' in messages[k]['_semantic'] for k in units[i-1][-3:]))
        conversational_reply = seconds <= 45 * 60 and (short_ack or question_answer)
        reason = None
        if seconds >= args.hard_gap * 60:
            if score is None or score < args.long_gap_similarity:
                reason = 'long_gap_and_weak_topic_continuity'
            elif count + len(units[i]) > args.max_messages:
                reason = 'size_limit'
        elif count + len(units[i]) > args.max_messages:
            reason = 'size_limit'
        elif not reply_bridge and not conversational_reply:
            if seconds >= args.idle_gap * 60 and (score is None or score < args.resume_similarity):
                reason = 'idle_gap_and_weak_topic_continuity'
            elif seconds >= args.topic_gap * 60 and score is not None and score < args.similarity:
                reason = 'topic_change_after_pause'
        if reason:
            starts.append((i, dict(reason=reason, gap_minutes=round(seconds / 60, 2),
                                   cosine_similarity=round(score, 4) if score is not None else None,
                                   explicit_reply_crosses_boundary=reply_bridge)))
            start_unit, count = i, 0
        count += len(units[i])
    chunks = []
    for n, (first, boundary) in enumerate(starts):
        last = starts[n+1][0] if n+1 < len(starts) else len(units)
        selected = [messages[k] for u in units[first:last] for k in u]
        chunks.append(dict(id=f'{selected[0]["id"].split(":")[0]}:c{n+1:05d}',
                           start=selected[0]['timestamp'], end=selected[-1]['timestamp'],
                           participants=list(dict.fromkeys(m['sender'] for m in selected)),
                           message_count=len(selected), boundary_before=boundary,
                           preview=next((m['_semantic'][:160] for m in selected if m['_semantic']), '[attachments / reactions]'),
                           transcript=[f'{m["timestamp"][11:]} {m["sender"]}: {m["text"]}' for m in selected],
                           messages=[{k: v for k, v in m.items() if not k.startswith('_')} for m in selected]))
    owner = {m['id']: c['id'] for c in chunks for m in c['messages']}
    for c in chunks:
        c['related_chunk_ids'] = sorted({owner[p] for m in c['messages'] for p in m['reply_to'] if owner[p] != c['id']})
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, default=Path('raw'), help='Export directory (default: raw)')
    ap.add_argument('--output-dir', type=Path, default=Path('segments'))
    ap.add_argument('--similarity', type=float, default=0.35)
    ap.add_argument('--resume-similarity', type=float, default=0.65)
    ap.add_argument('--long-gap-similarity', type=float, default=0.20)
    ap.add_argument('--topic-gap', type=float, default=5, help='Minimum pause in minutes for a semantic split')
    ap.add_argument('--idle-gap', type=float, default=45)
    ap.add_argument('--hard-gap', type=float, default=240, help='Long-pause threshold in minutes; uses --long-gap-similarity')
    ap.add_argument('--max-messages', type=int, default=120)
    args = ap.parse_args()
    if not (0 <= args.similarity <= 1 and 0 <= args.resume_similarity <= 1 and
            0 <= args.long_gap_similarity <= 1 and
            0 <= args.topic_gap <= args.idle_gap <= args.hard_gap and args.max_messages >= 8):
        ap.error('Invalid threshold, gap ordering, or max-messages (minimum 8)')
    import numpy as np
    from fastembed import TextEmbedding
    paths = sorted(p for p in args.input_dir.glob('*.txt') if p.name != 'requirements.txt')
    if not paths:
        ap.error('No input exports found')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = Path('.cache')
    cache.mkdir(exist_ok=True)
    model = None
    overview, review = [], []
    settings = {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
    for path in paths:
        messages, source = parse(path)
        if not messages:
            raise ValueError(f'No messages in {path}')
        screening = eligibility(messages)
        if not screening['eligible']:
            output = args.output_dir / (path.stem + '.chunks.json')
            stats = dict(chunks=0, median_messages=0, single_message_chunks=0, boundary_reasons={})
            doc = dict(schema_version=1, source=source, eligibility=screening,
                       timestamp_timezone='unknown; preserved export wall time',
                       statistics=stats, chunks=[],
                       messages=[{k: v for k, v in m.items() if not k.startswith('_')} for m in messages])
            output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
            overview.append(dict(source=source, output=output.name, eligibility=screening, statistics=stats))
            print(f'{path.name}: skipped (verification codes only)', flush=True)
            continue
        units = bursts(messages)
        texts = ['\n'.join(messages[k]['_semantic'] for k in u if messages[k]['_semantic']) or '[attachment]' for u in units]
        digest = hashlib.sha256((MODEL + json.dumps(texts, ensure_ascii=False)).encode()).hexdigest()
        cached = cache / f'{digest}.npy'
        print(f'{path.name}: {len(messages)} messages, {len(units)} embedding windows', flush=True)
        if cached.exists():
            vectors = np.load(cached)
        else:
            if model is None:
                model = TextEmbedding(model_name=MODEL, cache_dir=str(cache / 'models'), threads=2)
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            vectors = np.zeros((len(texts), 384), dtype=np.float32)
            for start in range(0, len(texts), 512):
                indices = order[start:start+512]
                batch_texts = [texts[i] for i in indices]
                batch_key = hashlib.sha256((MODEL + json.dumps(batch_texts, ensure_ascii=False)).encode()).hexdigest()
                batch_path = cache / f'batch-{batch_key}.npy'
                if batch_path.exists():
                    batch = np.load(batch_path)
                else:
                    batch = np.asarray(list(model.embed(batch_texts, batch_size=32)))
                    np.save(batch_path, batch)
                vectors[indices] = batch
                print(f'  embedded {min(start+512, len(texts))}/{len(texts)} windows', flush=True)
            np.save(cached, vectors)
        chunks = segment(messages, units, vectors, args)
        assert sum(c['message_count'] for c in chunks) == len(messages)
        assert len({m['id'] for c in chunks for m in c['messages']}) == len(messages)
        stats = dict(chunks=len(chunks), median_messages=float(np.median([c['message_count'] for c in chunks])),
                     single_message_chunks=sum(c['message_count'] == 1 for c in chunks),
                     boundary_reasons=dict(Counter(c['boundary_before']['reason'] for c in chunks)))
        doc = dict(schema_version=1, source=source, eligibility=screening, timestamp_timezone='unknown; preserved export wall time',
                   method=dict(embedding_model=MODEL, settings=settings, inference='local CPU; two threads'),
                   statistics=stats, chunks=chunks)
        output = args.output_dir / (path.stem + '.chunks.json')
        output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
        overview.append(dict(source=source, output=output.name, eligibility=screening, statistics=stats))
        # Consecutive chunks let reviewers inspect both sides of actual boundaries.
        indices = set(range(min(4, len(chunks))))
        for reason in ('topic_change_after_pause', 'idle_gap_and_weak_topic_continuity', 'size_limit'):
            matches = [i for i, c in enumerate(chunks) if c['boundary_before']['reason'] == reason]
            for i in matches[:2]:
                indices.update(range(max(0, i-1), min(len(chunks), i+1)))
        for i in sorted(indices):
            c = chunks[i]
            review.append(dict(source_file=path.name, chunk_id=c['id'], start=c['start'], end=c['end'],
                               message_count=c['message_count'], boundary_before=c['boundary_before'],
                               transcript=c['transcript'], related_chunk_ids=c['related_chunk_ids'],
                               assessment=None, reviewer_notes=''))
        print(f'  wrote {len(chunks)} chunks to {output}', flush=True)
    (args.output_dir / 'index.json').write_text(json.dumps(dict(files=overview), ensure_ascii=False, indent=2) + '\n')
    (args.output_dir / 'review-sample.json').write_text(json.dumps(dict(
        instructions='Consecutive opening chunks and first examples of each heuristic boundary. Compare neighboring chunks; set assessment to good, over_split, or under_split. Samples are diagnostic, not random.',
        chunks=review), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
