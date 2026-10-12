"""Shared loading and result format for chunk search."""
import json
import re
from pathlib import Path

from imsg.config import ROOT
DEFAULT_CHUNKS_DIR = ROOT / 'segments' / 'chunk_data'
DEFAULT_CACHE_DIR = ROOT / '.cache'


def load_chunks(chunks_dir):
    paths = sorted(Path(chunks_dir).glob('*.chunks.json'))
    if not paths:
        raise FileNotFoundError(f'No *.chunks.json files in {chunks_dir}')
    records = []
    for path in paths:
        doc = json.loads(path.read_text(encoding='utf-8'))
        if doc.get('schema_version') != 1:
            raise ValueError(f'{path}: unsupported schema_version')
        if not doc.get('eligibility', {}).get('eligible', True):
            continue
        for chunk in doc['chunks']:
            records.append(dict(source=doc['source'], chunks_file=str(path), chunk=chunk))
    return records


def chunk_text(record, semantic=False):
    texts = []
    for message in record['chunk']['messages']:
        if message['record_type'] != 'message':
            continue
        attachments = set(message.get('attachments', []))
        text = '\n'.join(line for line in message['text'].splitlines()
                         if line.strip() not in attachments)
        if semantic:
            text = re.sub(r'https?://\S+', ' ', text)
        texts.append(text)
    return '\n'.join(texts).strip()


def validate_query(query, top_k):
    if not isinstance(query, str):
        raise TypeError('query must be a string')
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k must be a positive integer')


def result(record, score):
    return dict(chunk_id=record['chunk']['id'], score=float(score),
                source_file=record['source']['file'],
                source_sha256=record['source']['sha256'],
                chunks_file=record['chunks_file'], chunk=record['chunk'])
