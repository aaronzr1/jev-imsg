"""Content-addressed summary sidecars; never rewrite source chunks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

PROMPT_VERSION = 1


def evidence(record):
    chunk = record['chunk']
    messages = []
    for message in chunk['messages']:
        attachments = set(message.get('attachments', []))
        text = '\n'.join(line for line in message['text'].splitlines()
                         if line.strip() not in attachments)
        messages.append(dict(timestamp=message['timestamp'], sender=message['sender'],
                             text=text, record_type=message.get('record_type', 'message'),
                             attachment_count=len(attachments)))
    return dict(chunk_id=chunk['id'], messages=messages)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


class SummaryStore:
    def __init__(self, directory, model):
        self.directory, self.model = Path(directory), model

    def identity(self, record):
        return dict(chunk_id=record['chunk']['id'], source_sha256=record['source']['sha256'],
                    evidence_sha256=digest(evidence(record)), requested_model=self.model,
                    prompt_version=PROMPT_VERSION)

    def path(self, record):
        return self.directory / (digest(self.identity(record)) + '.json')

    def read(self, record):
        path = self.path(record)
        if not path.exists():
            return None
        artifact = json.loads(path.read_text(encoding='utf-8'))
        if (not isinstance(artifact, dict) or artifact.get('schema_version') != 1 or
                any(artifact.get(k) != v for k, v in self.identity(record).items())):
            raise ValueError('Summary provenance mismatch; regenerate summaries.')
        text = artifact.get('summary')
        if not isinstance(text, str) or not text.strip() or len(text) > 1600:
            raise ValueError('Invalid cached summary; regenerate summaries.')
        return text.strip()

    def write(self, record, summary, model):
        if not isinstance(summary, str) or not summary.strip() or len(summary) > 1600:
            raise ValueError('Summary must contain 1–1,600 characters.')
        self.directory.mkdir(parents=True, exist_ok=True)
        artifact = dict(schema_version=1, **self.identity(record), summary=summary.strip(),
                        model=model, created_at=datetime.now(timezone.utc).isoformat())
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.directory,
                                             suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(artifact, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            temporary.replace(self.path(record))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def signature(self):
        files = []
        for path in sorted(self.directory.glob('*.json')):
            stat = path.stat()
            files.append((path.name, stat.st_mtime_ns, stat.st_size))
        return str(self.directory.resolve()), self.model, PROMPT_VERSION, tuple(files)

    def text_factory(self, records):
        texts, missing = {}, []
        for record in records:
            text = self.read(record)
            if text is None:
                missing.append(record['chunk']['id'])
            else:
                texts[record['chunk']['id']] = text
        if missing:
            raise RuntimeError(f'{len(missing)} chunks lack current summaries. '
                               'Run main.py --stage summarize with the same chunks directory, '
                               'summaries directory, and summary model before summary retrieval.')

        def text(record, semantic=False):
            return texts[record['chunk']['id']]

        return text
