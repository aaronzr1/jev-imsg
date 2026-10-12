"""Local web interface for the shared search pipeline (no extra dependencies)."""
import argparse
from functools import lru_cache
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .intent_search.api import Handler as APIHandler
from .intent_search.service import IntentSearchService
from .config import DEFAULTS, ROOT

WEB = ROOT / 'web'


def title(hit):
    people = [p for p in hit['chunk'].get('participants', [])
              if p.lower() not in ('me', '[system event]')]
    return ', '.join(people) or Path(hit['source_file']).stem


def summary(hit, index):
    chunk = hit['chunk']
    return dict(index=index, title=title(hit), source=hit['source_file'],
                preview=chunk.get('preview') or next(
                    (m['text'][:200] for m in chunk['messages'] if m['text']), 'Attachment'),
                start=chunk['start'], end=chunk['end'],
                message_count=len(chunk['messages']), probability=hit['match_probability'],
                summary=hit.get('summary'))


@lru_cache(maxsize=2)
def conversation(path, modified, size):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    messages = {m['id']: m for c in document['chunks'] for m in c['messages']}
    ordered = sorted(messages.values(), key=lambda m: (
        m['timestamp'], m['source_occurrences'][0]['lines'][0]))
    return document['source'], ordered


def context(hit, start=None, end=None, *, chunks_dir=None):
    # The browser supplies a result index, never a filesystem path.
    path = Path(hit['chunks_file']).resolve()
    if path.parent != (chunks_dir or DEFAULTS['chunks_dir']).resolve() or not path.name.endswith('.chunks.json'):
        raise ValueError('Conversation is outside the configured corpus.')
    stat = path.stat()
    source, messages = conversation(str(path), stat.st_mtime_ns, stat.st_size)
    if source['sha256'] != hit['source_sha256']:
        raise ValueError('Conversation changed since this search. Search again.')
    matched = {m['id'] for m in hit['chunk']['messages']}
    positions = [i for i, m in enumerate(messages) if m['id'] in matched]
    if not positions:
        raise ValueError('Conversation changed since this search. Search again.')
    if start is None:
        start, end = max(0, min(positions) - 20), min(len(messages), max(positions) + 21)
    else:
        if end is None or start < 0 or end <= start or end - start > 200:
            raise ValueError('Invalid context range.')
        start, end = min(start, len(messages)), min(end, len(messages))
    return dict(title=title(hit), source=hit['source_file'], start=start, end=end,
                total=len(messages), matched_ids=sorted(matched),
                messages=[{key: m.get(key) for key in (
                    'id', 'sender', 'text', 'timestamp', 'record_type',
                    'reactions', 'attachments', 'reply_to')} for m in messages[start:end]])


# The browser and API share the same service and single search worker.
Search = IntentSearchService


def browser_status(search, job_id):
    response = search.status(job_id)
    if response['status'] == 'done':
        output = response.pop('output')
        response.update(results=[summary(h, i) for i, h in enumerate(output['results'])],
                        warnings=output.get('warnings', []), complete=output.get('complete', True))
    return response


class Handler(APIHandler):
    def do_POST(self):
        if urlsplit(self.path).path == '/api/search':
            self.path = '/api/intent-search'
        super().do_POST()

    def do_GET(self):
        if not self.local_request():
            return
        url = urlsplit(self.path)
        if url.path == '/api/config':
            settings = self.server.search.settings
            return self.send(200, {k: settings[k] for k in ('retrieval_text', 'summary_weight')})
        assets = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                  '/style.css': ('style.css', 'text/css')}
        if url.path in assets:
            name, mime = assets[url.path]
            return self.send(200, (WEB / name).read_bytes(), mime + '; charset=utf-8')
        parts = url.path.strip('/').split('/')
        try:
            if len(parts) == 3 and parts[:2] == ['api', 'search']:
                return self.send(200, browser_status(self.server.search, parts[2]))
            if len(parts) == 5 and parts[:2] == ['api', 'search'] and parts[3] == 'context':
                job = self.server.search.get(parts[2])
                index = int(parts[4])
                if job['status'] != 'done' or index < 0:
                    raise KeyError()
                hit = job['output']['results'][index]
                params = parse_qs(url.query)
                start = int(params['start'][0]) if 'start' in params else None
                end = int(params['end'][0]) if 'end' in params else None
                return self.send(200, context(hit, start, end, chunks_dir=self.server.search.settings['chunks_dir']))
            super().do_GET()
        except (KeyError, IndexError):
            self.send(404, dict(error='Search or result expired. Run your query again.'))
        except ValueError as exc:
            self.send(400, dict(error=str(exc)))
        except OSError:
            self.send(404, dict(error='Conversation is unavailable. Run your query again.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    from imsg.intent_search.chunk_summaries.options import add_options, validate_options
    add_options(parser, generation=False)
    parser.set_defaults(**DEFAULTS)
    args = parser.parse_args()
    try:
        validate_options(args.retrieval_text, args.summary_weight)
    except ValueError as exc:
        parser.error(str(exc))
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.search = Search(vars(args))
    print(f'Message search: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.search.close()
        server.server_close()


if __name__ == '__main__':
    main()
