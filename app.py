"""Local web interface for the main.py search pipeline (no extra dependencies)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
import traceback
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from agent_query_operator.planner import QueryPlanError
from main import DEFAULTS, ROOT, SearchIndexes, run as run_pipeline

WEB = ROOT / 'web'
STAGES = ('segment', 'plan', 'retrieve', 'fuse', 'rerank')


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


def context(hit, start=None, end=None):
    # The browser supplies a result index, never a filesystem path.
    path = Path(hit['chunks_file']).resolve()
    if path.parent != DEFAULTS['chunks_dir'].resolve() or not path.name.endswith('.chunks.json'):
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


class Search:
    def __init__(self, settings=None):
        self.lock = Lock()
        self.job = None
        self.indexes = SearchIndexes()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='message-search')
        self.settings = dict(DEFAULTS, **(settings or {}))

    def close(self):
        self.worker.shutdown(wait=False, cancel_futures=True)

    def start(self, query, retrieval_text=None, summary_weight=None):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError('Enter a query between 1 and 2,000 characters.')
        from chunk_summaries.options import validate_options
        retrieval_text = self.settings['retrieval_text'] if retrieval_text is None else retrieval_text
        summary_weight = self.settings['summary_weight'] if summary_weight is None else summary_weight
        validate_options(retrieval_text, summary_weight)
        with self.lock:
            if self.job and self.job['status'] == 'running':
                return None
            job = dict(id=uuid4().hex, query=query.strip(), status='running',
                       directory=ROOT / '.cache' / ('web-' + uuid4().hex),
                       retrieval_text=retrieval_text, summary_weight=summary_weight)
            self.job = job
        self.worker.submit(self.run, job)
        return job['id']

    def run(self, job):
        directory = job['directory']
        try:
            directory.mkdir(parents=True)
            output_dir = directory / 'run'
            output_dir.mkdir()
            settings = dict(self.settings, retrieval_text=job.get('retrieval_text', 'messages'),
                            summary_weight=job.get('summary_weight', 0.5))
            args = argparse.Namespace(**settings, query=job['query'], out_dir=output_dir,
                                      stage='all', plan=None, input=None, aliases=None,
                                      resegment=False, no_cache=False)
            # The same worker keeps indices and the embedding model warm between queries.
            with (directory / 'pipeline.log').open('w', encoding='utf-8') as log:
                output = run_pipeline(args, indexes=self.indexes,
                                      log=lambda message: print(message, file=log, flush=True))
            with self.lock:
                job.update(status='done', output=output)
        except Exception as exc:
            # A failed query must not strand the persistent worker or expose its error in HTTP.
            try:
                with (directory / 'pipeline.log').open('a', encoding='utf-8') as log:
                    traceback.print_exc(file=log)
            except OSError:
                pass
            with self.lock:
                missing_summary = (isinstance(exc, RuntimeError) and
                                   'chunks lack current summaries' in str(exc))
                if missing_summary:
                    error = str(exc)
                elif isinstance(exc, QueryPlanError):
                    error = 'The query planner could not interpret this search. Try rephrasing your query.'
                else:
                    error = 'Search could not finish. Check that Ollama is running and JEV_KEY is set.'
                job.update(status='error', error=error + ' Details are saved in ' +
                           str(directory.relative_to(ROOT) / 'pipeline.log') + '.')

    def get(self, job_id):
        with self.lock:
            if not self.job or self.job['id'] != job_id:
                raise KeyError('Search expired. Run your query again.')
            return dict(self.job)

    def status(self, job_id):
        job = self.get(job_id)
        completed = [s for s in STAGES if (job['directory'] / 'run' / (s + '.json')).exists()]
        response = {k: job[k] for k in ('id', 'query', 'status')}
        response.update(retrieval_text=job.get('retrieval_text', 'messages'),
                        summary_weight=job.get('summary_weight', 0.5))
        response['completed_stages'] = completed
        if job['status'] == 'error':
            response['error'] = job['error']
        if job['status'] == 'done':
            output = job['output']
            response.update(results=[summary(h, i) for i, h in enumerate(output['results'])],
                            warnings=output.get('warnings', []), complete=output.get('complete', True))
        return response


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Queries and conversation identifiers do not belong in access logs.

    def send(self, status, body, content_type='application/json; charset=utf-8'):
        data = json.dumps(body, ensure_ascii=False).encode() if isinstance(body, dict) else body
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(data)

    def local_request(self):
        port = self.server.server_port
        allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
        host = self.headers.get('Host', '')
        origin = self.headers.get('Origin')
        if host not in allowed or (origin and origin != 'http://' + host):
            self.send(403, dict(error='Only requests from this local app are allowed.'))
            return False
        return True

    def do_POST(self):
        if not self.local_request():
            return
        if self.path != '/api/search':
            return self.send(404, dict(error='Not found.'))
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8192 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ValueError('Send a query as JSON.')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Send a query as JSON.')
            job_id = self.server.search.start(payload.get('query'), payload.get('retrieval_text'),
                                               payload.get('summary_weight'))
            if job_id is None:
                return self.send(409, dict(error='A search is already running. Please wait for it to finish.'))
            self.send(202, dict(id=job_id))
        except (ValueError, UnicodeDecodeError) as exc:
            self.send(400, dict(error=str(exc)))

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
                return self.send(200, self.server.search.status(parts[2]))
            if len(parts) == 5 and parts[:2] == ['api', 'search'] and parts[3] == 'context':
                job = self.server.search.get(parts[2])
                index = int(parts[4])
                if job['status'] != 'done' or index < 0:
                    raise KeyError()
                hit = job['output']['results'][index]
                params = parse_qs(url.query)
                start = int(params['start'][0]) if 'start' in params else None
                end = int(params['end'][0]) if 'end' in params else None
                return self.send(200, context(hit, start, end))
            self.send(404, dict(error='Not found.'))
        except (KeyError, IndexError):
            self.send(404, dict(error='Search or result expired. Run your query again.'))
        except ValueError as exc:
            self.send(400, dict(error=str(exc)))
        except OSError:
            self.send(404, dict(error='Conversation is unavailable. Run your query again.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    from chunk_summaries.options import add_options, validate_options
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
