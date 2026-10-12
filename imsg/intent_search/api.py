"""Local HTTP API for intent search, with no frontend dependency."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import urlsplit

from imsg.config import DEFAULTS
from .chunk_summaries.options import add_options, validate_options
from .service import IntentSearchService

SEARCH_PATH = '/api/intent-search'


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Queries and conversation identifiers do not belong in access logs.

    def send(self, status, body, content_type='application/json; charset=utf-8', *, location=None):
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode() if isinstance(body, dict) else body
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if location is not None:
            self.send_header('Location', location)
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
        if urlsplit(self.path).path != SEARCH_PATH:
            return self.send(404, dict(error='Not found.'))
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8192 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ValueError('Send a query as JSON.')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Send a query as JSON.')
            if set(payload) - {'query', 'retrieval_text', 'summary_weight', 'top_k'}:
                raise ValueError('Allowed fields: query, retrieval_text, summary_weight, top_k.')
            job_id = self.server.search.start(payload.get('query'), payload.get('retrieval_text'),
                                               payload.get('summary_weight'), payload.get('top_k'))
            if job_id is None:
                return self.send(409, dict(error='A search is already running. Please wait for it to finish.'))
            location = SEARCH_PATH + '/' + job_id
            self.send(202, dict(id=job_id, status='running', status_url=location, schema_version=1),
                      location=location)
        except (ValueError, UnicodeDecodeError) as exc:
            self.send(400, dict(error=str(exc)))
        except RuntimeError:
            self.send(503, dict(error='Search service is shutting down.'))

    def do_GET(self):
        if not self.local_request():
            return
        path = urlsplit(self.path).path
        if path == '/api/health':
            return self.send(200, dict(status='ok', service='intent-search', schema_version=1))
        if path == SEARCH_PATH + '/config':
            settings = self.server.search.settings
            return self.send(200, {k: settings[k] for k in ('retrieval_text', 'summary_weight', 'top_k')})
        parts = path.strip('/').split('/')
        if len(parts) == 3 and parts[:2] == ['api', 'intent-search']:
            try:
                return self.send(200, self.server.search.status(parts[2]))
            except KeyError:
                return self.send(404, dict(error='Search expired. Run your query again.'))
        self.send(404, dict(error='Not found.'))


class IntentSearchServer(ThreadingHTTPServer):
    def __init__(self, port=8001, *, settings=None, handler=Handler):
        self.search = IntentSearchService(settings)
        super().__init__(('127.0.0.1', port), handler)

    def server_close(self):
        self.search.close()
        super().server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8001)
    add_options(parser, generation=False)
    parser.set_defaults(**DEFAULTS)
    args = parser.parse_args(argv)
    try:
        validate_options(args.retrieval_text, args.summary_weight)
    except ValueError as exc:
        parser.error(str(exc))
    with IntentSearchServer(args.port, settings=vars(args)) as server:
        print(f'Intent search API: http://127.0.0.1:{server.server_port}{SEARCH_PATH}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
