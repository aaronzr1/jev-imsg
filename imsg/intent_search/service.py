"""Search job lifecycle and warm indexes shared by HTTP and Python clients."""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import traceback
from uuid import uuid4

from imsg.config import DEFAULTS
from .agent_query_operator.planner import QueryPlanError
from .chunk_summaries.options import validate_options
from .pipeline import SearchIndexes, load, run as run_pipeline

STAGES = ('segment', 'plan', 'retrieve', 'fuse', 'rerank')


class IntentSearchService:
    """Run one search at a time; retain the latest 16 jobs until restart.

    start() returns an ID, or None while busy. status() returns the job metadata
    and, on completion, the full pipeline output with messages and provenance.
    """

    def __init__(self, settings=None, *, max_jobs=16):
        if type(max_jobs) is not int or max_jobs < 1:
            raise ValueError('max_jobs must be a positive integer')
        self.settings = dict(DEFAULTS, **(settings or {}))
        self.max_jobs = max_jobs
        self.lock = Lock()
        self.jobs = OrderedDict()
        self.active_id = None
        self.closed = False
        self.indexes = SearchIndexes()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='intent-search')

    def close(self):
        with self.lock:
            self.closed = True
        self.worker.shutdown(wait=False, cancel_futures=True)

    def start(self, query, retrieval_text=None, summary_weight=None, top_k=None):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError('Enter a query between 1 and 2,000 characters.')
        retrieval_text = self.settings['retrieval_text'] if retrieval_text is None else retrieval_text
        summary_weight = self.settings['summary_weight'] if summary_weight is None else summary_weight
        top_k = self.settings['top_k'] if top_k is None else top_k
        validate_options(retrieval_text, summary_weight)
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError('top_k must be an integer between 1 and 100.')
        with self.lock:
            if self.closed:
                raise RuntimeError('Search service is shutting down.')
            if self.active_id is not None:
                return None
            job_id = uuid4().hex
            job = dict(id=job_id, query=query.strip(), status='running',
                       directory=self.settings['cache_dir'] / ('intent-search-' + job_id),
                       retrieval_text=retrieval_text, summary_weight=summary_weight, top_k=top_k)
            self.jobs[job_id] = job
            self.active_id = job_id
            while len(self.jobs) > self.max_jobs:
                self.jobs.popitem(last=False)
            # Submit under the lock so close() cannot shut the executor down here.
            self.worker.submit(self.run, job)
        return job_id

    def run(self, job):
        directory = job['directory']
        try:
            output_dir = directory / 'run'
            output_dir.mkdir(parents=True, exist_ok=False)
            settings = dict(self.settings, retrieval_text=job['retrieval_text'],
                            summary_weight=job['summary_weight'], top_k=job['top_k'])
            args = argparse.Namespace(**settings, query=job['query'], out_dir=output_dir,
                                      stage='all', plan=None, input=None, aliases=None,
                                      resegment=False, no_cache=False)
            with (directory / 'pipeline.log').open('w', encoding='utf-8') as log:
                output = run_pipeline(args, indexes=self.indexes,
                                      log=lambda message: print(message, file=log, flush=True))
            timings = load(output_dir / 'timings.json')
            with self.lock:
                job.update(status='done', output=output, timings=timings)
                self.active_id = None
        except Exception as exc:
            try:
                with (directory / 'pipeline.log').open('a', encoding='utf-8') as log:
                    traceback.print_exc(file=log)
            except OSError:
                pass
            missing_summary = (isinstance(exc, RuntimeError) and
                               'chunks lack current summaries' in str(exc))
            if missing_summary:
                error = str(exc)
            elif isinstance(exc, QueryPlanError):
                error = 'The query planner could not interpret this search. Try rephrasing your query.'
            else:
                error = 'Search could not finish. Check that Ollama is running and JEV_KEY is set.'
            with self.lock:
                job.update(status='error', error=error + ' Details are saved in the search run\'s pipeline.log.')
                self.active_id = None

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError('Search expired. Run your query again.')
            return dict(self.jobs[job_id])

    def status(self, job_id):
        job = self.get(job_id)
        response = {k: job[k] for k in (
            'id', 'query', 'status', 'retrieval_text', 'summary_weight', 'top_k')}
        response['schema_version'] = 1
        response['completed_stages'] = [
            s for s in STAGES if (job['directory'] / 'run' / (s + '.json')).exists()]
        if job['status'] == 'error':
            response['error'] = job['error']
        if job['status'] == 'done':
            response.update(output=job['output'], timings=job['timings'])
        return response
