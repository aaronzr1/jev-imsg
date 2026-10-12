"""Boundary and integration checks with synthetic conversations; no model calls."""
import json
from pathlib import Path
import tempfile
from threading import Event, get_ident
import unittest
from unittest.mock import patch

import app


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.chunks_dir = Path(self.temp.name)
        self.messages = [dict(
            id=f'm{i}', timestamp=f'2024-01-01T12:{i // 60:02}:{i % 60:02}',
            sender='Me' if i % 2 else 'Alex', text=f'Message {i}',
            record_type='message', reactions=[], attachments=[], reply_to=[],
            source_occurrences=[dict(lines=[i + 1, i + 1])]) for i in range(180)]
        self.chunks = [dict(id=f'c{i}', start=self.messages[i]['timestamp'],
                            end=self.messages[i + 29]['timestamp'],
                            participants=['Me', 'Alex'], messages=self.messages[i:i + 30])
                       for i in range(0, 180, 30)]
        self.document = dict(source=dict(file='Alex.txt', sha256='original'), chunks=self.chunks)
        self.path = self.chunks_dir / 'Alex.chunks.json'
        self.path.write_text(json.dumps(self.document))
        self.hit = dict(chunk_id='c60', source_file='Alex.txt', source_sha256='original',
                        chunks_file=str(self.path), chunk=self.chunks[2], match_probability=.9)
        self.patcher = patch.dict(app.DEFAULTS, chunks_dir=self.chunks_dir)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(app.conversation.cache_clear)

    def test_initial_context_includes_whole_match_and_neighbors(self):
        data = app.context(self.hit)
        self.assertEqual((data['start'], data['end'], data['total']), (40, 110, 180))
        self.assertEqual({m['id'] for m in self.chunks[2]['messages']}, set(data['matched_ids']))
        self.assertEqual([m['id'] for m in data['messages']], [f'm{i}' for i in range(40, 110)])

    def test_paging_has_no_gaps_or_duplicates(self):
        initial = app.context(self.hit)
        earlier = app.context(self.hit, 0, initial['start'])
        later = app.context(self.hit, initial['end'], 180)
        combined = earlier['messages'] + initial['messages'] + later['messages']
        self.assertEqual([m['id'] for m in combined], [f'm{i}' for i in range(180)])

    def test_rejects_outside_paths_and_invalid_ranges(self):
        with self.assertRaises(ValueError):
            app.context(dict(self.hit, chunks_file=str(self.chunks_dir.parent / '.env')))
        for start, end in [(-1, 10), (10, 10), (0, 201), (1, None)]:
            with self.assertRaises(ValueError):
                app.context(self.hit, start, end)

    def test_detects_replaced_source(self):
        with self.assertRaisesRegex(ValueError, 'changed'):
            app.context(dict(self.hit, source_sha256='replaced'))

    def test_summary_does_not_include_filesystem_path_or_full_messages(self):
        data = app.summary(self.hit, 0)
        self.assertEqual(data['title'], 'Alex')
        self.assertNotIn('chunks_file', data)
        self.assertNotIn('messages', data)


class SearchTests(unittest.TestCase):
    def test_worker_reuses_session_after_failed_search(self):
        search = app.Search()
        self.addCleanup(search.close)
        worker_threads = []
        sessions = []
        finished = Event()

        def pipeline(args, indexes, log):
            worker_threads.append(get_ident())
            sessions.append(indexes)
            if len(sessions) == 1:
                raise RuntimeError('A query failed')
            return dict(results=[], complete=True, warnings=[])

        original_run = search.run

        def tracked(job):
            original_run(job)
            finished.set()

        with tempfile.TemporaryDirectory(dir=app.ROOT / '.cache') as folder, \
                patch('app.ROOT', Path(folder)), patch('app.run_pipeline', side_effect=pipeline), \
                patch.object(search, 'run', side_effect=tracked):
            first = search.start('first query')
            self.assertTrue(finished.wait(5))
            self.assertEqual(search.status(first)['status'], 'error')
            finished.clear()
            second = search.start('second query')
            self.assertTrue(finished.wait(5))
            self.assertEqual(search.status(second)['status'], 'done')
        self.assertEqual(worker_threads[0], worker_threads[1])
        self.assertIs(sessions[0], sessions[1])

    def test_query_validation_and_only_one_running_search(self):
        search = app.Search()
        self.addCleanup(search.close)
        for query in [None, '', '  ', [], 'a' * 2001]:
            with self.assertRaises(ValueError):
                search.start(query)
        with patch.object(search.worker, 'submit'):
            job_id = search.start('  a moment  ')
            self.assertIsNotNone(job_id)
            self.assertEqual(search.get(job_id)['query'], 'a moment')
            self.assertIsNone(search.start('another search'))
        with self.assertRaises(KeyError):
            search.get('unknown')

    def test_retrieval_settings_are_validated_and_retained(self):
        search = app.Search()
        self.addCleanup(search.close)
        for mode, weight in [('invalid', .5), ('both', -1), ('both', float('nan')), ('both', True)]:
            with self.assertRaises(ValueError):
                search.start('query', mode, weight)
        with patch.object(search.worker, 'submit'):
            job = search.start('query', 'both', .7)
        status = search.status(job)
        self.assertEqual((status['retrieval_text'], status['summary_weight']), ('both', .7))

    def test_runs_main_and_keeps_partial_results(self):
        with tempfile.TemporaryDirectory() as folder:
            search = app.Search()
            self.addCleanup(search.close)
            job = dict(id='test', query='a moment', status='running', directory=Path(folder) / 'job')
            search.job = job

            def pipeline(args, indexes, log):
                self.assertEqual(args.query, 'a moment')
                self.assertEqual(args.stage, 'all')
                self.assertIs(indexes, search.indexes)
                self.assertTrue(args.out_dir.is_dir())
                log('rerank: test')
                return dict(results=[], complete=False, warnings=['One candidate failed.'])

            with patch('app.run_pipeline', side_effect=pipeline):
                search.run(job)
            output = search.status('test')
            self.assertEqual(output['status'], 'done')
            self.assertFalse(output['complete'])
            self.assertEqual(output['warnings'], ['One candidate failed.'])

    def test_failure_does_not_disclose_exception_details(self):
        with tempfile.TemporaryDirectory(dir=app.ROOT / '.cache') as folder:
            search = app.Search()
            self.addCleanup(search.close)
            job = dict(id='test', query='a moment', status='running', directory=Path(folder) / 'job')
            search.job = job
            with patch('app.run_pipeline', side_effect=OSError('SECRET provider response')):
                search.run(job)
            output = search.status('test')
            self.assertEqual(output['status'], 'error')
            self.assertNotIn('SECRET', output['error'])

    def test_planner_failure_reports_the_stage_without_exception_details(self):
        with tempfile.TemporaryDirectory(dir=app.ROOT / '.cache') as folder:
            search = app.Search()
            self.addCleanup(search.close)
            job = dict(id='test', query='a moment', status='running', directory=Path(folder) / 'job')
            search.job = job
            with patch('app.run_pipeline', side_effect=app.QueryPlanError('SECRET model output')):
                search.run(job)
            output = search.status('test')
            self.assertEqual(output['status'], 'error')
            self.assertIn('query planner', output['error'])
            self.assertNotIn('JEV_KEY', output['error'])
            self.assertNotIn('SECRET', output['error'])
            self.assertIn('pipeline.log', output['error'])
            self.assertIn('SECRET model output', (job['directory'] / 'pipeline.log').read_text())


if __name__ == '__main__':
    unittest.main()
