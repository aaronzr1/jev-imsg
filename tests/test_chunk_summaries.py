"""Summary experiments use real indices and fusion, with fake local models."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from agent_query_operator.fusion import fuse
from agent_query_operator.search import retrieve
from chunk_summaries.compare import compare, metrics
from chunk_summaries.generate import generate
from chunk_summaries.retrieval import retrieve_with_summaries
from chunk_summaries.store import SummaryStore, evidence
from chunk_summaries.summarizer import INPUT_BYTES, OllamaSummarizer
from jev_reranker.client import JevJudge
from main import DEFAULTS, SearchIndexes, run


class Embedder:
    def embed(self, texts, batch_size=1):
        for text in texts:
            yield np.array([text.lower().count('alpha'), text.lower().count('beta'), 1.0])


class SummariesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.chunks_dir = self.root / 'chunks'
        self.chunks_dir.mkdir()
        self.args = argparse.Namespace(**dict(DEFAULTS, chunks_dir=self.chunks_dir,
            cache_dir=self.root / 'cache', summaries_dir=self.root / 'summaries',
            per_query=2, top_k=2), no_cache=False, aliases=None, rerank=False)
        chunks = []
        for name, text in [('c1', 'alpha alpha'), ('c2', 'beta beta')]:
            message = dict(id=name + ':m1', timestamp='2024-01-01T12:00:00', sender='Me',
                           text=text, record_type='message', attachments=[],
                           source_occurrences=[dict(lines=[1, 3])])
            chunks.append(dict(id=name, start=message['timestamp'], end=message['timestamp'],
                               participants=['Me'], messages=[message]))
        (self.chunks_dir / 'thread.chunks.json').write_text(json.dumps(dict(schema_version=1,
            source=dict(file='thread.txt', sha256='export'), chunks=chunks)))
        self.indexes = SearchIndexes()
        self.records = self.indexes.snapshot(self.args)
        self.store = SummaryStore(self.args.summaries_dir, self.args.summary_model)
        self.plan = dict(query='alpha', rewrites=['alpha project', 'alpha goals'], keywords=['alpha'],
                         participants=[], requirements=[], answer_mode='relevance')
        self.model_patch = patch('fastembed.TextEmbedding', return_value=Embedder())
        self.model_factory = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def summaries(self):
        # Deliberately different texts prove retrieval uses the selected representation.
        for record, text in zip(self.records, ['beta beta topic', 'alpha alpha topic']):
            self.store.write(record, text, 'fake-summary-model')

    def results(self, mode, weight=.5):
        self.args.retrieval_text, self.args.summary_weight = mode, weight
        if mode == 'messages':
            dense, sparse = self.indexes.get(self.args)
            bundle = retrieve(self.plan, dense, sparse, 2)
        else:
            bundle = retrieve_with_summaries(self.plan, self.args, self.indexes)
        return fuse(bundle['runs'], 2), bundle

    def test_default_needs_no_summaries_and_selected_modes_change_ranking(self):
        hits, _ = self.results('messages')
        self.assertEqual(hits[0]['chunk_id'], 'c1')
        self.assertFalse(self.args.summaries_dir.exists())
        self.summaries()
        hits, _ = self.results('summaries')
        self.assertEqual(hits[0]['chunk_id'], 'c2')
        self.assertEqual(hits[0]['summary'], 'alpha alpha topic')
        self.assertEqual(hits[0]['chunk']['messages'][0]['text'], 'beta beta')
        self.assertEqual(self.results('both', .8)[0][0]['chunk_id'], 'c2')
        self.assertEqual(self.results('both', .2)[0][0]['chunk_id'], 'c1')
        self.assertEqual(self.model_factory.call_count, 1)  # all views share the embedder

    def test_weight_endpoints_match_pure_modes_without_requiring_unused_source(self):
        baseline = self.results('messages')[0]
        endpoint = self.results('both', 0)[0]
        self.assertEqual([(h['chunk_id'], h['rrf_score']) for h in endpoint],
                         [(h['chunk_id'], h['rrf_score']) for h in baseline])
        self.summaries()
        pure = self.results('summaries')[0]
        endpoint = self.results('both', 1)[0]
        self.assertEqual([(h['chunk_id'], h['rrf_score']) for h in endpoint],
                         [(h['chunk_id'], h['rrf_score']) for h in pure])

    def test_missing_or_stale_summaries_fail_instead_of_falling_back(self):
        with self.assertRaisesRegex(RuntimeError, '2 chunks lack'):
            self.results('summaries')
        self.summaries()
        changed = dict(self.records[0], chunk=dict(self.records[0]['chunk']))
        changed['chunk']['messages'] = [dict(changed['chunk']['messages'][0], text='Changed evidence')]
        self.assertIsNone(self.store.read(changed))
        with self.assertRaises(RuntimeError):
            self.store.text_factory([changed])

    def test_summary_changes_refresh_view_and_messages_stay_warm(self):
        self.summaries()
        self.results('summaries')
        old = self.indexes.views['summaries']['pair'][0]
        self.results('summaries')
        self.assertIs(self.indexes.views['summaries']['pair'][0], old)
        self.store.write(self.records[0], 'alpha alpha alpha corrected', 'fake')
        self.results('summaries')
        self.assertIsNot(self.indexes.views['summaries']['pair'][0], old)
        self.assertEqual(self.model_factory.call_count, 1)

    def test_generation_resumes_after_failure_and_repairs_corrupt_cache(self):
        class Fake:
            last_model = 'fake'
            calls = 0

            def summarize(self, record):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError('Provider failed')
                return 'Me discusses a project.'

        fake = Fake()
        first = generate(self.records, self.store, fake, log=lambda text: None)
        self.assertFalse(first['complete'])
        self.assertEqual(first['generated'], 1)
        second = generate(self.records, self.store, fake, log=lambda text: None)
        self.assertTrue(second['complete'])
        self.assertEqual((second['cached'], second['generated']), (1, 1))
        self.store.path(self.records[0]).write_text('broken JSON')
        fixed = generate(self.records, self.store, fake, log=lambda text: None)
        self.assertEqual((fixed['cached'], fixed['generated']), (1, 1))

    def test_artifacts_validate_provenance_and_model_version(self):
        self.summaries()
        artifact = json.loads(self.store.path(self.records[0]).read_text())
        artifact['source_sha256'] = 'wrong'
        self.store.path(self.records[0]).write_text(json.dumps(artifact))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.store.read(self.records[0])
        other_model = SummaryStore(self.store.directory, 'another-model')
        self.assertIsNone(other_model.read(self.records[1]))

    def test_verifier_receives_original_messages_without_summary_claims(self):
        self.summaries()
        hit = self.results('summaries')[0][0]
        candidate = JevJudge._question(hit)['instructions']['candidate']
        self.assertNotIn('summary', candidate)
        self.assertEqual(candidate['messages'][0]['text'], 'beta beta')

    def test_weighted_fusion_keeps_distinct_views_but_deduplicates_same_view(self):
        self.summaries()
        hits, bundle = self.results('both', .5)
        run = bundle['runs'][0]
        self.assertEqual(fuse(bundle['runs'] + [run], 2), hits)
        sources = {c['text_source'] for c in hits[0]['contributions']}
        self.assertEqual(sources, {'messages', 'summaries'})
        for bad in [-1, float('nan'), float('inf'), True]:
            with self.assertRaises(ValueError):
                fuse([dict(run, weight=bad)], 2)

    def test_summary_inputs_strip_paths_and_long_messages_are_not_truncated(self):
        record = dict(self.records[0], chunk=dict(self.records[0]['chunk']))
        attachment = '/Users/person/Library/Messages/Attachments/photo.jpg'
        text = 'prefix ' + '字' * 5000 + '\nTAIL_EVIDENCE\n' + attachment
        record['chunk']['messages'] = [dict(record['chunk']['messages'][0], text=text, attachments=[attachment])]
        self.assertNotIn(attachment, str(evidence(record)))
        payloads = []

        def chat(data):
            payloads.append(data)
            return 'Me discusses a topic.'

        summarizer = OllamaSummarizer()
        with patch.object(summarizer, '_chat', side_effect=chat):
            summarizer.summarize(record)
        original = [d for d in payloads if 'messages' in d]
        combined = ''.join(m['text'] for d in original for m in d['messages'])
        self.assertEqual(combined, text.replace('\n' + attachment, ''))
        self.assertIn('TAIL_EVIDENCE', combined)
        for data in payloads:
            self.assertLessEqual(len(json.dumps(list(data.values())[0], ensure_ascii=False).encode()), INPUT_BYTES)

    def test_comparison_shares_one_plan_and_scores_expected_chunk_ids(self):
        self.summaries()
        self.args.out_dir = self.root / 'comparison'
        with patch('chunk_summaries.compare.OllamaPlanner') as factory:
            factory.return_value.plan.return_value = self.plan
            report = compare([dict(query='alpha', relevant_chunk_ids=['c1'])], self.args,
                             log=lambda message: None)
        factory.return_value.plan.assert_called_once_with('alpha')
        case = report['cases'][0]
        self.assertEqual(set(case['modes']), {'messages', 'summaries', 'both'})
        self.assertEqual(case['modes']['messages']['metrics']['reciprocal_rank'], 1)
        self.assertEqual(case['modes']['summaries']['metrics']['reciprocal_rank'], .5)
        self.assertTrue((self.args.out_dir / 'comparison.json').exists())
        self.assertTrue((self.args.out_dir / 'case-001' / 'summaries' / 'fuse.json').exists())
        self.assertIsNone(metrics(['c1'], [], 2))
        self.assertEqual(metrics(['c1'], ['c1'], 2)['precision_at_k'], .5)

    def test_summary_full_pipeline_saves_settings_and_verifies_original_evidence(self):
        self.summaries()
        path = self.root / 'plan.json'
        path.write_text(json.dumps(self.plan))
        self.args.__dict__.update(stage='all', retrieval_text='summaries', plan=path,
                                  out_dir=self.root / 'full', resegment=False)
        with patch('jev_reranker.load_key', return_value='fake'), patch('jev_reranker.JevJudge') as judge:
            judge.return_value.settings = {}
            judge.return_value.evaluate.return_value = dict(probability=.9, model='fake')
            output = run(self.args, indexes=self.indexes, log=lambda text: None)
        self.assertTrue(output['complete'])
        self.assertEqual(len(output['results']), 2)
        saved = json.loads((self.args.out_dir / 'fuse.json').read_text())
        self.assertEqual(saved['settings']['retrieval_text'], 'summaries')
        self.assertEqual(output['results'][0]['chunk']['messages'][0]['text'], 'beta beta')


if __name__ == '__main__':
    unittest.main()
