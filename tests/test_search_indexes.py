"""Exercise real retrieval indices with a tiny corpus and a local fake embedder."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from chunk_search.chunk_search import load_chunks
from main import DEFAULTS, SearchIndexes, run


class Embedder:
    def embed(self, texts, batch_size=1):
        for text in texts:
            yield np.array([text.lower().count('alpha'), text.lower().count('beta'), 1.0])


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.chunks_dir = self.root / 'chunks'
        self.chunks_dir.mkdir()
        self.args = argparse.Namespace(**dict(DEFAULTS, chunks_dir=self.chunks_dir,
                                              cache_dir=self.root / 'cache'), no_cache=False)
        self.indexes = SearchIndexes()
        self.write('one', 'alpha alpha')
        self.model_patch = patch('fastembed.TextEmbedding', return_value=Embedder())
        self.model_factory = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def write(self, name, text):
        message = dict(id=name + ':m1', timestamp='2024-01-01T12:00:00', sender='Me',
                       text=text, record_type='message', attachments=[],
                       source_occurrences=[dict(lines=[1, 3])])
        chunk = dict(id=name + ':c1', start=message['timestamp'], end=message['timestamp'],
                     participants=['Me'], messages=[message])
        path = self.chunks_dir / (name + '.chunks.json')
        path.write_text(json.dumps(dict(schema_version=1,
                        source=dict(file=name + '.txt', sha256=text), chunks=[chunk])))
        return path

    def test_reuses_indices_and_model_and_loads_corpus_only_once(self):
        with patch('chunk_search.chunk_search.load_chunks', wraps=load_chunks) as loader:
            dense, sparse = self.indexes.get(self.args)
            first = dense.search('alpha')
            dense_again, sparse_again = self.indexes.get(self.args)
            self.assertIs(dense_again, dense)
            self.assertIs(sparse_again, sparse)
            self.assertIs(dense.records, sparse.records)
            self.assertEqual(dense_again.search('alpha'), first)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(self.model_factory.call_count, 1)

    def test_refreshes_after_addition_modification_and_removal(self):
        self.indexes.get(self.args)
        added = self.write('two', 'beta beta')
        dense, sparse = self.indexes.get(self.args)
        self.assertEqual(len(dense.records), 2)
        self.assertEqual(sparse.search('beta')[0]['source_file'], 'two.txt')
        self.write('one', 'beta changed content')
        updated, sparse = self.indexes.get(self.args)
        self.assertIsNot(updated, dense)
        self.assertEqual(sparse.search('alpha'), [])
        added.unlink()
        dense, sparse = self.indexes.get(self.args)
        self.assertEqual([r['source']['file'] for r in dense.records], ['one.txt'])
        self.assertEqual(len(sparse.records), 1)

    def test_no_cache_and_explicit_clear_rebuild(self):
        first, _ = self.indexes.get(self.args)
        self.args.no_cache = True
        second, _ = self.indexes.get(self.args)
        self.assertIsNot(first, second)
        self.args.no_cache = False
        self.indexes.clear()
        third, _ = self.indexes.get(self.args)
        self.assertIsNot(second, third)

    def test_bad_corpus_can_be_fixed_and_retried(self):
        self.indexes.get(self.args)
        (self.chunks_dir / 'one.chunks.json').write_text('invalid JSON')
        with self.assertRaises(ValueError):
            self.indexes.get(self.args)
        self.write('one', 'beta repaired')
        _, sparse = self.indexes.get(self.args)
        self.assertEqual(sparse.search('alpha'), [])
        self.assertEqual(len(sparse.search('beta')), 1)

    def test_pipeline_keeps_saved_stages_with_warm_indices(self):
        plan_path = self.root / 'plan.json'
        plan_path.write_text(json.dumps(dict(query='alpha', rewrites=['alpha project', 'alpha work'],
                          keywords=['alpha'], participants=[], requirements=[], answer_mode='relevance')))
        self.args.__dict__.update(stage='retrieve', plan=plan_path, aliases=None, resegment=False)
        for number in (1, 2):
            self.args.out_dir = self.root / str(number)
            messages = []
            output = run(self.args, indexes=self.indexes, log=messages.append)
            self.assertEqual(output['runs'][0]['results'][0]['source_file'], 'one.txt')
            self.assertTrue((self.args.out_dir / 'plan.json').exists())
            self.assertTrue((self.args.out_dir / 'retrieve.json').exists())
            self.assertTrue((self.args.out_dir / 'timings.json').exists())
            self.assertEqual(len(messages), 2)
        self.assertEqual(self.model_factory.call_count, 1)


if __name__ == '__main__':
    unittest.main()
