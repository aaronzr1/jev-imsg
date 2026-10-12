"""Planner recovery and lexical grounding, without model or network calls."""
import json
import unittest
from unittest.mock import Mock, patch

from agent_query_operator.fusion import fuse
from agent_query_operator.planner import (
    OllamaPlanner, QueryPlanError, grounded_keywords, validate_plan,
)
from agent_query_operator.search import retrieve


def generated(**changes):
    return dict(dict(rewrites=['I want to build a company', 'Exploring business ideas'],
                     keywords=['startups'], participants=[], requirements=[],
                     answer_mode='relevance'), **changes)


class PlannerTests(unittest.TestCase):
    def test_invented_keywords_are_removed_without_another_model_call(self):
        planner = OllamaPlanner()
        with patch.object(planner, '_chat', return_value=json.dumps(
                generated(keywords=['startup ambitions', 'pivot', 'startups']))) as chat:
            plan = planner.plan('find when I wanted to pivot into startups')
        self.assertEqual(plan['keywords'], ['pivot', 'startups'])
        self.assertEqual(chat.call_count, 1)
        validate_plan(plan)

    def test_only_invented_keywords_use_distinctive_query_words(self):
        planner = OllamaPlanner()
        with patch.object(planner, '_chat', return_value=json.dumps(
                generated(keywords=['entrepreneurship', 'business']))) as chat:
            plan = planner.plan('find me the first time I mentioned wanting to pivot into startups')
        self.assertEqual(plan['keywords'], ['pivot', 'startups'])
        self.assertEqual(chat.call_count, 1)

    def test_case_unicode_phrases_and_keyword_limit(self):
        self.assertEqual(grounded_keywords(['CAFÉ plans', 'housing'], 'find café plans'),
                         ['café', 'plans'])
        self.assertEqual(len(grounded_keywords([], ' '.join(f'word{i}' for i in range(20)))), 12)

    def test_saved_plans_still_reject_invented_keywords(self):
        with self.assertRaisesRegex(ValueError, 'without synonyms'):
            validate_plan(dict(query='startups', **generated(keywords=['business'])))

    def test_queries_without_distinctive_words_use_semantic_search_only(self):
        planner = OllamaPlanner()
        with patch.object(planner, '_chat', return_value=json.dumps(generated(keywords=['events']))):
            plan = planner.plan('find me the time')
        self.assertEqual(plan['keywords'], [])
        dense, sparse = Mock(records=[]), Mock(records=[])
        dense.search.return_value = []
        bundle = retrieve(plan, dense, sparse)
        self.assertEqual(len(bundle['runs']), 3)
        sparse.search.assert_not_called()
        self.assertEqual(fuse(bundle['runs']), [])

    def test_invalid_participants_still_require_repair_and_fail(self):
        planner = OllamaPlanner()
        with patch.object(planner, '_chat', return_value=json.dumps(
                generated(participants=['Morgan']))) as chat:
            with self.assertRaisesRegex(QueryPlanError, 'participants must be named'):
                planner.plan('find startups')
        self.assertEqual(chat.call_count, 2)

    def test_repair_can_fix_other_invalid_fields(self):
        planner = OllamaPlanner()
        with patch.object(planner, '_chat', side_effect=[
                json.dumps(generated(participants=['Morgan'])), json.dumps(generated())]) as chat:
            plan = planner.plan('find startups')
        self.assertEqual(plan['participants'], [])
        self.assertEqual(chat.call_count, 2)


if __name__ == '__main__':
    unittest.main()
