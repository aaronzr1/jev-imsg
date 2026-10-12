"""Compatibility imports for the intent-search pipeline."""
from .intent_search.pipeline import SearchIndexes, load, run, save

__all__ = ['SearchIndexes', 'load', 'run', 'save']
