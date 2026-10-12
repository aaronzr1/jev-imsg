"""Compatibility entry point for the intent-search command line."""
from .intent_search.cli import main


if __name__ == '__main__':
    raise SystemExit(main())
