"""Rerank saved candidates: python -m jev_reranker.example --input fuse.json."""
import sys

from main import main

if __name__ == '__main__':
    raise SystemExit(main(['--stage', 'rerank', *sys.argv[1:]]))
