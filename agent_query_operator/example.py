"""Plan and retrieve: python -m agent_query_operator.example [--plan saved.json]."""
import sys

from main import main

if __name__ == '__main__':
    raise SystemExit(main(['--stage', 'retrieve', *sys.argv[1:]]))
