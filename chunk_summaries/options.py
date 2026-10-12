"""Shared CLI switches for the optional summary experiment."""
import math
from pathlib import Path


def add_options(parser, *, modes=True, generation=True):
    if modes:
        parser.add_argument('--retrieval-text', choices=['messages', 'summaries', 'both'])
    parser.add_argument('--summary-weight', type=float, help='Summary share of weighted RRF in both mode (0–1)')
    parser.add_argument('--summaries-dir', type=Path)
    parser.add_argument('--summary-model', help='Ollama model for generating/identifying cached summaries')
    parser.add_argument('--summary-url')
    parser.add_argument('--summary-timeout', type=float)
    if generation:
        parser.add_argument('--summary-limit', type=positive, help='Generate only the first N chunks for a smoke test')
        parser.add_argument('--force-summaries', action='store_true', help='Regenerate existing summary artifacts')


def positive(value):
    number = int(value)
    if number < 1:
        raise ValueError('must be positive')
    return number


def validate_options(mode, weight):
    if mode not in ('messages', 'summaries', 'both'):
        raise ValueError('Choose messages, summaries, or both for retrieval text.')
    if (isinstance(weight, bool) or not isinstance(weight, (int, float)) or
            not math.isfinite(weight) or not 0 <= weight <= 1):
        raise ValueError('Summary weight must be a number between 0 and 1.')
