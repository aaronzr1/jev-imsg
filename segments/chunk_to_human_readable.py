"""Convert a full chunks JSON to readable review records, without re-segmenting."""
import argparse
import json
from pathlib import Path


def to_review(document, limit=None):
    chunks = document['chunks']
    selected = chunks if limit is None else chunks[:limit]
    return {
        'instructions': (
            'Chronological chunks converted from a full segmentation output. '
            'Includes all chunks unless --limit was used. '
            'Compare neighboring chunks; set assessment to good, over_split, or under_split. '
            'Consult the original chunks JSON for message metadata and source citations.'
        ),
        'chunks': [
            dict(source_file=document['source']['file'], chunk_id=chunk['id'],
                 start=chunk['start'], end=chunk['end'],
                 message_count=chunk['message_count'], boundary_before=chunk['boundary_before'],
                 transcript=chunk['transcript'], related_chunk_ids=chunk['related_chunk_ids'],
                 assessment=None, reviewer_notes='')
            for chunk in selected
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='A *.chunks.json file')
    parser.add_argument('-o', '--output', type=Path, help='Default: <input name>.review.json beside input')
    parser.add_argument('--limit', type=int, help='Include only the first N chunks (default: all)')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    output = args.output or args.input.with_name(args.input.stem.removesuffix('.chunks') + '.review.json')
    if output.resolve() == args.input.resolve():
        parser.error('Output must differ from input')
    try:
        document = json.loads(args.input.read_text(encoding='utf-8'))
        review = to_review(document, args.limit)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f'Could not read a chunks document: {exc}')
    # Exclusive creation protects existing reviewer assessments from accidental overwrite.
    try:
        with output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    except OSError as exc:
        parser.error(f'Could not create output: {exc}')
    print(f'Wrote {len(review["chunks"])} review chunks to {output}')


if __name__ == '__main__':
    main()
