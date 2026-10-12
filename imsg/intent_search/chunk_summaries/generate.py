"""Resumable generation: each successful chunk is saved immediately."""


def generate(records, store, summarizer, limit=None, force=False, log=print):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('Summary limit must be positive.')
    selected = records if limit is None else records[:limit]
    report = dict(schema_version=1, total_chunks=len(records), requested=len(selected),
                  generated=0, cached=0, complete=True, errors=[],
                  summaries_dir=str(store.directory), model=store.model)
    for number, record in enumerate(selected, 1):
        try:
            cached = None if force else store.read(record)
        except (ValueError, OSError):
            cached = None  # A generation run repairs corrupt sidecars atomically.
        if cached is not None:
            report['cached'] += 1
            continue
        try:
            text = summarizer.summarize(record)
            store.write(record, text, summarizer.last_model)
            report['generated'] += 1
            log(f'summary {number}/{len(selected)}: {record["chunk"]["id"]}')
        except (ValueError, RuntimeError, OSError, TypeError, KeyError):
            report['complete'] = False
            report['errors'].append(dict(chunk_id=record['chunk']['id'],
                error='Summary generation failed. Check Ollama/model settings and retry; completed chunks are cached.'))
            break  # Avoid thousands of failing calls when the provider is unavailable.
    return report
