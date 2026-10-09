"""Publish bounded historical semantic totals without phase or operator streams."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(path):
    path = Path(path)
    source = json.loads(path.read_text(encoding='utf-8'))
    assert source['status'] == 'PASS_TYPED_ACCOUNTING_AND_ARCHIVE_RECONCILIATION'
    rows = []
    for case in source['target_case_ids']:
        row = source['cases'][case]
        values = row['semantic_bytes']
        assert set(values) == {'read', 'write'}
        assert all(set(v) == {'Weights', 'KV cache', 'Activation', 'Other'} for v in values.values())
        assert all(type(b) is int and b >= 0 for v in values.values() for b in v.values())
        assert values['write']['Weights'] == 0
        rows.append(dict(case=case, prefill_tokens=row['prefill_tokens'], decode_steps=row['decode_steps'],
                         evidence=row['evidence'], semantic_bytes=values,
                         known_read_bytes=sum(values['read'].values()),
                         known_write_bytes=sum(values['write'].values()),
                         operator_count=row['operator_count']))
    assert sum(r['evidence'] == 'ARCHIVE_TOTALS_RECONCILED' for r in rows) == 5
    return dict(schema='LLAMA31_8B_LEGACY_SEMANTIC_KEY_TOTALS_V1',
                status=source['status'], claim=source['claim'], model=source['model'],
                source_aggregate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_aggregate_filename=path.name, definitions=source['definitions'],
                source_identity=source['source_identity'], checks=source['checks'],
                historical_case_ids=source['historical_case_ids'], cases=rows,
                hardware_comparison=None, native_ada_profile=False,
                timing_excluded='Legacy GA100/HBM and retired serial timing are not current Ada timing evidence.',
                raw_operator_or_phase_stream_published=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.source), indent=2) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
