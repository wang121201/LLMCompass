"""Independent, read-only verification of paired aggregate Qwen measurements."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

CASES = ((32, 2), (64, 2), (128, 2), (256, 2), (512, 2),
         (128, 4), (128, 8), (128, 16))
METRICS = ('dram__bytes_read.sum', 'dram__bytes_write.sum')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def raw_bytes(path, count):
    lines = Path(path).read_text().splitlines()
    starts = [i for i, value in enumerate(lines) if value.startswith('"ID",')]
    require(len(starts) == 1, 'ambiguous NCU raw table')
    table = csv.DictReader(io.StringIO('\n'.join(lines[starts[0]:])))
    units = next(table)
    require(all(units[m] == 'byte' for m in METRICS), 'wrong physical-byte units')
    rows = [(int(row['ID']), int(row[METRICS[0]].replace(',', '')),
             int(row[METRICS[1]].replace(',', '')))
            for row in table if row.get('ID')]
    require(len(rows) == count and len(set(r[0] for r in rows)) == count, 'range denominator')
    return sorted(rows)


def controls(row, p, d):
    records = row['execution']['phases']
    require(len(records) == 2 * (d + 1), 'complete warmup and measured workflow required')
    for ordinal, record in enumerate(records):
        index = ordinal % (d + 1)
        phase = 'Prefill' if index == 0 else f'Decode{index}'
        stage = 'Warmup' if ordinal < d + 1 else 'Measured'
        require((record['stage'], record['phase']) == (stage, phase), 'phase order')
        value = record['actual_controls']
        length = p + index
        require(value['seq_lens_sum'] == length and value['seq_lens'] == [length], 'KV context')
        expected_input = list(range(1000, 1000 + p)) if index == 0 else [(944, 291)[(index - 1) % 2]]
        require(value['input_ids'] == expected_input, 'fixed input IDs')
        positions = list(range(p)) if index == 0 else [p + index - 1]
        require(value['positions'] == positions, 'position IDs')
        write_slots = list(range(1, p + 1)) if index == 0 else [p + index]
        require(value['kv_write_slots'] == write_slots, 'append-only writes')
        require(value['kv_read_slots_in_logical_order'] == list(range(1, length + 1)), 'KV read prefix')
    require(all(e['final_gpu_table_matches_allocations'] for e in row['execution']['kv_evidence']),
            'actual final GPU KV table check')


def verify(root):
    collection = load(root / 'collection.json')
    gate = load(root / 'gate.json')
    require(sha(root / 'gate.json') == collection['gate_sha256'], 'gate identity')
    require(len(gate['cases']) == 8 and len(gate['groups']) == 12, 'numerical/control coverage')
    require(all(c['passed'] for c in gate['cases']) and
            all(g['check']['passed'] for g in gate['groups']), 'bounded numerical controls')
    for relative, expected in collection['files_sha256'].items():
        require(sha(root / relative) == expected, 'changed receipt ' + relative)
    total_timing, total_ranges, rows = 0, 0, []
    require(len(collection['cases']) == 8, 'eight full cases required')
    by_name = {r['case']: r for r in collection['cases']}
    require(len(by_name) == 8, 'duplicate case')
    for p, d in CASES:
        name = f'p{p:03d}d{d:02d}'
        entry = by_name[name]
        timing = load(root / name / 'timing.json')
        require(len(timing['rows']) == 40, 'twenty timing repeats per path')
        for path in ('native', 'explicit'):
            selected = [r for r in timing['rows'] if r['path'] == path]
            require(len(selected) == 20 and sorted(r['repeat'] for r in selected) == list(range(20)),
                    'timing repeat IDs')
            for row in selected:
                controls(row, p, d)
                require(not row['time_is_profiled'] and
                        abs(row['full_ms'] - sum(row['phase_ms'].values())) < 1e-9,
                        'unprofiled time closure')
            median = statistics.median(r['full_ms'] for r in selected)
            hardware = entry['hardware'][path]
            require(abs(median - hardware['full_ms']) < 1e-9, 'GPU time denominator')
            table = raw_bytes(root / name / f'{path}-counters.csv', 5)
            capture = load(root / name / f'{path}-capture.json')
            require(len(capture['rows']) == 5 and capture['path'] == path, 'counter host count')
            for row in capture['rows']:
                controls(row, p, d)
                require(len(row['profiler_events']) == 2 and row['phase_ms'] is None, 'range/time separation')
            read = statistics.median(x[1] for x in table)
            write = statistics.median(x[2] for x in table)
            require((read, write) == (hardware['read_bytes'], hardware['write_bytes']), 'DRAM reconstruction')
            bandwidth = (read + write) / (median * 1e6)
            require(abs(bandwidth - hardware['effective_GBps']) < 1e-9, 'effective bandwidth arithmetic')
            total_timing += len(selected)
            total_ranges += len(table)
        parts = entry['time_decomposition_ms']
        require(abs(parts['model_minus_native'] - parts['model_minus_explicit'] -
                    parts['explicit_minus_native']) < 1e-8, 'paired time decomposition')
        rows.append(name)
    groups = raw_bytes(root / 'groups-counters.csv', 120)
    group_controls = load(root / 'groups-capture.json')['ranges']
    group_rows = load(root / 'groups-counters.json')['rows']
    require(len(group_controls) == len(group_rows) == len(groups), 'group range association')
    counts = {}
    for index, (raw, control, row) in enumerate(zip(groups, group_controls, group_rows)):
        require(raw[0] == control['range_ordinal'] == row['range_ordinal'] == row['range_id'] == index,
                'exact group range ordinal join')
        require(raw[1:] == (row['read_bytes'], row['write_bytes']), 'group physical-byte reconstruction')
        key = tuple(control[k] for k in ('prefill', 'phase', 'group', 'path'))
        require(all(control[k] == row[k] for k in ('prefill', 'phase', 'group', 'path', 'repeat')),
                'group labels changed')
        counts.setdefault(key, []).append(control['repeat'])
    require(len(counts) == 24 and all(sorted(v) == list(range(5)) for v in counts.values()),
            'five repeats in each of 24 paired group conditions')
    require(len(load(root / 'groups-timing.json')['groups']) == 12, 'P128/P512 group timing coverage')
    forbidden = [str(p.relative_to(root)) for p in root.rglob('*')
                 if p.is_file() and p.suffix in ('.ncu-rep', '.nsys-rep', '.sqlite', '.jsonl')
                 and 'cache' not in p.relative_to(root).parts]
    require(not forbidden, 'unexpected trace/binary report')
    return dict(status='PASS_INDEPENDENT_PAIRING_AND_AGGREGATE_ACCOUNTING_NOT_ACCURACY',
                cases=rows, unprofiled_timing_workflows=total_timing,
                full_counter_ranges=total_ranges, local_group_counter_ranges=len(groups),
                strict_elementwise_logit_pass_cases=sum(c['strict_cross_implementation_logit_allclose_passed']
                                                       for c in gate['cases']),
                strict_cross_implementation_kv_pass_cases=sum(c['strict_cross_implementation_kv_2pct_passed']
                                                           for c in gate['cases']),
                maximum_phase_logit_relative_l2=max(x['logits']['relative_l2']
                    for c in gate['cases'] for x in c['checks']),
                trace_or_binary_reports=forbidden, collection_sha256=sha(root / 'collection.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == '__main__':
    main()
