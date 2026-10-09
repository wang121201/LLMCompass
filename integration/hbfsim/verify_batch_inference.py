"""Independent aggregate verification; no model compilation or hardware claims."""
import argparse
import hashlib
import json
import math
from pathlib import Path

BATCHES = (1, 2, 4, 8, 16, 32)
PHASES = ('prefill', *(f'decode_{i}' for i in range(1, 9)))
CATEGORIES = ('weights', 'kv_cache', 'activation', 'other')
DIRECTIONS = {'read_bytes': 'known_read_bytes', 'write_bytes': 'known_write_bytes',
              'unknown_direction_bytes': 'unclassified_io_bytes'}
FIELDS = ('operators', 'model_ns', *DIRECTIONS.values())
PROFILE_SHA = 'd7c2659d32b14c4cf0f5da876be09375a810bcdbfdc7b03c4b820cf2fc4288b1'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, message):
    check(isinstance(a, (int, float)) and math.isfinite(a)
          and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), message)


def verify_row(row):
    for name in FIELDS:
        check(type(row[name]) is int and row[name] >= 0, f'Invalid integer {name}')
    check(row['model_ns'] > 0, 'Nonpositive model time')
    check(set(row['semantic']) == set(CATEGORIES), 'Semantic category scope')
    for category in CATEGORIES:
        check(set(row['semantic'][category]) == set(DIRECTIONS), 'Semantic direction scope')
        for amount in row['semantic'][category].values():
            check(type(amount) is int and amount >= 0, 'Invalid semantic amount')
    for direction, field in DIRECTIONS.items():
        check(sum(row['semantic'][c][direction] for c in CATEGORIES) == row[field],
              f'Semantic {direction} does not close')
    check(row['semantic']['weights']['write_bytes'] == 0, 'Weights must be read-only')


def verify_cases(rows, baseline):
    check([r['batch_size'] for r in rows] == list(BATCHES), 'Incomplete or duplicate batch matrix')
    for row in rows:
        batch = row['batch_size']
        check(row['case'] == f'b{batch:02d}p128d08', 'Case identity')
        check(row['prefill_tokens'] == 128 and row['decode_steps'] == 8, 'Window mismatch')
        check(row['status'] == 'PASS_FULL_ANALYTICAL_GRAPH_NOT_HARDWARE_ACCEPTANCE', 'Case not complete')
        check(row['hardware_comparison'] is None and row['hbfsim_in_model_time'] is False,
              'Unrequested hardware/HBFSim timing claim')
        verify_row(row)
        check(row['operators'] == 4563 and set(row['phase_totals']) == set(PHASES), 'Full graph coverage')
        for group_name in ('phase_totals', 'family_totals'):
            group = row[group_name]
            check(bool(group), 'Empty aggregate group')
            for child in group.values():
                verify_row(child)
                if group_name == 'phase_totals':
                    check(child['operators'] == 507, 'Phase graph coverage')
            for field in FIELDS:
                check(sum(child[field] for child in group.values()) == row[field],
                      f'{group_name} {field} closure')
            for category in CATEGORIES:
                for direction in DIRECTIONS:
                    check(sum(child['semantic'][category][direction] for child in group.values())
                          == row['semantic'][category][direction], 'Grouped semantic closure')
        region = row['request_regions']
        expected = 2 * 28 * batch * 136 * 256 * 2
        check(region['independent_kv'] is True and region['last_token_rows_checked'] is True,
              'KV/last-token verification missing')
        check(region['request_count'] == batch and region['kv_regions'] == 56 * batch,
              'Request-region count')
        check(region['append_bytes'] == region['kv_payload_bytes'] == expected, 'KV payload mismatch')
        check(row['semantic']['kv_cache']['write_bytes'] == expected, 'KV semantic writes')
        # Official head expansion may repeat KV reads; do not equate those with unique payload.
        boundary = sum(row[name] for name in DIRECTIONS.values())
        check(row['total_boundary_bytes'] == boundary, 'Boundary total')
        close(row['model_ms'], row['model_ns'] / 1e6, 'Time units')
        close(row['model_GBps'], boundary / row['model_ns'], 'Bandwidth denominator')
        close(row['full_request_latency_ms'], row['model_ms'], 'Request latency was divided by batch')
        close(row['amortized_ms_per_request'], row['model_ms'] / batch, 'Amortized cost')
        close(row['requests_per_second'], batch * 1e9 / row['model_ns'], 'Request throughput')
        prefill = row['phase_totals']['prefill']['model_ns']
        decode = row['model_ns'] - prefill
        close(row['prefill_tokens_per_second'], batch * 128 * 1e9 / prefill, 'Prefill throughput')
        close(row['decode_tokens_per_second'], batch * 8 * 1e9 / decode, 'Decode throughput')
    check(baseline['case'] == 'p128d08', 'Frozen baseline identity')
    first = rows[0]
    check(first['operators'] == baseline['operator_count'], 'B1 operator regression')
    for field in (*DIRECTIONS.values(), 'total_boundary_bytes', 'model_ms', 'model_GBps'):
        check(first[field] == baseline[field], f'B1 regression: {field}')
    for phase in PHASES:
        for field in FIELDS:
            check(first['phase_totals'][phase][field] == baseline['phase_totals'][phase][field],
                  f'B1 phase regression: {phase} {field}')


def verify_shapes(shapes):
    keys = [tuple(row['key']) for row in shapes]
    check(len(keys) == len(set(keys)) and bool(keys), 'Shape uniqueness')
    # Required linear/head/attention shapes establish that larger batches are not B1 multiplication.
    for batch in BATCHES:
        for key in [('matmul', batch * 128, 1536, 8960), ('matmul', batch, 1536, 151936),
                    ('batch', batch * 12, 128, 128, 128), ('softmax', batch * 12, 128, 128)]:
            check(key in keys, f'Missing shape {key}')
        for context in range(129, 137):
            for key in [('batch', batch * 12, 1, 128, context),
                        ('batch', batch * 12, 1, context, 128), ('softmax', batch * 12, 1, context)]:
                check(key in keys, f'Missing decode shape {key}')
    for row in shapes:
        check(row['official_latency_parity'] is True, 'Official parity missing')
        close(row['seconds'], row['official_source_seconds'], 'Official shape-time mismatch')
        check(row['seconds'] > 0, 'Nonpositive shape time')
        acc, operands = row['accounting'], row['operands']
        for direction, field in [('read', 'main_read_bytes'), ('write', 'main_write_bytes')]:
            check(sum(operands[direction].values()) == acc[field], 'Operand-transfer closure')
        check(row['selected_candidate'] in (0, 1) if row['key'][0] == 'batch'
              else row['selected_candidate'] is None, 'Official candidate scope')


def verify(root, baseline_path, source_root=None):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = json.loads((root / 'comparison.json').read_text())
    shapes = json.loads((root / 'shapes.json').read_text())
    check(manifest['status'] == 'PASS_STATIC_BATCH_FULL_GRAPH_ACCOUNTING', 'Run is not terminal PASS')
    check(manifest['schema'] == 'LLMCOMPASS_ADA_STATIC_BATCH_V1', 'Manifest schema')
    check(manifest['model'] == 'Qwen/Qwen2.5-1.5B-Instruct', 'Model identity')
    check(manifest['target_batches'] == list(BATCHES[1:]) and manifest['baseline_batch'] == 1,
          'Manifest batch scope')
    check(manifest['prefill_tokens'] == 128 and manifest['decode_steps'] == 8, 'Manifest window')
    check(manifest['profile_sha256'] == PROFILE_SHA, 'Frozen Ada profile')
    check(manifest['dependency_versions']['scalesim'] == '2.0.2', 'Official ScaleSim version')
    for name in ('no_new_cache_or_scheduler', 'no_latency_seed'):
        check(manifest[name] is True, f'Model scope changed: {name}')
    for name in ('hbfsim_in_model_time', 'hardware_collected', 'operator_stream_saved', 'trace_saved'):
        check(manifest[name] is False, f'Unsupported claim: {name}')
    check(manifest['completed_cases'] == 6 and manifest['verified_operators'] == 27378
          and manifest['verified_phases'] == 54, 'Terminal count mismatch')
    check(manifest['unique_shapes'] == len(shapes), 'Terminal shape count')
    for name in ('comparison', 'shapes'):
        check(sha(root / f'{name}.json') == manifest[f'{name}_sha256'], f'Hash mismatch: {name}')
    baseline_all = json.loads(Path(baseline_path).read_text())
    baseline = next(r for r in baseline_all if r['case'] == 'p128d08')
    verify_cases(rows, baseline)
    verify_shapes(shapes)
    if source_root is not None:
        for name, expected in manifest['source_sha256'].items():
            check(sha(Path(source_root) / name) == expected, f'Source mismatch: {name}')
        check(sha(Path(source_root) / 'RTX4000Ada_xmu_profile_v3.json') == PROFILE_SHA, 'Profile file mismatch')
    # Only bounded aggregates and optional validation receipts belong in a run root.
    allowed = {'manifest.json', 'comparison.json', 'shapes.json', 'validation.json', 'plan-parity.json'}
    check(all(p.is_file() and p.name in allowed for p in root.iterdir()), 'Unexpected run artifact')
    return dict(status='PASS_STATIC_BATCH_INDEPENDENT_AGGREGATE_VERIFICATION',
                cases=6, operators=27378, phases=54, unique_shapes=len(shapes),
                b1_frozen_time_bytes_and_phase_parity=True,
                semantic_residual_bytes=0, no_raw_trace_artifacts=True,
                model_time_includes_hbfsim=False, hardware_accuracy='NOT_EVALUATED',
                manifest_sha256=sha(root / 'manifest.json'), baseline_sha256=sha(baseline_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--baseline', type=Path, default=Path(__file__).parent / 'checkpoint/official-comparison.json')
    parser.add_argument('--source-root', type=Path, default=Path(__file__).parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.baseline, args.source_root)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
