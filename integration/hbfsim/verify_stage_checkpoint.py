"""Verify the portable aggregate checkpoint; do not imply hardware accuracy.

This standard-library check validates file identities, eight-case coverage,
semantic byte/time closure, unchanged official sources and disclosed failures.
It does not rerun GPU measurements or recompile all official mapper shapes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

CASES = ('p032d02', 'p064d02', 'p128d02', 'p256d02', 'p512d02',
         'p128d04', 'p128d08', 'p128d16')
CATEGORIES = ('weights', 'kv_cache', 'activation', 'other')
DIRECTIONS = ('read_bytes', 'write_bytes', 'unknown_direction_bytes')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def close(a, b, label):
    require(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9), label)


def validate_payloads(official, semantic, paired, validation, gddr, gddr_validation):
    """Keep model conservation and hardware acceptance as different gates."""
    rows = {x['case']: x for x in official}
    semantic_rows = {x['case']: x for x in semantic['cases']}
    paired_rows = {x['case']: x for x in paired['cases']}
    gddr_rows = {x['case']: x for x in gddr}
    for name, mapping, length in [('official', rows, len(official)),
            ('semantic', semantic_rows, len(semantic['cases'])),
            ('paired', paired_rows, len(paired['cases'])), ('gddr', gddr_rows, len(gddr))]:
        require(length == 8 and set(mapping) == set(CASES), name + ' case coverage')
    operators = phases = 0
    fresh_time_passes = []
    for case in CASES:
        row, sem, pair, extension = rows[case], semantic_rows[case], paired_rows[case], gddr_rows[case]
        decode = int(case.split('d')[1])
        expected = 507 * (decode + 1)
        require(row['operator_count'] == sem['operator_count'] == extension['operators'] == expected,
                case + ' operator count')
        require(len(row['phase_totals']) == decode + 1, case + ' phase coverage')
        require(row['status'] == 'PASS_FULL_OPERATOR_ACCOUNTING_NOT_HARDWARE_ACCEPTANCE',
                case + ' official result status')
        for field in ('known_read_bytes', 'known_write_bytes', 'unclassified_io_bytes'):
            require(row[field] == sem[field] == pair['model'][field] == extension[field],
                    case + ' ' + field + ' source closure')
            require(sum(p[field] for p in row['phase_totals'].values()) == row[field],
                    case + ' ' + field + ' phase closure')
        expected_ns = sum(p['model_ns'] for p in row['phase_totals'].values())
        close(row['model_ms'] * 1e6, expected_ns, case + ' time closure')
        total = sum(row[f] for f in ('known_read_bytes', 'known_write_bytes', 'unclassified_io_bytes'))
        require(row['total_boundary_bytes'] == total, case + ' total traffic')
        close(row['model_GBps'], total / expected_ns, case + ' model effective bandwidth')
        require(set(sem['categories']) == set(CATEGORIES), case + ' categories')
        for direction, field in zip(DIRECTIONS, ('known_read_bytes', 'known_write_bytes', 'unclassified_io_bytes')):
            require(sum(sem['categories'][c][direction] for c in CATEGORIES) == row[field],
                    case + ' semantic ' + direction)
        require(sem['categories']['weights']['write_bytes'] == 0, case + ' weight writes')
        require(sem['categories']['other']['unknown_direction_bytes'] == row['unclassified_io_bytes'],
                case + ' unknown-direction IO')
        close(pair['model']['model_ms'], row['model_ms'], case + ' paired model time')
        for path in ('native', 'explicit'):
            hw = pair['hardware'][path]
            close(hw['effective_GBps'], (hw['read_bytes'] + hw['write_bytes']) / (hw['full_ms'] * 1e6),
                  case + ' ' + path + ' observed effective bandwidth')
            close(hw['model_time_error_pct'], 100 * (row['model_ms'] / hw['full_ms'] - 1),
                  case + ' ' + path + ' timing error')
            for direction in ('read', 'write'):
                close(hw['model_known_' + direction + '_error_pct'],
                      100 * (row['known_' + direction + '_bytes'] / hw[direction + '_bytes'] - 1),
                      case + ' ' + path + ' directional error')
        if abs(pair['hardware']['native']['model_time_error_pct']) <= 10:
            fresh_time_passes.append(case)
        operators += expected
        phases += decode + 1
    require(operators == 23322 and phases == 46, 'full matrix denominators')
    require(validation['accounting_status'] == 'PASS' and validation['completed_cases'] == 8,
            'official accounting verification')
    require(validation['verified_unique_official_shapes'] == 111, 'official source parity denominator')
    require(validation['hbfsim_primary_time_contribution_ns'] == 0, 'HBFSim must not enter official time')
    require(validation['full_window_time_gate'] == validation['every_phase_time_gate'] == 'FAIL',
            'frozen accuracy failures must remain disclosed')
    require(len(validation['passing_cases']) == 2 and len(validation['phase_passing_cases']) == 0,
            'historical time gate counts')
    require(gddr_validation['full_no_worse_gate'] == 'FAIL' and
            gddr_validation['every_phase_no_worse_gate'] == 'FAIL', 'GDDR regression disclosure')
    require(len(gddr_validation['full_no_worse_cases']) == 7 and
            len(gddr_validation['every_phase_no_worse_cases']) == 1, 'GDDR regression denominators')
    require(semantic['profile_sha256'] == validation['model_profile_sha256'], 'frozen profile consistency')
    return dict(status='PASS_STAGE_ANALYTICAL_REPRODUCTION_AND_ACCOUNTING_NOT_HARDWARE_ACCURACY',
                cases=8, operators=operators, phases=phases, official_shapes=111,
                fresh_native_full_time_within_10_percent_cases=fresh_time_passes,
                hardware_accuracy='NOT_ACCEPTED', hbfsim_in_official_time=False)


def verify(adapter):
    adapter = Path(adapter).resolve()
    checkpoint = adapter / 'checkpoint'
    manifest = load(checkpoint / 'manifest.json')
    for group, root in [('files_sha256', adapter), ('official_source_sha256', adapter.parents[1])]:
        for name, digest in manifest[group].items():
            path = (root / name).resolve()
            require(path.is_relative_to(root), 'Unsafe manifest path')
            require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'Changed file: ' + name)
    result = validate_payloads(*(load(checkpoint / name) for name in
        ('official-comparison.json', 'semantic-breakdown.json', 'paired-collection.json',
         'official-validation.json', 'gddr-comparison.json', 'gddr-validation.json')))
    if 'batch_support_checkpoint' in manifest:
        support = load(checkpoint / manifest['batch_support_checkpoint'])
        require(support['status'] == 'PASS_STATIC_BATCH_SUPPORT_NOT_FULL_MATRIX_ACCEPTANCE',
                'Batch support scope')
        require(support['unit_tests']['count'] == 65 and support['unit_tests']['status'] == 'PASS',
                'Batch support test denominator')
        require(support['b1_plan_parity']['operators'] == 23322
                and support['b1_plan_parity']['exact_access_bytes_addresses_and_shapes'] is True,
                'B1 plan regression')
        require([r['request_count'] for r in support['request_regions']] == [1, 2, 4, 8, 16, 32],
                'Static batch support scope')
        require(support['batch_timing_results_published'] is False and support['hardware_collected'] is False,
                'Support receipt must not claim matrix/hardware acceptance')
        for name, digest in support['source_sha256'].items():
            require(hashlib.sha256((adapter / name).read_bytes()).hexdigest() == digest,
                    'Batch support source changed: ' + name)
        result['static_batch_support'] = support['status']
    if 'static_batch_checkpoint' in manifest:
        from verify_batch_inference import verify as verify_batch
        batch_root = (checkpoint / manifest['static_batch_checkpoint']).resolve()
        require(batch_root.is_relative_to(checkpoint), 'Unsafe batch checkpoint path')
        result['static_batch'] = verify_batch(batch_root, checkpoint / 'official-comparison.json', adapter)
    if 'historical_semantic_checkpoint' in manifest:
        legacy = load(checkpoint / manifest['historical_semantic_checkpoint'])
        require(legacy['claim'] == 'MODEL_ESTIMATE_LEGACY_ADAPTER_NOT_OFFICIAL_MAPPER_OR_PHYSICAL_DRAM',
                'Legacy evidence must not be promoted')
        require(len(legacy['cases']) == 8 and legacy['native_ada_profile'] is False
                and legacy['hardware_comparison'] is None, 'Legacy comparison scope')
        require(sum(r['evidence'] == 'ARCHIVE_TOTALS_RECONCILED' for r in legacy['cases']) == 5,
                'Legacy archived case count')
        for row in legacy['cases']:
            for direction in ('read', 'write'):
                require(sum(row['semantic_bytes'][direction].values()) == row['known_' + direction + '_bytes'],
                        'Legacy semantic closure')
        require(legacy['checks']['read_write_delta_bytes_all_archived_windows'] == 0,
                'Legacy archive reconciliation')
        result['historical_semantic_cases'] = 8
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().adapter), indent=2))


if __name__ == '__main__':
    main()
