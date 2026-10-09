"""Independently verify frozen eight-case accounting and hardware comparisons.

Only validation.json is refreshed. Model results, hardware measurements and
source snapshots are read-only. Passing accounting is not hardware accuracy.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import pathlib
import statistics


CASES = ('p032d02', 'p064d02', 'p128d02', 'p256d02', 'p512d02',
         'p128d04', 'p128d08', 'p128d16')
PATH_COUNTS = {'OFFICIAL_MATMUL': 197, 'OFFICIAL_BATCHED_MATMUL': 56,
               'OFFICIAL_SOFTMAX': 28, 'QWEN_VECTOR_ADAPTER': 226}
METRICS = ('model_ns', 'known_read_bytes', 'known_write_bytes',
           'unclassified_io_bytes')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected) and
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10),
            f'{label}: {actual!r} != {expected!r}')


def distribution(values):
    require(len(values) >= 2 and all(math.isfinite(v) and v >= 0 for v in values),
            'Invalid measurement samples')
    values = sorted(values)
    mean = statistics.fmean(values)
    require(mean > 0, 'Zero measurement mean')
    quantiles = statistics.quantiles(values, n=10, method='inclusive')
    stdev = statistics.stdev(values)
    return dict(samples=values, count=len(values), median=statistics.median(values),
                mean=mean, p10=quantiles[0], p90=quantiles[8], stdev=stdev,
                coefficient_of_variation_pct=100 * stdev / mean)


def verify_distribution(saved, values, label):
    expected = distribution(values)
    require(saved['count'] == expected['count'], f'{label}: sample count')
    require(saved['samples'] == expected['samples'], f'{label}: sample identity')
    for name in ('median', 'mean', 'p10', 'p90', 'stdev',
                 'coefficient_of_variation_pct'):
        close(saved[name], expected[name], f'{label}/{name}')


def time_comparison(model_ns, durations):
    require(all(duration > 0 for duration in durations), 'Zero timing denominator')
    stats = distribution(durations)
    model_ms = model_ns / 1e6
    error = 100 * (model_ms / stats['median'] - 1)
    # Observed sample range, not a confidence interval or clock control claim.
    sample_errors = [100 * (model_ms / duration - 1) for duration in durations]
    return dict(model_ms=model_ms, hardware_ms=stats['median'], time_error_pct=error,
                time_within_10_percent=abs(error) <= 10,
                hardware_timing_statistics=stats,
                time_error_over_observed_samples_pct=[min(sample_errors), max(sample_errors)],
                passing_timing_samples=sum(abs(e) <= 10 for e in sample_errors))


def cancellation_status(full, phases):
    errors = [row['time_error_pct'] for row in phases.values()]
    mixed = min(errors) < 0 < max(errors)
    all_pass = all(row['time_within_10_percent'] for row in phases.values())
    return dict(opposite_signed_phase_errors=mixed,
                total_pass_with_failing_phase=full['time_within_10_percent'] and not all_pass,
                total_pass_with_cancellation=full['time_within_10_percent'] and not all_pass and mixed,
                all_phase_time_within_10_percent=all_pass,
                full_and_every_phase_time_gate='PASS' if full['time_within_10_percent'] and all_pass else 'FAIL')


def traffic_comparison(totals, observed, model_ns, hardware_ms):
    read, write, extra = (totals[k] for k in METRICS[1:])
    hr, hw = (observed[k]['median'] for k in ('dram_read_bytes', 'dram_write_bytes'))
    require(hr > 0 and hw > 0 and model_ns > 0 and hardware_ms > 0,
            'Counter and bandwidth denominators must be positive')
    total = read + write + extra
    model_bw = total / model_ns
    hardware_bw = (hr + hw) / (hardware_ms * 1e6)
    return dict(known_read_bytes=read, known_write_bytes=write,
                unclassified_io_bytes=extra, total_boundary_bytes=total,
                hardware_read_bytes=hr, hardware_write_bytes=hw,
                known_read_error_pct=100 * (read / hr - 1),
                known_write_error_pct=100 * (write / hw - 1),
                read_error_range_pct=[100 * (read / hr - 1), 100 * ((read + extra) / hr - 1)],
                write_error_range_pct=[100 * (write / hw - 1), 100 * ((write + extra) / hw - 1)],
                directional_bound_rule='Allocate extra IO once: extra_read + extra_write = unclassified_io_bytes; upper bounds are not simultaneous.',
                model_GBps=model_bw, hardware_GBps=hardware_bw,
                bandwidth_error_pct=100 * (model_bw / hardware_bw - 1))


def verify_root(root):
    root = pathlib.Path(root)
    files = {}

    def read_bytes(path):
        path = pathlib.Path(path)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(str(path) not in files or files[str(path)] == digest,
                f'Input changed during verification: {path}')
        files[str(path)] = digest
        return data

    def read_json(path):
        return json.loads(read_bytes(path))

    rows = read_json(root / 'comparison.json')
    manifest = read_json(root / 'manifest.json')
    require(len(rows) == 8 and {r['case'] for r in rows} == set(CASES),
            'Expected exactly the eight requested cases, without duplicates')
    require(manifest['status'] == 'PASS_FULL_MATRIX_ACCOUNTING' and
            manifest['completed_cases'] == 8, 'Incomplete matrix')
    require(manifest['hbfsim_in_primary_time'] is False and manifest['seed_path'] is None,
            'Primary time must not include HBFSim or a latency seed')
    snapshot = root / 'source' / pathlib.Path(manifest['profile']).name
    profile = read_json(snapshot)
    require(files[str(snapshot)] == manifest['profile_sha256'], 'Profile hash mismatch')
    for name, digest in manifest['source_sha256'].items():
        path = root / 'source' / name
        read_bytes(path)
        require(files[str(path)] == digest, f'Source snapshot hash mismatch: {name}')
    device = profile['device']
    observed_device = profile['calibration_status']['observed_on_xmu']
    require('RTX 4000 Ada' in profile['name'] and profile['device_count'] == 1 and
            device['compute_chiplet']['core_count'] == observed_device['sm_count'] == 48 and
            device['memory_protocol'] == 'GDDR6', 'Wrong target profile')
    require(all(device['operator_overhead_seconds'][k] == 0
                for k in ('matmul', 'softmax', 'layernorm', 'gelu')),
            'This receipt verifies the frozen body-only timing contract')
    shapes = read_json(root / 'shape-cache.json')
    require(len(shapes) == len({tuple(s['key']) for s in shapes}) == 111,
            'Expected 111 distinct compiled shapes')
    for shape in shapes:
        require(shape['official_latency_parity'] is True, 'Official latency parity failed')
        close(shape['seconds'], shape['official_source_seconds'], str(shape['key']))

    results = {}
    config_hashes, driver_hashes = set(), set()
    timing_receipts = counter_receipts = operator_count = 0
    for summary in rows:
        case = summary['case']
        p, d = int(case[1:4]), int(case.split('d')[1])
        hw_path = pathlib.Path(summary['hardware_source'])
        hardware = read_json(hw_path)
        require(files[str(hw_path)] == summary['hardware_sha256'], f'{case}: hardware hash')
        require(hardware['status'] == 'PASS_NATIVE_DRAM_COUNTERS_AND_TIMING', f'{case}: hardware status')
        require(hardware['definition']['units'] ==
                {'bandwidth': 'decimal GB/s', 'bytes': 'byte', 'duration': 'ms'}, f'{case}: hardware units')
        identity_path = hw_path.parents[2] / 'identity.txt'
        identity = dict(line.split('=', 1) for line in read_bytes(identity_path).decode().splitlines() if '=' in line)
        require(identity['gpu_uuid'] == observed_device['gpu_uuid'], f'{case}: hardware GPU identity')
        require(identity['counter_repeats'] == '5' and identity['timing_repeats'] == '20' and
                identity['counter_cache_control'] == identity['counter_clock_control'] == 'none',
                f'{case}: hardware acquisition contract')
        workload = hardware['workload']
        require(workload['prefill_length'] == p and workload['decode_steps'] == d and
                workload['batch_size'] == workload['pp_size'] == workload['tp_size'] == 1 and
                workload['layers'] == 28 and workload['model_key'] == 'qwen25_1p5b' and
                workload['dtype'] == 'bfloat16', f'{case}: workload identity')
        require(workload['native_eager'] and workload['disable_overlap_schedule'] and
                workload['disable_radix_cache'] and not workload['cuda_graph'] and
                not workload['torch_compile'] and not workload['output_feedback'],
                f'{case}: native workflow contract')
        native_phases = ['Prefill'] + [f'Decode{i}' for i in range(1, d + 1)]
        require(workload['phases'] == native_phases, f'{case}: hardware phase sequence')
        ledger = read_json(root / case / 'operators.json')
        require(read_json(root / case / 'summary.json') == summary, f'{case}: duplicated summary differs')
        expected = 507 * (d + 1)
        require(summary['operator_count'] == len(ledger) == expected, f'{case}: operator count')
        require([o['index'] for o in ledger] == list(range(expected)), f'{case}: operator indices')
        require(all(isinstance(o[k], int) and o[k] >= 0 for o in ledger for k in METRICS) and
                all(o['model_ns'] > 0 for o in ledger), f'{case}: invalid ledger values')
        model_phases = ['prefill'] + [f'decode_{i}' for i in range(1, d + 1)]
        require(list(summary['phase_totals']) == model_phases, f'{case}: model phase sequence')
        require([o['phase'] for o in ledger] == [phase for phase in model_phases for _ in range(507)],
                f'{case}: noncontiguous operator phases')
        totals = {k: sum(o[k] for o in ledger) for k in METRICS}
        close(totals['model_ns'] / 1e6, summary['model_ms'], f'{case}: total time')
        for k in METRICS[1:]:
            require(totals[k] == summary[k], f'{case}: total {k}')
        require(dict(collections.Counter(o['path'] for o in ledger)) == summary['paths'], f'{case}: path totals')
        for phase in model_phases:
            members = [o for o in ledger if o['phase'] == phase]
            require(dict(collections.Counter(o['path'] for o in members)) == PATH_COUNTS,
                    f'{case}/{phase}: official/adapter path coverage')
            phase_totals = summary['phase_totals'][phase]
            require(phase_totals['operators'] == 507 and
                    all(sum(o[k] for o in members) == phase_totals[k] for k in METRICS),
                    f'{case}/{phase}: phase closure')

        durations = {phase: [] for phase in native_phases + ['full', 'DecodeAll']}
        walls = {phase: [] for phase in durations}
        finishes = sorted((hw_path.parent / 'timing').glob('repeat-*/host/process-*/finish.json'))
        require(len(finishes) == 20 and
                {int(path.parents[2].name.split('-')[1]) for path in finishes} == set(range(1, 21)),
                f'{case}: expected one finish receipt per timing repeat')
        for path in finishes:
            finish = read_json(path)
            require(finish['status'] == 'PASS_NATIVE_WORKFLOW_AND_ROI' and
                    finish['mode'] == 'validate' and finish['input_contract'] == workload and
                    finish['source_unchanged'] is True and not finish['profiler_api_invoked'] and
                    not finish['profiler_events'] and not finish['event_time_is_NCU_duration'] and
                    not finish['native_cuda_kernels_modified'] and not finish['instruction_memory_trace'],
                    f'{path}: invalid unprofiled timing receipt')
            aligned = finish['aligned_execution']
            require(not aligned['control_D2H_inside_forward'] and
                    not aligned['control_D2H_between_measured_phases'] and
                    aligned['sampling_retained'] and not aligned['predictions_fed_back'],
                    f'{path}: workflow window changed')
            events = finish['natural_cuda_event_ms']
            require(list(events) == native_phases and all(v > 0 for v in events.values()), f'{path}: event phases')
            measured = [row for row in aligned['phases'] if row['stage'] == 'Measured']
            require([row['phase'] for row in measured] == native_phases, f'{path}: measured phase order')
            wall = {row['phase']: row['instrumented_wall_seconds'] * 1000 for row in measured}
            for phase in native_phases:
                durations[phase].append(events[phase]); walls[phase].append(wall[phase])
            durations['full'].append(sum(events.values())); walls['full'].append(sum(wall.values()))
            durations['DecodeAll'].append(sum(events[phase] for phase in native_phases[1:]))
            walls['DecodeAll'].append(sum(wall[phase] for phase in native_phases[1:]))
            config_hashes.add(finish['model_config_sha256']); driver_hashes.add(finish['driver_sha256'])
        timing_receipts += len(finishes)

        expected_rois = {'Prefill', 'D1', 'D2', 'full'} | ({'DecodeAll'} if d > 2 else set())
        require(set(hardware['rois']) == expected_rois, f'{case}: missing/extra counter ROIs')
        roi_comparisons = {}
        for roi, observation in hardware['rois'].items():
            native_roi = {'D1': 'Decode1', 'D2': 'Decode2'}.get(roi, roi)
            verify_distribution(observation['natural_cuda_event_ms'], durations[native_roi], f'{case}/{roi}/event')
            verify_distribution(observation['synchronized_wall_ms_crosscheck'], walls[native_roi], f'{case}/{roi}/wall')
            counter_files = sorted((hw_path.parent / 'counters' / roi).glob('repeat-*/raw.csv'))
            require(len(counter_files) == 5 and
                    {int(path.parent.name.split('-')[1]) for path in counter_files} == set(range(1, 6)),
                    f'{case}/{roi}: counter repeats')
            counters = {'dram_read_bytes': [], 'dram_write_bytes': []}
            actions = []
            for path in counter_files:
                records = list(csv.DictReader(read_bytes(path).decode().splitlines()))
                require(len(records) == 2 and records[1]['Kernel Name'] == 'range', f'{path}: range action')
                row = records[1]
                number = lambda k: int(row[k].replace(',', ''))
                require(row['device__attribute_display_name'] == 'NVIDIA RTX 4000 Ada Generation' and
                        number('device__attribute_multiprocessor_count') == 48 and
                        number('device__attribute_l2_cache_size') == observed_device['l2_cache_bytes'] and
                        number('profiler__replayer_passes') == 1, f'{path}: target/replay identity')
                counters['dram_read_bytes'].append(number('dram__bytes_read.sum'))
                counters['dram_write_bytes'].append(number('dram__bytes_write.sum'))
                actions.append(dict(path=str(path.relative_to(hw_path.parent)), replay_passes=1))
            require(actions == observation['counter_actions'], f'{case}/{roi}: counter action identity')
            for name, samples in counters.items():
                verify_distribution(observation[name], samples, f'{case}/{roi}/{name}')
            ht = statistics.median(durations[native_roi])
            for name, keys in [('read_gbps', ['dram_read_bytes']), ('write_gbps', ['dram_write_bytes']),
                               ('total_gbps', ['dram_read_bytes', 'dram_write_bytes'])]:
                close(observation[name], sum(observation[k]['median'] for k in keys) / ht / 1e6,
                      f'{case}/{roi}/{name}')
            selected = model_phases if roi == 'full' else model_phases[1:] if roi == 'DecodeAll' else [
                {'Prefill': 'prefill', 'D1': 'decode_1', 'D2': 'decode_2'}[roi]]
            model_totals = {k: sum(summary['phase_totals'][phase][k] for phase in selected) for k in METRICS}
            comparison = time_comparison(model_totals['model_ns'], durations[native_roi])
            comparison.update(traffic_comparison(model_totals, observation, model_totals['model_ns'], ht))
            roi_comparisons[roi] = comparison
            counter_receipts += len(counter_files)

        full = roi_comparisons['full']
        for k in ('model_ms', 'hardware_ms', 'hardware_read_bytes', 'hardware_write_bytes',
                  'total_boundary_bytes', 'model_GBps', 'hardware_GBps', 'time_error_pct',
                  'bandwidth_error_pct', 'known_read_error_pct', 'known_write_error_pct'):
            close(summary[k], full[k], f'{case}: saved full {k}')
        require(summary['read_error_range_pct'] == full['read_error_range_pct'] and
                summary['write_error_range_pct'] == full['write_error_range_pct'], f'{case}: directional bounds')
        phases = {}
        for model_phase, native_phase in zip(model_phases, native_phases):
            comparison = time_comparison(summary['phase_totals'][model_phase]['model_ns'], durations[native_phase])
            comparison['hardware_counter_roi'] = {'Prefill': 'Prefill', 'Decode1': 'D1', 'Decode2': 'D2'}.get(native_phase)
            comparison['individual_dram_comparison_status'] = 'AVAILABLE' if comparison['hardware_counter_roi'] else 'NOT_COLLECTED'
            phases[model_phase] = comparison
        decode_ns = sum(summary['phase_totals'][phase]['model_ns'] for phase in model_phases[1:])
        results[case] = dict(operator_count=expected, phases=phases, rois=roi_comparisons,
                             decode_all=time_comparison(decode_ns, durations['DecodeAll']),
                             **cancellation_status(full, phases))
        operator_count += expected
        print(case, 'PASS_OPERATOR_PHASE_BYTE_TIME_BANDWIDTH_RAW_AGGREGATE_CLOSURE', expected)

    require(len(config_hashes) == len(driver_hashes) == 1, 'Native model/driver differs across cases')
    errors = {case: results[case]['rois']['full']['time_error_pct'] for case in CASES}
    passing = [case for case in CASES if results[case]['rois']['full']['time_within_10_percent']]
    phase_passing = [case for case in CASES if results[case]['all_phase_time_within_10_percent']]
    # Bind every consumed input and reject a verification-time modification.
    for path, digest in files.items():
        require(hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() == digest,
                f'Input changed before receipt completion: {path}')
    return dict(schema='LLMCOMPASS_RTX4000ADA_EIGHT_CASE_ACCEPTANCE_V2',
                accounting_status='PASS', completed_cases=8,
                verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                verifier_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                full_window_time_gate='PASS' if len(passing) == 8 else 'FAIL',
                criterion='Every requested full-window case must satisfy abs(model/hardware-1)*100 <= 10; no average-only acceptance',
                passing_cases=passing, per_case_error_pct=errors,
                max_abs_error_pct=max(abs(v) for v in errors.values()),
                model_profile_sha256=manifest['profile_sha256'],
                every_phase_time_gate='PASS' if len(phase_passing) == 8 else 'FAIL',
                phase_passing_cases=phase_passing,
                total_pass_with_cancellation_cases=[case for case in CASES if results[case]['total_pass_with_cancellation']],
                phase_criterion='Each case must also satisfy the same ten-percent bound at every measured Prefill and Decode step; a diagnostic anti-cancellation guard, not the original paper criterion.',
                verified_operators=operator_count, verified_unique_official_shapes=len(shapes),
                verified_phase_comparisons=sum(len(r['phases']) for r in results.values()),
                verified_unprofiled_timing_receipts=timing_receipts,
                verified_ncu_aggregate_receipts=counter_receipts,
                native_model_config_sha256=next(iter(config_hashes)), native_driver_sha256=next(iter(driver_hashes)),
                target_gpu_uuid=observed_device['gpu_uuid'], hbfsim_primary_time_contribution_ns=0,
                model_timing_scope='Official operator bodies plus labeled Qwen/bias adapters; no official transformer overhead aggregation.',
                hardware_timing_scope='Median of same-run phase CUDA-event sums; not host wall, NCU replay time, or sum of separate phase medians.',
                cycle_scope='Primary ledger is integer ns, not a maintained native GPU cycle clock. Hardware cycle counters were not collected.',
                traffic_bandwidth_acceptance='NOT_ESTABLISHED: mapper/adapter boundary estimates vs observed DRAM counters and separately measured effective event-window rates; no agreed directional/bandwidth threshold or physical counter equivalence.',
                directional_bandwidth_threshold=None, numerical_acceptance='NOT_ASSESSED',
                input_sha256=files, cases=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=pathlib.Path)
    root = parser.parse_args().root
    receipt = verify_root(root)
    (root / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('VERIFIED_COMPLETED_CASES', receipt['completed_cases'])
    print('FULL_WINDOW_TIME_10_PERCENT_GATE', receipt['full_window_time_gate'], len(receipt['passing_cases']), '/ 8')
    print('EVERY_PHASE_TIME_10_PERCENT_GATE', receipt['every_phase_time_gate'], len(receipt['phase_passing_cases']), '/ 8')
    print('TOTAL_PASS_WITH_CANCELLATION', receipt['total_pass_with_cancellation_cases'])
    print('VERIFIED_NATIVE_TIMING_AND_NCU_AGGREGATES', receipt['verified_unprofiled_timing_receipts'], receipt['verified_ncu_aggregate_receipts'])


if __name__ == '__main__':
    main()
