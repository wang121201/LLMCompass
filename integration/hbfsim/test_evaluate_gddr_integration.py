"""No-regression arithmetic and canonical proxy-address contracts."""
import types
import unittest
import hashlib
import json
import math
import pathlib

from evaluate_gddr_integration import canonical_views, compare, frozen_plan, percent
from mapper_tensor_addresses import MatrixView


def run_completed_result_checks(root):
    """Read-only independent closure, full-window and every-phase audit."""
    root = pathlib.Path(root)
    receipt = json.loads((root / 'receipt.json').read_text())
    if receipt['status'] != 'PASS_EIGHT_CASE_GDDR_EXTENSION_ACCOUNTING':
        raise ValueError('Incomplete GDDR matrix')
    for name, digest in receipt['input_sha256'].items():
        if hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen input changed: ' + name)
    baseline_root = pathlib.Path(receipt['source_matrix'])
    original = {r['case']: r for r in json.loads((baseline_root / 'comparison.json').read_text())}
    frozen = json.loads((baseline_root / 'validation.json').read_text())
    rows = json.loads((root / 'comparison.json').read_text())
    if len(rows) != 8 or {r['case'] for r in rows} != set(original):
        raise ValueError('Wrong eight-case denominator')
    checked = 0
    cases = {}
    for row in rows:
        case = row['case']
        ledger = json.loads((root / case / 'operators.json').read_text())
        before = json.loads((baseline_root / case / 'operators.json').read_text())
        if not (len(ledger) == len(before) == row['operators']):
            raise ValueError('Incomplete operator graph')
        for current, old in zip(ledger, before):
            for key in ('index', 'name', 'layer', 'phase', 'path', 'known_read_bytes', 'known_write_bytes', 'unclassified_io_bytes'):
                if current[key] != old[key]:
                    raise ValueError('Operator identity/traffic changed: ' + key)
            if current['baseline_ns'] != old['model_ns'] or not math.isfinite(current['model_ns']) or current['model_ns'] <= 0:
                raise ValueError('Invalid operator time')
            if current['service_read_bytes'] < current['known_read_bytes'] or current['service_write_bytes'] < current['known_write_bytes']:
                raise ValueError('Backend burst service loses logical bytes')
            checked += 1
        for field in ('model_ns', 'known_read_bytes', 'known_write_bytes', 'unclassified_io_bytes',
                      'service_read_bytes', 'service_write_bytes'):
            value = sum(x[field] for x in ledger)
            if not math.isclose(value, row[field], rel_tol=1e-12, abs_tol=1e-5):
                raise ValueError('Full sum does not close: ' + field)
        phase_checks = {}
        for phase, totals in row['phase_totals'].items():
            members = [x for x in ledger if x['phase'] == phase]
            for field in totals:
                value = len(members) if field == 'operators' else sum(x[field] for x in members)
                if not math.isclose(value, totals[field], rel_tol=1e-12, abs_tol=1e-5):
                    raise ValueError('Phase sum does not close: ' + phase)
            hw = frozen['cases'][case]['phases'][phase]['hardware_ms'] * 1e6
            new_error = 100 * (totals['model_ns'] / hw - 1)
            old_error = frozen['cases'][case]['phases'][phase]['time_error_pct']
            phase_checks[phase] = dict(candidate_ms=totals['model_ns'] / 1e6, hardware_ms=hw / 1e6,
                candidate_error_percent=new_error, baseline_error_percent=old_error,
                no_worse=abs(new_error) <= abs(old_error) + 1e-9, within_10_percent=abs(new_error) <= 10)
        old = original[case]
        htime = old['hardware_ms'] * 1e6
        hb = (old['hardware_read_bytes'] + old['hardware_write_bytes']) / htime
        expected = {
            'time': (row['model_ns'], old['model_ms'] * 1e6, htime),
            'read': (row['known_read_bytes'], old['known_read_bytes'], old['hardware_read_bytes']),
            'write': (row['known_write_bytes'], old['known_write_bytes'], old['hardware_write_bytes']),
            'bandwidth': ((row['known_read_bytes'] + row['known_write_bytes'] + row['unclassified_io_bytes']) / row['model_ns'],
                          old['model_GBps'], hb),
        }
        for metric, (new, prior, observed) in expected.items():
            ne, oe = 100 * (new / observed - 1), 100 * (prior / observed - 1)
            record = row['comparison']['metrics'][metric]
            if not math.isclose(ne, record['candidate_error_percent'], abs_tol=1e-9) or not math.isclose(oe, record['baseline_error_percent'], abs_tol=1e-9):
                raise ValueError('Comparison arithmetic differs: ' + metric)
            if record['no_worse'] != (abs(ne) <= abs(oe) + 1e-9):
                raise ValueError('No-regression gate differs: ' + metric)
        service = dict(read_error_percent=100 * (row['service_read_bytes'] / old['hardware_read_bytes'] - 1),
                       write_error_percent=100 * (row['service_write_bytes'] / old['hardware_write_bytes'] - 1),
                       read_burst_padding_bytes=row['service_read_bytes'] - row['known_read_bytes'],
                       write_burst_padding_bytes=row['service_write_bytes'] - row['known_write_bytes'],
                       scope='Backend burst-service estimates, not validated physical GPU counters')
        cases[case] = dict(full_no_worse=row['comparison']['no_worse'],
                          every_phase_no_worse=all(x['no_worse'] for x in phase_checks.values()),
                          full_within_10_percent=abs(row['comparison']['metrics']['time']['candidate_error_percent']) <= 10,
                          every_phase_within_10_percent=all(x['within_10_percent'] for x in phase_checks.values()),
                          phases=phase_checks, backend_service=service)
    result = dict(status='PASS_INDEPENDENT_GDDR_MATRIX_ACCOUNTING_NOT_HARDWARE_ACCEPTANCE',
                  verified_operators=checked, verified_cases=len(cases), verified_phases=sum(len(x['phases']) for x in cases.values()),
                  full_no_worse_gate='PASS' if all(x['full_no_worse'] for x in cases.values()) else 'FAIL',
                  every_phase_no_worse_gate='PASS' if all(x['every_phase_no_worse'] for x in cases.values()) else 'FAIL',
                  full_no_worse_cases=[c for c, x in cases.items() if x['full_no_worse']],
                  every_phase_no_worse_cases=[c for c, x in cases.items() if x['every_phase_no_worse']],
                  full_within_10_percent_cases=[c for c, x in cases.items() if x['full_within_10_percent']],
                  every_phase_within_10_percent_cases=[c for c, x in cases.items() if x['every_phase_within_10_percent']],
                  verifier_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), cases=cases)
    if checked != 23322 or result['verified_phases'] != 46:
        raise ValueError('Wrong full-graph denominator')
    return result


def run_request_granularity_checks(backend_root):
    """Same traffic and address coverage, differing parent request granularity.

    A third variant uses the P128 gate-weight row stride. This is diagnosis,
    not an address rewrite or alternative used in the eight-case results.
    """
    backend_root = pathlib.Path(backend_root)
    import sys
    sys.path.insert(0, str(backend_root))
    from hbfsim_client import SimulationSession, ResolvedSystemConfig, Transaction
    config = ResolvedSystemConfig.load((backend_root / 'configs/systems/eight-stack-baseline.cfg',
        backend_root / 'configs/overlays/dram/rtx4000-ada-gddr6.cfg')).resolve(backend_root / 'build/hbfsim', enable_hbf=False)
    cases = [('one_contiguous_parent', [(0, 131072)]),
             ('1024_contiguous_128byte_parents', [(i * 128, 128) for i in range(1024)]),
             ('1024_strided_128byte_parents', [(i * 8960 * 2, 128) for i in range(1024)])]
    rows = []
    for name, spans in cases:
        txs = [Transaction(id=f'r{i}', target='HBM', op='R', addr=1024**3 + address,
                            bytes=size, issue_ns=0) for i, (address, size) in enumerate(spans)]
        with SimulationSession(simulator_path=backend_root / 'build/hbfsim', system_config=config,
                enable_hbm=True, enable_hbf=False, read_timeout_s=60) as session:
            result = session.run(txs, retain=())
            physical = result.receipt['device_delta']['hbm']['read_bytes']
        if physical != 131072 or sum(t.bytes for t in txs) != 131072:
            raise ValueError('Fixed-byte granularity control did not close')
        source = session.source_receipt()
        rows.append(dict(case=name, parents=len(txs), read_bytes=physical, service_ns=result.elapsed_ns,
                         effective_GBps=physical / result.elapsed_ns,
                         engine_source=source['engine_source'],
                         resources=source['final_measurement']['device_workload_totals']['hbm']['resource_busy']))
    return dict(status='PASS_FIXED_BYTE_GRANULARITY_CONTROLS_NOT_HARDWARE_VALIDATION',
                contiguous_fragmentation_time_ratio=rows[1]['service_ns'] / rows[0]['service_ns'],
                strided_vs_fragmented_time_ratio=rows[2]['service_ns'] / rows[1]['service_ns'], rows=rows)


def run_mapper_stage_diagnostic(root):
    """Re-run two frozen mappings and aggregate their causal stage intervals.

    Completion records exist only in memory. No alternate configuration,
    mapper optimization, trace file or eight-case result is generated.
    """
    import os
    import shutil
    import sys
    import tempfile
    from evaluate_gddr_integration import load
    root = pathlib.Path(root)
    receipt = json.loads((root / 'receipt.json').read_text())
    baseline = pathlib.Path(receipt['source_matrix'])
    adapter = baseline.parent.parent
    repo = adapter.parents[1]
    backend = pathlib.Path(receipt['backend_root'])
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(backend))
    from hbfsim_client import SimulationSession, ResolvedSystemConfig, Transaction
    q = load('diagnostic_frozen_qwen', baseline / 'source/qwen_hbfsim_cosim.py')
    cost = q.LLMCompassCostModel(repo, baseline / 'source/RTX4000Ada_xmu_profile_v3.json')
    mat = load('diagnostic_frozen_matmul', baseline / 'source/mapper_matmul.py')
    coupling = load('diagnostic_frozen_coupling', root / 'mapper_event_coupling.py')
    selected = {tuple(x['key']): x for x in json.loads((adapter / 'results/semantic-traffic-20261008-r1/selected-operands.json').read_text())}
    templates = json.loads((root / 'prototype-summary.json').read_text())
    cfg = ResolvedSystemConfig.load((backend / 'configs/systems/eight-stack-baseline.cfg',
        backend / 'configs/overlays/dram/rtx4000-ada-gddr6.cfg')).resolve(backend / 'build/hbfsim', enable_hbf=False)
    work = pathlib.Path(tempfile.mkdtemp(prefix='llmcompass-gddr-stage-diagnostic-'))
    shutil.copytree(repo / 'systolic_array_model', work / 'systolic_array_model',
                    ignore=shutil.ignore_patterns('temp', '*.gz', '*.npy'))
    (work / 'systolic_array_model/temp').mkdir()
    previous_cwd = pathlib.Path.cwd()
    rows = []
    try:
        os.chdir(work)
        for m in (128, 512):
            k, n = 1536, 8960
            record = selected[('matmul', m, k, n)]
            op = mat.Matmul(cost.dtype)
            op(cost.Tensor([m, k], cost.dtype), cost.Tensor([k, n], cost.dtype))
            cycles = op.simulate(op.computational_graph, mat.Matmul.Mapping(**record['mapping']), cost.device)
            clock = cost.device.compute_module.clock_freq
            views = canonical_views(MatrixView, m, k, n)
            txs, frontier = coupling.lower_stages(Transaction, op.execution_stages,
                op.main_memory_events, clock, lambda e: views[e['tensor']].requests(e))
            with SimulationSession(simulator_path=backend / 'build/hbfsim', system_config=cfg,
                    enable_hbm=True, enable_hbf=False, read_timeout_s=120) as session:
                result = session.run(txs, frontier=frontier, retain=(), completions=True)
                ends = coupling.audit_completion_dependencies(txs, {x.id: x.finish_ns for x in result.completions},
                                                               frontier, result.blocking_finish_ns)
            key = json.dumps(['matmul', m, k, n, record['mapping']], sort_keys=True)
            saved = templates[key]
            if not math.isclose(result.elapsed_ns, saved['ns'], abs_tol=1e-6):
                raise ValueError('Frozen prototype replay is not deterministic')
            sums = dict(analytical_read_work_ns=0, backend_read_work_ns=0,
                        analytical_read_critical_ns=0, backend_read_critical_ns=0,
                        analytical_write_critical_ns=0, backend_write_critical_ns=0,
                        onchip_work_ns=0)
            event_finishes = {}
            for name, finish in ends.items():
                if name.startswith('mem'):
                    event = int(name[3:name.index('-row')])
                    event_finishes[event] = max(event_finishes.get(event, 0), finish)
            phase_frontier = 0
            for index, stage in enumerate(op.execution_stages):
                read_finish = max([phase_frontier] + [event_finishes[e] for e in stage['read_events']])
                write_finish = max([read_finish, ends[f'tile{index}-onchip']] +
                                   [event_finishes[e] for e in stage['write_events']])
                read_ns = read_finish - phase_frontier
                compute_ns = stage['compute_cycles'] * 1e9 / clock
                analytical_read_ns = stage['read_cycles'] * 1e9 / clock
                sums['analytical_read_work_ns'] += analytical_read_ns
                sums['backend_read_work_ns'] += read_ns
                sums['onchip_work_ns'] += compute_ns
                sums['analytical_read_critical_ns'] += max(analytical_read_ns - compute_ns, 0) if stage['overlap'] else analytical_read_ns
                sums['backend_read_critical_ns'] += max(read_ns - compute_ns, 0) if stage['overlap'] else read_ns
                sums['analytical_write_critical_ns'] += stage['write_cycles'] * 1e9 / clock
                sums['backend_write_critical_ns'] += write_finish - max(read_finish, ends[f'tile{index}-onchip'])
                phase_frontier = write_finish
            analytical = sums['onchip_work_ns'] + sums['analytical_read_critical_ns'] + sums['analytical_write_critical_ns']
            modeled = sums['onchip_work_ns'] + sums['backend_read_critical_ns'] + sums['backend_write_critical_ns']
            if not math.isclose(analytical, cycles * 1e9 / clock, abs_tol=1e-5) or not math.isclose(modeled, result.elapsed_ns, abs_tol=1e-5):
                raise ValueError('Critical-stage decomposition does not close')
            rows.append(dict(shape=[m,k,n], stages=len(op.execution_stages),
                read_events=sum(e['direction']=='read' for e in op.main_memory_events),
                write_events=sum(e['direction']=='write' for e in op.main_memory_events),
                analytical_ns=analytical, candidate_ns=modeled, delta_ns=modeled-analytical,
                read_critical_delta_ns=sums['backend_read_critical_ns']-sums['analytical_read_critical_ns'],
                write_critical_delta_ns=sums['backend_write_critical_ns']-sums['analytical_write_critical_ns'],
                logical_read_bytes=saved['known_read_bytes'], logical_write_bytes=saved['known_write_bytes'],
                decomposition=sums, deterministic_completion_digest=result.receipt['transaction_completions_digest']['sha256']))
            if rows[-1]['deterministic_completion_digest'] != saved['completion_digest']:
                raise ValueError('Replay completion digest changed')
            print('STAGE_DIAGNOSTIC_COMPLETE', m, flush=True)
    finally:
        os.chdir(previous_cwd)
    return dict(status='PASS_DETERMINISTIC_STAGE_DECOMPOSITION_NOT_HARDWARE_CAUSAL_ATTRIBUTION',
                scope='Exact critical read/write interval decomposition; does not isolate backend latency, queue and address effects',
                rows=rows, isolated_working_directory=str(work))


class GddrIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(phase_totals={'prefill': {'model_ns': 1000}}, known_read_bytes=1000,
                         known_write_bytes=100, total_boundary_bytes=1100)
        self.hw = dict(natural_cuda_event_ms={'median': .00125},
                       dram_read_bytes={'median': 1000}, dram_write_bytes={'median': 100})

    def test_equal_results_pass_each_metric(self):
        result = compare(1000, 1000, 100, 0, self.base, self.hw)
        self.assertTrue(result['no_worse'])
        self.assertAlmostEqual(result['metrics']['time']['candidate_error_percent'], -20)

    def test_timing_improvement_does_not_hide_write_regression(self):
        result = compare(1250, 1000, 110, 0, self.base, self.hw)
        self.assertFalse(result['no_worse'])
        self.assertTrue(result['metrics']['time']['no_worse'])
        self.assertFalse(result['metrics']['write']['no_worse'])

    def test_bandwidth_and_time_have_separate_gates(self):
        result = compare(900, 1000, 100, 0, self.base, self.hw)
        self.assertFalse(result['metrics']['time']['no_worse'])
        self.assertFalse(result['metrics']['bandwidth']['no_worse'])

    def test_actual_tile_extents_and_proxy_regions(self):
        views = canonical_views(MatrixView, 3, 5, 7)
        event = dict(row=1, rows=2, column=2, columns=3)
        self.assertEqual(sum(n for _, _, n in views['A'].requests(event)), 12)
        self.assertEqual(views['B'].base, 1024**3)
        self.assertEqual(views['C'].allocation_bytes, 42)
        with self.assertRaises(ValueError):
            list(views['A'].requests(dict(row=0, rows=4, column=0, columns=1)))

    def test_invalid_denominator_rejected(self):
        with self.assertRaises(ValueError):
            percent(1, 0)


if __name__ == '__main__':
    unittest.main()
