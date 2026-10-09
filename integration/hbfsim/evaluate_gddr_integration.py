"""Fixed-mapper Qwen matrix with independently calibrated GDDR6 service.

The official analytical operator aggregation stays intact. Selected tiled
main-memory stages use HBFSim completions instead of their original IO costs.
Vector/Matmul shortcuts retain the official max(compute, IO) approximation.
Independent batches and concatenated batch surrogates retain their official
selected candidate. Unclassified extra IO retains its original analytical
cost once and is never assigned a fabricated read/write direction.
Canonical matrix addresses are modeling proxies, not captured CUDA addresses.
No cache model, Qwen fitting, raw trace output or primary-result overwrite.
"""
import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

GIB = 1024**3
CASES = ('p032d02', 'p064d02', 'p128d02', 'p256d02', 'p512d02', 'p128d04', 'p128d08', 'p128d16')


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def write(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def percent(model, hardware):
    check(hardware > 0, 'Invalid hardware denominator')
    return 100 * (model / hardware - 1)


def compare(candidate_ns, candidate_read, candidate_write, unknown, baseline, hardware):
    hardware_ns = hardware['natural_cuda_event_ms']['median'] * 1e6
    hr, hw = hardware['dram_read_bytes']['median'], hardware['dram_write_bytes']['median']
    hardware_bw = (hr + hw) / hardware_ns
    baseline_ns = sum(p['model_ns'] for p in baseline['phase_totals'].values())
    values = {
        'time': (candidate_ns, baseline_ns, hardware_ns),
        'read': (candidate_read, baseline['known_read_bytes'], hr),
        'write': (candidate_write, baseline['known_write_bytes'], hw),
        'bandwidth': ((candidate_read + candidate_write + unknown) / candidate_ns,
                      baseline['total_boundary_bytes'] / baseline_ns, hardware_bw),
    }
    metrics = {}
    for name, (new, old, observed) in values.items():
        ne, oe = percent(new, observed), percent(old, observed)
        metrics[name] = dict(candidate=new, baseline=old, hardware=observed,
                             candidate_error_percent=ne, baseline_error_percent=oe,
                             no_worse=abs(ne) <= abs(oe) + 1e-9)
    return dict(metrics=metrics, no_worse=all(x['no_worse'] for x in metrics.values()))


def canonical_views(View, m, k, n):
    shapes = {'A': (0, m, k), 'B': (GIB, k, n), 'C': (2 * GIB, m, n)}
    return {key: View(base, rows, columns, columns * 2, 2, base, rows * columns * 2)
            for key, (base, rows, columns) in shapes.items()}


class Engine:
    def __init__(self, root, output):
        self.root, self.output = root, output
        sys.path.insert(0, str(root))
        from hbfsim_client import SimulationSession, Transaction, ResolvedSystemConfig
        self.Session, self.Tx = SimulationSession, Transaction
        self.binary = root / 'build/hbfsim'
        self.configs = (root / 'configs/systems/eight-stack-baseline.cfg',
                        root / 'configs/overlays/dram/rtx4000-ada-gddr6.cfg')
        self.cfg = ResolvedSystemConfig.load(self.configs).resolve(self.binary, enable_hbf=False)
        identity = subprocess.run([str(self.binary), '--system-config', str(self.configs[1]),
                                  '--enable-hbf', 'false'], input='', capture_output=True,
                                  text=True, check=True)
        self.ready = json.loads(identity.stdout.splitlines()[0])
        check(self.ready['dram_memory_type'] == 'GDDR6' and self.ready['hbm_standard'] == 'GDDR6-aggregate-v1',
              'Backend is not the explicit GDDR6 aggregate implementation')
        self.expected_head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        self.checked = 0
        self.source = None

    def execute(self, transactions, frontier, audit):
        # Compilation-style isolated operator service, consistent with the
        # official independent operator aggregation; no cross-op cache/state.
        with self.Session(simulator_path=self.binary, system_config=self.cfg,
                          enable_hbm=True, enable_hbf=False, read_timeout_s=120) as session:
            result = session.run(transactions, frontier=frontier, retain=(), completions=True)
            finishes = {row.id: row.finish_ns for row in result.completions}
            check(len(finishes) == sum(t.target != 'BARRIER' for t in transactions), 'Incomplete memory completions')
            audit(transactions, finishes, frontier, result.blocking_finish_ns)
            latency = result.receipt['transaction_latency_by_target']['HBM']
            device = result.receipt['device_delta']['hbm']
            op_by_id = {t.id: t.op for t in transactions}
            for direction, op in [('read', 'R'), ('write', 'W')]:
                logical = sum(t.bytes for t in transactions if t.target == 'HBM' and t.op == op)
                physical = sum(c.physical_bytes for c in result.completions
                               if op_by_id[c.id] == op)
                check(logical == latency[direction]['logical_bytes'], 'Logical byte closure failure')
                check(physical == latency[direction]['physical_bytes'] == device[direction + '_bytes'],
                      'Backend service byte closure failure')
            check(result.finish_ns == result.blocking_finish_ns, 'Detached work remains after final frontier')
            saved = dict(ns=result.elapsed_ns,
                         known_read_bytes=latency['read']['logical_bytes'], known_write_bytes=latency['write']['logical_bytes'],
                         service_read_bytes=device['read_bytes'], service_write_bytes=device['write_bytes'],
                         transactions=result.receipt['memory_transactions'],
                         transaction_digest=result.receipt['transaction_trace_sha256'],
                         completion_digest=result.receipt['transaction_completions_digest']['sha256'],
                         bus_busy_work_ns=device['bus_busy_ns'], service_busy_work_ns=device['service_busy_ns'])
        source = session.source_receipt()
        check(source['engine_source']['git_commit'] == self.expected_head and not source['engine_source']['git_dirty'],
              'Executable source identity differs from clean target checkout')
        check(source['final_measurement']['result'] == 'stopped', 'Missing clean terminal drain receipt')
        if self.source is None:
            self.source = source
            write(self.output / 'backend-source-receipt.json', source)
        self.checked += 1
        return saved


class Compiler:
    def __init__(self, engine, q, mat, soft, cost, selected, coupling, View):
        self.engine, self.q, self.mat, self.soft, self.cost = engine, q, mat, soft, cost
        self.selected, self.coupling, self.View = selected, coupling, View
        self.cache, self.results = {}, {}
        self.clock = cost.device.compute_module.clock_freq

    def memory_only(self, accesses):
        txs, frontier = [], ()
        for index, (direction, address, count) in enumerate(accesses):
            name = f'access{index}'
            txs.append(self.engine.Tx(id=name, target='HBM', op=direction, addr=address, bytes=count,
                                      issue_ns=0, dependencies=frontier))
            frontier = (name,)
        return self.engine.execute(txs, set(frontier), self.coupling.audit_completion_dependencies)

    def prototype(self, m, k, n, record, kind='matmul'):
        cache_key = json.dumps([kind, m, k, n, record['mapping']], sort_keys=True)
        if cache_key in self.cache:
            return copy.deepcopy(self.cache[cache_key])
        if kind == 'softmax':
            op = self.soft.Softmax(self.cost.dtype)
            op(self.cost.Tensor([m, n], self.cost.dtype))
        else:
            op = self.mat.Matmul(self.cost.dtype)
            op(self.cost.Tensor([m, k], self.cost.dtype), self.cost.Tensor([k, n], self.cost.dtype))
        if record['mapping'] is None:
            op.compile_and_simulate(self.cost.device, 'heuristic-GPU')
            check(math.isclose(op.latency, record['seconds'], abs_tol=1e-12), 'Frozen shortcut time changed')
            views = canonical_views(self.View, m, k, n)
            spans = [('R', views['A'].base, m * k * 2), ('R', views['B'].base, k * n * 2),
                     ('W', views['C'].base, m * n * 2)]
            result = self.memory_only(spans)
            compute_ns = 2 * m * k * n * 1e9 / (self.cost.device.compute_module.total_vector_flops_per_cycle * self.clock)
            result.update(ns=max(result['ns'], compute_ns), compute_only_ns=compute_ns,
                          timing_policy='OFFICIAL_SHORTCUT_MAX_COMPUTE_GDDR_SERVICE')
        else:
            mapping = (self.soft.Softmax.Mapping if kind == 'softmax' else self.mat.Matmul.Mapping)(**record['mapping'])
            cycles = op.simulate(op.computational_graph, mapping, self.cost.device)
            check(math.isclose(cycles / self.clock, record['seconds'], abs_tol=1e-12), 'Frozen selected-mapping time changed')
            check(math.isclose(self.coupling.analytical_cycles(op.execution_stages), cycles, abs_tol=1e-6),
                  'Original stage recurrence does not close')
            views = canonical_views(self.View, m, k, n)
            if kind == 'softmax':
                views['A'] = self.View(0, m, n, n * 2, 2, 0, m * n * 2)
            txs, frontier = self.coupling.lower_stages(self.engine.Tx, op.execution_stages,
                op.main_memory_events, self.clock, lambda event: views[event['tensor']].requests(event))
            result = self.engine.execute(txs, frontier, self.coupling.audit_completion_dependencies)
            result.update(compute_only_ns=sum(s['compute_cycles'] for s in op.execution_stages) * 1e9 / self.clock,
                          timing_policy='SELECTED_MAPPER_MAIN_IO_REPLACED_BY_GDDR_COMPLETIONS')
        check(result['known_read_bytes'] == op.memory_accounting['main_read_bytes'] and
              result['known_write_bytes'] == op.memory_accounting['main_write_bytes'], 'Mapper transfer bytes changed')
        result['original_ns'] = record['seconds'] * 1e9
        self.cache[cache_key] = copy.deepcopy(result)
        self.results[cache_key] = result
        print('PROTOTYPE_COMPLETE', len(self.cache), kind, m, k, n, flush=True)
        return copy.deepcopy(result)

    def shape(self, key):
        record = self.selected[tuple(key)]
        if key[0] == 'matmul':
            _, m, k, n = key
            return self.prototype(m, k, n, record)
        if key[0] == 'softmax':
            _, b, m, n = key
            return self.prototype(b * m, 0, n, dict(record, seconds=record['seconds']), 'softmax')
        _, b, m, k, n = key
        candidate = record['selected_candidate']
        result = self.prototype(m, k if candidate == 0 else k * b, n, record['batch_candidates'][candidate])
        if candidate == 0:
            for field in ('ns', 'known_read_bytes', 'known_write_bytes', 'service_read_bytes', 'service_write_bytes',
                          'transactions', 'compute_only_ns', 'bus_busy_work_ns', 'service_busy_work_ns'):
                result[field] *= b
            result['batch_policy'] = 'OFFICIAL_INDEPENDENT_BATCH_LATENCY_MULTIPLIER'
        else:
            result['batch_policy'] = 'OFFICIAL_K_CONCATENATED_PERFORMANCE_SURROGATE_NOT_NATIVE_BATCH_GRAPH'
        unknown = record['accounting'].get('unclassified_extra_io_bytes', 0)
        result['unclassified_io_bytes'] = unknown
        result['unclassified_analytical_ns'] = unknown * 1e9 / self.cost.device.io_module.bandwidth
        result['ns'] += result['unclassified_analytical_ns']
        check(result['known_read_bytes'] == record['accounting']['main_read_bytes'] and
              result['known_write_bytes'] == record['accounting']['main_write_bytes'], 'Batch selected-traffic mismatch')
        return result

    def vector(self, operator, flops):
        accesses = operator.reads + operator.writes
        key = json.dumps(['vector', operator.name, flops,
                         [(x.kind, re.sub(r'\.layers\.\d+\.', '.layers.N.', x.label), x.bytes) for x in accesses]])
        if key not in self.cache:
            result = self.memory_only([(x.kind[0].upper(), 4 * GIB + i * 256 * 1024**2, x.bytes)
                                       for i, x in enumerate(accesses)])
            compute_ns = flops * 1e9 / (self.cost.device.compute_module.total_vector_flops_per_cycle * self.clock)
            result.update(ns=max(result['ns'], compute_ns), compute_only_ns=compute_ns,
                          timing_policy='QWEN_ADAPTER_MAX_COMPUTE_GDDR_SERVICE')
            self.cache[key], self.results[key] = result, result
        return copy.deepcopy(self.cache[key])


def frozen_plan(q, model, ledger):
    class FrozenCost:
        def __init__(self):
            self.calls = []

        def take(self, key, family, flops):
            row = ledger[len(self.calls)]
            self.calls.append((key, flops))
            return q.Timing(family, flops, row['model_ns'])

        def matmul(self, m, k, n, bias=False, family='matmul'):
            return self.take(('matmul', m, k, n), family, 2 * m * k * n + (m * n if bias else 0))

        def batched_matmul(self, b, m, k, n, family):
            return self.take(('batch', b, m, k, n), family, 2 * b * m * k * n)

        def softmax(self, b, m, n):
            return self.take(('softmax', b, m, n), 'softmax', b * m * n * 5)

        def vector(self, elements, ops_per_element, family, bytes_moved=0):
            return self.take(None, family, elements * ops_per_element)

    frozen = FrozenCost()
    plan = q.build_plan(model, frozen)
    check(len(plan.operators) == len(ledger), 'Full graph is incomplete')
    for op, row in zip(plan.operators, ledger):
        check((op.index, op.phase, op.layer, op.name) == (row['index'], row['phase'], row['layer'], row['name']),
              'Frozen graph identity changed')
    return plan, frozen.calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-root', type=pathlib.Path, required=True)
    parser.add_argument('--output-root', type=pathlib.Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='One layer and two phases before full-matrix execution')
    args = parser.parse_args()
    adapter = pathlib.Path(__file__).resolve().parent
    repo = adapter.parents[1]
    matrix = adapter / 'results/official-inference-matrix-20261007-r3'
    semantic = adapter / 'results/semantic-traffic-20261008-r1'
    validation = json.loads((semantic / 'receipt.json').read_text())
    inputs = dict(validation['input_sha256'])
    for path in [semantic / 'receipt.json', semantic / 'selected-operands.json', pathlib.Path(__file__),
                 adapter / 'mapper_event_coupling.py', adapter / 'mapper_tensor_addresses.py']:
        inputs[str(path)] = sha(path)
    backend = args.backend_root.resolve()
    for path in [backend / 'build/hbfsim', backend / 'configs/overlays/dram/rtx4000-ada-gddr6.cfg',
                 backend / 'configs/systems/eight-stack-baseline.cfg',
                 backend / 'evidence/hardware/rtx4000_ada_gddr6/measurement.json']:
        inputs[str(path)] = sha(path)
    for path, digest in inputs.items():
        check(sha(path) == digest, 'Input identity changed: ' + path)
    root = args.output_root.resolve()
    check(not root.exists(), 'Fresh output required; historical data must be preserved')
    root.mkdir(parents=True)
    for path in [pathlib.Path(__file__), adapter / 'mapper_event_coupling.py', adapter / 'mapper_tensor_addresses.py']:
        shutil.copy2(path, root / path.name)
    receipt = dict(status='RUNNING', scope='FIXED_MAPPER_ANALYTICAL_GDDR_COMPLETION_EXTENSION_MODEL_ESTIMATE',
                   source_matrix=str(matrix), backend_root=str(backend), input_sha256=inputs,
                   raw_trace_exported=False, primary_results_modified=False, completed_cases=0,
                   canonical_address_proxy=True, operator_template_reset_policy='isolated service compilation, official per-operator aggregation',
                   batch_surrogate_policy='frozen official selection; extra unclassified IO kept analytically once')
    write(root / 'receipt.json', receipt)
    sys.path.insert(0, str(repo))
    q = load('gddr_frozen_qwen', matrix / 'source/qwen_hbfsim_cosim.py')
    profile = matrix / 'source/RTX4000Ada_xmu_profile_v3.json'
    cost = q.LLMCompassCostModel(repo, profile)
    mat = load('gddr_frozen_matmul', matrix / 'source/mapper_matmul.py')
    soft = load('gddr_frozen_softmax', matrix / 'source/mapper_softmax.py')
    coupling = load('gddr_coupling', adapter / 'mapper_event_coupling.py')
    View = load('gddr_views', adapter / 'mapper_tensor_addresses.py').MatrixView
    work = pathlib.Path(tempfile.mkdtemp(prefix='llmcompass-gddr-'))
    shutil.copytree(repo / 'systolic_array_model', work / 'systolic_array_model',
                    ignore=shutil.ignore_patterns('temp', '*.gz', '*.npy'))
    (work / 'systolic_array_model/temp').mkdir()
    os.chdir(work)
    selected = {tuple(row['key']): row for row in json.loads((semantic / 'selected-operands.json').read_text())}
    engine = Engine(backend, root)
    compiler = Compiler(engine, q, mat, soft, cost, selected, coupling, View)
    baseline = {row['case']: row for row in json.loads((matrix / 'comparison.json').read_text())}
    base_model = q.ModelSpec.from_json(semantic / 'qwen25_1p5b.json')
    rows = []
    for case in (CASES[:1] if args.smoke else CASES):
        model = dataclasses.replace(base_model, prefill_tokens=int(case[1:4]), decode_steps=int(case.split('d')[1]))
        ledger = json.loads((matrix / case / 'operators.json').read_text())
        plan, calls = frozen_plan(q, model, ledger)
        operators, phases = [], {}
        for op, original, (key, flops) in zip(plan.operators, ledger, calls):
            if args.smoke and (op.layer not in (None, 0) or op.phase not in ('prefill', 'decode_1')):
                continue
            result = compiler.vector(op, flops) if key is None else compiler.shape(key)
            if original['bias_adapter_seconds']:
                bias_bytes = sum(x.bytes for x in op.reads if x.label.endswith('.bias'))
                bias_key = json.dumps(['bias', bias_bytes])
                if bias_key not in compiler.cache:
                    compiler.cache[bias_key] = compiler.memory_only([('R', 3 * GIB, bias_bytes)])
                bias = compiler.cache[bias_key]
                result['ns'] += bias['ns'] + original['bias_adapter_seconds'] * 1e9
                for field in ('known_read_bytes', 'service_read_bytes', 'transactions'):
                    result[field] += bias[field]
            result.setdefault('unclassified_io_bytes', 0)
            check(result['known_read_bytes'] == original['known_read_bytes'] and
                  result['known_write_bytes'] == original['known_write_bytes'] and
                  result['unclassified_io_bytes'] == original['unclassified_io_bytes'], 'Operator semantic traffic changed')
            row = dict(index=op.index, phase=op.phase, layer=op.layer, name=op.name, path=original['path'],
                       baseline_ns=original['model_ns'], model_ns=result['ns'],
                       **{k: v for k, v in result.items() if k != 'ns'})
            operators.append(row)
            phase = phases.setdefault(op.phase, dict(model_ns=0, known_read_bytes=0, known_write_bytes=0,
                service_read_bytes=0, service_write_bytes=0, unclassified_io_bytes=0, operators=0))
            for field in phase:
                phase[field] += 1 if field == 'operators' else row[field]
        total = {field: sum(p[field] for p in phases.values()) for field in next(iter(phases.values()))}
        hardware_path = pathlib.Path('/home/xmu/nvidiagds/simulators/wgslogs/qwen-ncu/qwen1.5-series/collection-r2/cases') / case / 'summary.json'
        hw = json.loads(hardware_path.read_text())
        gate = None if args.smoke else compare(total['model_ns'], total['known_read_bytes'], total['known_write_bytes'],
            total['unclassified_io_bytes'], baseline[case], hw['rois']['full'])
        if not args.smoke:
            check(total['operators'] == 507 * (model.decode_steps + 1), 'Full operator denominator mismatch')
            for field in ('known_read_bytes', 'known_write_bytes', 'unclassified_io_bytes'):
                check(total[field] == baseline[case][field], 'Full directional byte closure failure')
        row = dict(case=case, **total, phase_totals=phases, comparison=gate)
        rows.append(row)
        out = root / case
        out.mkdir()
        write(out / 'operators.json', operators)
        write(out / 'summary.json', row)
        write(root / 'comparison.json', rows)
        write(root / 'prototype-summary.json', compiler.results)
        receipt.update(completed_cases=len(rows), checked_backend_templates=engine.checked,
                       completed_operators=sum(r['operators'] for r in rows))
        write(root / 'receipt.json', receipt)
        print('CASE_COMPLETE', case, total['model_ns'] / 1e6, 'NO_WORSE', None if gate is None else gate['no_worse'], flush=True)
    for path, digest in inputs.items():
        check(sha(path) == digest, 'Input changed during evaluation: ' + path)
    receipt.update(status='PASS_SMOKE_ACCOUNTING_NOT_FULL_MATRIX' if args.smoke else 'PASS_EIGHT_CASE_GDDR_EXTENSION_ACCOUNTING',
        no_regression_gate=None if args.smoke else ('PASS' if all(r['comparison']['no_worse'] for r in rows) else 'FAIL'),
        terminal_backend_source=engine.source['engine_source'], isolated_working_directory=str(work),
        comparison_sha256=sha(root / 'comparison.json'), prototype_summary_sha256=sha(root / 'prototype-summary.json'))
    write(root / 'receipt.json', receipt)
    print(receipt['status'], 'NO_REGRESSION_GATE', receipt['no_regression_gate'], flush=True)


if __name__ == '__main__':
    main()
