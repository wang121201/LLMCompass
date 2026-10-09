"""Static Qwen P128D8 full-graph batches; official time, aggregate output only.

No scheduler, fitted efficiency, numerical CUDA execution, or HBFSim time is
introduced. A request owns an independent KV region; learned weights are shared.
"""
import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TARGET_BATCHES = (2, 4, 8, 16, 32)
PROFILE_SHA256 = 'd7c2659d32b14c4cf0f5da876be09375a810bcdbfdc7b03c4b820cf2fc4288b1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def validate_request_regions(plan):
    """Validate every request/layer key and value lifetime, not just a byte sum."""
    model = plan.model
    capacity = (model.prefill_tokens + model.decode_steps) * model.kv_hidden_size * model.bytes_per_element
    regions = []
    bases = {}
    for op in plan.operators:
        if op.phase == 'prefill' and op.name == 'kv_append':
            for label in ('key_cache', 'value_cache'):
                group = [a for a in op.writes if a.label == label]
                assert len(group) == model.batch_size
                bases[op.layer, label] = [a.address for a in group]
                regions.extend((a.address, a.address + capacity) for a in group)
    assert len(regions) == 2 * model.layers * model.batch_size
    ordered = sorted(regions)
    assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))
    step_bytes = model.kv_hidden_size * model.bytes_per_element
    total_append = 0
    for op in plan.operators:
        step = 0 if op.phase == 'prefill' else int(op.phase.split('_')[1])
        context = model.prefill_tokens + step
        start = 0 if step == 0 else context - 1
        count = model.prefill_tokens if step == 0 else 1
        if op.name == 'kv_append':
            for label in ('key_cache', 'value_cache'):
                accesses = [a for a in op.writes if a.label == label]
                assert len(accesses) == model.batch_size
                for base, access in zip(bases[op.layer, label], accesses):
                    assert access.address == base + start * step_bytes
                    assert access.bytes == count * step_bytes
                    total_append += access.bytes
        elif op.name in ('attention_score', 'attention_value'):
            label = 'key_cache' if op.name == 'attention_score' else 'value_cache'
            accesses = [a for a in op.reads if a.label == label]
            assert len(accesses) == model.batch_size
            for base, access in zip(bases[op.layer, label], accesses):
                assert access.address == base and access.bytes == context * step_bytes
        elif op.name == 'lm_head':
            inputs = [a for a in op.reads if a.label == 'last_hidden']
            assert len(inputs) == model.batch_size
            tokens = model.prefill_tokens if step == 0 else 1
            assert all(a.bytes == model.hidden_size * model.bytes_per_element for a in inputs)
            assert all(b.address - a.address == tokens * model.hidden_size * model.bytes_per_element
                       for a, b in zip(inputs, inputs[1:]))
    expected = 2 * model.layers * model.batch_size * (model.prefill_tokens + model.decode_steps) * step_bytes
    assert total_append == expected
    return dict(request_count=model.batch_size, kv_regions=len(regions), independent_kv=True,
                kv_payload_bytes=expected, append_bytes=total_append, last_token_rows_checked=True)


class Compiler:
    """Reuse the existing selected-mapper event accounting and official parity rule."""
    def __init__(self, q, mat, soft, semantic, profile):
        self.q, self.mat, self.soft, self.semantic = q, mat, soft, semantic
        self.base = q.LLMCompassCostModel(REPO, profile)
        self.device, self.dtype, self.Tensor = self.base.device, self.base.dtype, self.base.Tensor
        self.cache, self.inner_cache, self.observed = {}, {}, []
        original_compile = mat.Matmul.compile_and_simulate

        def observed_compile(op, device, mode='exhaustive'):
            key = (op.M, op.K, op.N, mode)
            if key not in self.inner_cache:
                seconds = original_compile(op, device, mode)
                if op.best_mapping is not None:
                    cycles = op.simulate(op.computational_graph, op.best_mapping, device)
                    assert math.isclose(seconds, cycles / device.compute_module.clock_freq, rel_tol=1e-12, abs_tol=1e-15)
                operands = semantic.operand_counts(op.main_memory_events, op.data_type.word_size)
                assert sum(operands['read'].values()) == op.memory_accounting['main_read_bytes']
                assert sum(operands['write'].values()) == op.memory_accounting['main_write_bytes']
                self.inner_cache[key] = (dict(seconds=seconds, operands=operands), op)
            record, cached = self.inner_cache[key]
            if op is not cached:
                for name in ('best_mapping', 'best_cycle_count', 'best_latency', 'latency', 'memory_accounting', 'look_up_table', 'main_memory_events'):
                    if hasattr(cached, name):
                        setattr(op, name, getattr(cached, name))
            self.observed.append(copy.deepcopy(record))
            return record['seconds']
        mat.Matmul.compile_and_simulate = observed_compile

    def shape(self, key):
        if key in self.cache:
            return self.cache[key]
        print(json.dumps(dict(event='compile_start', shape=key)), flush=True)
        self.observed.clear()
        kind = key[0]
        if kind == 'matmul':
            _, m, k, n = key
            op = self.mat.Matmul(self.dtype)
            inputs = [self.Tensor([m,k],self.dtype), self.Tensor([k,n],self.dtype)]
        elif kind == 'batch':
            _, b, m, k, n = key
            op = self.mat.BatchedMatmul(self.dtype)
            inputs = [self.Tensor([b,m,k],self.dtype), self.Tensor([b,k,n],self.dtype)]
        else:
            _, b, m, n = key
            op = self.soft.Softmax(self.dtype)
            inputs = [self.Tensor([b,m,n],self.dtype)]
        op(*inputs)
        seconds = op.compile_and_simulate(self.device, 'heuristic-GPU')
        selected = getattr(op, 'selected_accounting_candidate', None)
        if kind == 'batch':
            assert len(self.observed) == 2
            operands = copy.deepcopy(self.observed[selected]['operands'])
            if selected == 0:
                for direction in operands:
                    for operand in operands[direction]:
                        operands[direction][operand] *= b
        elif kind == 'matmul':
            operands = copy.deepcopy(self.observed[-1]['operands'])
        else:
            operands = self.semantic.operand_counts(op.main_memory_events, self.dtype.word_size)
        accounting = copy.deepcopy(op.memory_accounting)
        assert sum(operands['read'].values()) == accounting['main_read_bytes']
        assert sum(operands['write'].values()) == accounting['main_write_bytes']
        original_class = {'matmul':self.base.Matmul, 'batch':self.base.BatchedMatmul, 'softmax':self.base.Softmax}[kind]
        original = original_class(self.dtype)
        original(*inputs)
        reference = original.compile_and_simulate(self.device, 'heuristic-GPU')
        assert math.isclose(seconds, reference, rel_tol=1e-12, abs_tol=1e-15), key
        self.cache[key] = dict(key=list(key), seconds=seconds, official_source_seconds=reference,
                               official_latency_parity=True, accounting=accounting,
                               selected_candidate=selected, operands=operands)
        print(json.dumps(dict(event='shape_compiled', shape=key, seconds=seconds)), flush=True)
        return self.cache[key]

    def cost(self):
        engine, q = self, self.q
        class OfficialBatchCost(q.LLMCompassCostModel):
            def __init__(self):
                self.calls = []
                self.device = engine.device
                self.timing_mode = 'llmcompass_roofline'
            def matmul(self, m, k, n, bias=False, family='matmul'):
                r = engine.shape(('matmul',m,k,n))
                cm = self.device.compute_module
                bias_s = m*n/(cm.total_vector_flops_per_cycle*cm.clock_freq) if bias else 0.0
                self.calls.append(dict(path='OFFICIAL_MATMUL', **r, bias_seconds=bias_s, bias=bias))
                return q.Timing(family, 2*m*k*n+(m*n if bias else 0), math.ceil((r['seconds']+bias_s)*1e9))
            def batched_matmul(self, b, m, k, n, family):
                r = engine.shape(('batch',b,m,k,n))
                self.calls.append(dict(path='OFFICIAL_BATCHED_MATMUL', **r, bias_seconds=0))
                return q.Timing(family, 2*b*m*k*n, math.ceil(r['seconds']*1e9))
            def softmax(self, b, m, n):
                r = engine.shape(('softmax',b,m,n))
                self.calls.append(dict(path='OFFICIAL_SOFTMAX', **r, bias_seconds=0))
                return q.Timing('softmax', b*m*n*5, math.ceil(r['seconds']*1e9))
            def vector(self, elements, ops_per_element, family, bytes_moved=0):
                timing = engine.base.vector(elements, ops_per_element, family, bytes_moved)
                self.calls.append(dict(path='QWEN_VECTOR_ADAPTER', accounting=None))
                return timing
        return OfficialBatchCost()


def evaluate_case(q, semantic, engine, base, batch):
    model = dataclasses.replace(base, batch_size=batch, prefill_tokens=128, decode_steps=8)
    cost = engine.cost()
    plan = q.build_plan(model, cost)
    regions = validate_request_regions(plan)
    assert len(plan.operators) == len(cost.calls) == 4563
    phases, families = {}, {}
    for op, call in zip(plan.operators, cost.calls):
        acc = call['accounting']
        if acc is None:
            read, write_bytes, extra = sum(a.bytes for a in op.reads), sum(a.bytes for a in op.writes), 0
        else:
            read, write_bytes, extra = acc['main_read_bytes'], acc['main_write_bytes'], acc.get('unclassified_extra_io_bytes',0)
            if call.get('bias'):
                read += sum(a.bytes for a in op.reads if a.label.endswith('.bias'))
        ledger = dict(path=call['path'], known_read_bytes=read, known_write_bytes=write_bytes, unclassified_io_bytes=extra)
        categories, _ = semantic.allocate_operator(op, ledger, call if acc else None)
        for group, key in [(phases,op.phase),(families,op.name)]:
            row = group.setdefault(key, dict(operators=0, model_ns=0, known_read_bytes=0, known_write_bytes=0,
                                            unclassified_io_bytes=0, semantic=semantic.empty_categories()))
            row['operators'] += 1
            row['model_ns'] += op.timing.compute_ns
            row['known_read_bytes'] += read
            row['known_write_bytes'] += write_bytes
            row['unclassified_io_bytes'] += extra
            semantic.add_categories(row['semantic'], categories)
    total = {key:sum(p[key] for p in phases.values()) for key in ['operators','model_ns','known_read_bytes','known_write_bytes','unclassified_io_bytes']}
    total['semantic'] = semantic.empty_categories()
    for row in phases.values():
        assert row['operators'] == 507
        semantic.add_categories(total['semantic'],row['semantic'])
    for direction, name in [('read_bytes','known_read_bytes'),('write_bytes','known_write_bytes'),('unknown_direction_bytes','unclassified_io_bytes')]:
        assert semantic.category_total(total['semantic'],direction) == total[name]
    assert sum(row['model_ns'] for row in families.values()) == total['model_ns']
    boundary = total['known_read_bytes'] + total['known_write_bytes'] + total['unclassified_io_bytes']
    decode_ns = sum(row['model_ns'] for phase,row in phases.items() if phase != 'prefill')
    return dict(case=f'b{batch:02d}p128d08', batch_size=batch, prefill_tokens=128, decode_steps=8,
                status='PASS_FULL_ANALYTICAL_GRAPH_NOT_HARDWARE_ACCEPTANCE', **total,
                model_ms=total['model_ns']/1e6, model_GBps=boundary/total['model_ns'],
                total_boundary_bytes=boundary, full_request_latency_ms=total['model_ns']/1e6,
                amortized_ms_per_request=total['model_ns']/1e6/batch,
                prefill_tokens_per_second=batch*128*1e9/phases['prefill']['model_ns'],
                decode_tokens_per_second=batch*8*1e9/decode_ns,
                requests_per_second=batch*1e9/total['model_ns'],
                request_regions=regions, phase_totals=phases, family_totals=families,
                hardware_comparison=None, hbfsim_in_model_time=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    args = parser.parse_args()
    profile = HERE/'RTX4000Ada_xmu_profile_v3.json'
    assert sha(profile) == PROFILE_SHA256, 'Use the frozen accepted Ada profile'
    versions = {name:importlib.metadata.version(name) for name in ['torch','numpy','pandas','scalesim']}
    assert versions['scalesim'] == '2.0.2', 'Use the frozen official ScaleSim release'
    subprocess.run(['git','-C',str(REPO),'diff','--exit-code','62321b1ee28ddbdba8a2eb7475d7caa30f75e8be',
                    '--','software_model','hardware_model','ae/figure5'],check=True,capture_output=True)
    root = args.output_root.resolve()
    assert not root.exists(), 'Use a fresh result root; preserve prior evidence'
    sys.path.insert(0,str(REPO))
    q = load('batch_qwen',HERE/'qwen_hbfsim_cosim.py')
    mat = load('batch_matmul',HERE/'mapper_matmul.py')
    soft = load('batch_softmax',HERE/'mapper_softmax.py')
    semantic = load('batch_semantic',HERE/'semantic_traffic_breakdown.py')
    root.mkdir(parents=True,exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix='llmcompass-static-batch-'))
    shutil.copytree(REPO/'systolic_array_model',work/'systolic_array_model',ignore=shutil.ignore_patterns('temp','*.gz','*.npy'))
    (work/'systolic_array_model/temp').mkdir()
    os.chdir(work)
    engine = Compiler(q,mat,soft,semantic,profile)
    assert engine.device.compute_module.core_count == 48
    assert engine.device.compute_module.clock_freq == 2175000000.0
    assert engine.device.io_module.bandwidth == 342e9
    manifest = dict(schema='LLMCOMPASS_ADA_STATIC_BATCH_V1',status='RUNNING',target_batches=list(TARGET_BATCHES),
                    baseline_batch=1, model='Qwen/Qwen2.5-1.5B-Instruct',prefill_tokens=128,decode_steps=8,
                    started_at_utc=datetime.now(timezone.utc).isoformat(),profile_sha256=sha(profile),
                    source_sha256={name:sha(HERE/name) for name in ['qwen_hbfsim_cosim.py','mapper_matmul.py','mapper_softmax.py','semantic_traffic_breakdown.py','evaluate_batch_inference.py']},
                    boundary='Static synchronized batch; all 28 layers; official chosen mapper; Qwen vector/bias adapters explicit.',
                    no_new_cache_or_scheduler=True,no_latency_seed=True,hbfsim_in_model_time=False,
                    hardware_collected=False,operator_stream_saved=False,trace_saved=False,working_directory=str(work),
                    python=sys.executable,dependency_versions=versions,
                    completed_cases=0)
    write(root/'manifest.json',manifest)
    rows=[]
    try:
        base=q.ModelSpec.from_json(HERE/'qwen25_1p5b.json')
        for batch in (1,*TARGET_BATCHES):
            row=evaluate_case(q,semantic,engine,base,batch)
            rows.append(row)
            write(root/'comparison.json',rows)
            write(root/'shapes.json',list(engine.cache.values()))
            manifest.update(completed_cases=len(rows),unique_shapes=len(engine.cache))
            write(root/'manifest.json',manifest)
            print(json.dumps(dict(event='case_completed',case=row['case'],model_ms=row['model_ms'],model_GBps=row['model_GBps'])),flush=True)
        manifest.update(status='PASS_STATIC_BATCH_FULL_GRAPH_ACCOUNTING',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        comparison_sha256=sha(root/'comparison.json'),shapes_sha256=sha(root/'shapes.json'),
                        verified_operators=sum(row['operators'] for row in rows),verified_phases=sum(len(row['phase_totals']) for row in rows))
        write(root/'manifest.json',manifest)
        print(manifest['status'],flush=True)
    except Exception as exc:
        manifest.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
        write(root/'manifest.json',manifest)
        raise


if __name__ == '__main__':
    main()
