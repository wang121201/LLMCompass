"""Aggregate-only native timing diagnosis; never export activation or trace data.

Reuses the pinned native workload unchanged. CUDA-event partitions and in-memory
profiler kernel groups are diagnostics, not additive model corrections.
"""
import argparse
import collections
import contextlib
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
import threading
import time
import ast

HW = pathlib.Path('/home/xmu/nvidiagds/simulators/wgslogs/qwen-ncu/qwen1.5-series/collection-r2')
sys.path.insert(0, str(HW/'package'))
import matrix_common as common
import matrix_workload as workload
import shared_phase_execution as execution
from native_host import Boundary

def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    import torch
    import sglang
    import sglang.bench_one_batch as bo
    import pynvml
    pynvml.nvmlInit()
    nvml_gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    assert pynvml.nvmlDeviceGetUUID(nvml_gpu) == 'GPU-18ace299-5348-e6e4-d48c-1ee5a602859b'
    sources = common.native_source_inventory(pathlib.Path(sglang.__file__).resolve().parent)
    workload.check_packages()
    workload.check_native_sources(sources)
    (args.output/'input-identities.json').write_text(json.dumps(dict(
        driver_sha256=digest(__file__),shared_execution_sha256=digest(execution.__file__),
        native_sources=sources,gpu_uuid=pynvml.nvmlDeviceGetUUID(nvml_gpu)),indent=2)+'\n')
    pathlib.Path(args.output/'diagnose_inference_timing.py').write_bytes(pathlib.Path(__file__).read_bytes())
    frozen0 = workload.contract('qwen25_1p5b', 32, 2)
    server = bo.ServerArgs(model_path=frozen0['model'], dtype='bfloat16', load_format='safetensors',
        device='cuda', tp_size=1, pp_size=1, attention_backend='flashinfer',
        disable_cuda_graph=True, cuda_graph_max_bs=1, enable_torch_compile=False,
        disable_overlap_schedule=True, disable_radix_cache=True, mem_fraction_static=0.90,
        max_total_tokens=frozen0['max_total_tokens'], max_running_requests=1,
        random_seed=0, cpu_offload_gb=0)
    bo._set_envs_and_config(server)
    runner, _ = bo.load_model(server, bo.PortArgs.init_new(server), 0)
    assert runner.cuda_graph_runner is None
    assert type(runner.model).__name__ == frozen0['model_class']
    assert len(runner.model.model.layers) == 28
    original_forward, original_sample = runner.forward, runner.sample
    mode = {'events': False, 'profile': False, 'partitions': []}
    clock_state = {'phase': None, 'rows': []}
    stop_sampler = threading.Event()
    def sample_clocks():
        while not stop_sampler.wait(0.002):
            phase = clock_state['phase']
            if phase:
                clock_state['rows'].append((phase,
                    pynvml.nvmlDeviceGetClockInfo(nvml_gpu, pynvml.NVML_CLOCK_SM),
                    pynvml.nvmlDeviceGetClockInfo(nvml_gpu, pynvml.NVML_CLOCK_MEM)))
    sampler = threading.Thread(target=sample_clocks, daemon=True)
    sampler.start()
    class ClockBoundary(Boundary):
        def before(self, phase):
            super().before(phase)
            clock_state['phase'] = phase
        def after(self, phase):
            super().after(phase)
            clock_state['phase'] = None

    def boundary_call(name, original, *a, **kw):
        if not mode['events'] and not mode['profile']:
            return original(*a, **kw)
        pair = None
        if mode['events']:
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            pair[0].record()
        scope = torch.profiler.record_function('diagnostic/'+name) if mode['profile'] else contextlib.nullcontext()
        with scope:
            value = original(*a, **kw)
        if pair:
            pair[1].record()
            mode['partitions'].append((name, pair))
        return value

    runner.forward = lambda *a, **kw: boundary_call('forward', original_forward, *a, **kw)
    runner.sample = lambda *a, **kw: boundary_call('sampling', original_sample, *a, **kw)
    wrapped = []
    def install_groups():
        targets = [('embedding', runner.model.model.embed_tokens), ('final_norm', runner.model.model.norm),
                   ('logits', runner.model.logits_processor)]
        for layer in runner.model.model.layers:
            targets.extend([('input_norm', layer.input_layernorm), ('post_norm', layer.post_attention_layernorm),
                ('qkv', layer.self_attn.qkv_proj), ('rope', layer.self_attn.rotary_emb),
                ('attention', layer.self_attn.attn), ('o_proj', layer.self_attn.o_proj),
                ('gate_up', layer.mlp.gate_up_proj), ('silu_mul', layer.mlp.act_fn),
                ('down', layer.mlp.down_proj)])
        for name, module in targets:
            old = module.forward
            def wrap(*a, _old=old, _name=name, **kw):
                with torch.profiler.record_function('diagnostic/group/'+_name):
                    return _old(*a, **kw)
            module.forward = wrap
            wrapped.append((module, old))

    def run(frozen, profiler=None, profile_scope='full'):
        clock_state['rows'] = []
        fixed = [torch.tensor([x], device=runner.device, dtype=torch.int64) for x in frozen['decode_input_ids']]
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in frozen['phases']]
        mode['partitions'] = []
        execution.execute(torch, bo, runner, frozen, fixed,
            ClockBoundary(profile_scope, frozen['phases'],
                profiler.start if profiler is not None else lambda: None,
                profiler.stop if profiler is not None else lambda: None), events=events)
        phases = {name: pair[0].elapsed_time(pair[1]) for name, pair in zip(frozen['phases'], events)}
        partitions = mode['partitions'][2*len(frozen['phases']):] if mode['events'] else []
        assert not mode['events'] or len(partitions) == 2*len(frozen['phases'])
        windows = {}
        for i, phase in enumerate(frozen['phases']):
            if mode['events']:
                (fn, fp), (sn, sp) = partitions[2*i:2*i+2]
                assert (fn, sn) == ('forward', 'sampling')
                start, stop = events[i]
                windows[phase] = dict(prepare_ms=start.elapsed_time(fp[0]),
                    forward_ms=fp[0].elapsed_time(fp[1]), between_ms=fp[1].elapsed_time(sp[0]),
                    sampling_ms=sp[0].elapsed_time(sp[1]), tail_ms=sp[1].elapsed_time(stop))
                assert abs(sum(windows[phase].values())-phases[phase]) < 0.005
        clocks = {}
        for phase in frozen['phases']:
            samples = [r for r in clock_state['rows'] if r[0] == phase]
            clocks[phase] = dict(samples=len(samples),
                sm_MHz=dict(min=min(r[1] for r in samples),median=statistics.median(r[1] for r in samples),max=max(r[1] for r in samples)),
                memory_MHz=dict(min=min(r[2] for r in samples),median=statistics.median(r[2] for r in samples),max=max(r[2] for r in samples))) if samples else None
        return dict(phases=phases, partitions=windows, clocks=clocks)

    cases = [(32,2),(64,2),(128,2),(256,2),(512,2),(128,4),(128,8),(128,16)]
    rows = []
    device_rows=[]
    prior=HW.parent/'timing-diagnosis-20261007-r7'
    if (prior/'device-activity-windows.json').exists():
        prior_identity=json.loads((prior/'input-identities.json').read_text())
        assert prior_identity['native_sources']==sources
        assert prior_identity['shared_execution_sha256']==digest(execution.__file__)
        assert digest(prior/'diagnose_inference_timing.py')==prior_identity['driver_sha256']
        def control_ast(path):
            tree=ast.parse(pathlib.Path(path).read_text())
            names={'run','boundary_call','sample_clocks','ClockBoundary'}
            return {node.name:ast.dump(node,include_attributes=False) for node in ast.walk(tree)
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names}
        assert control_ast(prior/'diagnose_inference_timing.py')==control_ast(__file__)
        rows=json.loads((prior/'native-partitions.json').read_text())
        device_rows=json.loads((prior/'device-activity-windows.json').read_text())
        assert len(rows)==8 and len(device_rows)==18
        (args.output/'reuse-manifest.json').write_text(json.dumps(dict(source_root=str(prior),
            source_driver_sha256=prior_identity['driver_sha256'],unchanged_control_ast=True,
            native_partitions_sha256=digest(prior/'native-partitions.json'),
            device_activity_sha256=digest(prior/'device-activity-windows.json'),
            completed_baseline_cases=len(rows),completed_device_activity_windows=len(device_rows)),indent=2)+'\n')
        (args.output/'native-partitions.json').write_text(json.dumps(rows,indent=2)+'\n')
        (args.output/'device-activity-windows.json').write_text(json.dumps(device_rows,indent=2)+'\n')
    for p, d in cases:
        if any(row['case']==f'p{p:03d}d{d:02d}' for row in rows):continue
        frozen = workload.contract('qwen25_1p5b', p, d)
        baseline = [run(frozen) for _ in range(5)]
        mode['events'] = True
        partitioned = [run(frozen) for _ in range(5)]
        mode['events'] = False
        case = f'p{p:03d}d{d:02d}'
        archived = json.loads((HW/'cases'/case/'summary.json').read_text())
        row = dict(case=case, baseline_full_ms=statistics.median(sum(x['phases'].values()) for x in baseline),
            event_partition_full_ms=statistics.median(sum(x['phases'].values()) for x in partitioned),
            archived_full_ms=archived['rois']['full']['natural_cuda_event_ms']['median'],
            baseline_phase_ms={ph:statistics.median(x['phases'][ph] for x in baseline) for ph in frozen['phases']},
            native_partitions={ph:{k:statistics.median(x['partitions'][ph][k] for x in partitioned)
                for k in partitioned[0]['partitions'][ph]} for ph in frozen['phases']},
            raw_phase_samples=baseline, workload_sha256=frozen['sha256'])
        rows.append(row)
        (args.output/'native-partitions.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ['raw_phase_samples','native_partitions']}), flush=True)

    # Low-instrumentation, one-phase device activity capture. Each event interval
    # and its activity union come from the same invocation; no semantic hooks.
    for p in (32,512):
      for scope,phase in [('Prefill','Prefill'),('D1','Decode1'),('D2','Decode2')]:
        for repeat in range(3):
            if any(x['case']==f'p{p:03d}d02' and x['phase']==phase and x['repeat']==repeat for x in device_rows):continue
            frozen=workload.contract('qwen25_1p5b',p,2)
            profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,profile_memory=False,with_stack=False)
            timings=run(frozen,profiler=profiler,profile_scope=scope)
            activity=[e for e in profiler.profiler.kineto_results.events()
                if e.device_type()==torch.autograd.DeviceType.CUDA and not e.is_user_annotation()]
            intervals=sorted((e.start_ns(),e.end_ns()) for e in activity);merged=[]
            for lo,hi in intervals:
                if merged and lo<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],hi)
                else:merged.append([lo,hi])
            union_ms=sum(hi-lo for lo,hi in merged)/1e6
            assert union_ms<=timings['phases'][phase]+0.005
            device_rows.append(dict(case=f'p{p:03d}d02',phase=phase,repeat=repeat,
                cuda_event_ms=timings['phases'][phase],gpu_busy_union_ms=union_ms,
                outside_recorded_activity_ms=timings['phases'][phase]-union_ms,
                device_activity_count=len(activity),kernel_names=dict(collections.Counter(e.name() for e in activity)),
                clocks=timings['clocks'][phase],
                scope='Same-run single measured phase; device-only in-memory profiler; no group hooks, no trace export; residual is not pure CPU overhead'))
            (args.output/'device-activity-windows.json').write_text(json.dumps(device_rows,indent=2)+'\n')
            print('DEVICE_ACTIVITY_WINDOW',json.dumps({k:v for k,v in device_rows[-1].items() if k!='kernel_names'}),flush=True)
            del profiler,activity

    # Capture only the measured stage, after native warmup. GPU correlation IDs
    # identify unique CUDA launch events; external CPU-op IDs can be reused.
    # No profiler trace or tensor payload is exported.
    install_groups()
    mode['profile'] = True
    profile_rows = []
    for p in [32,128,512]:
      for repeat in range(3):
        frozen = workload.contract('qwen25_1p5b', p, 2)
        profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False,
            with_stack=False)
        timings = run(frozen,profiler=profiler)
        groups = collections.defaultdict(lambda: dict(kernel_us=0.0, kernels=0))
        functions = profiler.events()
        cpu_by_id = collections.defaultdict(list)
        for event in functions:
            if event.device_type == torch.autograd.DeviceType.CPU and event.name.startswith('cu'):
                cpu_by_id[event.id].append(event)
        def semantic_owner(event):
            phase, group = 'UNATTRIBUTED', 'unclassified'
            ancestor = event
            while ancestor is not None:
                if ancestor.name.startswith('diagnostic/group/') and group == 'unclassified':
                    group = ancestor.name.rsplit('/',1)[-1]
                if ancestor.name == 'diagnostic/sampling' and group == 'unclassified':
                    group = 'sampling'
                if ancestor.name.startswith('phase/'):
                    phase = ancestor.name[len('phase/'):]
                ancestor = ancestor.cpu_parent
            return phase, group
        # Do not sum CPU FunctionEvent.kernels: repeated frontend correlation
        # IDs can attach a GPU event more than once. Own every raw CUDA event
        # once, then use only unambiguous CPU-context labels. Unlinked or
        # ambiguous events stay UNATTRIBUTED; never guess a semantic owner.
        gpu_events = [e for e in profiler.profiler.kineto_results.events()
            if e.device_type() == torch.autograd.DeviceType.CUDA and not e.is_user_annotation()]
        raw_total_us = 0.0
        phase_intervals=collections.defaultdict(list)
        kernel_names=collections.defaultdict(collections.Counter)
        for event in gpu_events:
            # A GPU kernel and its runtime launch share correlation_id. The
            # linked external ID is not a unique launch join key.
            owners = {semantic_owner(x) for x in cpu_by_id[event.correlation_id()]}
            phase, group = next(iter(owners)) if len(owners) == 1 else ('UNATTRIBUTED','unclassified')
            if group=='unclassified' and phase.startswith('Measured/'):
                group='framework_or_other'
            v = groups[(phase,group)]
            duration = (event.end_ns()-event.start_ns())/1000
            v['kernel_us'] += duration
            v['kernels'] += 1
            raw_total_us += duration
            phase_intervals[phase].append((event.start_ns(),event.end_ns()))
            kernel_names[(phase,group)][event.name()]+=1
        group_list = [dict(phase=k[0], group=k[1], **v) for k,v in groups.items()]
        grouped_us = sum(v['kernel_us'] for v in groups.values())
        grouped_count = sum(v['kernels'] for v in groups.values())
        assert grouped_count == len(gpu_events)
        assert abs(raw_total_us-grouped_us) < 0.001
        parsed_gpu = [e for e in functions if e.device_type == torch.autograd.DeviceType.CUDA and not e.is_user_annotation]
        assert len(parsed_gpu) == len(gpu_events)
        parsed_us = sum(e.time_range.elapsed_us() for e in parsed_gpu)
        assert abs(raw_total_us-parsed_us) < 0.001
        print('KERNEL_CORRELATION_CLOSURE',json.dumps(dict(cuda_count=len(gpu_events),grouped_count=grouped_count,
            cuda_us=raw_total_us, grouped_us=grouped_us, unattributed=groups[('UNATTRIBUTED','unclassified')])),flush=True)
        group_list = [dict(x,phase=x['phase'].split('/',1)[-1],kernel_names=dict(kernel_names[(x['phase'],x['group'])])) for x in group_list
            if x['phase'].startswith('Measured/') or x['phase']=='UNATTRIBUTED']
        union={}
        for phase,intervals in phase_intervals.items():
            merged=[]
            for lo,hi in sorted(intervals):
                if merged and lo<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],hi)
                else:merged.append([lo,hi])
            union[phase.split('/',1)[-1]]=sum(hi-lo for lo,hi in merged)/1e6
        assert all(phase.startswith('Measured/') for phase in phase_intervals),dict(groups)
        assert sum(x['kernels'] for x in group_list)==len(gpu_events)
        assert all(union[phase]<=ms+0.005 for phase,ms in timings['phases'].items())
        row = dict(case=f'p{p:03d}d02', repeat=repeat,instrumented_phase_ms=timings['phases'], groups=group_list,
                   attribution='Measured-stage-only raw CUDA events; unique runtime-launch correlation_id join; no external-ID fallback',
                   cuda_event_count=len(gpu_events), clock_summary=timings['clocks'],
                   all_cuda_us=raw_total_us, all_grouped_us=grouped_us,
                   gpu_busy_union_ms=union,
                   gpu_interval_outside_recorded_activity_ms={phase:ms-union[phase] for phase,ms in timings['phases'].items()},
                   gap_scope='Instrumented same-run CUDA-event interval minus union of recorded device activity; not pure CPU overhead or an additive correction to unprofiled native timing')
        profile_rows.append(row)
        (args.output/'kernel-groups.json').write_text(json.dumps(profile_rows,indent=2)+'\n')
        print('PROFILE_GROUPS',json.dumps(row),flush=True)
        del profiler, gpu_events
    for module, old in wrapped:
        module.forward = old
    runner.forward, runner.sample = original_forward, original_sample
    stop_sampler.set()
    sampler.join(timeout=1.0)
    pynvml.nvmlShutdown()
    assert sources == common.native_source_inventory(pathlib.Path(sglang.__file__).resolve().parent)
    receipt = dict(status='PASS_AGGREGATE_TIMING_DIAGNOSIS_NOT_NEW_ACCEPTANCE', completed_cases=len(rows),
        source_unchanged=True, trace_exported=False, native_sources=sources,
        driver_sha256=digest(__file__), shared_execution_sha256=digest(execution.__file__),
        warning='Profiler runs and event partitions are diagnostic, not additive corrections to official time.')
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['status'],flush=True)

if __name__ == '__main__':
    main()
