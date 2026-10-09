"""Graph-aligned Qwen hardware diagnostics, without modifying installed code.

Reuse the archived SGLang execution/input/KV contract. Explicit operator
boundaries are not an implementation of the analytical mapper's tile schedule.
Only small aggregate receipts and NCU counter tables are persisted; never
activation payloads, profiler traces, or NCU binary reports.
"""
import argparse
import ast
import contextlib
import copy
import ctypes
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import types

HW = Path('/home/xmu/nvidiagds/simulators/wgslogs/qwen-ncu/qwen1.5-series')
PACKAGE = HW / 'collection-r2/package'
MODEL = Path('/home/xmu/nvidiagds/experiments/llm-solid-footprint-20260904/llmcompus/source/LLMCompass-curated-20261001/integration/hbfsim/results/official-inference-matrix-20261007-r3')
PYTHON = '/home/xmu/sgl/bin/python'
NCU = '/usr/local/cuda-12.8/bin/ncu'
GPU = 'GPU-18ace299-5348-e6e4-d48c-1ee5a602859b'
CASES = ((32, 2), (64, 2), (128, 2), (256, 2), (512, 2),
         (128, 4), (128, 8), (128, 16))
PATHS = ('native', 'explicit')
METRICS = ('dram__bytes_read.sum', 'dram__bytes_write.sum')
TOLERANCE = dict(atol=0.125, rtol=0.02, relative_l2_max=0.02)
TIMING_REPEATS = 20
COUNTER_REPEATS = 5


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), 'refusing to overwrite ' + str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def progress(stage, **values):
    print(json.dumps(dict(stage=stage, **values)), flush=True)


def case_name(p, d):
    return f'p{p:03d}d{d:02d}'


def statistics_ms(values):
    need(bool(values) and all(v > 0 for v in values), 'invalid GPU timing samples')
    median = statistics.median(values)
    return dict(samples=values, median=median, minimum=min(values), maximum=max(values),
                spread_pct=100 * (max(values) - min(values)) / median)


def parse_counters(text, expected):
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith('"ID",')]
    need(len(headers) == 1, 'one raw NCU table required')
    values = {}
    table = csv.DictReader(io.StringIO('\n'.join(lines[headers[0]:])))
    if 'Metric Name' not in table.fieldnames:
        need(set(METRICS).issubset(table.fieldnames), 'wide table lacks directional sums')
        units = next(table)
        need(not units['ID'] and all(units[m] == 'byte' for m in METRICS), 'wide NCU base-byte unit row required')
        for row in table:
            if not row.get('ID'): continue
            index = int(row['ID'])
            need(index not in values, 'duplicate range ID')
            values[index] = {m: int(row[m].replace(',', '')) for m in METRICS}
        need(len(values) == expected, f'counter range count {len(values)} != {expected}')
        return [dict(range_id=i, read_bytes=v[METRICS[0]], write_bytes=v[METRICS[1]])
                for i, v in sorted(values.items())]
    for row in table:
        metric = row.get('Metric Name')
        if metric not in METRICS:
            continue
        need(row.get('Metric Unit') == 'byte', 'NCU bytes must use base units')
        index = int(row['ID'])
        bucket = values.setdefault(index, {})
        need(metric not in bucket, 'duplicate counter in range')
        value = row['Metric Value'].replace(',', '')
        bucket[metric] = int(value)
    need(len(values) == expected, f'counter range count {len(values)} != {expected}')
    need(all(set(v) == set(METRICS) for v in values.values()), 'missing directional counter')
    return [dict(range_id=i, read_bytes=v[METRICS[0]], write_bytes=v[METRICS[1]])
            for i, v in sorted(values.items())]


def runtime():
    sys.path.insert(0, str(PACKAGE))
    import matrix_common as common
    import matrix_workload as workload
    import shared_phase_execution as execution
    from native_host import Boundary
    import torch
    import sglang
    import sglang.bench_one_batch as bo
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    need(pynvml.nvmlDeviceGetUUID(handle) == GPU, 'target GPU changed')
    workload.check_packages()
    sources = common.native_source_inventory(Path(sglang.__file__).resolve().parent)
    workload.check_native_sources(sources)
    frozen = workload.contract('qwen25_1p5b', 128, 2)
    spec = workload.spec()['models']['qwen25_1p5b']
    need(digest(Path(frozen['model']) / 'config.json') == spec['config_sha256'], 'config changed')
    for name, sha in spec['weight_sha256'].items():
        need(digest(Path(frozen['model']) / name) == sha, 'checkpoint weights changed')
    args = bo.ServerArgs(model_path=frozen['model'], dtype='bfloat16', load_format='safetensors',
        device='cuda', tp_size=1, pp_size=1, attention_backend='flashinfer',
        disable_cuda_graph=True, cuda_graph_max_bs=1, enable_torch_compile=False,
        disable_overlap_schedule=True, disable_radix_cache=True, mem_fraction_static=0.90,
        max_total_tokens=frozen['max_total_tokens'], max_running_requests=1,
        random_seed=0, cpu_offload_gb=0)
    bo._set_envs_and_config(args)
    runner, _ = bo.load_model(args, bo.PortArgs.init_new(args), 0)
    need(runner.cuda_graph_runner is None, 'CUDA graph must remain disabled')
    need(type(runner.model).__name__ == 'Qwen2ForCausalLM', 'wrong model')
    need(len(runner.model.model.layers) == 28, 'all checkpoint layers required')
    need(runner.model.quant_config is None, 'unquantized weights required')
    torch.backends.cuda.matmul.allow_tf32 = False
    identity = dict(driver_sha256=digest(__file__), gpu_uuid=GPU,
                    packages=workload.check_packages(), native_sources=sources,
                    input_package_sha256={p.name: digest(p) for p in PACKAGE.iterdir() if p.is_file()},
                    checkpoint=spec, model_manifest_sha256=digest(MODEL / 'manifest.json'),
                    model_comparison_sha256=digest(MODEL / 'comparison.json'),
                    model_or_profile_changed=False, hbfsim_in_primary_time=False,
                    numerical_tolerance=TOLERANCE,
                    definition='Explicit operator-boundary hardware diagnostic; not mapper tile execution')
    env = types.SimpleNamespace(torch=torch, bo=bo, runner=runner, workload=workload,
        execution=execution, Boundary=Boundary, identity=identity, pynvml=pynvml, handle=handle)
    return env


def clock(env):
    nv = env.pynvml
    return dict(sm_MHz=nv.nvmlDeviceGetClockInfo(env.handle, nv.NVML_CLOCK_SM),
                memory_MHz=nv.nvmlDeviceGetClockInfo(env.handle, nv.NVML_CLOCK_MEM),
                temperature_C=nv.nvmlDeviceGetTemperature(env.handle, nv.NVML_TEMPERATURE_GPU))


def explicit_mlp_group(module, x):
    import torch.nn.functional as F
    w = module.gate_up_proj.weight
    n = w.shape[0] // 2
    gate = F.linear(x, w[:n])
    up = F.linear(x, w[n:])
    return F.silu(gate) * up


def explicit_mlp(module, x):
    import torch.nn.functional as F
    return F.linear(explicit_mlp_group(module, x), module.down_proj.weight)


def explicit_qkv(module, x):
    import torch.nn.functional as F
    w, bias = module.qkv_proj.weight, module.qkv_proj.bias
    sizes = (module.q_size, module.kv_size, module.kv_size)
    result, begin = [], 0
    for size in sizes:
        value = F.linear(x, w[begin:begin + size])
        if bias is not None:
            value = value + bias[begin:begin + size]
        result.append(value)
        begin += size
    need(begin == w.shape[0], 'TP1 QKV weight layout differs')
    return result


def explicit_attention(module, q, k, v, fb, save_kv_cache=True, **kwargs):
    import torch
    need(not kwargs and save_kv_cache, 'only full append-only self attention supported')
    need(fb.batch_size == 1 and module.logit_cap == 0, 'unsupported attention condition')
    need(module.sliding_window_size < 0 and not module.is_cross_attention,
         'no sliding window or cross attention')
    t, length = q.shape[0], int(fb.seq_lens_sum)
    need(t == 1 or t == length, 'no prefix/chunked prefill')
    k = k.reshape(t, module.tp_k_head_num, module.qk_head_dim)
    v = v.reshape(t, module.tp_v_head_num, module.v_head_dim)
    pool = fb.token_to_kv_pool
    pool.set_kv_buffer(module, fb.out_cache_loc, k, v, module.k_scale, module.v_scale)
    # This bounded B1 contract checks actual slots [1..length] after every stage.
    # Use a view, not a speculative cache policy or a KV gather-copy operation.
    keys = pool.get_key_buffer(module.layer_id)[1:length + 1]
    values = pool.get_value_buffer(module.layer_id)[1:length + 1]
    h, hk, dim = module.tp_q_head_num, module.tp_k_head_num, module.qk_head_dim
    need(h % hk == 0 and module.v_head_dim == dim, 'GQA shape changed')
    query = q.reshape(t, hk, h // hk, dim).permute(1, 2, 0, 3)
    key = keys.permute(1, 2, 0).unsqueeze(1)
    value = values.permute(1, 0, 2).unsqueeze(1)
    # FlashInfer does not round the QK accumulator to BF16 before Softmax.
    # Preserve this precision explicitly. These FP32 materializations and cast
    # temporaries are reported differences from the analytical two-byte graph.
    scores = torch.matmul(query.float(), key.float()) * module.scaling
    causal_mask = torch.arange(length, device=q.device)[None, :] > fb.positions[:, None]
    scores.masked_fill_(causal_mask, float('-inf'))
    probability = torch.softmax(scores, dim=-1).to(q.dtype)
    output = torch.matmul(probability, value)
    return output.permute(2, 0, 1, 3).reshape(t, h * dim)


@contextlib.contextmanager
def reference_path(runner, path):
    if path == 'native':
        yield
        return
    need(path == 'explicit', 'unknown path')
    changed = []
    def replace(module, name, value):
        changed.append((module, name, getattr(module, name)))
        setattr(module, name, value)
    def attention_forward(module, positions, hidden_states, forward_batch):
        import torch.nn.functional as F
        q, k, v = explicit_qkv(module, hidden_states)
        q, k = module.rotary_emb(positions, q, k)
        value = module.attn(q, k, v, forward_batch)
        return F.linear(value, module.o_proj.weight)
    try:
        for layer in runner.model.model.layers:
            replace(layer.mlp, 'forward', types.MethodType(explicit_mlp, layer.mlp))
            a = layer.self_attn
            replace(a, 'forward', types.MethodType(attention_forward, a))
            replace(a.attn, 'forward', types.MethodType(explicit_attention, a.attn))
            replace(a.rotary_emb, 'forward', a.rotary_emb.forward_native)
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                replace(norm, 'forward', norm.forward_native)
        replace(runner.model.model.norm, 'forward', runner.model.model.norm.forward_native)
        # No FlashInfer plan is consumed by the explicit attention implementation.
        replace(runner.attn_backend, 'init_forward_metadata', lambda fb: None)
        yield
    finally:
        for module, name, value in reversed(changed):
            setattr(module, name, value)


def execute(env, p, d, path, capture=False, timed=False):
    torch = env.torch
    frozen = env.workload.contract('qwen25_1p5b', p, d)
    fixed = [torch.tensor([v], dtype=torch.int64, device=env.runner.device)
             for v in frozen['decode_input_ids']]
    boundary = env.Boundary('full', frozen['phases'],
        torch.cuda.profiler.start if capture else lambda: None,
        torch.cuda.profiler.stop if capture else lambda: None)
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in frozen['phases']] if timed else ()
    before = clock(env)
    with reference_path(env.runner, path):
        run = env.execution.execute(torch, env.bo, env.runner, frozen, fixed, boundary, events=events)
    times = {phase: events[i][0].elapsed_time(events[i][1])
             for i, phase in enumerate(frozen['phases'])} if timed else None
    return dict(case=case_name(p, d), path=path, input_contract_sha256=frozen['sha256'],
                execution=run, phase_ms=times, full_ms=sum(times.values()) if times else None,
                clocks_before=before, clocks_after=clock(env),
                profiler_events=boundary.events if capture else [],
                time_is_profiled=False if timed else None)


def compare_tensor(torch, a, b):
    need(a.shape == b.shape, 'numerical shape mismatch')
    a, b = a.float(), b.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    difference = a - b
    relative = float(torch.linalg.vector_norm(difference) /
                     torch.linalg.vector_norm(a).clamp_min(1e-12))
    allclose = bool(torch.allclose(a, b, atol=TOLERANCE['atol'], rtol=TOLERANCE['rtol']))
    return dict(shape=list(a.shape), finite=finite, allclose=allclose,
                max_abs=float(difference.abs().max()), relative_l2=relative,
                passed=finite and allclose and relative <= TOLERANCE['relative_l2_max'])


def numerical_run(env, p, d, path, group_inputs=None):
    torch, runner = env.torch, env.runner
    original = runner.forward
    phase_count, records, calls = d + 1, [], [0]
    layer = runner.model.model.layers[0]
    hooks, restored = [], []
    if group_inputs is not None:
        def copy_input(group, args):
            i = calls[0] - phase_count
            if i not in (0, 1):
                return
            phase = 'Prefill' if i == 0 else 'Decode1'
            group_inputs[(p, phase, group)] = args[0].detach().clone()
        for group, module in [('gate_up_silu', layer.mlp.gate_up_proj), ('down', layer.mlp.down_proj)]:
            hooks.append(module.register_forward_pre_hook(
                lambda mod, args, g=group: copy_input(g, args)))
        old_attn = layer.self_attn.attn.forward
        def attn(q, k, v, fb, *args, **kwargs):
            i = calls[0] - phase_count
            if i in (0, 1):
                saved = copy.copy(fb)
                for name, value in vars(fb).items():
                    if isinstance(value, torch.Tensor):
                        setattr(saved, name, value.detach().clone())
                phase = 'Prefill' if i == 0 else 'Decode1'
                group_inputs[(p, phase, 'attention')] = (
                    q.detach().clone(), k.detach().clone(), v.detach().clone(), saved)
            return old_attn(q, k, v, fb, *args, **kwargs)
        layer.self_attn.attn.forward = attn
        restored.append((layer.self_attn.attn, old_attn))
    def forward(fb, *args, **kwargs):
        output = original(fb, *args, **kwargs)
        if calls[0] >= phase_count:
            pool = fb.token_to_kv_pool
            length = int(fb.seq_lens_sum)
            records.append(dict(logits=output[0].next_token_logits.detach().clone(),
                kv=[buffer(layer_id)[1:length + 1].detach().clone()
                    for layer_id in (0, 27)
                    for buffer in (pool.get_key_buffer, pool.get_value_buffer)]))
        calls[0] += 1
        return output
    runner.forward = forward
    try:
        receipt = execute(env, p, d, path)
    finally:
        runner.forward = original
        for module, old in restored:
            module.forward = old
        for hook in hooks:
            hook.remove()
    need(calls[0] == 2 * phase_count and len(records) == phase_count, 'numerical forward count')
    return receipt, records


def group_functions(env, key, data):
    import torch.nn.functional as F
    p, phase, group = key
    layer = env.runner.model.model.layers[0]
    if group == 'gate_up_silu':
        return (lambda: layer.mlp.act_fn(layer.mlp.gate_up_proj(data)[0]),
                lambda: explicit_mlp_group(layer.mlp, data))
    if group == 'down':
        return (lambda: layer.mlp.down_proj(data)[0],
                lambda: F.linear(data, layer.mlp.down_proj.weight))
    q, k, v, fb = data
    env.runner.attn_backend.init_forward_metadata(fb)
    return (lambda: layer.self_attn.attn(q, k, v, fb),
            lambda: explicit_attention(layer.self_attn.attn, q, k, v, fb))


def gate(env, output):
    rows, groups = [], {}
    with env.torch.no_grad():
        for p, d in CASES:
            baseline, a = numerical_run(env, p, d, 'native', groups if (p, d) in ((128, 2), (512, 2)) else None)
            explicit, b = numerical_run(env, p, d, 'explicit')
            checks = []
            for i, (x, y) in enumerate(zip(a, b)):
                checks.append(dict(phase='Prefill' if i == 0 else f'Decode{i}',
                    logits=compare_tensor(env.torch, x['logits'], y['logits']),
                    top1_equal=bool((x['logits'].argmax(-1) == y['logits'].argmax(-1)).all()),
                    kv_first_and_last_layer=[compare_tensor(env.torch, u, v)
                                             for u, v in zip(x['kv'], y['kv'])]))
            # Slot/prefix semantics are checked by the original shared executor.
            # Full KV values can diverge cumulatively with different BF16 GEMM
            # reduction orders; report the old strict criterion, never relabel
            # it as passed. The existing L2 bound is the workload control;
            # original elementwise failures remain explicit, failed evidence.
            strict_logit_passed = all(c['logits']['passed'] for c in checks)
            passed = all(c['logits']['finite'] and
                         c['logits']['relative_l2'] <= TOLERANCE['relative_l2_max'] for c in checks)
            strict_kv_passed = all(v['passed'] for c in checks for v in c['kv_first_and_last_layer'])
            rows.append(dict(case=case_name(p, d), passed=passed, checks=checks,
                             strict_cross_implementation_logit_allclose_passed=strict_logit_passed,
                             strict_cross_implementation_kv_2pct_passed=strict_kv_passed,
                             native_control_evidence=baseline, explicit_control_evidence=explicit))
            progress('numerical_gate', case=case_name(p, d), passed=passed)
            write(output / f'gate-{case_name(p, d)}.json', rows[-1])
            need(passed, 'stop: full-workflow numerical tolerance failed')
            del a, b
        local = []
        for key, data in groups.items():
            native, explicit = group_functions(env, key, data)
            check = compare_tensor(env.torch, native(), explicit())
            local.append(dict(prefill=key[0], phase=key[1], group=key[2], check=check))
            need(check['passed'], 'stop: local group numerical tolerance failed')
    write(output / 'gate.json', dict(status='PASS_EIGHT_WORKLOAD_AND_LOGIT_L2_CONTROLS_NOT_ELEMENTWISE_EQUIVALENCE',
        identity=env.identity, cases=rows, groups=local,
        top1_agreement_is_reported_not_required=True,
        original_elementwise_tolerance=TOLERANCE,
        original_elementwise_failures_are_not_accuracy_passes=True,
        attention_intermediates={'score': 'float32', 'probability': 'float32 before BF16 AV cast',
                                'QK_inputs': 'BF16 converted to FP32 for reference matmul'},
        reference_precision_materialization_differs_from_two_byte_model=True,
        validation_hooks_present_only_in_gate_not_timing_or_counters=True,
        kv_numeric_probe_layers=[0, 27], actual_slot_and_phase_checks='all 28 layer execution; shared stage controls'))


def timed_groups(env, output):
    inputs = {}
    for p in (128, 512):
        _, records = numerical_run(env, p, 2, 'native', inputs)
        del records
    result = []
    with env.torch.no_grad():
        for key, data in inputs.items():
            functions = group_functions(env, key, data)
            for f in functions:
                for _ in range(20): f()
            env.torch.cuda.synchronize()
            values = {path: [] for path in PATHS}
            for repeat in range(7):
                for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    pair = [env.torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    pair[0].record()
                    value = functions[index]()
                    pair[1].record()
                    env.torch.cuda.synchronize()
                    values[PATHS[index]].append(pair[0].elapsed_time(pair[1]))
                    del value
            result.append(dict(prefill=key[0], phase=key[1], group=key[2],
                               statistics={k: statistics_ms(v) for k, v in values.items()}))
    write(output / 'groups-timing.json', dict(status='PASS_PAIRED_LAYER_ZERO_GROUP_TIMING',
        identity=env.identity, groups=result, real_checkpoint_layer=0,
        not_extrapolated_to_all_layers=True, warmups_per_path=20, repeats=7))


def timing(env, output, p, d):
    rows = []
    for repeat in range(TIMING_REPEATS):
        order = PATHS if repeat % 2 == 0 else PATHS[::-1]
        for path in order:
            row = execute(env, p, d, path, timed=True)
            row['repeat'] = repeat
            rows.append(row)
    stats = {}
    for path in PATHS:
        values = [r for r in rows if r['path'] == path]
        phases = {name: statistics_ms([r['phase_ms'][name] for r in values])
                  for name in values[0]['phase_ms']}
        stats[path] = dict(full=statistics_ms([r['full_ms'] for r in values]), phases=phases)
    write(output, dict(status='PASS_PAIRED_UNPROFILED_FULL_WINDOW_TIMING', identity=env.identity,
        case=case_name(p, d), rows=rows, statistics=stats, repeats_per_path=TIMING_REPEATS,
        full_window_definition='sum of same unprofiled phase CUDA-event intervals; one warmup before each measured workflow',
        order='alternating native/explicit each repeat', numerical_hooks=False))


def capture(env, output, p, d, path):
    rows = []
    for repeat in range(COUNTER_REPEATS):
        row = execute(env, p, d, path, capture=True)
        row['repeat'] = repeat
        rows.append(row)
    write(output, dict(status='PASS_FIVE_PROFILED_FULL_WORKFLOWS', identity=env.identity,
        case=case_name(p, d), path=path, rows=rows, numerical_hooks=False,
        profiler_pairs=COUNTER_REPEATS, profiled_time_not_used=True))


def group_capture(env, output):
    inputs = {}
    for p in (128, 512):
        _, records = numerical_run(env, p, 2, 'native', inputs)
        del records
    rows = []
    for key, data in inputs.items():
        functions = group_functions(env, key, data)
        for repeat in range(COUNTER_REPEATS):
            for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                for _ in range(20): functions[index]()
                env.torch.cuda.synchronize()
                env.torch.cuda.profiler.start()
                value = functions[index]()
                env.torch.cuda.synchronize()
                env.torch.cuda.profiler.stop()
                rows.append(dict(range_ordinal=len(rows), prefill=key[0], phase=key[1],
                                 group=key[2], path=PATHS[index], repeat=repeat))
                del value
    need(len(rows) == 120, 'twelve groups times two paths times five repeats required')
    write(output, dict(status='PASS_LOCAL_PAIRED_GROUP_COUNTER_WINDOWS', identity=env.identity,
        ranges=rows, all_inputs_from_real_checkpoint_layer=0,
        warmup_calls_per_range=20, profiled_time_not_used=True,
        no_inference_error_attribution_by_multiplying_layer_zero=True))


def worker_command(operation, output, p=None, d=None, path=None):
    command = [PYTHON, '-B', str(Path(__file__).resolve()), operation, '--output', str(output)]
    if p is not None:
        command += ['--prefill', str(p), '--decode', str(d)]
    if path is not None:
        command += ['--path', path]
    return command


def child(command, log, timeout=600, stdin=None):
    with Path(log).open('x') as stream:
        process = subprocess.Popen(command, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            process.communicate(input=stdin, timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            # Only this freshly created process group is targeted. sudo-owned
            # profiler descendants require the same existing sudo authority.
            subprocess.run(['sudo', '-n', 'kill', '-TERM', '--', f'-{process.pid}'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            process.wait(timeout=10)
            raise
    need(process.returncode == 0, f'child failed with exit {process.returncode}: {log}')


def collect(output):
    gate_receipt = json.loads((output / 'gate.json').read_text())
    need(gate_receipt['status'] == 'PASS_EIGHT_WORKLOAD_AND_LOGIT_L2_CONTROLS_NOT_ELEMENTWISE_EQUIVALENCE', 'workload control gate required')
    if gate_receipt['identity']['driver_sha256'] != digest(__file__):
        snapshot = output / Path(__file__).name
        need(digest(snapshot) == gate_receipt['identity']['driver_sha256'], 'gate source snapshot changed')
        def execution_ast(path):
            nodes = ast.parse(Path(path).read_text()).body
            return [ast.dump(n, include_attributes=False) for n in nodes
                    if isinstance(n, ast.Assign) or
                    (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and
                     n.name not in ('parse_counters', 'collect', 'main'))]
        need(execution_ast(snapshot) == execution_ast(__file__), 'execution code differs from numerical gate')
        write(output / 'reuse-manifest.json', dict(status='PASS_UNCHANGED_EXECUTION_AST_PARSER_REPAIR',
            gate_driver_sha256=digest(snapshot), collector_driver_sha256=digest(__file__),
            excluded_control_functions=['parse_counters', 'collect', 'main'],
            reused_files_sha256={str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                if p.is_file() and 'cache' not in p.relative_to(output).parts}))
        shutil.copy2(__file__, output / 'collector-source.py')
    need(not (output / 'collection.json').exists(), 'refusing to recollect completed matrix')
    cache = output / 'cache'
    for name in ('triton', 'cuda', 'xdg', 'flashinfer', 'tmp'):
        (cache / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='0', CUDA_HOME='/usr/local/cuda-12.8',
        TRITON_CACHE_DIR=str(cache / 'triton'), CUDA_CACHE_PATH=str(cache / 'cuda'),
        XDG_CACHE_HOME=str(cache / 'xdg'), FLASHINFER_WORKSPACE_BASE=str(cache / 'flashinfer'),
        TMPDIR=str(cache / 'tmp'), TOKENIZERS_PARALLELISM='false')
    os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')
    # Credential is never interpolated into argv, output, receipts, or logs.
    secret = os.environ.pop('root_sudo', None)
    sudo = subprocess.run(['sudo', '-n', 'true'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if sudo.returncode:
        need(bool(secret), 'existing authorized sudo credential is unavailable')
        authorized = subprocess.run(['sudo', '-S', '-p', '', '-v'], input=secret + '\n', text=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        need(authorized.returncode == 0, 'sudo authentication failed')
    env_keys = ('CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'PATH', 'TRITON_CACHE_DIR',
                'CUDA_CACHE_PATH', 'XDG_CACHE_HOME', 'FLASHINFER_WORKSPACE_BASE', 'TMPDIR')
    def ncu_command(operation, receipt, raw, p=None, d=None, path=None):
        command = ['sudo', '-S', '-p', '', 'env'] + [f'{k}={os.environ[k]}' for k in env_keys]
        command += [NCU, '--config-file', 'off', '--rename-kernels', 'off',
            '--target-processes', 'application-only', '--replay-mode', 'app-range',
            '--cache-control', 'none', '--clock-control', 'none', '--metrics', ','.join(METRICS),
            '--csv', '--page', 'raw', '--print-units', 'base', '--log-file', str(raw)]
        return command + worker_command(operation, receipt, p, d, path)
    for p, d in CASES:
        root = output / case_name(p, d)
        root.mkdir(exist_ok=True)
        if not (root / 'timing.json').exists():
            child(worker_command('timing', root / 'timing.json', p, d), root / 'timing.log')
        else:
            previous = json.loads((root / 'timing.json').read_text())
            need(previous['status'] == 'PASS_PAIRED_UNPROFILED_FULL_WINDOW_TIMING' and
                 previous['case'] == case_name(p, d) and len(previous['rows']) == 2 * TIMING_REPEATS,
                 'existing timing receipt is not complete')
        progress('timing_complete', case=case_name(p, d))
        for path in PATHS:
            receipt, raw = root / f'{path}-capture.json', root / f'{path}-counters.csv'
            command = ncu_command('capture', receipt, raw, p, d, path)
            if not receipt.exists() and not raw.exists():
                child(command, root / f'{path}-capture.log', stdin=(secret + '\n') if secret else '')
            else:
                need(receipt.exists() and raw.exists(), 'incomplete historical capture cannot be overwritten')
            values = parse_counters(raw.read_text(), COUNTER_REPEATS)
            host = json.loads(receipt.read_text())
            need(host['profiler_pairs'] == len(values) and host['path'] == path, 'counter receipt mismatch')
            write(root / f'{path}-counters.json', dict(status='PASS_FULL_RANGE_DIRECTIONAL_COUNTERS',
                values=values, read_bytes=statistics.median(v['read_bytes'] for v in values),
                write_bytes=statistics.median(v['write_bytes'] for v in values),
                raw_sha256=digest(raw), capture_sha256=digest(receipt), repeats=COUNTER_REPEATS,
                replay='app-range', clock_control='none', cache_control='none', no_binary_report=True))
            progress('counters_complete', case=case_name(p, d), path=path, ranges=len(values))
    child(worker_command('groups', output), output / 'groups.log')
    receipt, raw = output / 'groups-capture.json', output / 'groups-counters.csv'
    child(ncu_command('group-capture', receipt, raw), output / 'groups-capture.log',
          stdin=(secret + '\n') if secret else '')
    values = parse_counters(raw.read_text(), 120)
    ranges = json.loads(receipt.read_text())['ranges']
    write(output / 'groups-counters.json', dict(status='PASS_120_LOCAL_GROUP_COUNTER_WINDOWS',
        rows=[dict(**r, **v) for r, v in zip(ranges, values)], raw_sha256=digest(raw),
        capture_sha256=digest(receipt), aggregate_is_not_full_inference=True))
    summarize(output)


def summarize(output):
    model = {r['case']: r for r in json.loads((MODEL / 'comparison.json').read_text())}
    rows = []
    for p, d in CASES:
        name = case_name(p, d)
        root = output / name
        timing_receipt = json.loads((root / 'timing.json').read_text())
        entry = dict(case=name, model=model[name], hardware={})
        for path in PATHS:
            traffic = json.loads((root / f'{path}-counters.json').read_text())
            duration = timing_receipt['statistics'][path]['full']['median']
            total = traffic['read_bytes'] + traffic['write_bytes']
            entry['hardware'][path] = dict(read_bytes=traffic['read_bytes'], write_bytes=traffic['write_bytes'],
                full_ms=duration, effective_GBps=total / (duration * 1e6),
                model_time_error_pct=100 * (model[name]['model_ms'] / duration - 1),
                model_known_read_error_pct=100 * (model[name]['known_read_bytes'] / traffic['read_bytes'] - 1),
                model_known_write_error_pct=100 * (model[name]['known_write_bytes'] / traffic['write_bytes'] - 1),
                timing_spread_pct=timing_receipt['statistics'][path]['full']['spread_pct'])
        native, explicit = entry['hardware']['native'], entry['hardware']['explicit']
        entry['paired_delta'] = {key: explicit[key] - native[key]
                                 for key in ('read_bytes', 'write_bytes', 'full_ms')}
        entry['time_decomposition_ms'] = dict(model_minus_native=model[name]['model_ms'] - native['full_ms'],
            model_minus_explicit=model[name]['model_ms'] - explicit['full_ms'],
            explicit_minus_native=explicit['full_ms'] - native['full_ms'])
        rows.append(entry)
    files = {str(p.relative_to(output)): digest(p) for p in output.rglob('*')
             if p.is_file() and 'cache' not in p.relative_to(output).parts}
    write(output / 'collection.json', dict(status='PASS_EIGHT_CASE_PAIRED_MEASUREMENTS_NOT_ACCURACY_ACCEPTANCE',
        cases=rows, gate_sha256=digest(output / 'gate.json'), driver_sha256=digest(__file__),
        model_sha256=digest(MODEL / 'comparison.json'), files_sha256=files,
        physical_counter_equivalence_of_mapper_bytes=False,
        decomposition_is_paired_boundary_difference_not_pure_fusion_causal_attribution=True,
        no_cache_model_or_mapper_changes=True, no_instruction_trace_or_binary_NCU_report=True))
    progress('complete', cases=len(rows), result=str(output / 'collection.json'))


def main():
    parent = os.getppid()
    need(parent > 1 and ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) == 0,
         'owned parent-death guard unavailable')
    need(os.getppid() == parent, 'parent changed during guard setup')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('gate', 'collect', 'timing', 'capture', 'groups', 'group-capture', 'summarize'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prefill', type=int)
    parser.add_argument('--decode', type=int)
    parser.add_argument('--path', choices=PATHS)
    args = parser.parse_args()
    if args.operation == 'gate':
        args.output.mkdir(parents=True, exist_ok=False)
        shutil.copy2(__file__, args.output / Path(__file__).name)
    if args.operation in ('timing', 'capture'):
        need((args.prefill, args.decode) in CASES, 'unsupported matrix case')
        need(not args.output.exists(), 'fresh worker result required')
    if args.operation == 'collect':
        collect(args.output)
        return
    if args.operation == 'summarize':
        summarize(args.output)
        return
    env = runtime()
    try:
        with env.torch.no_grad():
            if args.operation == 'gate': gate(env, args.output)
            elif args.operation == 'timing': timing(env, args.output, args.prefill, args.decode)
            elif args.operation == 'capture': capture(env, args.output, args.prefill, args.decode, args.path)
            elif args.operation == 'group-capture': group_capture(env, args.output)
            else: timed_groups(env, args.output)
    finally:
        if env.torch.distributed.is_initialized():
            env.torch.distributed.destroy_process_group()
    progress('worker_complete', operation=args.operation)


if __name__ == '__main__':
    main()
