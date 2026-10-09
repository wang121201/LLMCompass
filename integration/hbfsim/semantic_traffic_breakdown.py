"""Semantic accounting of frozen LLMCompass boundary transfers, not NCU labels.

Recover operand bytes from unchanged mapper transfer events under the frozen
profile. No latency-to-bytes inversion, request deletion, or new cache rule.
The analyze command requires the official software-model dependencies; plot
requires Matplotlib. The offline HTML report uses only the standard library.
"""
import argparse
import collections
import copy
import dataclasses
import datetime
import hashlib
import html
import importlib.util
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


CATEGORIES = ('weights', 'kv_cache', 'activation', 'other')
DIRECTIONS = ('read_bytes', 'write_bytes', 'unknown_direction_bytes')
CASE_IDS = ('p032d02', 'p064d02', 'p128d02', 'p256d02', 'p512d02',
            'p128d04', 'p128d08', 'p128d16')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    pathlib.Path(path).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def empty_categories():
    return {category: {direction: 0 for direction in DIRECTIONS} for category in CATEGORIES}


def classify_access(access):
    """Persistent ownership, not whether an operand happens to be K or V."""
    label = access.label
    if access.target == 'HBF_STATIC':
        check(access.kind == 'read', 'Inference must not write model weights')
        group = ('mlp_weights' if '.mlp.' in label else
                 'attention_weights' if '.self_attn.' in label else
                 'embedding_and_lm_head_weights' if label == 'model.embed_tokens' else
                 'normalization_weights')
        return 'weights', group
    check(access.target == 'HBM', f'Unrecognized source address target: {access.target}')
    if label in ('key_cache', 'value_cache'):
        return 'kv_cache', label
    if label in ('attention_scores', 'attention_probabilities'):
        return 'activation', 'attention_intermediates'
    if label in ('gate', 'up', 'gate_silu', 'gated_up'):
        return 'activation', 'mlp_intermediates'
    if label in ('q', 'k', 'v', 'q_rope', 'k_rope', 'attention_output'):
        return 'activation', 'qkv_attention_activations'
    if label in ('hidden', 'norm_hidden', 'final_hidden', 'last_hidden',
                 'attention_projected', 'mlp_output'):
        return 'activation', 'hidden_and_residual_activations'
    if label == 'logits':
        return 'activation', 'logits'
    raise ValueError(f'New logical object needs an explicit semantic rule: {label}')


def operand_counts(events, word_size):
    counts = {direction: {operand: 0 for operand in ('A', 'B', 'C')}
              for direction in ('read', 'write')}
    for event in events:
        check(event['direction'] in counts and event['tensor'] in counts['read'], 'Unknown mapper event')
        amount = event['rows'] * event['columns'] * word_size
        check(amount > 0, 'Invalid mapper event size')
        counts[event['direction']][event['tensor']] += amount
    return counts


def add_categories(destination, source):
    for category in CATEGORIES:
        for direction in DIRECTIONS:
            destination[category][direction] += source[category][direction]


def category_total(categories, direction=None):
    directions = DIRECTIONS if direction is None else (direction,)
    return sum(categories[c][d] for c in CATEGORIES for d in directions)


def allocate_operator(op, ledger, recovered):
    categories = empty_categories()
    subobjects = {}

    def add(access, direction, amount):
        check(isinstance(amount, int) and amount >= 0, 'Semantic byte count must be an integer')
        category, group = classify_access(access)
        categories[category][direction] += amount
        key = f'{category}/{group}'
        subobjects.setdefault(key, {d: 0 for d in DIRECTIONS})[direction] += amount

    if ledger['path'] == 'QWEN_VECTOR_ADAPTER':
        for access in op.reads + op.writes:
            add(access, access.kind + '_bytes', access.bytes)
    else:
        check(recovered is not None, 'Missing selected-mapper operand accounting')
        counts = recovered['operands']
        check(counts['write']['A'] == counts['write']['B'] == 0, 'Unexpected input operand writes')
        if ledger['path'] == 'OFFICIAL_SOFTMAX':
            check(counts['read']['B'] == counts['read']['C'] == 0, 'Unexpected Softmax operand')
            add(op.reads[0], 'read_bytes', counts['read']['A'])
        else:
            check(len(op.reads) >= 2, 'Missing matrix operands')
            add(op.reads[0], 'read_bytes', counts['read']['A'])
            add(op.reads[1], 'read_bytes', counts['read']['B'])
            add(op.writes[0], 'read_bytes', counts['read']['C'])
            # Bias is an explicit frozen adapter addition, not a mapper operand.
            for access in op.reads[2:]:
                check(access.target == 'HBF_STATIC' and access.label.endswith('.bias'), 'Unknown extra projection input')
                add(access, 'read_bytes', access.bytes)
        add(op.writes[0], 'write_bytes', counts['write']['C'])
    extra = ledger['unclassified_io_bytes']
    categories['other']['unknown_direction_bytes'] += extra
    if extra:
        subobjects['other/unclassified_batch_surrogate_io'] = dict(read_bytes=0, write_bytes=0, unknown_direction_bytes=extra)
    for direction, source in [('read_bytes', 'known_read_bytes'), ('write_bytes', 'known_write_bytes'),
                              ('unknown_direction_bytes', 'unclassified_io_bytes')]:
        check(category_total(categories, direction) == ledger[source],
              f"{op.index}/{op.name}: semantic {direction} does not close")
    return categories, subobjects


def analyze(matrix, output):
    matrix, output = pathlib.Path(matrix).resolve(), pathlib.Path(output).resolve()
    adapter = matrix.parents[1]
    repo = adapter.parents[1]
    baseline = json.loads((matrix / 'validation.json').read_text())
    check(baseline['accounting_status'] == 'PASS' and baseline['completed_cases'] == 8, 'Incomplete validated source matrix')
    inputs = dict(baseline['input_sha256'])
    inputs[str(matrix / 'validation.json')] = sha(matrix / 'validation.json')
    manifest = json.loads((matrix / 'manifest.json').read_text())
    profile = matrix / 'source' / pathlib.Path(manifest['profile']).name
    check(sha(profile) == manifest['profile_sha256'] and not manifest['hbfsim_in_primary_time'], 'Wrong frozen timing inputs')
    original_diff = subprocess.run(['git', 'diff', '--exit-code', 'origin/ISCA_AE', '--',
                                   'software_model', 'hardware_model', 'ae/figure5'], cwd=repo, capture_output=True)
    check(original_diff.returncode == 0 and not original_diff.stdout, 'Official source differs from frozen reference')
    official_names = subprocess.check_output(['git', 'ls-files', 'software_model', 'hardware_model', 'ae/figure5', 'utils.py'], cwd=repo, text=True).splitlines()
    for name in official_names:
        path = repo / name
        inputs[str(path)] = sha(path)
    model_path = adapter / 'qwen25_1p5b.json'
    inputs[str(model_path)] = sha(model_path)
    for name, digest in manifest['source_sha256'].items():
        check(sha(matrix / 'source' / name) == digest, f'Frozen source mismatch: {name}')
    for name, digest in manifest['geometry_lut_sha256'].items():
        path = repo / 'systolic_array_model' / name
        check(sha(path) == digest, f'Geometry lookup mismatch: {name}')
        inputs[str(path)] = digest
    for path, digest in inputs.items():
        check(sha(path) == digest, f'Input hash changed: {path}')
    check(not output.exists(), 'Use a fresh analysis directory; preserve old results')
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / pathlib.Path(__file__).name)
    shutil.copy2(model_path, output / model_path.name)
    receipt = dict(status='RUNNING', source_matrix=str(matrix), profile_sha256=manifest['profile_sha256'],
                   driver_sha256=sha(__file__), input_sha256=inputs, completed_shapes=0, completed_cases=0,
                   trace_exported=False, model_or_hardware_measurements_modified=False)
    write_json(output / 'receipt.json', receipt)
    sys.path.insert(0, str(repo))
    q = load_module('semantic_frozen_qwen', matrix / 'source/qwen_hbfsim_cosim.py')
    cost = q.LLMCompassCostModel(repo, profile)
    mat = load_module('semantic_frozen_matmul', matrix / 'source/mapper_matmul.py')
    soft = load_module('semantic_frozen_softmax', matrix / 'source/mapper_softmax.py')
    work = pathlib.Path(tempfile.mkdtemp(prefix='llmcompass-semantic-'))
    shutil.copytree(repo / 'systolic_array_model', work / 'systolic_array_model',
                    ignore=shutil.ignore_patterns('temp', '*.gz', '*.npy'))
    (work / 'systolic_array_model/temp').mkdir()
    os.chdir(work)
    receipt['isolated_working_directory'] = str(work)
    write_json(output / 'receipt.json', receipt)

    # A fresh in-process cache is qualified by this one immutable profile/source.
    # It is populated by original compilation, never imported latency seed data.
    compiled = {}
    observed_calls = []
    original_compile = mat.Matmul.compile_and_simulate

    def observed_compile(op, device, mode='exhaustive'):
        key = (op.M, op.K, op.N, mode)
        if key not in compiled:
            seconds = original_compile(op, device, mode)
            if op.best_mapping is not None:
                cycles = op.simulate(op.computational_graph, op.best_mapping, device)
                check(math.isclose(seconds, cycles / device.compute_module.clock_freq, rel_tol=1e-12, abs_tol=1e-15), 'Selected mapping changes latency')
            counts = operand_counts(op.main_memory_events, op.data_type.word_size)
            check(sum(counts['read'].values()) == op.memory_accounting['main_read_bytes'] and
                  sum(counts['write'].values()) == op.memory_accounting['main_write_bytes'], 'Operand event bytes do not close')
            record = dict(seconds=seconds, accounting=copy.deepcopy(op.memory_accounting), operands=counts,
                          mapping=None if op.best_mapping is None else vars(op.best_mapping).copy(),
                          transfer_event_count=len(op.main_memory_events))
            compiled[key] = (record, op)
        record, cached = compiled[key]
        if op is not cached:
            for name in ('best_mapping', 'best_cycle_count', 'best_latency', 'latency',
                         'memory_accounting', 'look_up_table', 'main_memory_events'):
                if hasattr(cached, name):
                    setattr(op, name, getattr(cached, name))
        observed_calls.append(copy.deepcopy(record))
        return record['seconds']

    mat.Matmul.compile_and_simulate = observed_compile
    frozen_shapes = json.loads((matrix / 'shape-cache.json').read_text())
    check(len(frozen_shapes) == 111, 'Unexpected source shape matrix')
    recovered = {}
    for expected in frozen_shapes:
        key = expected['key']
        observed_calls.clear()
        if key[0] == 'matmul':
            _, m, k, n = key
            op = mat.Matmul(cost.dtype)
            op(cost.Tensor([m, k], cost.dtype), cost.Tensor([k, n], cost.dtype))
            seconds = op.compile_and_simulate(cost.device, 'heuristic-GPU')
            selected = observed_calls[-1]
            selected_candidate = None
        elif key[0] == 'batch':
            _, b, m, k, n = key
            op = mat.BatchedMatmul(cost.dtype)
            op(cost.Tensor([b, m, k], cost.dtype), cost.Tensor([b, k, n], cost.dtype))
            seconds = op.compile_and_simulate(cost.device, 'heuristic-GPU')
            check(len(observed_calls) == 2, 'Official batch must evaluate two Matmul candidates')
            selected_candidate = op.selected_accounting_candidate
            selected = copy.deepcopy(observed_calls[selected_candidate])
            if selected_candidate == 0:
                for direction in selected['operands']:
                    for operand in selected['operands'][direction]:
                        selected['operands'][direction][operand] *= b
                selected['transfer_event_count'] *= b
            selected['batch_candidates'] = copy.deepcopy(observed_calls)
        else:
            _, b, m, n = key
            check(key[0] == 'softmax', 'Unknown shape kind')
            op = soft.Softmax(cost.dtype)
            op(cost.Tensor([b, m, n], cost.dtype))
            seconds = op.compile_and_simulate(cost.device, 'heuristic-GPU')
            cycles = op.simulate(op.computational_graph, op.best_mapping, cost.device)
            check(math.isclose(seconds, cycles / cost.device.compute_module.clock_freq, rel_tol=1e-12, abs_tol=1e-15), 'Softmax mapping changes latency')
            selected = dict(operands=operand_counts(op.main_memory_events, cost.dtype.word_size),
                            mapping=vars(op.best_mapping).copy(), transfer_event_count=len(op.main_memory_events))
            selected_candidate = None
        check(math.isclose(seconds, expected['seconds'], rel_tol=1e-12, abs_tol=1e-15), f'{key}: frozen latency mismatch')
        check(op.memory_accounting == expected['accounting'], f'{key}: frozen byte accounting mismatch')
        check(selected_candidate == expected['selected_candidate'], f'{key}: batch selection mismatch')
        counts = selected['operands']
        check(sum(counts['read'].values()) == op.memory_accounting['main_read_bytes'] and
              sum(counts['write'].values()) == op.memory_accounting['main_write_bytes'], f'{key}: selected operand closure')
        selected.update(key=key, seconds=seconds, accounting=copy.deepcopy(op.memory_accounting), selected_candidate=selected_candidate)
        recovered[tuple(key)] = selected
        receipt['completed_shapes'] = len(recovered)
        write_json(output / 'selected-operands.json', list(recovered.values()))
        write_json(output / 'receipt.json', receipt)
        print('PASS_FROZEN_SHAPE_OPERAND_RECOVERY', len(recovered), '/ 111', key, flush=True)

    rows = json.loads((matrix / 'comparison.json').read_text())
    check({r['case'] for r in rows} == set(CASE_IDS), 'Wrong case set')
    base = q.ModelSpec.from_json(model_path)
    results = []
    for source in rows:
        case = source['case']
        model = dataclasses.replace(base, prefill_tokens=int(case[1:4]), decode_steps=int(case.split('d')[1]))
        ledger = json.loads((matrix / case / 'operators.json').read_text())

        class FrozenCost:
            def __init__(self):
                self.position = 0
                self.keys = []

            def take(self, path, key, family):
                row = ledger[self.position]
                check(row['path'] == path, 'Source graph operator-path mismatch')
                if key is not None:
                    check(tuple(key) in recovered, f'Unrecovered graph shape: {key}')
                    shape = recovered[tuple(key)]
                    check(math.ceil((shape['seconds'] + row['bias_adapter_seconds']) * 1e9) == row['model_ns'], 'Source graph timing mismatch')
                    check(shape['accounting'] == row['mapper_accounting'], 'Source graph mapper-byte mismatch')
                self.keys.append(key)
                self.position += 1
                return q.Timing(family, 0, row['model_ns'])

            def matmul(self, m, k, n, bias=False, family='matmul'):
                return self.take('OFFICIAL_MATMUL', ('matmul', m, k, n), family)

            def batched_matmul(self, b, m, k, n, family):
                return self.take('OFFICIAL_BATCHED_MATMUL', ('batch', b, m, k, n), family)

            def softmax(self, b, m, n):
                return self.take('OFFICIAL_SOFTMAX', ('softmax', b, m, n), 'softmax')

            def vector(self, elements, ops_per_element, family, bytes_moved=0):
                return self.take('QWEN_VECTOR_ADAPTER', None, family)

        frozen = FrozenCost()
        plan = q.build_plan(model, frozen)
        check(len(plan.operators) == len(ledger) == frozen.position, 'Source graph size mismatch')
        full = empty_categories()
        phases, subobjects, by_operator = {}, {}, {}
        for op, row, key in zip(plan.operators, ledger, frozen.keys):
            check((op.index, op.phase, op.layer, op.name) == (row['index'], row['phase'], row['layer'], row['name']), 'Source graph/ledger identity mismatch')
            categories, objects = allocate_operator(op, row, None if key is None else recovered[key])
            add_categories(full, categories)
            add_categories(phases.setdefault(op.phase, empty_categories()), categories)
            add_categories(by_operator.setdefault(op.name, empty_categories()), categories)
            for name, values in objects.items():
                group = subobjects.setdefault(name, {direction: 0 for direction in DIRECTIONS})
                for direction in DIRECTIONS:
                    group[direction] += values[direction]
        for direction, field in [('read_bytes', 'known_read_bytes'), ('write_bytes', 'known_write_bytes'),
                                 ('unknown_direction_bytes', 'unclassified_io_bytes')]:
            check(category_total(full, direction) == source[field], f'{case}: full semantic bytes do not close')
            for phase, values in phases.items():
                check(category_total(values, direction) == source['phase_totals'][phase][field], f'{case}/{phase}: phase semantic bytes do not close')
        check(category_total(full) == source['total_boundary_bytes'], f'{case}: total semantic bytes do not close')
        results.append(dict(case=case, operator_count=len(ledger), categories=full, phases=phases,
                            subobjects=subobjects, by_operator=by_operator,
                            known_read_bytes=source['known_read_bytes'], known_write_bytes=source['known_write_bytes'],
                            unclassified_io_bytes=source['unclassified_io_bytes'], total_boundary_bytes=source['total_boundary_bytes'],
                            hardware_semantic_breakdown_status='NOT_AVAILABLE_FROM_AGGREGATE_COUNTERS'))
        receipt['completed_cases'] = len(results)
        write_json(output / 'receipt.json', receipt)
        print('PASS_CASE_SEMANTIC_CLOSURE', case, json.dumps(full), flush=True)
    for path, digest in inputs.items():
        check(sha(path) == digest, f'Frozen input changed during analysis: {path}')
    payload = dict(schema='LLMCOMPASS_MAIN_BOUNDARY_SEMANTIC_TRAFFIC_V1', status='PASS_EIGHT_CASE_SEMANTIC_ACCOUNTING_MODEL_ESTIMATE',
                   source_matrix=str(matrix), profile_sha256=manifest['profile_sha256'], units='byte; decimal GB/MB for plotting',
                   scope='Full Prefill plus every Decode step; direct selected-mapper boundary transfers plus explicit Qwen adapters, MODEL_ESTIMATE, not native physical DRAM attribution.',
                   definitions={
                       'weights': 'Learned projection, tied embedding/lm-head, normalization and bias accesses; traffic, not unique parameter footprint; no inference writes.',
                       'kv_cache': 'Persistent key_cache/value_cache appends and attention reads. Temporary projected K/V and their rotary transforms are activation.',
                       'activation': 'Hidden/residual, Q/K/V temporaries, MLP tensors, attention scores/probabilities, logits and official mapper C intermediate/partial-output transfers.',
                       'other': 'Unclassified official batch-surrogate extra IO; direction and exact tensor ownership unavailable; counted once in combined traffic only.'},
                   read_write_pie_rule='Read and Write pies use their own classified directional totals; unknown-direction IO excluded, disclosed separately. Combined pie includes this IO exactly once as Other.',
                   batch_semantic_rule='Preserve the official chosen candidate and its operand identities/multiplicity. K-concatenated surrogate bytes remain surrogate accounting, not a native executable batch graph or unique KV footprint.',
                   hardware_semantic_breakdown_status='NOT_AVAILABLE: archived NCU counters lack per-object attribution; no proportional allocation or model shares applied to hardware.',
                   cases=results)
    write_json(output / 'semantic-breakdown.json', payload)
    receipt.update(status=payload['status'], verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   source_unique_shape_latency_and_byte_parity=111, operator_semantic_closure_count=sum(x['operator_count'] for x in results),
                   semantic_breakdown_sha256=sha(output / 'semantic-breakdown.json'), selected_operands_sha256=sha(output / 'selected-operands.json'))
    write_json(output / 'receipt.json', receipt)
    print(receipt['status'], '8 cases', receipt['operator_semantic_closure_count'], 'operators', flush=True)


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    summary, output = pathlib.Path(summary), pathlib.Path(output)
    data = json.loads(summary.read_text())
    check(data['status'] == 'PASS_EIGHT_CASE_SEMANTIC_ACCOUNTING_MODEL_ESTIMATE', 'Incomplete semantic analysis')
    check(not output.exists(), 'Use a fresh plot directory; preserve old figures')
    output.mkdir(parents=True)
    # Static scientific figure palette; invariant semantic colors in all panels.
    colors = {'weights': '#4C78A8', 'kv_cache': '#59A14F', 'activation': '#F28E2B', 'other': '#9D9DA1'}
    names = {'weights': 'Weights', 'kv_cache': 'KV cache', 'activation': 'Activation', 'other': 'Other'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'svg.fonttype': 'none',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

    def amount(value):
        if value >= 1e9:
            return f'{value / 1e9:.3f} GB'
        if value >= 1e6:
            return f'{value / 1e6:.3f} MB'
        if value >= 1e3:
            return f'{value / 1e3:.2f} kB'
        return f'{value} B'

    def panel(ax, row, direction):
        values = [sum(row['categories'][category].values()) if direction == 'combined' else
                  row['categories'][category][direction + '_bytes'] for category in CATEGORIES]
        total = sum(values)
        check(total > 0, 'Cannot plot an empty directional total')
        wedges, _ = ax.pie(values, colors=[colors[c] for c in CATEGORIES], startangle=90,
                           counterclock=False, radius=0.90,
                           wedgeprops={'width': 0.36, 'edgecolor': 'white', 'linewidth': 1.0})
        ax.text(0, 0.06, amount(total), ha='center', va='center', fontsize=13)
        ax.text(0, -0.15, 'R + W + unknown' if direction == 'combined' else direction.upper(),
                ha='center', va='center', fontsize=10.5)
        title = row['case'].upper().replace('P0', 'P').replace('D0', 'D')
        ax.set_title(title, fontsize=15, pad=4)
        legend_pairs = []
        for index, (category, value) in enumerate(zip(CATEGORIES, values)):
            y = -1.20 - index * 0.19
            ax.plot([-1.12, -1.02], [y, y], color=colors[category], lw=8, solid_capstyle='butt')
            label = ax.text(-0.96, y, names[category], fontsize=11.5, va='center')
            percent = 100 * value / total
            number = ax.text(1.15, y, f'{percent:.2f}%  |  {amount(value)}', ha='right', va='center', fontsize=11.5)
            legend_pairs.append((label, number))
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-2.00, 1.08)
        ax.set_aspect('equal')
        ax._semantic_legend_pairs = legend_pairs
        return len(wedges)

    def check_layout(fig):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            for label, number in ax._semantic_legend_pairs:
                left = label.get_window_extent(renderer)
                right = number.get_window_extent(renderer)
                check(left.x1 + 4 <= right.x0, 'Semantic legend label/value overlap')

    outputs = []
    for direction in ('read', 'write', 'combined'):
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        for ax, row in zip(axes.flat, data['cases']):
            panel(ax, row, direction)
        title = {'read': 'Classified main-boundary READ traffic', 'write': 'Classified main-boundary WRITE traffic',
                 'combined': 'Total main-boundary traffic: Read + Write + unknown IO'}[direction]
        fig.suptitle('Qwen2.5-1.5B | RTX 4000 Ada | ' + title, fontsize=18, y=0.98)
        fig.text(0.5, 0.943, 'Full inference (Prefill + all Decode steps) | MODEL_ESTIMATE | Semantic accounting, not measured physical DRAM attribution',
                 ha='center', fontsize=12)
        footer = ('Unknown-direction batch IO is excluded from directional pies and retained once in the combined pie. Weights are read-only; zero slices are not inflated.'
                  if direction != 'combined' else 'Other contains unclassified batch-surrogate IO, counted once. KV cache excludes temporary K/V. Mapper partial outputs remain Activation.')
        fig.text(0.5, 0.025, footer, ha='center', fontsize=12)
        fig.subplots_adjust(left=0.035, right=0.985, top=0.89, bottom=0.06, hspace=0.18, wspace=0.23)
        check_layout(fig)
        for suffix in ('png', 'svg'):
            path = output / f'semantic-{direction}-8cases.{suffix}'
            fig.savefig(path, dpi=160)
            outputs.append(path)
        plt.close(fig)
    for row in data['cases']:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        for ax, direction in zip(axes, ('read', 'write', 'combined')):
            panel(ax, row, direction)
            ax.set_title({'read': 'READ (classified)', 'write': 'WRITE (classified)', 'combined': 'TOTAL (includes unknown IO)'}[direction], fontsize=13)
        case_title = row['case'].upper().replace('P0', 'P').replace('D0', 'D')
        fig.suptitle(f'{case_title} | Qwen2.5-1.5B | RTX 4000 Ada | MODEL_ESTIMATE', fontsize=16)
        fig.text(0.5, 0.05, f"Unknown-direction batch IO: {amount(row['unclassified_io_bytes'])}; not assigned to Read or Write. No hardware semantic counter attribution.",
                 ha='center', fontsize=11.5)
        fig.subplots_adjust(top=0.86, bottom=0.12, left=0.04, right=0.98, wspace=0.18)
        check_layout(fig)
        for suffix in ('png', 'svg'):
            path = output / f"{row['case']}-semantic-pies.{suffix}"
            fig.savefig(path, dpi=180)
            outputs.append(path)
        plt.close(fig)
    write_json(output / 'plot-manifest.json', dict(status='PASS_8_CASES_READ_WRITE_COMBINED_PIES',
               source=str(summary.resolve()), source_sha256=sha(summary), matplotlib_version=matplotlib.__version__,
               renderer_source=str(pathlib.Path(__file__).resolve()), renderer_sha256=sha(pathlib.Path(__file__)),
               legend_label_value_bounds_checked=True,
               category_colors=colors, percentages_use_actual_slice_denominator=True,
               plots={path.name: dict(sha256=sha(path), bytes=path.stat().st_size) for path in outputs}))
    print('PASS_8_CASES_READ_WRITE_COMBINED_PIES', len(outputs), 'PNG/SVG files')


HTML_COLORS = {'weights': '#4C78A8', 'kv_cache': '#59A14F',
               'activation': '#F28E2B', 'other': '#9D9DA1'}
HTML_NAMES = {'weights': 'Weights', 'kv_cache': 'KV cache',
              'activation': 'Activation', 'other': 'Other / unknown IO'}
SUBOBJECT_NAMES = {
    'mlp_intermediates': 'MLP intermediates, including mapper partial outputs',
    'hidden_and_residual_activations': 'Hidden states and residuals',
    'qkv_attention_activations': 'Q/K/V and attention-output temporaries',
    'attention_intermediates': 'Attention scores and probabilities',
    'logits': 'Logits',
}


def validate_html_inputs(semantic, paired):
    """Validate aggregate partitions; never infer hardware semantic shares."""
    check(semantic['status'] == 'PASS_EIGHT_CASE_SEMANTIC_ACCOUNTING_MODEL_ESTIMATE',
          'Incomplete semantic source')
    check(paired['status'] == 'PASS_EIGHT_CASE_PAIRED_MEASUREMENTS_NOT_ACCURACY_ACCEPTANCE',
          'Incomplete paired hardware source')
    check(len(semantic['cases']) == len(paired['cases']) == len(CASE_IDS),
          'Eight cases required')
    sem = {r['case']: r for r in semantic['cases']}
    native = {r['case']: r for r in paired['cases']}
    check(set(sem) == set(native) == set(CASE_IDS), 'Duplicate or missing case')
    for case in CASE_IDS:
        row, model = sem[case], native[case]['model']
        check(set(row['categories']) == set(CATEGORIES), 'Unexpected semantic category')
        for category, values in row['categories'].items():
            check(set(values) == set(DIRECTIONS), 'Unexpected byte direction')
            check(all(type(v) is int and v >= 0 for v in values.values()),
                  'Byte amounts must be nonnegative integers')
            if category != 'other':
                check(values['unknown_direction_bytes'] == 0, 'Unknown IO ownership changed')
        check(row['categories']['weights']['write_bytes'] == 0, 'Inference weight writes')
        check(row['categories']['other'] == {
            'read_bytes': 0, 'write_bytes': 0,
            'unknown_direction_bytes': row['unclassified_io_bytes']},
            'Unclassified IO must remain Other with no invented direction')
        for direction, field in zip(DIRECTIONS, ('known_read_bytes', 'known_write_bytes',
                                               'unclassified_io_bytes')):
            check(category_total(row['categories'], direction) == row[field] == model[field],
                  f'{case}: semantic/model directional mismatch')
            check(sum(category_total(v, direction) for v in row['phases'].values()) == row[field],
                  f'{case}: phase partition mismatch')
            check(sum(v[direction] for v in row['subobjects'].values()) == row[field],
                  f'{case}: subobject partition mismatch')
        for category in CATEGORIES:
            for direction in DIRECTIONS:
                check(sum(v[direction] for k, v in row['subobjects'].items()
                          if k.split('/')[0] == category) == row['categories'][category][direction],
                      f'{case}: subobject ownership mismatch')
                check(sum(v[category][direction] for v in row['phases'].values()) ==
                      row['categories'][category][direction], f'{case}: phase ownership mismatch')
        check(category_total(row['categories']) == row['total_boundary_bytes'] ==
              model['total_boundary_bytes'], f'{case}: total mismatch')
        check(row['operator_count'] == model['operator_count'], 'Operator count mismatch')
        check(math.isclose(model['model_GBps'],
              row['total_boundary_bytes'] / (model['model_ms'] * 1e6), rel_tol=1e-10),
              'Model bandwidth denominator mismatch')
        for path in ('native', 'explicit'):
            hardware = native[case]['hardware'][path]
            check(hardware['full_ms'] > 0 and hardware['read_bytes'] >= 0 and
                  hardware['write_bytes'] >= 0, 'Invalid measured totals')
            check(math.isclose(hardware['effective_GBps'],
                  (hardware['read_bytes'] + hardware['write_bytes']) /
                  (hardware['full_ms'] * 1e6), rel_tol=1e-10),
                  'Hardware effective bandwidth denominator mismatch')
    return sem, native


def html_amount(value):
    for scale, unit in ((1e9, 'GB'), (1e6, 'MB'), (1e3, 'kB')):
        if value >= scale:
            return f'{value / scale:,.3f} {unit}'
    return f'{value:,} B'


def html_case_name(case):
    return f"P{int(case[1:4])}D{int(case[5:])}"


def html_pie(categories, direction, case):
    """Draw actual angular shares, including arbitrarily small positive slices."""
    check(direction in ('read', 'write', 'combined'), 'Unknown pie direction')
    values = [sum(categories[c].values()) if direction == 'combined' else
              categories[c][direction + '_bytes'] for c in CATEGORIES]
    total = sum(values)
    check(total > 0, 'Empty pie denominator')
    title = {'read': 'Read', 'write': 'Write', 'combined': 'Total'}[direction]
    boundary = 'Read + Write + unknown IO' if direction == 'combined' else 'Classified ' + direction
    segments, legend, angle = [], [], -math.pi / 2
    def point(radius, a):
        return f'{120 + radius * math.cos(a):.8f},{106 + radius * math.sin(a):.8f}'
    for category, value in zip(CATEGORIES, values):
        share = value / total
        if value:
            end = angle + share * 2 * math.pi
            if share == 1:
                path = ('M120,24 A82,82 0 1 1 120,188 A82,82 0 1 1 120,24 '
                        'M120,54 A52,52 0 1 0 120,158 A52,52 0 1 0 120,54 Z')
            else:
                large = int(share > 0.5)
                path = (f'M{point(82, angle)} A82,82 0 {large} 1 {point(82, end)} '
                        f'L{point(52, end)} A52,52 0 {large} 0 {point(52, angle)} Z')
            segments.append(f'<path data-category="{category}" data-bytes="{value}" '
                f'data-share="{share:.16g}" d="{path}" fill="{HTML_COLORS[category]}" '
                f'fill-rule="evenodd"><title>{HTML_NAMES[category]}: {value:,} bytes; '
                f'{share * 100:.6f}% of {total:,} bytes</title></path>')
            angle = end
        pct = '&lt;0.01%' if 0 < share < 0.0001 else f'{share * 100:.2f}%'
        legend.append(f'<tr data-category="{category}" data-bytes="{value}" '
            f'data-share="{share:.16g}"><th scope="row"><span class="swatch" '
            f'style="background:{HTML_COLORS[category]}"></span>{HTML_NAMES[category]}</th>'
            f'<td title="{value:,} bytes">{html_amount(value)}</td><td>{pct}</td></tr>')
    return (f'<figure class="pie" data-case="{case}" data-direction="{direction}" '
            f'data-total-bytes="{total}"><figcaption>{html_case_name(case)}</figcaption>'
            f'<svg viewBox="0 0 240 214" role="img" aria-label="{html_case_name(case)} '
            f'{title} model semantic traffic: {total:,} bytes">'
            f'<title>{html_case_name(case)} {title}: MODEL_ESTIMATE</title>'
            f'{"".join(segments)}<text x="120" y="105" text-anchor="middle" class="pie-total">'
            f'{html_amount(total)}</text><text x="120" y="124" text-anchor="middle" '
            f'class="pie-subtitle">{title.upper()}</text></svg>'
            f'<table class="legend"><caption class="sr-only">{title} category bytes and shares</caption>'
            f'<tbody>{"".join(legend)}</tbody></table></figure>')


def build_html_report(semantic, paired, provenance, report_date):
    sem, measured = validate_html_inputs(semantic, paired)
    panels = []
    for case in CASE_IDS:
        row, comparison = sem[case], measured[case]
        model = comparison['model']
        native, explicit = comparison['hardware']['native'], comparison['hardware']['explicit']
        metric_rows = []
        for label, key, formatter in (
                ('Read', 'read_bytes', html_amount), ('Write', 'write_bytes', html_amount),
                ('Full-window time', 'full_ms', lambda v: f'{v:,.3f} ms'),
                ('Effective bandwidth', 'effective_GBps', lambda v: f'{v:,.2f} GB/s')):
            model_key = {'read_bytes': 'known_read_bytes', 'write_bytes': 'known_write_bytes',
                         'full_ms': 'model_ms', 'effective_GBps': 'model_GBps'}[key]
            metric_rows.append(f'<tr><th scope="row">{label}</th><td>{formatter(model[model_key])}</td>'
                              f'<td>{formatter(native[key])}</td><td>{formatter(explicit[key])}</td></tr>')
        metric_rows.append(f'<tr><th scope="row">Unknown-direction IO</th>'
                           f'<td>{html_amount(row["unclassified_io_bytes"])}</td>'
                           '<td>Not separately identified</td><td>Not separately identified</td></tr>')
        activation = row['categories']['activation']['write_bytes']
        details = []
        for name, amount in sorted(row['subobjects'].items(),
                                  key=lambda item: item[1]['write_bytes'], reverse=True):
            if not name.startswith('activation/'):
                continue
            short = name.split('/')[1]
            check(short in SUBOBJECT_NAMES, 'New activation object needs a report label')
            details.append(f'<tr><th scope="row">{html.escape(SUBOBJECT_NAMES[short])}</th>'
                f'<td title="{amount["write_bytes"]:,} bytes">{html_amount(amount["write_bytes"])}</td>'
                f'<td>{100 * amount["write_bytes"] / activation:.2f}%</td></tr>')
        panels.append(f'<section class="case" id="{case}"><header class="case-header">'
            f'<h2>{html_case_name(case)}</h2><span>{row["operator_count"]:,} model operators '
            f'&middot; Prefill + {int(case[5:])} Decode steps</span></header>'
            f'<p class="unknown">Unknown-direction IO: '
            f'<strong>{html_amount(row["unclassified_io_bytes"])}</strong>. Excluded from Read and '
            'Write pies; included exactly once as Other in Total.</p>'
            '<div class="table-wrap"><table class="comparison"><caption>Matched workload window; '
            'hardware counters have no semantic ownership labels</caption><thead><tr><th>Metric</th>'
            '<th>LLMCompass model</th><th>Native SGLang</th><th>Explicit reference &mdash; diagnostic</th>'
            f'</tr></thead><tbody>{"".join(metric_rows)}</tbody></table></div>'
            '<details class="activation-detail"><summary>Activation Write subobjects</summary>'
            f'<p>Denominator: {html_amount(activation)} of Activation Write only. These components '
            'are already included in Activation above, not additional traffic.</p>'
            '<div class="table-wrap"><table><thead><tr><th>Subobject</th><th>Write</th>'
            f'<th>Share of Activation Write</th></tr></thead><tbody>{"".join(details)}</tbody>'
            '</table></div></details></section>')
    compact = dict(semantic_status=semantic['status'], source_provenance=provenance,
                   hardware_semantic_attribution=False,
                   cases=[dict(case=case, categories=sem[case]['categories'],
                               subobjects=sem[case]['subobjects'],
                               model=measured[case]['model'], hardware=measured[case]['hardware'])
                          for case in CASE_IDS])
    inline = json.dumps(compact, separators=(',', ':')).replace('<', '\\u003c')
    overview_sections = []
    for direction, title, denominator in (
            ('read', 'Read traffic', 'Classified Read bytes; unknown-direction IO excluded'),
            ('write', 'Write traffic', 'Classified Write bytes; unknown-direction IO excluded'),
            ('combined', 'Total traffic', 'Read + Write + unknown-direction IO, counted once')):
        charts = ''.join(html_pie(sem[case]['categories'], direction, case) for case in CASE_IDS)
        overview_sections.append(f'<section class="traffic-overview" id="{direction}-traffic">'
            f'<h2>{title}</h2><div class="overview-frame"><p class="overview-caption">'
            f'Qwen2.5-1.5B &middot; RTX 4000 Ada &middot; Full inference &middot; MODEL_ESTIMATE</p>'
            f'<div class="overview-grid" data-direction="{direction}">{charts}</div>'
            f'<p class="overview-footnote">{denominator}. Weights are read-only; '
            'zero slices are not inflated.</p></div></section>')
    navigation = ('<a href="#read-traffic">Read</a><a href="#write-traffic">Write</a>'
                  '<a href="#combined-traffic">Total</a>'
                  '<a href="#hardware-comparisons">Hardware totals and details</a>')
    definitions = ''.join(f'<dt>{HTML_NAMES[c]}</dt><dd>{html.escape(semantic["definitions"][c])}</dd>'
                          for c in CATEGORIES)
    identities = ''.join(f'<dt>{html.escape(k.replace("_", " "))}</dt>'
                         f'<dd><code>{html.escape(str(v))}</code></dd>'
                         for k, v in provenance.items())
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'none'; img-src data:">
<title>Qwen2.5-1.5B semantic traffic | RTX 4000 Ada | Eight cases</title>
<style>
:root {{color-scheme:light dark;--bg:#f4f6f9;--surface:#fff;--text:#1f2937;--muted:#586577;--line:#dce2ea;--accent:#315f91;--note:#fff2dd}}
@media(prefers-color-scheme:dark){{:root{{--bg:#151a23;--surface:#202733;--text:#e6edf7;--muted:#b0bfd1;--line:#3b4758;--accent:#96bde5;--note:#332c20}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1220px;margin:auto;padding:34px 30px 48px}}h1{{font-size:30px;line-height:1.2;font-weight:600;margin:8px 0 12px}}h2{{font-size:23px;margin:0;font-weight:600}}
p{{margin:10px 0}}a{{color:var(--accent)}}.eyebrow{{font-size:13px;color:var(--muted);letter-spacing:.03em}}.scope{{max-width:1050px}}
.notice{{background:var(--note);padding:14px 18px;margin:20px 0}}nav{{display:flex;flex-wrap:wrap;gap:9px 20px;margin:20px 0 28px}}nav a{{font-weight:600;text-decoration:none}}
.definitions{{margin:18px 0}}summary{{cursor:pointer;font-weight:600;padding:8px 0}}dl{{display:grid;grid-template-columns:minmax(150px,1fr) 4fr;gap:9px 18px;margin:12px 0}}
dt{{font-weight:600}}dd{{margin:0;overflow-wrap:anywhere}}.case{{background:var(--surface);margin:20px 0;padding:24px 26px;scroll-margin-top:16px}}
.case-header{{display:flex;align-items:baseline;justify-content:space-between;gap:10px;flex-wrap:wrap;padding-bottom:17px}}.case-header span{{font-size:13px;color:var(--muted)}}
.traffic-overview{{margin:38px 0 48px;scroll-margin-top:20px}}.traffic-overview>h2{{font-size:28px;margin-bottom:16px}}
.overview-frame{{background:var(--surface);border:1px solid var(--line);border-radius:14px;box-shadow:0 3px 9px rgb(0 0 0 / 5%);padding:20px 22px}}
.overview-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:28px 20px}}.overview-caption{{text-align:center;color:var(--muted);font-size:12px;margin:0 0 22px}}
.overview-footnote{{text-align:center;color:var(--muted);font-size:12px;margin:22px 0 0}}figure{{margin:0;min-width:0}}figcaption{{text-align:center;font-size:15px;font-weight:600}}
.pie svg{{display:block;width:100%;max-width:220px;margin:3px auto 0;overflow:visible}}
.pie-total{{fill:var(--text);font-size:16px;font-weight:600}}.pie-subtitle{{fill:var(--muted);font-size:13px;letter-spacing:.02em}}
.case-comparisons{{margin:30px 0}}.case-comparisons>summary{{font-size:20px}}
table{{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}}
th:first-child{{text-align:left}}tbody th{{font-weight:400}}thead th{{font-weight:600;color:var(--muted)}}caption{{text-align:left;padding:8px 0;color:var(--muted);font-size:12px}}
.legend{{font-size:12px}}.legend th,.legend td{{padding:5px 2px;border:0;white-space:nowrap}}.legend th{{font-weight:400;white-space:normal}}.swatch{{display:inline-block;width:9px;height:9px;margin-right:5px}}
.unknown{{font-size:13px;color:var(--muted);margin:17px 0 9px}}.table-wrap{{overflow-x:auto}}.comparison{{min-width:540px}}.activation-detail{{margin-top:16px}}
.activation-detail p{{font-size:13px;color:var(--muted)}}.activation-detail table{{min-width:460px}}footer{{margin-top:28px;color:var(--muted);font-size:13px}}code{{font-size:12px;overflow-wrap:anywhere}}
.sr-only{{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}}
@media(max-width:980px){{main{{padding:24px 20px}}.overview-grid{{grid-template-columns:repeat(2,minmax(0,1fr));gap:26px 20px}}.case{{padding:22px 18px}}}}
@media(max-width:560px){{main{{padding:18px 12px}}h1{{font-size:25px}}.overview-grid{{grid-template-columns:1fr;gap:26px}}.overview-frame{{padding:18px 16px}}.pie{{max-width:410px;width:100%;margin:auto}}.legend{{font-size:13px}}.legend th,.legend td{{padding:5px 4px}}dl{{grid-template-columns:1fr;gap:5px}}dd{{margin-bottom:10px}}.traffic-overview>h2{{font-size:25px}}}}
@media print{{:root{{--bg:white;--surface:white;--text:black;--muted:#444;--line:#ccc;--note:#f4f4f4}}main{{max-width:none;padding:0}}nav{{display:none}}.traffic-overview{{break-inside:avoid}}.overview-grid{{grid-template-columns:repeat(4,minmax(0,1fr))}}.overview-frame{{box-shadow:none}}}}
</style></head><body><main>
<div class="eyebrow">COMPILED {html.escape(report_date)} &middot; ASIA/DUBAI &middot; MEASUREMENTS: 2026-10-08</div>
<h1>Semantic traffic across eight Qwen inference cases</h1>
<p class="scope">Qwen2.5-1.5B-Instruct &middot; NVIDIA RTX 4000 Ada &middot; batch size 1 &middot; all 28 layers.
P is the prompt token count; D is the number of subsequent separate one-token Decode passes.
Every chart covers the complete Prefill-plus-all-Decode workload, not a single layer or a unique tensor footprint.</p>
<div class="notice"><strong>MODEL_ESTIMATE, not measured physical-DRAM semantic attribution.</strong>
Charts partition official selected-mapper main-memory transfers plus labeled Qwen adapter accesses.
Hardware totals are separate; model percentages are never applied to hardware.</div>
<nav aria-label="Traffic directions and detailed comparisons">{navigation}</nav>
<details class="definitions"><summary>Categories, denominators and units</summary><dl>{definitions}</dl>
<p>Read and Write percentages use their own classified directional totals. Total uses Read + Write + unknown-direction IO,
counted once. A zero-byte category has no visible slice and is never enlarged. Unknown IO is not silently assigned to a direction.</p>
<p>KV means persistent key/value cache; temporary projected K/V remains Activation. MLP means multilayer perceptron.
DRAM means dynamic random-access memory; IO means input/output. GB = 10<sup>9</sup> bytes, MB = 10<sup>6</sup> bytes,
kB = 10<sup>3</sup> bytes, ms = 10<sup>-3</sup> seconds. GB/s is workload-effective bandwidth.
Hardware bandwidth uses DRAM counter bytes divided by an independently measured, unprofiled CUDA-event median for the same workload;
model bandwidth additionally includes unknown IO and uses official model time. Neither is instantaneous controller throughput.
BF16 means brain floating point, FP16/FP32 mean 16/32-bit floating point; the model uses a two-byte FP16 storage proxy for BF16.</p></details>
{"".join(overview_sections)}
<details class="case-comparisons" id="hardware-comparisons"><summary>Hardware totals and activation Write details</summary>
<p>The latest native/explicit hardware counters appear only as totals. The explicit reference is diagnostic,
with FP32 materialized attention and unresolved strict numerical/timing gates.</p>{"".join(panels)}</details>
<footer><details><summary>Provenance and validation boundary</summary>
<p>All eight semantic partitions match the latest paired model Read, Write, unknown IO, total bytes and operator counts exactly.
The semantic source closes 23,322 model operators. No simulation, cache policy, mapper transfer, hardware profile or HBFSim parameter is changed.
HBFSim is the independent memory backend extension; it contributes zero to official model time here.
SGLang is the native inference framework. NCU means NVIDIA Nsight Compute, the hardware-counter profiler.
GPU means graphics processing unit; SM means streaming multiprocessor.</p>
<p>Latest paired collection: 320 unprofiled workflows, 80 full counter windows and 120 warm isolated group windows.
Hardware per-object semantic attribution is unavailable. The strict cross-implementation KV check passes 0/8,
and the strict elementwise logit check passes 5/8. All full-phase logit relative L2 errors are below 2%.
The frozen model SM clock is 2175 MHz; sampled hardware endpoints are usually 2325 MHz, not fixed-clock validation.
P128D8 explicit timing rechecks differ by +2.78% to +13.35%. Original samples and failed evidence remain intact.
No all-cases-under-ten-percent or pure-fusion causal claim is made.</p><dl>{identities}</dl>
</details><p>Self-contained offline report &middot; consistent historical category colors &middot; no external scripts, images or fonts.</p></footer>
<script type="application/json" id="report-data">{inline}</script>
</main></body></html>'''


def offline_html(summary, paired_path, output, report_date):
    summary, paired_path, output = map(pathlib.Path, (summary, paired_path, output))
    check(not output.exists(), 'Use a fresh HTML result directory; preserve historical files')
    semantic = json.loads(summary.read_text(encoding='utf-8'))
    paired = json.loads(paired_path.read_text(encoding='utf-8'))
    provenance = dict(semantic_source=str(summary.resolve()), semantic_sha256=sha(summary),
                      paired_source=str(paired_path.resolve()), paired_sha256=sha(paired_path),
                      source_matrix=semantic['source_matrix'], profile_sha256=semantic['profile_sha256'])
    rendered = build_html_report(semantic, paired, provenance, report_date)
    output.mkdir(parents=True)
    target = output / 'qwen-semantic-traffic.html'
    target.write_text(rendered, encoding='utf-8')
    write_json(output / 'manifest.json', dict(status='PASS_8_CASE_OFFLINE_SEMANTIC_HTML_ACCOUNTING',
        cases=list(CASE_IDS), charts=24, provenance=provenance,
        renderer_sha256=sha(__file__), html=dict(path=target.name, bytes=target.stat().st_size,
        sha256=sha(target)), source_read_write_unknown_total_closure=True,
        per_category_subobject_and_phase_closure=True,
        model_percentages_applied_to_hardware=False, external_dependencies=False,
        layout='DIRECTION_GROUPED_OVERVIEW', direction_groups=3,
        desktop_grid_columns=4, cases_per_direction=8,
        report_date=report_date, report_timezone='Asia/Dubai'))
    print('PASS_8_CASE_OFFLINE_SEMANTIC_HTML_ACCOUNTING', target, target.stat().st_size, 'bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, argument in [('analyze', 'matrix'), ('plot', 'summary')]:
        command = commands.add_parser(name)
        command.add_argument(argument, type=pathlib.Path)
        command.add_argument('output', type=pathlib.Path)
    command = commands.add_parser('html')
    command.add_argument('summary', type=pathlib.Path)
    command.add_argument('paired', type=pathlib.Path)
    command.add_argument('output', type=pathlib.Path)
    command.add_argument('--report-date', required=True, help='Client-side date, YYYY-MM-DD')
    args = parser.parse_args()
    if args.command == 'analyze':
        analyze(args.matrix, args.output)
    elif args.command == 'plot':
        plot(args.summary, args.output)
    else:
        datetime.date.fromisoformat(args.report_date)
        offline_html(args.summary, args.paired, args.output, args.report_date)


if __name__ == '__main__':
    main()
