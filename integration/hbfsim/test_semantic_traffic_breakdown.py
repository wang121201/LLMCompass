"""Semantic ownership and direct operand byte-accounting regression tests."""
import types
import unittest
import json
import pathlib
import copy

import semantic_traffic_breakdown as semantic


def access(kind, label, size=2, target='HBM'):
    return types.SimpleNamespace(kind=kind, label=label, bytes=size, target=target)


class SemanticTests(unittest.TestCase):
    def test_kv_cache_is_not_temporary_kv(self):
        self.assertEqual(semantic.classify_access(access('read', 'key_cache'))[0], 'kv_cache')
        self.assertEqual(semantic.classify_access(access('write', 'value_cache'))[0], 'kv_cache')
        for label in ('k', 'v', 'k_rope'):
            self.assertEqual(semantic.classify_access(access('read', label))[0], 'activation')

    def test_learned_norm_bias_and_tied_head_are_weights(self):
        for label in ('model.norm.weight', 'model.layers.0.self_attn.q_proj.bias', 'model.embed_tokens'):
            self.assertEqual(semantic.classify_access(access('read', label, target='HBF_STATIC'))[0], 'weights')
        with self.assertRaises(ValueError):
            semantic.classify_access(access('write', 'model.norm.weight', target='HBF_STATIC'))

    def test_partial_output_reread_is_activation_not_weight(self):
        op = types.SimpleNamespace(index=0, name='gate_proj',
             reads=(access('read', 'norm_hidden'), access('read', 'model.layers.0.mlp.gate_proj.weight', target='HBF_STATIC')),
             writes=(access('write', 'gate'),))
        row = dict(path='OFFICIAL_MATMUL', known_read_bytes=18, known_write_bytes=8, unclassified_io_bytes=0)
        recovered = dict(operands={'read': {'A': 2, 'B': 12, 'C': 4}, 'write': {'A': 0, 'B': 0, 'C': 8}})
        cats, _ = semantic.allocate_operator(op, row, recovered)
        self.assertEqual(cats['weights']['read_bytes'], 12)
        self.assertEqual(cats['activation']['read_bytes'], 6)
        self.assertEqual(cats['activation']['write_bytes'], 8)

    def test_unknown_io_is_once_and_never_given_a_direction(self):
        op = types.SimpleNamespace(index=0, name='attention_score',
             reads=(access('read', 'q_rope'), access('read', 'key_cache')), writes=(access('write', 'attention_scores'),))
        row = dict(path='OFFICIAL_BATCHED_MATMUL', known_read_bytes=10, known_write_bytes=4, unclassified_io_bytes=6)
        recovered = dict(operands={'read': {'A': 4, 'B': 6, 'C': 0}, 'write': {'A': 0, 'B': 0, 'C': 4}})
        cats, _ = semantic.allocate_operator(op, row, recovered)
        self.assertEqual(cats['other'], dict(read_bytes=0, write_bytes=0, unknown_direction_bytes=6))
        self.assertEqual(semantic.category_total(cats), 20)

    def test_operand_bytes_use_actual_remainder_tile_extents(self):
        counts = semantic.operand_counts([dict(direction='read', tensor='B', rows=3, columns=5)], 2)
        self.assertEqual(counts['read']['B'], 30)

    def test_new_objects_and_nonclosing_accounting_fail(self):
        with self.assertRaises(ValueError):
            semantic.classify_access(access('read', 'unknown_object'))
        op = types.SimpleNamespace(index=0, name='silu', reads=(access('read', 'gate', 4),), writes=(access('write', 'gate_silu', 4),))
        with self.assertRaises(ValueError):
            semantic.allocate_operator(op, dict(path='QWEN_VECTOR_ADAPTER', known_read_bytes=5, known_write_bytes=4, unclassified_io_bytes=0), None)


class OfflineHTMLTests(unittest.TestCase):
    def fixtures(self):
        rows, paired = [], []
        for case in semantic.CASE_IDS:
            cats = {
                'weights': dict(read_bytes=80, write_bytes=0, unknown_direction_bytes=0),
                'kv_cache': dict(read_bytes=5, write_bytes=3, unknown_direction_bytes=0),
                'activation': dict(read_bytes=15, write_bytes=7, unknown_direction_bytes=0),
                'other': dict(read_bytes=0, write_bytes=0, unknown_direction_bytes=2),
            }
            objects = {f'{c}/{n}': copy.deepcopy(cats[c]) for c, n in (
                ('weights', 'mlp_weights'), ('kv_cache', 'key_cache'),
                ('activation', 'mlp_intermediates'), ('other', 'unclassified_batch_surrogate_io'))}
            totals = dict(known_read_bytes=100, known_write_bytes=10,
                          unclassified_io_bytes=2, total_boundary_bytes=112,
                          operator_count=(int(case[5:]) + 1) * 507)
            rows.append(dict(case=case, categories=cats, subobjects=objects,
                             phases={'prefill': copy.deepcopy(cats)}, **totals))
            paired.append(dict(case=case, model=dict(model_ms=1, model_GBps=112 / 1e6, **totals),
                hardware={k: dict(read_bytes=r, write_bytes=w, full_ms=t,
                                  effective_GBps=(r + w) / (t * 1e6))
                          for k, r, w, t in [('native', 90, 8, 5), ('explicit', 91, 9, 6)]}))
        return (dict(status='PASS_EIGHT_CASE_SEMANTIC_ACCOUNTING_MODEL_ESTIMATE', cases=rows,
                     definitions={c: c + ' definition' for c in semantic.CATEGORIES}),
                dict(status='PASS_EIGHT_CASE_PAIRED_MEASUREMENTS_NOT_ACCURACY_ACCEPTANCE', cases=paired))

    def test_read_write_and_combined_have_distinct_denominators(self):
        payload, _ = self.fixtures()
        cats = payload['cases'][0]['categories']
        for direction, total in [('read', 100), ('write', 10), ('combined', 112)]:
            output = semantic.html_pie(cats, direction, 'p032d02')
            self.assertIn(f'data-total-bytes="{total}"', output)
        svg = semantic.html_pie(cats, 'write', 'p032d02').split('<table')[0]
        self.assertNotIn('data-category="weights"', svg)
        self.assertNotIn('data-category="other"', svg)

    def test_full_offline_report_preserves_all_cases_and_no_remote_resources(self):
        payload, paired = self.fixtures()
        output = semantic.build_html_report(payload, paired, {}, '2026-10-09')
        self.assertEqual(output.count('<figure class="pie"'), 24)
        self.assertEqual(output.count('class="case" id='), 8)
        self.assertEqual(output.count('Activation Write subobjects</summary>'), 8)
        self.assertEqual(output.count('class="overview-grid"'), 3)
        for direction in ('read', 'write', 'combined'):
            self.assertIn(f'id="{direction}-traffic"', output)
            self.assertEqual(output.count(f'data-direction="{direction}" data-total-bytes='), 8)
        self.assertIn('id="hardware-comparisons"', output)
        self.assertNotIn(' src=', output)
        self.assertNotIn('https://', output)
        self.assertIn('model percentages are never applied to hardware', output)
        for case in semantic.CASE_IDS:
            self.assertIn(f'id="{case}"', output)

    def test_stale_paired_model_or_nonclosing_phase_rejected(self):
        payload, paired = self.fixtures()
        paired['cases'][0]['model']['known_write_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'directional mismatch'):
            semantic.validate_html_inputs(payload, paired)
        payload, paired = self.fixtures()
        payload['cases'][0]['phases']['prefill']['activation']['write_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'phase partition mismatch'):
            semantic.validate_html_inputs(payload, paired)

    def test_subobject_reclassification_and_hardware_bandwidth_rejected(self):
        payload, paired = self.fixtures()
        row = payload['cases'][0]['subobjects']
        row['weights/mlp_weights']['read_bytes'] -= 1
        row['activation/mlp_intermediates']['read_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'ownership mismatch'):
            semantic.validate_html_inputs(payload, paired)
        payload, paired = self.fixtures()
        paired['cases'][0]['hardware']['native']['effective_GBps'] *= 2
        with self.assertRaisesRegex(ValueError, 'bandwidth denominator'):
            semantic.validate_html_inputs(payload, paired)

    def test_tiny_slices_are_not_inflated_and_single_positive_category_works(self):
        cats = semantic.empty_categories()
        cats['weights']['read_bytes'] = 10**12
        cats['kv_cache']['read_bytes'] = 1
        output = semantic.html_pie(cats, 'read', 'p032d02')
        self.assertIn('data-bytes="1" data-share=', output)
        self.assertIn('&lt;0.01%', output)
        cats['kv_cache']['read_bytes'] = 0
        output = semantic.html_pie(cats, 'read', 'p032d02')
        self.assertIn('data-share="1"', output)
        self.assertNotIn('nan', output.lower())


def run_completed_result_checks(root):
    """Independent role/shape cross-check without using the graph allocator."""
    root = pathlib.Path(root)
    payload = json.loads((root / 'semantic-breakdown.json').read_text())
    receipt = json.loads((root / 'receipt.json').read_text())
    matrix = pathlib.Path(payload['source_matrix'])
    source = {row['case']: row for row in json.loads((matrix / 'comparison.json').read_text())}
    recovered = {tuple(row['key']): row for row in json.loads((root / 'selected-operands.json').read_text())}
    semantic.check(payload['status'] == receipt['status'] == 'PASS_EIGHT_CASE_SEMANTIC_ACCOUNTING_MODEL_ESTIMATE', 'Incomplete output')
    semantic.check(len(payload['cases']) == len(source) == 8 and len(recovered) == 111, 'Incorrect case/shape denominator')
    h, kv, f, heads, dim, vocab, layers, word = 1536, 256, 8960, 12, 128, 151936, 28, 2
    checked = 0
    for row in payload['cases']:
        case = row['case']; original = source[case]
        ledger = json.loads((matrix / case / 'operators.json').read_text())
        p, d = int(case[1:4]), int(case.split('d')[1])
        weights_read = cache_read = cache_write = 0
        for op in ledger:
            step = 0 if op['phase'] == 'prefill' else int(op['phase'].split('_')[1])
            tokens, context = (p if step == 0 else 1), p + step
            if op['path'] == 'OFFICIAL_MATMUL':
                k, n = {
                    'q_proj': (h, h), 'k_proj': (h, kv), 'v_proj': (h, kv),
                    'o_proj': (h, h), 'gate_proj': (h, f), 'up_proj': (h, f),
                    'down_proj': (f, h), 'lm_head': (h, vocab),
                }[op['name']]
                m = 1 if op['name'] == 'lm_head' else tokens
                shape = recovered[('matmul', m, k, n)]
                weights_read += shape['operands']['read']['B']
                if op['name'] in ('q_proj', 'k_proj', 'v_proj'):
                    weights_read += n * word
            elif op['name'] in ('input_rmsnorm', 'post_attention_rmsnorm', 'final_rmsnorm'):
                weights_read += h * word
            elif op['name'] == 'token_embedding':
                weights_read += tokens * h * word
            elif op['name'] in ('attention_score', 'attention_value'):
                k, n = (dim, context) if op['name'] == 'attention_score' else (context, dim)
                cache_read += recovered[('batch', heads, tokens, k, n)]['operands']['read']['B']
            elif op['name'] == 'kv_append':
                cache_write += 2 * tokens * kv * word
        cats = row['categories']
        semantic.check(set(cats) == set(semantic.CATEGORIES), f'{case}: category set')
        semantic.check(cats['weights']['read_bytes'] == weights_read, f'{case}: independent weight-read total')
        semantic.check(cats['weights']['write_bytes'] == 0, f'{case}: nonzero inference weight writes')
        semantic.check(cats['kv_cache']['read_bytes'] == cache_read and cats['kv_cache']['write_bytes'] == cache_write,
                       f'{case}: independent persistent-KV totals')
        semantic.check(cache_write == 2 * (p + d) * kv * word * layers, f'{case}: useful KV append bytes')
        semantic.check(cats['other'] == dict(read_bytes=0, write_bytes=0, unknown_direction_bytes=original['unclassified_io_bytes']),
                       f'{case}: unknown IO given a false direction')
        for direction, field in [('read_bytes', 'known_read_bytes'), ('write_bytes', 'known_write_bytes'),
                                 ('unknown_direction_bytes', 'unclassified_io_bytes')]:
            semantic.check(semantic.category_total(cats, direction) == original[field], f'{case}: directional closure')
            semantic.check(sum(values[direction] for values in row['subobjects'].values()) == original[field], f'{case}: subobject closure')
            semantic.check(sum(semantic.category_total(values, direction) for values in row['by_operator'].values()) == original[field],
                           f'{case}: operator-group closure')
            semantic.check(sum(semantic.category_total(values, direction) for values in row['phases'].values()) == original[field],
                           f'{case}: phase closure')
        semantic.check(semantic.category_total(cats) == original['total_boundary_bytes'], f'{case}: combined closure')
        print('PASS_INDEPENDENT_WEIGHT_KV_AND_PARTITION_CHECKS', case)
        checked += 1
    return checked


if __name__ == '__main__':
    unittest.main()
