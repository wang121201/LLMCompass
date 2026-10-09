"""Negative controls for the independent static-batch aggregate verifier."""
import copy
import unittest

import verify_batch_inference as v


def fixture():
    rows = []
    for batch in v.BATCHES:
        phases = {}
        for phase in v.PHASES:
            kv = 2 * 28 * batch * (128 if phase == 'prefill' else 1) * 256 * 2
            semantic = {c: {d: 0 for d in v.DIRECTIONS} for c in v.CATEGORIES}
            semantic['weights']['read_bytes'] = 100
            semantic['kv_cache']['write_bytes'] = kv
            phases[phase] = dict(operators=507, model_ns=507, known_read_bytes=100,
                                 known_write_bytes=kv, unclassified_io_bytes=0, semantic=semantic)
        total = {f: sum(p[f] for p in phases.values()) for f in v.FIELDS}
        total['semantic'] = {c: {d: sum(p['semantic'][c][d] for p in phases.values())
                                for d in v.DIRECTIONS} for c in v.CATEGORIES}
        size = total['known_read_bytes'] + total['known_write_bytes']
        rows.append(dict(case=f'b{batch:02d}p128d08', batch_size=batch, prefill_tokens=128, decode_steps=8,
                         status='PASS_FULL_ANALYTICAL_GRAPH_NOT_HARDWARE_ACCEPTANCE', **total,
                         phase_totals=phases, family_totals={'synthetic': copy.deepcopy(total)},
                         request_regions=dict(request_count=batch, kv_regions=56 * batch, independent_kv=True,
                                              last_token_rows_checked=True, append_bytes=total['known_write_bytes'],
                                              kv_payload_bytes=total['known_write_bytes']),
                         total_boundary_bytes=size, model_ms=4563 / 1e6, model_GBps=size / 4563,
                         full_request_latency_ms=4563 / 1e6, amortized_ms_per_request=4563 / 1e6 / batch,
                         requests_per_second=batch * 1e9 / 4563,
                         prefill_tokens_per_second=batch * 128 * 1e9 / 507,
                         decode_tokens_per_second=batch * 8 * 1e9 / 4056,
                         hardware_comparison=None, hbfsim_in_model_time=False))
    baseline = copy.deepcopy(rows[0])
    baseline.update(case='p128d08', operator_count=4563)
    return rows, baseline


class VerificationTests(unittest.TestCase):
    def test_complete_synthetic_aggregate_passes(self):
        v.verify_cases(*fixture())

    def test_missing_or_duplicate_batch_fails(self):
        rows, baseline = fixture()
        with self.assertRaises(ValueError): v.verify_cases(rows[:-1], baseline)
        rows[-1]['batch_size'] = 16
        with self.assertRaises(ValueError): v.verify_cases(rows, baseline)

    def test_bad_semantic_or_group_total_fails(self):
        rows, baseline = fixture()
        rows[1]['semantic']['activation']['write_bytes'] += 1
        with self.assertRaises(ValueError): v.verify_cases(rows, baseline)
        rows, baseline = fixture()
        rows[1]['family_totals']['synthetic']['model_ns'] += 1
        with self.assertRaises(ValueError): v.verify_cases(rows, baseline)

    def test_bad_kv_region_count_fails(self):
        rows, baseline = fixture()
        rows[2]['request_regions']['kv_regions'] -= 1
        with self.assertRaises(ValueError): v.verify_cases(rows, baseline)

    def test_bad_bandwidth_or_latency_denominator_fails(self):
        for field in ('model_GBps', 'full_request_latency_ms'):
            rows, baseline = fixture()
            rows[1][field] /= 2
            with self.assertRaises(ValueError): v.verify_cases(rows, baseline)

    def test_baseline_drift_and_hbfsim_time_fail(self):
        rows, baseline = fixture()
        baseline['known_read_bytes'] += 1
        with self.assertRaises(ValueError): v.verify_cases(rows, baseline)
        rows, baseline = fixture()
        rows[2]['hbfsim_in_model_time'] = True
        with self.assertRaises(ValueError): v.verify_cases(rows, baseline)


if __name__ == '__main__':
    unittest.main()
