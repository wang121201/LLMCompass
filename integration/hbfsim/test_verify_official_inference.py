"""Small regression tests for acceptance arithmetic and claim boundaries."""
import copy
import json
import pathlib
import unittest
from unittest import mock

import verify_official_inference as verifier


class AcceptanceTests(unittest.TestCase):
    def test_distribution_matches_archived_statistics(self):
        samples = [1.0, 3.0, 2.0, 5.0, 4.0]
        result = verifier.distribution(samples)
        self.assertEqual(result['median'], 3.0)
        self.assertAlmostEqual(result['p10'], 1.4)
        self.assertAlmostEqual(result['p90'], 4.6)
        verifier.verify_distribution(result, samples[::-1], 'valid')

    def test_changed_measurement_or_summary_is_rejected(self):
        saved = verifier.distribution([1, 2, 3, 4, 5])
        changed = copy.deepcopy(saved)
        changed['median'] = 3.1
        with self.assertRaises(ValueError):
            verifier.verify_distribution(changed, saved['samples'], 'median')
        with self.assertRaises(ValueError):
            verifier.verify_distribution(saved, [1, 2, 3, 4, 6], 'samples')

    def test_phase_cancellation_does_not_pass(self):
        full = verifier.time_comparison(30_000_000, [30, 30])
        phases = {'prefill': verifier.time_comparison(13_000_000, [10, 10]),
                  'decode': verifier.time_comparison(17_000_000, [20, 20])}
        result = verifier.cancellation_status(full, phases)
        self.assertTrue(result['total_pass_with_cancellation'])
        self.assertEqual(result['full_and_every_phase_time_gate'], 'FAIL')

    def test_total_only_and_phase_only_passes_are_distinct(self):
        full = verifier.time_comparison(40_000_000, [50, 50])
        phases = {'prefill': verifier.time_comparison(20_000_000, [20, 20])}
        result = verifier.cancellation_status(full, phases)
        self.assertTrue(result['all_phase_time_within_10_percent'])
        self.assertEqual(result['full_and_every_phase_time_gate'], 'FAIL')
        self.assertFalse(result['total_pass_with_cancellation'])

    def test_unknown_io_is_counted_once(self):
        totals = dict(known_read_bytes=100, known_write_bytes=50, unclassified_io_bytes=30)
        hardware = {'dram_read_bytes': {'median': 100}, 'dram_write_bytes': {'median': 50}}
        result = verifier.traffic_comparison(totals, hardware, 100, 0.0001)
        self.assertEqual(result['total_boundary_bytes'], 180)
        self.assertEqual(result['read_error_range_pct'][0], 0)
        self.assertAlmostEqual(result['read_error_range_pct'][1], 30)
        self.assertAlmostEqual(result['write_error_range_pct'][1], 60)
        self.assertEqual(result['model_GBps'], 1.8)
        self.assertAlmostEqual(result['hardware_GBps'], 1.5)

    def test_nan_and_empty_or_zero_denominators_fail(self):
        for samples in ([], [0, 0], [1, float('nan')]):
            with self.assertRaises(ValueError):
                verifier.distribution(samples)
        with self.assertRaises(ValueError):
            verifier.close(float('nan'), 1, 'nonfinite')
        with self.assertRaises(ValueError):
            verifier.time_comparison(1, [0, 1])

    def test_observed_range_is_not_a_confidence_interval(self):
        row = verifier.time_comparison(10_000_000, [9, 10, 11])
        self.assertTrue(row['time_within_10_percent'])
        self.assertEqual(row['passing_timing_samples'], 2)
        self.assertEqual(row['hardware_timing_statistics']['count'], 3)
        self.assertGreater(row['time_error_over_observed_samples_pct'][1], 10)


def run_matrix_rejection_checks(root):
    """Inject faults into reads only; never modify frozen experiment files."""
    root = pathlib.Path(root)
    original_read = pathlib.Path.read_bytes
    summary = json.loads(original_read(root / 'comparison.json'))[0]
    hw_root = pathlib.Path(summary['hardware_source']).parent
    finish_path = next((hw_root / 'timing').glob('repeat-1/host/process-*/finish.json'))
    faults = [
        ('duplicate_case', root / 'comparison.json',
         lambda data: data.__setitem__(1, copy.deepcopy(data[0])), 'eight requested cases'),
        ('hbfsim_additive_time', root / 'manifest.json',
         lambda data: data.__setitem__('hbfsim_in_primary_time', True), 'must not include HBFSim'),
        ('operator_time', root / summary['case'] / 'operators.json',
         lambda data: data[0].__setitem__('model_ns', data[0]['model_ns'] + 1000), 'total time'),
        ('operator_phase', root / summary['case'] / 'operators.json',
         lambda data: data[0].__setitem__('phase', 'decode_1'), 'noncontiguous operator phases'),
        ('profiled_hardware_time', finish_path,
         lambda data: data.__setitem__('profiler_api_invoked', True), 'invalid unprofiled timing receipt'),
        ('hardware_phase_time', finish_path,
         lambda data: data['natural_cuda_event_ms'].__setitem__('Decode1', data['natural_cuda_event_ms']['Decode1'] + 1), 'sample identity'),
    ]
    for name, target, mutate, expected in faults:
        def altered_read(path):
            raw = original_read(path)
            if path == target:
                data = json.loads(raw)
                mutate(data)
                return json.dumps(data).encode()
            return raw

        with mock.patch.object(pathlib.Path, 'read_bytes', altered_read):
            try:
                verifier.verify_root(root)
            except ValueError as error:
                if expected not in str(error):
                    raise AssertionError(f'{name}: unexpected rejection: {error}') from error
            else:
                raise AssertionError(f'{name}: corrupted input was accepted')
        print('PASS_READ_ONLY_MATRIX_FAULT_REJECTION', name)
    return len(faults)


if __name__ == '__main__':
    unittest.main()
