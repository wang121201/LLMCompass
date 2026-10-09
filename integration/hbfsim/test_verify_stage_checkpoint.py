"""Negative controls for the archived stage acceptance boundary."""
import copy
import tempfile
import types
import unittest
from pathlib import Path
from verify_stage_checkpoint import load, validate_payloads


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent / 'checkpoint'
        self.payloads = [load(root / name) for name in
            ('official-comparison.json', 'semantic-breakdown.json', 'paired-collection.json',
             'official-validation.json', 'gddr-comparison.json', 'gddr-validation.json')]

    def test_frozen_checkpoint_closes(self):
        receipt = validate_payloads(*self.payloads)
        self.assertEqual(receipt['operators'], 23322)
        self.assertEqual(receipt['hardware_accuracy'], 'NOT_ACCEPTED')

    def reject(self, edit):
        payloads = copy.deepcopy(self.payloads)
        edit(payloads)
        with self.assertRaises(ValueError):
            validate_payloads(*payloads)

    def test_extra_case_rejected(self):
        self.reject(lambda p: p[0].append(p[0][0]))

    def test_semantic_byte_mutation_rejected(self):
        self.reject(lambda p: p[1]['cases'][0]['categories']['activation'].__setitem__('write_bytes', 1))

    def test_serial_time_contamination_rejected(self):
        self.reject(lambda p: p[3].__setitem__('hbfsim_primary_time_contribution_ns', 1))

    def test_accuracy_relabeling_rejected(self):
        self.reject(lambda p: p[3].__setitem__('full_window_time_gate', 'PASS'))

    def test_bad_bandwidth_denominator_rejected(self):
        self.reject(lambda p: p[2]['cases'][0]['hardware']['native'].__setitem__('effective_GBps', 1))

    def test_gddr_regression_hidden_rejected(self):
        self.reject(lambda p: p[5].__setitem__('full_no_worse_gate', 'PASS'))

    def test_retired_serial_path_rejects_before_output_creation(self):
        from qwen_hbfsim_cosim import CosimError, run_cosimulation
        plan = types.SimpleNamespace(operators=[types.SimpleNamespace(
            timing=types.SimpleNamespace(compute_ns=1))])
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            output = adapter / 'must-not-exist'
            with self.assertRaises(CosimError):
                run_cosimulation(adapter, adapter, adapter, adapter, (), output,
                                 plan, 'test', adapter, adapter)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
