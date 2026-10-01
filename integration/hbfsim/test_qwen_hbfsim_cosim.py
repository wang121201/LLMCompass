#!/usr/bin/env python3
from __future__ import annotations

import math
import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qwen_hbfsim_cosim import HBF_BASE, KV_BASE, LLMCompassCostModel, ModelSpec, _add_batch_traffic, _empty_traffic_counts, _transactions, allocate_weights, build_analytical_cache_accounting, build_plan, build_traffic_report, compare_to_archived_ncu_traffic, compare_to_hardware, plan_summary, remap_plan_for_gddr_abstract


class FakeTransaction:
    def __init__(self, id, target, op, addr, bytes, issue_ns, duration_ns=0, dependencies=(), stack=None):
        self.id, self.target, self.op, self.addr, self.bytes = id, target, op, addr, bytes
        self.issue_ns, self.duration_ns = issue_ns, duration_ns
        self.dependencies, self.stack = dependencies, stack


class QwenCosimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = HERE.parents[1]
        cls.model = ModelSpec.from_json(HERE / "qwen25_1p5b.json")
        cls.costs = LLMCompassCostModel(root, root / "configs" / "GA100.json")
        cls.plan = build_plan(cls.model, cls.costs)

    def test_rtx4000ada_component_profile_uses_observed_and_bounded_fields(self):
        root = HERE.parents[1]
        costs = LLMCompassCostModel(root, HERE / "RTX4000Ada_xmu_profile_v1.json")
        cm = costs.device.compute_module
        self.assertEqual(costs.architecture_name, "NVIDIA RTX 4000 Ada xmu component profile v1")
        self.assertEqual(cm.core_count, 48)
        self.assertEqual(cm.l2_size, 40 * 1024 * 1024)
        self.assertAlmostEqual(costs.device.io_module.bandwidth, 360.04e9)
        self.assertAlmostEqual(cm.total_systolic_array_flops, 213.8112e12)
        self.assertAlmostEqual(cm.total_vector_flops, 26.7264e12)
        self.assertEqual((cm.overhead.matmul, cm.overhead.softmax, cm.overhead.layernorm, cm.overhead.gelu), (0.0, 0.0, 0.0, 0.0))

    def test_model_identity_and_p32d2_contract(self):
        self.assertEqual((self.model.hidden_size, self.model.intermediate_size, self.model.layers), (1536, 8960, 28))
        self.assertEqual((self.model.attention_heads, self.model.kv_heads, self.model.head_dim), (12, 2, 128))
        self.assertEqual((self.model.prefill_tokens, self.model.decode_steps), (32, 2))
        summary = plan_summary(self.plan, self.costs.architecture_name)
        self.assertEqual(set(summary["operator_count_by_phase"]), {"prefill", "decode_1", "decode_2"})

    def test_weights_are_aligned_non_overlapping_and_tied(self):
        weights = allocate_weights(self.model)
        ordered = sorted(weights.objects.values(), key=lambda item: item.address)
        self.assertEqual(ordered[0].address, HBF_BASE)
        self.assertNotIn("lm_head.weight", weights.objects)
        for left, right in zip(ordered, ordered[1:]):
            self.assertEqual(left.address % 4096, 0)
            self.assertLessEqual(left.address + left.bytes, right.address)
        expected_parameters = self.model.vocab_size*self.model.hidden_size + self.model.hidden_size + self.model.layers*(
            2*self.model.hidden_size + self.model.hidden_size*self.model.hidden_size + 2*self.model.hidden_size*self.model.kv_hidden_size
            + self.model.hidden_size + 2*self.model.kv_hidden_size + self.model.hidden_size*self.model.hidden_size
            + 3*self.model.hidden_size*self.model.intermediate_size)
        self.assertEqual(sum(item.bytes for item in weights.objects.values()), expected_parameters*self.model.bytes_per_element)

    def test_all_timings_are_finite_and_positive(self):
        for op in self.plan.operators:
            self.assertGreater(op.timing.compute_ns, 0)
            self.assertTrue(math.isfinite(op.timing.compute_ns))

    def test_qwen_operator_families_are_present(self):
        names = {op.name for op in self.plan.operators}
        required = {"input_rmsnorm","q_proj","k_proj","v_proj","rope","kv_append","attention_score","attention_softmax","attention_value","o_proj","post_attention_rmsnorm","gate_proj","up_proj","silu","gated_multiply","down_proj","lm_head"}
        self.assertTrue(required.issubset(names))

    def test_live_ffn_scratch_regions_do_not_overlap(self):
        gate = next(op for op in self.plan.operators if op.phase == "prefill" and op.layer == 0 and op.name == "gate_proj").writes[0]
        up = next(op for op in self.plan.operators if op.phase == "prefill" and op.layer == 0 and op.name == "up_proj").writes[0]
        self.assertLessEqual(gate.address + gate.bytes, up.address)
        self.assertLessEqual(up.address + up.bytes, KV_BASE)

    def test_transaction_dag_orders_reads_compute_and_writes(self):
        op = self.plan.operators[1]
        txs, expected, barrier_id = _transactions(FakeTransaction, op, 123)
        barrier = next(tx for tx in txs if tx.id == barrier_id)
        reads, writes = [tx for tx in txs if tx.op == "R"], [tx for tx in txs if tx.op == "W"]
        self.assertEqual(set(barrier.dependencies), {tx.id for tx in reads})
        self.assertEqual(barrier.duration_ns, op.timing.compute_ns)
        self.assertTrue(all(tx.dependencies == (barrier_id,) for tx in writes))
        self.assertEqual(expected, {tx.id for tx in reads+writes})

    def test_hbf_is_read_only_and_kv_writes_cover_every_layer_and_phase(self):
        self.assertEqual([a for op in self.plan.operators for a in op.writes if a.target == "HBF_STATIC"], [])
        kv_writes = [a for op in self.plan.operators for a in op.writes if op.name == "kv_append" and a.target == "HBM"]
        self.assertEqual(len(kv_writes), self.model.layers*3*2)
        self.assertTrue(all(access.address >= KV_BASE for access in kv_writes))

    def test_gddr_abstract_remap_uses_one_non_overlapping_hbm_address_space(self):
        plan = remap_plan_for_gddr_abstract(self.plan)
        self.assertEqual(plan.memory_layout, "gddr_abstract_all_hbm")
        self.assertEqual(plan.static_blocks_per_plane, 0)
        self.assertEqual(plan.hbm_address_shift_bytes, self.plan.weights.footprint)
        self.assertGreater(plan.hbm_arena_bytes, plan.hbm_address_shift_bytes)
        for op in plan.operators:
            for access in (*op.reads, *op.writes):
                self.assertEqual(access.target, "HBM")
                self.assertLess(access.address + access.bytes, plan.hbm_arena_bytes + 1)

    def test_phase1_cache_accounting_is_an_explicit_model_estimate(self):
        report = build_analytical_cache_accounting(self.plan)
        self.assertEqual(report["status"], "MODEL_ESTIMATE")
        self.assertEqual(report["claim_class"], "MODEL_ESTIMATE")
        self.assertEqual(report["phase_coverage"], ["prefill", "decode_1", "decode_2"])
        self.assertEqual(report["total"]["operator_count"], len(self.plan.operators))
        for row in report["phase_rows"]:
            self.assertEqual(row["l1_lookup"], row["l2_lookup"])
            for level in (row["l1_lookup"], row["l2_lookup"]):
                self.assertGreater(level["aggregate"]["total"]["logical_bytes"], 0)
                self.assertGreaterEqual(
                    level["aggregate"]["total"]["estimated_sector_bytes"],
                    level["aggregate"]["total"]["logical_bytes"],
                )
        self.assertEqual(report["assumptions"]["hit_miss_state"], "NOT_MODELED")
        self.assertEqual(report["assumptions"]["dram_traffic"], "NOT_REPORTED")
        self.assertEqual(report["assumptions"]["hardware_accuracy"], "NOT_CLAIMED")

    def test_workload_overrides_generate_all_requested_decode_phases(self):
        model = dataclasses.replace(self.model, prefill_tokens=128, decode_steps=4)
        plan = build_plan(model, self.costs)
        phases = tuple(dict.fromkeys(op.phase for op in plan.operators))
        self.assertEqual(phases, ("prefill", "decode_1", "decode_2", "decode_3", "decode_4"))
        self.assertEqual(len(plan.operators), 5 * (self.model.layers * 18 + 3))

    def test_candidate_comparison_uses_elapsed_simulated_phase_times(self):
        hardware = {
            "status": "PASS",
            "input_contract": {"case_id": "qwen25_1p5b-p32-d2", "batch_size": 1, "prefill_length": 32, "decode_steps": 2},
            "phase_rows": [
                {"role": "measurement", "phase": "Prefill", "seconds": 0.010},
                {"role": "measurement", "phase": "Decode1", "seconds": 0.020},
                {"role": "measurement", "phase": "Decode2", "seconds": 0.030},
            ],
        }
        simulation = {
            "schema": "llmcompass-hbfsim-qwen-p32d2-summary-v1", "status": "PASS",
            "phase_finish_ns": {"prefill": 5_000_000, "decode_1": 12_000_000, "decode_2": 21_000_000},
            "simulated_finish_ns": 21_000_000,
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hardware_path, simulation_path = root / "hardware.json", root / "simulation.json"
            hardware_path.write_text(json.dumps(hardware), encoding="utf-8")
            simulation_path.write_text(json.dumps(simulation), encoding="utf-8")
            report = compare_to_hardware(root, root / "comparison", hardware_path, simulation_path)
            self.assertEqual([row["simulated_operator_boundary_ms"] for row in report["phase_rows"]], [5.0, 7.0, 9.0])
            self.assertEqual(report["aggregate"]["hardware_host_wall_ms"], 60.0)
            self.assertEqual(report["aggregate"]["simulated_operator_boundary_ms"], 21.0)
            self.assertTrue((root / "comparison" / "comparison.json").is_file())

    def test_traffic_report_uses_hbfsim_completion_bytes_and_marks_caches_unmodeled(self):
        counts = _empty_traffic_counts()
        transactions = [
            FakeTransaction("read", "HBM", "R", 0, 33, 0),
            FakeTransaction("write", "HBF_STATIC", "W", 0, 17, 0),
            FakeTransaction("barrier", "BARRIER", None, 0, 0, 0),
        ]
        records = [
            type("Record", (), {"id": "read", "logical_bytes": 33, "physical_bytes": 64})(),
            type("Record", (), {"id": "write", "logical_bytes": 17, "physical_bytes": 4096})(),
        ]
        _add_batch_traffic(counts, transactions, records)
        report = build_traffic_report(
            {"prefill": counts, "decode_1": _empty_traffic_counts(), "decode_2": _empty_traffic_counts()},
            {"prefill": 1_000_000, "decode_1": 2_000_000, "decode_2": 3_000_000},
        )
        self.assertEqual(report["phase_rows"][0]["targets"]["HBM"]["read"]["physical_bytes"], 64)
        self.assertEqual(report["total"]["targets"]["HBF_STATIC"]["write"]["physical_bytes"], 4096)
        self.assertTrue(report["cache_level_coverage"]["L1"].startswith("NOT_MODELED"))
        smoke = build_traffic_report(
            {"prefill": counts, "decode_1": _empty_traffic_counts(), "decode_2": _empty_traffic_counts()},
            {"prefill": 1_000_000},
        )
        self.assertEqual(smoke["phase_coverage"], ["prefill"])
        gddr = build_traffic_report(
            {"prefill": counts, "decode_1": _empty_traffic_counts(), "decode_2": _empty_traffic_counts()},
            {"prefill": 1_000_000, "decode_1": 2_000_000, "decode_2": 3_000_000},
            memory_layout="gddr_abstract_all_hbm",
        )
        self.assertIn("GDDR-inspired", gddr["cache_level_coverage"]["DRAM"])

    def test_archived_ncu_comparison_is_limited_to_a_non_accuracy_screening_proxy(self):
        cache_metrics = (
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
            "lts__t_sectors_srcunit_tex_op_read.sum",
            "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
            "lts__t_sectors_srcunit_tex_op_write.sum",
        )
        def ncu_row(metric, unit, median):
            return {
                "workload": "qwen1p5b-P32D2", "model": "r4-small-shared", "phase": "whole", "scope": "whole",
                "metric": metric, "unit": unit, "launches": 1030,
                "hardware_samples": [median - 1, median, median + 1],
                "hardware_min": median - 1, "hardware_median": median, "hardware_max": median + 1,
            }
        archived = {
            "status": "PARTIAL_COMPARISON", "hardware_accuracy_accepted": False,
            "rows": [
                ncu_row("dram__bytes_read.sum", "byte", 1000),
                ncu_row("dram__bytes_write.sum", "byte", 200),
                *(ncu_row(metric, "32_byte_sector", 10) for metric in cache_metrics),
            ],
        }
        traffic = {
            "schema": "llmcompass-hbfsim-qwen-p32d2-traffic-v1", "status": "PASS",
            "total": {"elapsed_ns": 100.0, "targets": {
                "HBF_STATIC": {"read": {"transactions": 1, "logical_bytes": 900, "physical_bytes": 1000}, "write": {"transactions": 0, "logical_bytes": 0, "physical_bytes": 0}},
                "HBM": {"read": {"transactions": 2, "logical_bytes": 100, "physical_bytes": 100}, "write": {"transactions": 3, "logical_bytes": 50, "physical_bytes": 50}},
            }},
        }
        manifest = {
            "schema": "llmcompass-hbfsim-qwen-p32d2-run-v1", "status": "PASS",
            "plan": {"operator_count": 1521, "model": {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "prefill_tokens": 32, "decode_steps": 2}},
        }
        hardware = {
            "status": "PASS",
            "input_contract": {"case_id": "qwen25_1p5b-p32-d2", "batch_size": 1, "prefill_length": 32, "decode_steps": 2},
            "phase_rows": [
                {"role": "measurement", "phase": "Prefill", "seconds": 0.010},
                {"role": "measurement", "phase": "Decode1", "seconds": 0.020},
                {"role": "measurement", "phase": "Decode2", "seconds": 0.030},
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archived_path, traffic_path, manifest_path, hardware_path = root / "archived.json", root / "traffic.json", root / "manifest.json", root / "hardware.json"
            archived_path.write_text(json.dumps(archived), encoding="utf-8")
            traffic_path.write_text(json.dumps(traffic), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            hardware_path.write_text(json.dumps(hardware), encoding="utf-8")
            report = compare_to_archived_ncu_traffic(root, root / "comparison", archived_path, traffic_path, manifest_path)
            self.assertEqual(report["status"], "SCREENING_PROXY_ONLY_NOT_HARDWARE_ACCURACY")
            self.assertEqual(report["dram_traffic_screening"][0]["signed_screening_delta_percent"], 10.0)
            self.assertEqual(report["cache_metrics"][0]["simulator"], "NOT_MODELED")
            self.assertEqual(report["bandwidth"]["comparison_class"], "NOT_COMPARABLE")
            self.assertTrue((root / "comparison" / "ncu-compatibility.json").is_file())
            proxy_report = compare_to_archived_ncu_traffic(root, root / "comparison-with-proxy", archived_path, traffic_path, manifest_path, hardware_path)
            proxy = proxy_report["bandwidth"]["cross_receipt_host_time_normalized_proxy"]
            self.assertEqual(proxy["comparison_class"], "CROSS_RECEIPT_HOST_TIME_NORMALIZED_PROXY_ONLY")
            self.assertEqual(proxy["rows"][0]["simulator_operator_boundary_GBps"], 11.0)
            self.assertAlmostEqual(proxy["rows"][0]["hardware_host_time_normalized_GBps"], 1000 / 60_000_000)


if __name__ == "__main__":
    unittest.main()
