#!/usr/bin/env python3
"""Qwen2.5-1.5B P32D2 operator-boundary LLMCompass + HBFSim co-simulation."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB
HBF_BASE = 0
KV_BASE = 64 * MiB
ALIGNMENT = 4096
MEMORY_TARGETS = ("HBF_STATIC", "HBM")
MEMORY_OPERATIONS = ("read", "write")
MEMORY_LAYOUT_HBF_HBM = "hbf_hbm"
MEMORY_LAYOUT_GDDR_ABSTRACT_ALL_HBM = "gddr_abstract_all_hbm"
MEMORY_LAYOUTS = (MEMORY_LAYOUT_HBF_HBM, MEMORY_LAYOUT_GDDR_ABSTRACT_ALL_HBM)
ARCHIVED_NCU_WORKLOAD = "qwen1p5b-P32D2"
ARCHIVED_NCU_MODEL = "r4-small-shared"
ARCHIVED_NCU_WHOLE = ("whole", "whole")
ARCHIVED_NCU_DRAM_METRICS = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)
ARCHIVED_NCU_CACHE_METRICS = (
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_write.sum",
)


class CosimError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_sha(repo: Path) -> str:
    import subprocess
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    bytes_per_element: int
    prefill_tokens: int = 32
    decode_steps: int = 2

    @property
    def kv_hidden_size(self) -> int:
        return self.kv_heads * self.head_dim

    @classmethod
    def from_json(cls, path: Path) -> "ModelSpec":
        raw = json.loads(path.read_text(encoding="utf-8"))
        arch = raw["architecture"]
        precision = raw["precision"]
        workload = raw["workload"]
        spec = cls(
            model_id=raw["model_id"],
            hidden_size=arch["hidden_size"],
            intermediate_size=arch["intermediate_size"],
            layers=arch["num_hidden_layers"],
            attention_heads=arch["num_attention_heads"],
            kv_heads=arch["num_key_value_heads"],
            head_dim=arch["head_dim"],
            vocab_size=arch["vocab_size"],
            bytes_per_element=precision["bytes_per_element"],
            prefill_tokens=workload["prefill_tokens"],
            decode_steps=workload["decode_steps"],
        )
        if spec.hidden_size != spec.attention_heads * spec.head_dim:
            raise ValueError("hidden size must equal attention_heads * head_dim")
        if arch["tie_word_embeddings"] is not True:
            raise ValueError("this adapter requires tied input/output embeddings")
        if workload["batch_size"] != 1 or workload["tensor_parallelism"] != 1:
            raise ValueError("this adapter is restricted to batch size 1 and TP1")
        if (spec.prefill_tokens, spec.decode_steps) != (32, 2):
            raise ValueError("this adapter is restricted to P32D2")
        return spec


@dataclass(frozen=True)
class WeightObject:
    name: str
    address: int
    bytes: int


@dataclass(frozen=True)
class Access:
    kind: str
    target: str
    address: int
    bytes: int
    label: str


@dataclass(frozen=True)
class Timing:
    family: str
    flops: int
    compute_ns: int


@dataclass(frozen=True)
class OperatorPlan:
    index: int
    phase: str
    layer: int | None
    name: str
    timing: Timing
    reads: tuple[Access, ...]
    writes: tuple[Access, ...]


@dataclass(frozen=True)
class InferencePlan:
    model: ModelSpec
    weights: "WeightAllocator"
    operators: tuple[OperatorPlan, ...]
    static_blocks_per_plane: int
    hbm_arena_bytes: int
    memory_layout: str = MEMORY_LAYOUT_HBF_HBM
    hbm_address_shift_bytes: int = 0


class WeightAllocator:
    def __init__(self, bytes_per_element: int) -> None:
        self.bytes_per_element = bytes_per_element
        self.cursor = HBF_BASE
        self.objects: dict[str, WeightObject] = {}

    def add(self, name: str, elements: int) -> WeightObject:
        if name in self.objects:
            raise ValueError(f"duplicate weight object: {name}")
        self.cursor = _align_up(self.cursor)
        obj = WeightObject(name, self.cursor, elements * self.bytes_per_element)
        self.objects[name] = obj
        self.cursor += obj.bytes
        return obj

    @property
    def footprint(self) -> int:
        return _align_up(self.cursor)


def allocate_weights(model: ModelSpec) -> WeightAllocator:
    a = WeightAllocator(model.bytes_per_element)
    h, kv, f = model.hidden_size, model.kv_hidden_size, model.intermediate_size
    a.add("model.embed_tokens", model.vocab_size * h)
    for layer in range(model.layers):
        p = f"model.layers.{layer}"
        a.add(f"{p}.input_layernorm.weight", h)
        a.add(f"{p}.self_attn.q_proj.weight", h * h)
        a.add(f"{p}.self_attn.q_proj.bias", h)
        a.add(f"{p}.self_attn.k_proj.weight", h * kv)
        a.add(f"{p}.self_attn.k_proj.bias", kv)
        a.add(f"{p}.self_attn.v_proj.weight", h * kv)
        a.add(f"{p}.self_attn.v_proj.bias", kv)
        a.add(f"{p}.self_attn.o_proj.weight", h * h)
        a.add(f"{p}.post_attention_layernorm.weight", h)
        a.add(f"{p}.mlp.gate_proj.weight", h * f)
        a.add(f"{p}.mlp.up_proj.weight", h * f)
        a.add(f"{p}.mlp.down_proj.weight", f * h)
    a.add("model.norm.weight", h)
    return a


def _install_analytical_import_shims() -> None:
    """Permit analytical-only LLMCompass imports when torch/ScaleSim are absent."""
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        class _TorchStub:
            def __getattr__(self, name: str) -> Any:
                raise RuntimeError(f"torch operation {name} is unavailable in analytical-only mode")
        torch.Tensor = _TorchStub
        torch.nn = _TorchStub()
        torch.cuda = _TorchStub()
        sys.modules["torch"] = torch
    if "scalesim" not in sys.modules:
        scalesim = types.ModuleType("scalesim")
        scale_sim = types.ModuleType("scalesim.scale_sim")
        class _ScaleSimStub:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("ScaleSim is unavailable in analytical-only mode")
        scale_sim.scalesim = _ScaleSimStub
        scalesim.scale_sim = scale_sim
        sys.modules["scalesim"] = scalesim
        sys.modules["scalesim.scale_sim"] = scale_sim


class LLMCompassCostModel:
    """Use LLMCompass operator roofline models and device components analytically."""

    def __init__(self, repo: Path, architecture: Path) -> None:
        sys.path.insert(0, str(repo))
        _install_analytical_import_shims()
        from hardware_model.compute_module import ComputeModule, Core, SystolicArray, VectorUnit, overhead_dict
        from hardware_model.device import Device
        from hardware_model.io_module import IOModule
        from hardware_model.memory_module import MemoryModule
        from software_model.matmul import BatchedMatmul, Matmul
        from software_model.softmax import Softmax
        from software_model.utils import Tensor, data_type_dict

        raw = json.loads(architecture.read_text(encoding="utf-8"))
        device_cfg = raw["device"]
        chip = device_cfg["compute_chiplet"]
        core_cfg = chip["core"]
        sa_cfg = core_cfg["systolic_array"]
        sa_word_size = int(re.search(r"(\d+)", sa_cfg["data_type"]).group(1)) // 8
        array = SystolicArray(sa_cfg["array_height"], sa_cfg["array_width"], sa_cfg["mac_per_cycle"], sa_word_size, sa_word_size)
        vu = core_cfg["vector_unit"]
        sublanes = core_cfg["sublane_count"]
        vu_word_size = int(re.search(r"(\d+)", vu["data_type"]).group(1)) // 8
        vector = VectorUnit(sublanes*vu["vector_width"]*vu["flop_per_cycle"], vu_word_size, 35, vu["vector_width"], sublanes)
        core = Core(vector, array, sublanes, core_cfg["SRAM_KB"]*KiB)
        io_cfg = device_cfg["io"]
        compute = ComputeModule(core, chip["core_count"]*device_cfg["compute_chiplet_count"], device_cfg["frequency_Hz"], io_cfg["global_buffer_MB"]*MiB, io_cfg["global_buffer_bandwidth_per_cycle_byte"], overhead_dict["A100"])
        io_bandwidth = io_cfg["memory_channel_active_count"]*io_cfg["pin_count_per_channel"]*io_cfg["bandwidth_per_pin_bit"]//8
        io = IOModule(io_bandwidth, 1e-6)
        memory = MemoryModule(device_cfg["memory"]["total_capacity_GB"] * GiB)
        self.device = Device(compute, io, memory)
        self.Matmul, self.BatchedMatmul, self.Softmax = Matmul, BatchedMatmul, Softmax
        self.Tensor, self.dtype = Tensor, data_type_dict["fp16"]
        self.architecture_name = raw["name"]

    def _timing(self, family: str, flops: int, op: Any | None = None, extra_compute_s: float = 0.0) -> Timing:
        cm = self.device.compute_module
        compute_s = flops / max(cm.total_systolic_array_flops, 1.0) + extra_compute_s
        roofline_s = op.roofline_model(self.device) if op is not None else compute_s
        return Timing(family, flops, max(1, math.ceil(max(compute_s, roofline_s) * 1e9)))

    def matmul(self, m: int, k: int, n: int, bias: bool = False, family: str = "matmul") -> Timing:
        op = self.Matmul(self.dtype)
        op(self.Tensor([m, k], self.dtype), self.Tensor([k, n], self.dtype))
        flops = 2 * m * k * n + (m * n if bias else 0)
        cm = self.device.compute_module
        extra = m * n / max(cm.total_vector_flops_per_cycle * cm.clock_freq, 1.0) if bias else 0.0
        return self._timing(family, flops, op, extra)

    def batched_matmul(self, b: int, m: int, k: int, n: int, family: str) -> Timing:
        op = self.BatchedMatmul(self.dtype)
        op(self.Tensor([b, m, k], self.dtype), self.Tensor([b, k, n], self.dtype))
        return self._timing(family, 2 * b * m * k * n, op)

    def softmax(self, b: int, m: int, n: int) -> Timing:
        op = self.Softmax(self.dtype)
        op(self.Tensor([b, m, n], self.dtype))
        return self._timing("softmax", b * m * n * 5, op)

    def vector(self, elements: int, ops_per_element: int, family: str, bytes_moved: int = 0) -> Timing:
        cm = self.device.compute_module
        flops = elements * ops_per_element
        compute_s = flops / max(cm.total_vector_flops_per_cycle * cm.clock_freq, 1.0)
        bandwidth_s = bytes_moved / max(cm.l2_bandwidth_per_cycle * cm.clock_freq, 1.0)
        return Timing(family, flops, max(1, math.ceil(max(compute_s, bandwidth_s) * 1e9)))


def _weight_read(weights: WeightAllocator, name: str, logical_bytes: int | None = None) -> Access:
    weight = weights.objects[name]
    return Access("read", "HBF_STATIC", weight.address, weight.bytes if logical_bytes is None else logical_bytes, name)


def _hbm(kind: str, address: int, amount: int, label: str) -> Access:
    return Access(kind, "HBM", address, amount, label)


def _slot(slot: int, maximum_tokens: int, hidden: int, bytes_per_element: int) -> int:
    return slot * _align_up(maximum_tokens * hidden * bytes_per_element)


def build_plan(model: ModelSpec, costs: LLMCompassCostModel, *, layer_limit: int | None = None, phase_limit: int | None = None) -> InferencePlan:
    weights = allocate_weights(model)
    layers = model.layers if layer_limit is None else layer_limit
    if not 1 <= layers <= model.layers:
        raise ValueError("layer_limit must be within the model layer count")
    phases = [
        ("prefill", model.prefill_tokens, model.prefill_tokens, 0, model.prefill_tokens),
        ("decode_1", 1, model.prefill_tokens + 1, model.prefill_tokens, 1),
        ("decode_2", 1, model.prefill_tokens + 2, model.prefill_tokens + 1, 1),
    ]
    if phase_limit is not None:
        phases = phases[:phase_limit]
    bpe, h, kv = model.bytes_per_element, model.hidden_size, model.kv_hidden_size
    max_tokens = model.prefill_tokens + model.decode_steps
    scratch_width = max(model.hidden_size, model.intermediate_size)
    scratch_end = _slot(8, max_tokens, scratch_width, bpe)
    if scratch_end >= KV_BASE:
        raise CosimError("activation scratch overlaps the KV-cache arena")
    kv_layer_stride = _align_up(2 * max_tokens * kv * bpe)
    ops: list[OperatorPlan] = []

    def add(phase: str, layer: int | None, name: str, timing: Timing, reads: Iterable[Access], writes: Iterable[Access]) -> None:
        ops.append(OperatorPlan(len(ops), phase, layer, name, timing, tuple(reads), tuple(writes)))

    for phase, tokens, context, append_start, append_count in phases:
        active = tokens * h * bpe
        add(phase, None, "token_embedding", costs.vector(tokens*h, 1, "embedding_lookup", 2*active),
            [_weight_read(weights, "model.embed_tokens", active)], [_hbm("write", _slot(0,max_tokens,scratch_width,bpe), active, "hidden")])
        for layer in range(layers):
            p = f"model.layers.{layer}"
            s0, s1 = _slot(0,max_tokens,scratch_width,bpe), _slot(1,max_tokens,scratch_width,bpe)
            q_addr, k_addr, v_addr = _slot(2,max_tokens,scratch_width,bpe), _slot(3,max_tokens,scratch_width,bpe), _slot(4,max_tokens,scratch_width,bpe)
            score_addr, gate_addr, up_addr = _slot(5,max_tokens,scratch_width,bpe), _slot(6,max_tokens,scratch_width,bpe), _slot(7,max_tokens,scratch_width,bpe)
            q_bytes, kv_bytes = tokens*h*bpe, tokens*kv*bpe
            score_bytes = model.attention_heads*tokens*context*bpe
            layer_kv = KV_BASE + layer*kv_layer_stride
            key_cache, value_cache = layer_kv, layer_kv + max_tokens*kv*bpe
            add(phase, layer, "input_rmsnorm", costs.vector(tokens*h, 6, "rmsnorm", 2*active),
                [_hbm("read",s0,active,"hidden"), _weight_read(weights,f"{p}.input_layernorm.weight")], [_hbm("write",s1,active,"norm_hidden")])
            add(phase, layer, "q_proj", costs.matmul(tokens,h,h,True,"q_projection"),
                [_hbm("read",s1,active,"norm_hidden"), _weight_read(weights,f"{p}.self_attn.q_proj.weight"), _weight_read(weights,f"{p}.self_attn.q_proj.bias")], [_hbm("write",q_addr,q_bytes,"q")])
            add(phase, layer, "k_proj", costs.matmul(tokens,h,kv,True,"k_projection"),
                [_hbm("read",s1,active,"norm_hidden"), _weight_read(weights,f"{p}.self_attn.k_proj.weight"), _weight_read(weights,f"{p}.self_attn.k_proj.bias")], [_hbm("write",k_addr,kv_bytes,"k")])
            add(phase, layer, "v_proj", costs.matmul(tokens,h,kv,True,"v_projection"),
                [_hbm("read",s1,active,"norm_hidden"), _weight_read(weights,f"{p}.self_attn.v_proj.weight"), _weight_read(weights,f"{p}.self_attn.v_proj.bias")], [_hbm("write",v_addr,kv_bytes,"v")])
            add(phase, layer, "rope", costs.vector(tokens*(h+kv),8,"rotary_position_embedding",2*(q_bytes+kv_bytes)),
                [_hbm("read",q_addr,q_bytes,"q"), _hbm("read",k_addr,kv_bytes,"k")], [_hbm("write",q_addr,q_bytes,"q_rope"), _hbm("write",k_addr,kv_bytes,"k_rope")])
            add(phase, layer, "kv_append", costs.vector(append_count*kv*2,1,"kv_cache_append",4*kv_bytes),
                [_hbm("read",k_addr,kv_bytes,"k_rope"), _hbm("read",v_addr,kv_bytes,"v")],
                [_hbm("write",key_cache+append_start*kv*bpe,append_count*kv*bpe,"key_cache"), _hbm("write",value_cache+append_start*kv*bpe,append_count*kv*bpe,"value_cache")])
            add(phase, layer, "attention_score", costs.batched_matmul(model.attention_heads,tokens,model.head_dim,context,"grouped_query_attention_score"),
                [_hbm("read",q_addr,q_bytes,"q_rope"), _hbm("read",key_cache,context*kv*bpe,"key_cache")], [_hbm("write",score_addr,score_bytes,"attention_scores")])
            add(phase, layer, "attention_softmax", costs.softmax(model.attention_heads,tokens,context),
                [_hbm("read",score_addr,score_bytes,"attention_scores")], [_hbm("write",score_addr,score_bytes,"attention_probabilities")])
            add(phase, layer, "attention_value", costs.batched_matmul(model.attention_heads,tokens,context,model.head_dim,"grouped_query_attention_value"),
                [_hbm("read",score_addr,score_bytes,"attention_probabilities"), _hbm("read",value_cache,context*kv*bpe,"value_cache")], [_hbm("write",q_addr,q_bytes,"attention_output")])
            add(phase, layer, "o_proj", costs.matmul(tokens,h,h,False,"o_projection"),
                [_hbm("read",q_addr,q_bytes,"attention_output"), _weight_read(weights,f"{p}.self_attn.o_proj.weight")], [_hbm("write",s1,active,"attention_projected")])
            add(phase, layer, "attention_residual", costs.vector(tokens*h,1,"residual_add",3*active),
                [_hbm("read",s0,active,"hidden"), _hbm("read",s1,active,"attention_projected")], [_hbm("write",s0,active,"hidden")])
            add(phase, layer, "post_attention_rmsnorm", costs.vector(tokens*h,6,"rmsnorm",2*active),
                [_hbm("read",s0,active,"hidden"), _weight_read(weights,f"{p}.post_attention_layernorm.weight")], [_hbm("write",s1,active,"norm_hidden")])
            ffn_bytes = tokens*model.intermediate_size*bpe
            add(phase, layer, "gate_proj", costs.matmul(tokens,h,model.intermediate_size,False,"gate_projection"),
                [_hbm("read",s1,active,"norm_hidden"), _weight_read(weights,f"{p}.mlp.gate_proj.weight")], [_hbm("write",gate_addr,ffn_bytes,"gate")])
            add(phase, layer, "up_proj", costs.matmul(tokens,h,model.intermediate_size,False,"up_projection"),
                [_hbm("read",s1,active,"norm_hidden"), _weight_read(weights,f"{p}.mlp.up_proj.weight")], [_hbm("write",up_addr,ffn_bytes,"up")])
            add(phase, layer, "silu", costs.vector(tokens*model.intermediate_size,8,"silu",2*ffn_bytes),
                [_hbm("read",gate_addr,ffn_bytes,"gate")], [_hbm("write",gate_addr,ffn_bytes,"gate_silu")])
            add(phase, layer, "gated_multiply", costs.vector(tokens*model.intermediate_size,1,"elementwise_multiply",3*ffn_bytes),
                [_hbm("read",gate_addr,ffn_bytes,"gate_silu"), _hbm("read",up_addr,ffn_bytes,"up")], [_hbm("write",gate_addr,ffn_bytes,"gated_up")])
            add(phase, layer, "down_proj", costs.matmul(tokens,model.intermediate_size,h,False,"down_projection"),
                [_hbm("read",gate_addr,ffn_bytes,"gated_up"), _weight_read(weights,f"{p}.mlp.down_proj.weight")], [_hbm("write",s1,active,"mlp_output")])
            add(phase, layer, "mlp_residual", costs.vector(tokens*h,1,"residual_add",3*active),
                [_hbm("read",s0,active,"hidden"), _hbm("read",s1,active,"mlp_output")], [_hbm("write",s0,active,"hidden")])
        add(phase,None,"final_rmsnorm",costs.vector(tokens*h,6,"rmsnorm",2*active),
            [_hbm("read",_slot(0,max_tokens,scratch_width,bpe),active,"hidden"),_weight_read(weights,"model.norm.weight")], [_hbm("write",_slot(1,max_tokens,scratch_width,bpe),active,"final_hidden")])
        add(phase,None,"lm_head",costs.matmul(1,h,model.vocab_size,False,"tied_lm_head"),
            [_hbm("read",_slot(1,max_tokens,scratch_width,bpe)+(tokens-1)*h*bpe,h*bpe,"last_hidden"),_weight_read(weights,"model.embed_tokens")], [_hbm("write",_slot(2,max_tokens,scratch_width,bpe),model.vocab_size*bpe,"logits")])

    hbm_limit = KV_BASE + model.layers*kv_layer_stride
    for op in ops:
        for access in op.reads + op.writes:
            if access.bytes <= 0:
                raise CosimError(f"non-positive access in {op.name}")
            end = access.address + access.bytes
            if access.target == "HBF_STATIC" and (access.kind != "read" or end > weights.footprint):
                raise CosimError(f"invalid HBF_STATIC access in {op.name}")
            if access.target == "HBM" and end > hbm_limit:
                raise CosimError(f"HBM address exceeds declared arena in {op.name}")
    return InferencePlan(model, weights, tuple(ops), math.ceil(weights.footprint/GiB), hbm_limit)


def remap_plan_for_gddr_abstract(plan: InferencePlan) -> InferencePlan:
    """Place every analytical access in one HBM device with no cache model.

    The current HBFSim session exposes a generic HBM device, not a native
    GDDR6 backend.  This function makes that limitation explicit: it maps the
    planned HBF_STATIC weights and existing HBM scratch/KV spans into one
    non-overlapping address space for a numeric GDDR-inspired HBM overlay.
    """
    if plan.memory_layout != MEMORY_LAYOUT_HBF_HBM:
        raise CosimError("GDDR abstract remapping requires the original HBF/HBM plan")
    shift = _align_up(plan.weights.footprint)
    remapped_operators: list[OperatorPlan] = []
    hbm_end = 0
    for op in plan.operators:
        accesses: list[Access] = []
        for access in (*op.reads, *op.writes):
            if access.target == "HBF_STATIC":
                remapped = dataclasses.replace(access, target="HBM")
            elif access.target == "HBM":
                remapped = dataclasses.replace(access, address=access.address + shift)
            else:
                raise CosimError(f"unsupported target while building GDDR abstract plan: {access.target}")
            hbm_end = max(hbm_end, remapped.address + remapped.bytes)
            accesses.append(remapped)
        read_count = len(op.reads)
        remapped_operators.append(dataclasses.replace(
            op, reads=tuple(accesses[:read_count]), writes=tuple(accesses[read_count:]),
        ))
    if hbm_end <= shift:
        raise CosimError("GDDR abstract plan did not produce a non-empty HBM address space")
    return dataclasses.replace(
        plan,
        operators=tuple(remapped_operators),
        static_blocks_per_plane=0,
        hbm_arena_bytes=hbm_end,
        memory_layout=MEMORY_LAYOUT_GDDR_ABSTRACT_ALL_HBM,
        hbm_address_shift_bytes=shift,
    )


def plan_summary(plan: InferencePlan, architecture_name: str) -> dict[str, Any]:
    by_phase: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for op in plan.operators:
        by_phase[op.phase] = by_phase.get(op.phase, 0) + 1
        by_family[op.timing.family] = by_family.get(op.timing.family, 0) + 1
    return {
        "schema": "llmcompass-hbfsim-qwen-p32d2-plan-v1", "model": dataclasses.asdict(plan.model),
        "llmcompass_architecture": architecture_name, "operator_count": len(plan.operators),
        "operator_count_by_phase": by_phase, "operator_count_by_family": by_family,
        "weight_bytes": plan.weights.footprint, "static_blocks_per_plane": plan.static_blocks_per_plane,
        "hbm_arena_bytes": plan.hbm_arena_bytes, "weight_object_count": len(plan.weights.objects),
        "memory_layout": plan.memory_layout, "hbm_address_shift_bytes": plan.hbm_address_shift_bytes,
    }


def _load_hbfsim_api(source: Path) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(source))
    from hbfsim_client import ResolvedSystemConfig, SimulationSession, Transaction
    return SimulationSession, Transaction, ResolvedSystemConfig


def _transactions(Transaction: Any, op: OperatorPlan, origin_ns: int) -> tuple[list[Any], set[str], str]:
    txs: list[Any] = []
    read_ids: list[str] = []
    completion_ids: set[str] = set()
    prefix = f"op{op.index:06d}"
    for index, access in enumerate(op.reads):
        identifier = f"{prefix}-r{index:03d}"
        txs.append(Transaction(id=identifier, target=access.target, op="R", addr=access.address, bytes=access.bytes, issue_ns=origin_ns))
        read_ids.append(identifier)
        completion_ids.add(identifier)
    barrier_id = f"{prefix}-compute"
    txs.append(Transaction(id=barrier_id, target="BARRIER", op=None, addr=0, bytes=0, issue_ns=origin_ns, duration_ns=op.timing.compute_ns, dependencies=tuple(read_ids)))
    for index, access in enumerate(op.writes):
        identifier = f"{prefix}-w{index:03d}"
        txs.append(Transaction(id=identifier, target=access.target, op="W", addr=access.address, bytes=access.bytes, issue_ns=origin_ns, dependencies=(barrier_id,)))
        completion_ids.add(identifier)
    return txs, completion_ids, barrier_id


def _ensure_output_scope(adapter: Path, output: Path) -> None:
    try:
        output.resolve().relative_to(adapter.resolve())
    except ValueError as exc:
        raise CosimError(f"output must be inside {adapter.resolve()}") from exc
    if output.resolve().exists():
        raise CosimError(f"refusing to reuse existing output directory: {output.resolve()}")


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CosimError(f"cannot read {description}: {path}") from exc
    if not isinstance(data, dict):
        raise CosimError(f"{description} must contain a JSON object: {path}")
    return data


def _hardware_phase_ms(finish: dict[str, Any]) -> dict[str, float]:
    """Extract the single measured P32D2 sample written by the SGLang driver."""
    contract = finish.get("input_contract")
    if finish.get("status") != "PASS" or not isinstance(contract, dict):
        raise CosimError("hardware finish must be a passing SGLang result with input_contract")
    expected_contract = {"case_id": "qwen25_1p5b-p32-d2", "batch_size": 1, "prefill_length": 32, "decode_steps": 2}
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise CosimError("hardware finish is not the fixed Qwen2.5-1.5B P32D2 contract")
    phase_map = {"Prefill": "prefill", "Decode1": "decode_1", "Decode2": "decode_2"}
    result: dict[str, float] = {}
    rows = finish.get("phase_rows")
    if not isinstance(rows, list):
        raise CosimError("hardware finish has no phase_rows list")
    for row in rows:
        if not isinstance(row, dict) or row.get("role") != "measurement":
            continue
        phase = phase_map.get(row.get("phase"))
        seconds = row.get("seconds")
        if phase is None or not isinstance(seconds, (int, float)) or seconds <= 0 or phase in result:
            raise CosimError("hardware finish has invalid or duplicate measured phase rows")
        result[phase] = float(seconds) * 1e3
    if set(result) != set(phase_map.values()):
        raise CosimError("hardware finish must contain exactly one measured Prefill, Decode1, and Decode2 row")
    return result


def _simulation_phase_ms(summary: dict[str, Any]) -> dict[str, float]:
    """Convert cumulative operator-boundary finish times into phase elapsed time."""
    if summary.get("schema") != "llmcompass-hbfsim-qwen-p32d2-summary-v1" or summary.get("status") != "PASS":
        raise CosimError("simulation summary must be a passing Qwen P32D2 co-simulation result")
    finishes = summary.get("phase_finish_ns")
    order = ("prefill", "decode_1", "decode_2")
    if not isinstance(finishes, dict) or set(finishes) != set(order):
        raise CosimError("simulation summary has an incomplete phase_finish_ns map")
    result: dict[str, float] = {}
    previous = 0.0
    for phase in order:
        finish = finishes[phase]
        if not isinstance(finish, (int, float)) or finish <= previous:
            raise CosimError("simulation cumulative phase finish times must be strictly increasing")
        result[phase] = (float(finish) - previous) / 1e6
        previous = float(finish)
    if not math.isclose(previous, float(summary.get("simulated_finish_ns", -1)), rel_tol=0.0, abs_tol=1e-6):
        raise CosimError("simulation finish time does not match its final phase")
    return result


def compare_to_hardware(adapter: Path, output: Path, hardware_finish_path: Path, simulation_summary_path: Path) -> dict[str, Any]:
    """Write a transparent candidate-profile error report from immutable run receipts."""
    _ensure_output_scope(adapter, output)
    hardware_finish = _load_json_object(hardware_finish_path, "hardware finish")
    simulation_summary = _load_json_object(simulation_summary_path, "simulation summary")
    hardware_ms = _hardware_phase_ms(hardware_finish)
    simulation_ms = _simulation_phase_ms(simulation_summary)
    phases = ("prefill", "decode_1", "decode_2")
    phase_rows = []
    for phase in phases:
        hardware = hardware_ms[phase]
        simulated = simulation_ms[phase]
        signed = simulated - hardware
        phase_rows.append({
            "phase": phase,
            "hardware_host_wall_ms": hardware,
            "simulated_operator_boundary_ms": simulated,
            "signed_error_ms": signed,
            "signed_error_percent": signed / hardware * 100.0,
            "absolute_error_ms": abs(signed),
            "absolute_error_percent": abs(signed) / hardware * 100.0,
        })
    hardware_total = sum(hardware_ms.values())
    simulated_total = sum(simulation_ms.values())
    total_signed = simulated_total - hardware_total
    gddr_abstract = simulation_summary.get("memory_layout") == MEMORY_LAYOUT_GDDR_ABSTRACT_ALL_HBM
    report = {
        "schema": "llmcompass-hbfsim-qwen-p32d2-rtx4000-ada-candidate-comparison-v1",
        "status": (
            "GDDR_ABSTRACT_CROSS_CONFIG_DIAGNOSTIC_ONLY"
            if gddr_abstract else "CANDIDATE_PROFILE_COMPARISON_ONLY"
        ),
        "completed_utc": _utc_now(),
        "hardware_receipt": {"path": str(hardware_finish_path.resolve()), "sha256": _sha256(hardware_finish_path)},
        "simulation_receipt": {"path": str(simulation_summary_path.resolve()), "sha256": _sha256(simulation_summary_path)},
        "phase_rows": phase_rows,
        "aggregate": {
            "hardware_host_wall_ms": hardware_total,
            "simulated_operator_boundary_ms": simulated_total,
            "signed_error_ms": total_signed,
            "signed_error_percent": total_signed / hardware_total * 100.0,
            "absolute_error_ms": abs(total_signed),
            "absolute_error_percent": abs(total_signed) / hardware_total * 100.0,
        },
        "definitions": {
            "hardware_host_wall_ms": "one SGLang measurement per phase, measured by its driver around synchronized execution; it includes framework/runtime overhead",
            "simulated_operator_boundary_ms": "elapsed time from the analytical LLMCompass operator plan and HBFSim completion frontier",
            "signed_error_percent": "(simulated_operator_boundary_ms - hardware_host_wall_ms) / hardware_host_wall_ms * 100",
        },
        "calibration_verdict": (
            "NOT_CALIBRATED: the simulation uses the generic HBFSim HBM device with a GDDR-inspired numeric overlay and may use a non-matching LLMCompass architecture. It is not a native GDDR6 backend, cycle, kernel, cache, numerical, or statistical accuracy."
            if gddr_abstract else
            "NOT_CALIBRATED: this is one fixed-shape untraced hardware sample and a candidate device profile; it is not cycle, kernel, cache, GDDR6, numerical, or statistical accuracy."
        ),
    }
    output.mkdir(parents=True)
    _write_json_atomic(output / "comparison.json", report)
    return report


def _empty_traffic_counts() -> dict[str, dict[str, dict[str, int]]]:
    return {
        target: {
            operation: {"transactions": 0, "logical_bytes": 0, "physical_bytes": 0}
            for operation in MEMORY_OPERATIONS
        }
        for target in MEMORY_TARGETS
    }


def _add_batch_traffic(counts: dict[str, dict[str, dict[str, int]]], transactions: Iterable[Any], records: Iterable[Any]) -> None:
    """Accumulate HBFSim-returned bytes; barriers deliberately contribute no traffic."""
    by_id = {transaction.id: transaction for transaction in transactions}
    for record in records:
        transaction = by_id.get(record.id)
        if transaction is None:
            raise CosimError(f"traffic record has unknown transaction id: {record.id}")
        if transaction.target not in MEMORY_TARGETS:
            continue
        operation = {"R": "read", "W": "write"}.get(transaction.op)
        if operation is None:
            raise CosimError(f"memory transaction {record.id} has invalid operation {transaction.op}")
        logical, physical = record.logical_bytes, record.physical_bytes
        if not isinstance(logical, int) or not isinstance(physical, int) or logical < 0 or physical < 0:
            raise CosimError(f"memory transaction {record.id} has invalid byte counters")
        row = counts[transaction.target][operation]
        row["transactions"] += 1
        row["logical_bytes"] += logical
        row["physical_bytes"] += physical


def _summed_traffic(counts: Iterable[dict[str, dict[str, dict[str, int]]]]) -> dict[str, dict[str, dict[str, int]]]:
    total = _empty_traffic_counts()
    for item in counts:
        for target in MEMORY_TARGETS:
            for operation in MEMORY_OPERATIONS:
                for field in ("transactions", "logical_bytes", "physical_bytes"):
                    total[target][operation][field] += item[target][operation][field]
    return total


def _traffic_with_bandwidth(counts: dict[str, dict[str, dict[str, int]]], elapsed_ns: float) -> dict[str, Any]:
    if elapsed_ns <= 0:
        raise CosimError("traffic interval must have a positive elapsed time")
    result: dict[str, Any] = {}
    for target in MEMORY_TARGETS:
        result[target] = {}
        for operation in MEMORY_OPERATIONS:
            row = counts[target][operation]
            # Decimal GB/s equals bytes/ns; this is an interval average, not
            # a physical link peak or cache-level bandwidth.
            result[target][operation] = {
                **row,
                "effective_logical_GBps": row["logical_bytes"] / elapsed_ns,
                "effective_physical_GBps": row["physical_bytes"] / elapsed_ns,
            }
    return result


def build_traffic_report(
    phase_counts: dict[str, dict[str, dict[str, dict[str, int]]]],
    phase_finish_ns: dict[str, float],
    *,
    memory_layout: str = MEMORY_LAYOUT_HBF_HBM,
) -> dict[str, Any]:
    """Create phase and total HBFSim traffic receipts without inventing GPU caches."""
    all_phases = ("prefill", "decode_1", "decode_2")
    if set(phase_counts) != set(all_phases) or not phase_finish_ns or not set(phase_finish_ns).issubset(all_phases):
        raise CosimError("traffic report requires P32D2 phase counters and a non-empty ordered phase prefix")
    phase_order = tuple(phase for phase in all_phases if phase in phase_finish_ns)
    previous = 0.0
    rows = []
    ordered_counts = []
    for phase in phase_order:
        finish = float(phase_finish_ns[phase])
        if finish <= previous:
            raise CosimError("traffic report requires strictly increasing phase finish times")
        counts = phase_counts[phase]
        elapsed = finish - previous
        rows.append({"phase": phase, "elapsed_ns": elapsed, "targets": _traffic_with_bandwidth(counts, elapsed)})
        ordered_counts.append(counts)
        previous = finish
    total_counts = _summed_traffic(ordered_counts)
    if memory_layout not in MEMORY_LAYOUTS:
        raise CosimError(f"unknown memory layout: {memory_layout}")
    gddr_abstract = memory_layout == MEMORY_LAYOUT_GDDR_ABSTRACT_ALL_HBM
    return {
        "schema": "llmcompass-hbfsim-qwen-p32d2-traffic-v1",
        "status": "PASS",
        "memory_layout": memory_layout,
        "phase_coverage": list(phase_order),
        "phase_rows": rows,
        "total": {"elapsed_ns": previous, "targets": _traffic_with_bandwidth(total_counts, previous)},
        "definitions": {
            "logical_bytes": "sum of the adapter-requested bytes returned in HBFSim completion receipts",
            "physical_bytes": "sum of HBFSim physical completion bytes after its target-specific transfer granularity",
            "effective_physical_GBps": "physical_bytes divided by this serial operator-boundary interval in ns; it is an interval average, not a measured Ada link peak",
            "phase_coverage": "the completed contiguous P32D2 phase prefix; a smoke receipt can contain prefill only and must not be interpreted as full P32D2",
        },
        "cache_level_coverage": {
            "L1": "NOT_MODELED: no CTA/warp/instruction cache accesses or L1 policy exist in this adapter",
            "L2": "NOT_MODELED: the LLMCompass L2 bandwidth parameter is a roofline input only; it does not produce cache hit/miss or byte counters",
            "DRAM": (
                "HBFSim generic HBM-device traffic with a GDDR-inspired numeric overlay; it is not a native GDDR6 backend or RTX 4000 Ada GDDR6 traffic"
                if gddr_abstract else
                "HBFSim HBM/HBF target traffic only; it is not RTX 4000 Ada GDDR6 traffic"
            ),
        },
        "claim_boundary": (
            "exact for this adapter's generic-HBM completion receipts under a GDDR-inspired numeric overlay, not a native GDDR6, hardware L1/L2/DRAM, or bandwidth measurement"
            if gddr_abstract else
            "exact for this adapter's HBFSim completion receipts, not a hardware L1/L2/DRAM traffic or bandwidth measurement"
        ),
    }


def _archived_ncu_rows(comparison: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Select the one archived P32D2/r4/whole NCU window without mixing ROIs."""
    if comparison.get("status") not in {"PARTIAL_COMPARISON", "PASS"}:
        raise CosimError("archived NCU comparison must be a completed comparison artifact")
    rows = comparison.get("rows")
    if not isinstance(rows, list):
        raise CosimError("archived NCU comparison is missing rows")
    phase, scope = ARCHIVED_NCU_WHOLE
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if (row.get("workload"), row.get("model"), row.get("phase"), row.get("scope")) != (
            ARCHIVED_NCU_WORKLOAD, ARCHIVED_NCU_MODEL, phase, scope,
        ):
            continue
        metric = row.get("metric")
        if metric in ARCHIVED_NCU_DRAM_METRICS + ARCHIVED_NCU_CACHE_METRICS:
            if metric in selected:
                raise CosimError(f"archived NCU comparison has duplicate {metric} whole rows")
            selected[metric] = row
    expected = set(ARCHIVED_NCU_DRAM_METRICS + ARCHIVED_NCU_CACHE_METRICS)
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        raise CosimError(f"archived NCU comparison is missing P32D2/r4/whole rows: {missing}")
    launches = {row.get("launches") for row in selected.values()}
    if len(launches) != 1 or not isinstance(next(iter(launches)), int):
        raise CosimError("archived NCU whole metrics must have one integer launch count")
    for metric, row in selected.items():
        samples = row.get("hardware_samples")
        if not isinstance(samples, list) or len(samples) != 3 or not all(isinstance(value, int) and value >= 0 for value in samples):
            raise CosimError(f"archived NCU {metric} must retain three non-negative hardware samples")
        for key in ("hardware_min", "hardware_median", "hardware_max"):
            if not isinstance(row.get(key), int) or row[key] < 0:
                raise CosimError(f"archived NCU {metric} has invalid {key}")
        if not (row["hardware_min"] <= row["hardware_median"] <= row["hardware_max"]):
            raise CosimError(f"archived NCU {metric} has an invalid observed range")
    return selected


def _aggregate_simulated_target_traffic(traffic: dict[str, Any]) -> dict[str, Any]:
    if traffic.get("schema") != "llmcompass-hbfsim-qwen-p32d2-traffic-v1" or traffic.get("status") != "PASS":
        raise CosimError("simulation traffic must be a passing Qwen P32D2 traffic receipt")
    total = traffic.get("total")
    if not isinstance(total, dict) or not isinstance(total.get("elapsed_ns"), (int, float)) or total["elapsed_ns"] <= 0:
        raise CosimError("simulation traffic is missing a positive total elapsed_ns")
    targets = total.get("targets")
    if not isinstance(targets, dict) or set(targets) != set(MEMORY_TARGETS):
        raise CosimError("simulation traffic must contain exactly HBF_STATIC and HBM targets")
    aggregate: dict[str, dict[str, int]] = {}
    for operation in MEMORY_OPERATIONS:
        aggregate[operation] = {"transactions": 0, "logical_bytes": 0, "physical_bytes": 0}
        for target in MEMORY_TARGETS:
            record = targets[target].get(operation) if isinstance(targets[target], dict) else None
            if not isinstance(record, dict):
                raise CosimError(f"simulation traffic is missing {target} {operation}")
            for field in aggregate[operation]:
                value = record.get(field)
                if not isinstance(value, int) or value < 0:
                    raise CosimError(f"simulation traffic has invalid {target} {operation} {field}")
                aggregate[operation][field] += value
    return {
        "elapsed_ns": float(total["elapsed_ns"]),
        "aggregate": aggregate,
        "per_target": targets,
    }


def _observed_ncu_value(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "unit": row["unit"],
        "hardware_samples": row["hardware_samples"],
        "hardware_min": row["hardware_min"],
        "hardware_median": row["hardware_median"],
        "hardware_max": row["hardware_max"],
        "launches": row["launches"],
    }


def compare_to_archived_ncu_traffic(
    adapter: Path,
    output: Path,
    archived_ncu_comparison_path: Path,
    simulation_traffic_path: Path,
    simulation_manifest_path: Path,
    hardware_finish_path: Path | None = None,
) -> dict[str, Any]:
    """Write a fail-closed NCU/adapter compatibility screening report.

    This deliberately reports the aggregate HBF/HBM byte projection only as a
    screening proxy.  It never promotes it to an Ada GDDR6/cache accuracy claim.
    """
    _ensure_output_scope(adapter, output)
    archived = _load_json_object(archived_ncu_comparison_path, "archived NCU comparison")
    traffic = _load_json_object(simulation_traffic_path, "simulation traffic")
    manifest = _load_json_object(simulation_manifest_path, "simulation manifest")
    hardware_finish = (
        _load_json_object(hardware_finish_path, "hardware finish")
        if hardware_finish_path is not None else None
    )
    ncu_rows = _archived_ncu_rows(archived)
    simulated = _aggregate_simulated_target_traffic(traffic)
    if manifest.get("schema") != "llmcompass-hbfsim-qwen-p32d2-run-v1" or manifest.get("status") != "PASS":
        raise CosimError("simulation manifest must be a passing Qwen P32D2 co-simulation receipt")
    plan = manifest.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("model"), dict):
        raise CosimError("simulation manifest is missing its P32D2 plan")
    model = plan["model"]
    if (model.get("prefill_tokens"), model.get("decode_steps")) != (32, 2):
        raise CosimError("simulation manifest is not P32D2")

    traffic_rows: list[dict[str, Any]] = []
    for metric, operation in zip(ARCHIVED_NCU_DRAM_METRICS, MEMORY_OPERATIONS):
        hardware = _observed_ncu_value(ncu_rows[metric])
        projected = simulated["aggregate"][operation]
        median = hardware["hardware_median"]
        signed_bytes = projected["physical_bytes"] - median
        traffic_rows.append({
            "metric": metric,
            "comparison_class": "SCREENING_PROXY_ONLY",
            "hardware_observed": hardware,
            "simulator_aggregate_hbf_hbm": projected,
            "signed_screening_delta_bytes": signed_bytes,
            "signed_screening_delta_percent": signed_bytes / median * 100.0 if median else None,
            "boundary": "The simulator value sums HBF_STATIC and HBM completion bytes. It is not a single RTX 4000 Ada GDDR6 DRAM domain, so this delta is diagnostic only and is not an accuracy error.",
        })
    cache_rows = [{
        "metric": metric,
        "comparison_class": "NOT_COMPARABLE_SIM_NOT_MODELED",
        "hardware_observed": _observed_ncu_value(ncu_rows[metric]),
        "simulator": "NOT_MODELED",
        "boundary": "The adapter has no CTA, warp, instruction-cache, cache-tag, cache-replacement, or 32-byte-sector model. L1/L2 NCU counters therefore have no simulator denominator.",
    } for metric in ARCHIVED_NCU_CACHE_METRICS]

    elapsed_ns = simulated["elapsed_ns"]
    host_time_proxy: dict[str, Any] | str
    if hardware_finish is None:
        host_time_proxy = "NOT_COLLECTED: pass --hardware-finish to compute a cross-receipt host-time-normalized rate proxy."
    else:
        hardware_phase_ms = _hardware_phase_ms(hardware_finish)
        hardware_elapsed_ns = sum(hardware_phase_ms.values()) * 1e6
        if hardware_elapsed_ns <= 0:
            raise CosimError("hardware finish has no positive total P32D2 elapsed time")
        proxy_rows: list[dict[str, Any]] = []
        for metric, operation in zip(ARCHIVED_NCU_DRAM_METRICS, MEMORY_OPERATIONS):
            hardware = _observed_ncu_value(ncu_rows[metric])
            hardware_rate = hardware["hardware_median"] / hardware_elapsed_ns
            simulator_rate = simulated["aggregate"][operation]["physical_bytes"] / elapsed_ns
            proxy_rows.append({
                "metric": metric,
                "unit": "GBps_decimal",
                "hardware_ncu_bytes": hardware,
                "hardware_host_time_ms": sum(hardware_phase_ms.values()),
                "hardware_host_time_normalized_GBps": hardware_rate,
                "simulator_operator_boundary_time_ms": elapsed_ns / 1e6,
                "simulator_operator_boundary_GBps": simulator_rate,
                "signed_proxy_delta_GBps": simulator_rate - hardware_rate,
                "signed_proxy_delta_percent": (simulator_rate - hardware_rate) / hardware_rate * 100.0 if hardware_rate else None,
            })
        host_time_proxy = {
            "comparison_class": "CROSS_RECEIPT_HOST_TIME_NORMALIZED_PROXY_ONLY",
            "hardware_finish": {"path": str(hardware_finish_path.resolve()), "sha256": _sha256(hardware_finish_path)},
            "phase_ms": hardware_phase_ms,
            "rows": proxy_rows,
            "boundary": "This divides archived NCU whole-window bytes by a separate fixed-P32D2 SGLang host-synchronized time receipt. The receipts do not prove identical launch membership or a GPU NCU range duration, so it is a diagnostic rate proxy, not observed hardware DRAM bandwidth or an accuracy metric.",
        }
    report = {
        "schema": "llmcompass-hbfsim-qwen-p32d2-archived-ncu-compatibility-v1",
        "status": "SCREENING_PROXY_ONLY_NOT_HARDWARE_ACCURACY",
        "completed_utc": _utc_now(),
        "provenance": {
            "archived_ncu_comparison": {"path": str(archived_ncu_comparison_path.resolve()), "sha256": _sha256(archived_ncu_comparison_path)},
            "simulation_traffic": {"path": str(simulation_traffic_path.resolve()), "sha256": _sha256(simulation_traffic_path)},
            "simulation_manifest": {"path": str(simulation_manifest_path.resolve()), "sha256": _sha256(simulation_manifest_path)},
        },
        "contract_join": {
            "status": "PARTIAL_LABEL_MATCH_ONLY",
            "hardware_workload_label": ARCHIVED_NCU_WORKLOAD,
            "simulation_model_id": model.get("model_id"),
            "simulation_prefill_tokens": model.get("prefill_tokens"),
            "simulation_decode_steps": model.get("decode_steps"),
            "simulation_operator_count": plan.get("operator_count"),
            "hardware_whole_kernel_launches": ncu_rows[ARCHIVED_NCU_DRAM_METRICS[0]]["launches"],
            "boundary": "The archive names P32D2 but does not ship the SGLang app.config, issue.config, model-weight digest, or runtime contract. It proves neither exact model/runtime identity nor one-to-one operator/kernel attribution.",
        },
        "source_status": {
            "archived_comparison_status": archived.get("status"),
            "source_hardware_accuracy_accepted": archived.get("hardware_accuracy_accepted"),
            "boundary": "The source comparison is retained as observed NCU samples, but its own hardware-accuracy acceptance is false and its cache replay is not used as this adapter's model.",
        },
        "comparison_window": {
            "hardware_phase": ARCHIVED_NCU_WHOLE[0],
            "hardware_scope": ARCHIVED_NCU_WHOLE[1],
            "hardware_kernel_launches": ncu_rows[ARCHIVED_NCU_DRAM_METRICS[0]]["launches"],
            "simulation_scope": "one analytical P32D2 operator-boundary run",
            "simulation_operator_count": plan.get("operator_count"),
        },
        "dram_traffic_screening": traffic_rows,
        "cache_metrics": cache_rows,
        "bandwidth": {
            "comparison_class": "NOT_COMPARABLE",
            "hardware": "NOT_AVAILABLE: the archived comparison rows contain aggregate byte/sector counters but no matched duration for the whole NCU range.",
            "simulator_aggregate_target_interval_average_GBps": {
                "read": simulated["aggregate"]["read"]["physical_bytes"] / elapsed_ns,
                "write": simulated["aggregate"]["write"]["physical_bytes"] / elapsed_ns,
                "elapsed_ns": elapsed_ns,
            },
            "boundary": "The simulator values are serial operator-boundary interval averages over abstract HBF/HBM targets, not observed Ada GDDR6 link bandwidth or an NCU range bandwidth.",
            "cross_receipt_host_time_normalized_proxy": host_time_proxy,
        },
        "calibration_verdict": "NOT_CALIBRATED: this report is a reproducible, aggregate traffic compatibility screening only. It does not establish L1, L2, DRAM, GDDR6, kernel, timing, numerical, or hardware accuracy.",
        "definitions": {
            "P32D2": "P32D2 means prompt length 32 and two decode steps. It is a shape label and does not by itself prove identical weights, runtime, or kernel sequence.",
            "NCU": "NVIDIA Nsight Compute hardware counters. This report uses archived sample minimum, median, and maximum values.",
            "L1": "Level 1 cache hardware sector counters; the current adapter has no L1 model.",
            "L2": "Level 2 cache hardware sector counters; the current adapter has no L2 model.",
            "DRAM": "Dynamic Random Access Memory aggregate byte counters from NCU; the simulation side can only provide an aggregate HBF_STATIC plus HBM completion-byte proxy.",
            "HBF_STATIC": "HBFSim static HBF target used for mapped weights; it is not RTX 4000 Ada GDDR6 memory.",
            "HBM": "HBFSim High Bandwidth Memory target; it is not RTX 4000 Ada GDDR6 memory.",
            "GDDR6": "Graphics Double Data Rate 6 memory used by RTX 4000 Ada; the current comparison has no HBFSim transaction, address-mapping, or completion-feedback coupling to native GDDR6.",
            "SCREENING_PROXY_ONLY": "Only shape-matched aggregate magnitude and direction may be screened; percentage differences must not be called hardware-accuracy errors.",
        },
    }
    output.mkdir(parents=True)
    _write_json_atomic(output / "ncu-compatibility.json", report)
    return report


def run_cosimulation(
    adapter: Path,
    hbfsim_source: Path,
    binary: Path,
    config: Path,
    config_overlays: tuple[Path, ...],
    output: Path,
    plan: InferencePlan,
    architecture_name: str,
    model_path: Path,
    architecture_path: Path,
) -> dict[str, Any]:
    _ensure_output_scope(adapter, output)
    output.mkdir(parents=True)
    manifest_path, operator_path = output/"manifest.json", output/"operators.jsonl"
    manifest: dict[str, Any] = {
        "schema": "llmcompass-hbfsim-qwen-p32d2-run-v1", "status": "RUNNING", "started_utc": _utc_now(),
        "llmcompass_head": _git_sha(adapter.parents[1]), "hbfsim_source_head": _git_sha(hbfsim_source),
        "hbfsim_binary_sha256": _sha256(binary), "hbfsim_config": str(config.resolve()),
        "hbfsim_config_sha256": _sha256(config), "output": str(output.resolve()), "plan": plan_summary(plan, architecture_name),
        "adapter_script_sha256": _sha256(Path(__file__)),
        "model_spec": str(model_path.resolve()), "model_spec_sha256": _sha256(model_path),
        "llmcompass_architecture": str(architecture_path.resolve()), "llmcompass_architecture_sha256": _sha256(architecture_path),
        "hbfsim_config_overlays": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in config_overlays
        ],
        "memory_layout": plan.memory_layout,
    }
    _write_json_atomic(manifest_path, manifest)
    session = None
    phase_finish: dict[str, int] = {}
    phase_traffic = {phase: _empty_traffic_counts() for phase in ("prefill", "decode_1", "decode_2")}
    completed_count = 0
    try:
        SimulationSession, Transaction, ResolvedSystemConfig = _load_hbfsim_api(hbfsim_source)
        enable_hbf = plan.memory_layout == MEMORY_LAYOUT_HBF_HBM
        system_config = ResolvedSystemConfig.load([config, *config_overlays]).resolve(binary, enable_hbf=enable_hbf)
        if plan.hbm_arena_bytes > system_config.hbm_application_capacity(enable_hbf=enable_hbf):
            raise CosimError("LLMCompass plan exceeds the configured HBM application capacity")
        session = SimulationSession(
            simulator_path=binary, system_config=system_config,
            enable_hbm=True, enable_hbf=enable_hbf,
            static_hbf_blocks_per_plane=plan.static_blocks_per_plane,
            hbf_wear_output_prefix=(output/"hbf-wear") if enable_hbf else None,
        )
        manifest["hbfsim_engine_source"] = session.engine_source
        manifest["hbfsim_resolved_geometry"] = {
            "hbf": system_config.hbf_geometry.canonical(),
            "hbf_logical_capacity_bytes": system_config.logical_hbf_capacity_bytes,
            "hbm_capacity_bytes": system_config.hbm_capacity_bytes,
            "hbm_burst_bytes": system_config.hbm_burst_bytes,
        }
        _write_json_atomic(manifest_path, manifest)
        with operator_path.open("w", encoding="utf-8", newline="\n") as stream:
            origin_ns = 0
            for op in plan.operators:
                # HBFSim issue_ns is relative to the current batch origin.  A
                # zero offset starts this operator at the completed frontier
                # established by the preceding blocking batch.
                txs, expected, barrier_id = _transactions(Transaction, op, 0)
                read_ids = {tx.id for tx in txs if tx.op == "R"}
                write_ids = {tx.id for tx in txs if tx.op == "W"}
                result = session.run(txs, frontier=write_ids)
                records = result.completions
                returned = {record.id for record in records}
                if returned != expected:
                    raise CosimError(f"completion mismatch for operator {op.index}: expected {sorted(expected)}, got {sorted(returned)}")
                _add_batch_traffic(phase_traffic[op.phase], txs, records)
                read_finish = max((record.finish_ns for record in records if record.id in read_ids), default=origin_ns)
                write_finish = max((record.finish_ns for record in records if record.id in write_ids), default=read_finish+op.timing.compute_ns)
                if write_finish < read_finish + op.timing.compute_ns:
                    raise CosimError(f"causality violation for operator {op.index}")
                origin_ns = result.blocking_finish_ns
                phase_finish[op.phase] = origin_ns
                completed_count += len(records)
                stream.write(json.dumps({
                    "operator_index": op.index, "phase": op.phase, "layer": op.layer, "name": op.name,
                    "timing_family": op.timing.family, "flops": op.timing.flops, "compute_barrier_ns": op.timing.compute_ns,
                    "batch_origin_ns": result.batch_origin_ns, "read_finish_ns": read_finish, "operator_finish_ns": origin_ns,
                    "completion_count": len(records), "barrier_transaction_id": barrier_id,
                }, sort_keys=True) + "\n")
        session.close()
        wear_paths = session.hbf_wear_artifacts or {}
        wear_artifacts = {
            name: {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size, "sha256": _sha256(Path(path))}
            for name, path in wear_paths.items()
        }
        session = None
        finish_ns = max(phase_finish.values(), default=0)
        traffic = build_traffic_report(phase_traffic, phase_finish, memory_layout=plan.memory_layout)
        _write_json_atomic(output/"traffic.json", traffic)
        traffic_artifact = {"path": str((output/"traffic.json").resolve()), "sha256": _sha256(output/"traffic.json")}
        summary = {
            "schema": "llmcompass-hbfsim-qwen-p32d2-summary-v1", "status": "PASS", "completed_utc": _utc_now(),
            "simulated_finish_ns": finish_ns, "simulated_finish_ms": finish_ns/1e6, "phase_finish_ns": phase_finish,
            "operator_count": len(plan.operators), "completion_count": completed_count,
            "hbf_wear_artifacts": wear_artifacts,
            "traffic_artifact": traffic_artifact,
            "memory_layout": plan.memory_layout,
            "claim_boundary": (
                "operator-boundary analytical closed loop through the generic HBFSim HBM device with a GDDR-inspired numeric overlay; not a native GDDR6, CTA/warp/cycle/cache, or hardware-calibrated model"
                if plan.memory_layout == MEMORY_LAYOUT_GDDR_ABSTRACT_ALL_HBM else
                "operator-boundary analytical closed loop; not CTA/warp/cycle/cache or hardware calibrated"
            ),
        }
        _write_json_atomic(output/"summary.json", summary)
        manifest.update({"status":"PASS", "completed_utc":summary["completed_utc"], "summary_sha256":_sha256(output/"summary.json"), "operators_sha256":_sha256(operator_path), "hbf_wear_artifacts":wear_artifacts, "traffic_artifact":traffic_artifact})
        _write_json_atomic(manifest_path, manifest)
        return summary
    except Exception as exc:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        manifest.update({"status":"FAILED", "completed_utc":_utc_now(), "error_type":type(exc).__name__, "error":str(exc)})
        _write_json_atomic(manifest_path, manifest)
        raise


def _default_paths(adapter: Path) -> dict[str, Path]:
    llmcompass = adapter.parents[1]
    hbfsim = llmcompass.parent/"HBFSim"
    return {
        "model": adapter/"qwen25_1p5b.json", "architecture": llmcompass/"configs"/"GA100.json", "hbfsim_source": hbfsim,
        "binary": adapter/"_build"/"hbfsim-current"/"hbfsim", "config": hbfsim/"configs"/"systems"/"server-hbm128-hbf512.cfg",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    adapter = Path(__file__).resolve().parent
    defaults = _default_paths(adapter)
    parser = argparse.ArgumentParser(description="Qwen2.5-1.5B P32D2 LLMCompass + HBFSim operator-boundary co-simulation")
    parser.add_argument("command", choices=("plan", "smoke", "run", "compare", "compare-ncu-traffic"))
    parser.add_argument("--model", type=Path, default=defaults["model"])
    parser.add_argument("--architecture", type=Path, default=defaults["architecture"])
    parser.add_argument("--hbfsim-source", type=Path, default=defaults["hbfsim_source"])
    parser.add_argument("--hbfsim-binary", type=Path, default=defaults["binary"])
    parser.add_argument("--hbfsim-config", type=Path, default=defaults["config"])
    parser.add_argument("--hbfsim-overlay", action="append", type=Path, default=[], help="additional HBFSim key=value overlay; later overlays win")
    parser.add_argument("--memory-layout", choices=MEMORY_LAYOUTS, default=MEMORY_LAYOUT_HBF_HBM)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hardware-finish", type=Path, help="passing SGLang finish.json for fixed P32D2 timing comparison or an explicitly labeled cross-receipt bandwidth-rate proxy")
    parser.add_argument("--simulation-summary", type=Path, help="passing Qwen P32D2 co-simulation summary.json")
    parser.add_argument("--archived-ncu-comparison", type=Path, help="archived NCU comparison.json containing qwen1p5b-P32D2 r4 whole rows")
    parser.add_argument("--simulation-traffic", type=Path, help="passing Qwen P32D2 traffic.json")
    parser.add_argument("--simulation-manifest", type=Path, help="passing Qwen P32D2 manifest.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adapter = Path(__file__).resolve().parent
    if args.command == "compare":
        if args.output is None or args.hardware_finish is None or args.simulation_summary is None:
            raise CosimError("--output, --hardware-finish, and --simulation-summary are required for compare")
        report = compare_to_hardware(adapter, args.output, args.hardware_finish, args.simulation_summary)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "compare-ncu-traffic":
        if args.output is None or args.archived_ncu_comparison is None or args.simulation_traffic is None or args.simulation_manifest is None:
            raise CosimError("--output, --archived-ncu-comparison, --simulation-traffic, and --simulation-manifest are required for compare-ncu-traffic")
        report = compare_to_archived_ncu_traffic(
            adapter, args.output, args.archived_ncu_comparison, args.simulation_traffic, args.simulation_manifest,
            args.hardware_finish,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    model = ModelSpec.from_json(args.model)
    costs = LLMCompassCostModel(adapter.parents[1], args.architecture)
    plan = build_plan(model, costs, layer_limit=1, phase_limit=1) if args.command == "smoke" else build_plan(model, costs)
    if args.memory_layout == MEMORY_LAYOUT_GDDR_ABSTRACT_ALL_HBM:
        plan = remap_plan_for_gddr_abstract(plan)
    if args.command == "plan":
        print(json.dumps(plan_summary(plan, costs.architecture_name), indent=2, sort_keys=True))
        return 0
    if args.output is None:
        raise CosimError("--output is required for smoke and run")
    summary = run_cosimulation(
        adapter, args.hbfsim_source, args.hbfsim_binary, args.hbfsim_config,
        tuple(args.hbfsim_overlay), args.output, plan, costs.architecture_name,
        args.model, args.architecture,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CosimError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
