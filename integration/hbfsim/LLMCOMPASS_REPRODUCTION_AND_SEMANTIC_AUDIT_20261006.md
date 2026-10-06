# LLMCompass Reproduction and SGLang Semantic Audit

Date: 2026-10-06
Repository branch: `codex/integration-key-acceptance-20261001`
Repository commit: `23bd6ca`

## Scope and claim boundary

This audit restores the official LLMCompass operator-validation path, compares it with the current Qwen2.5-1.5B integration, and documents the semantic relation between the LLMCompass operator graph, the SGLang execution graph, and HBFSim completion receipts.

The audit does not modify the original model, add cache modeling, add a kernel scheduler, add launch-time fitting, or tune parameters to hardware. Existing Qwen/HBFSim results remain unchanged. LLMCompass outputs are `MODEL_ESTIMATE`; NCU and CUDA-event outputs are `OBSERVED`.

## Official paper and ISCA_AE reference

The official paper defines LLMCompass as a computational graph plus hardware description, mapper, architecture simulator, and performance report. Its mapper recursively partitions operators into global-buffer and local-buffer tiles, searches mapping and scheduling parameters, and reports simulated latency. The paper's AE appendix specifies Python 3.9, `scalesim`, PyTorch 2.0, and the `ae/figure5` through `ae/figure12` scripts. It also states that real-hardware profiling is supplied in advance and is outside the artifact workflow.

Reference sources:

- Official repository and `ISCA_AE` branch: https://github.com/PrincetonUniversity/LLMCompass
- Official paper: https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2024_2025/papers/LLMCOMPASS_ISCA_2024.pdf

## Official operator validation status

The original `ae/figure5` validation scripts are present and were executed through the official software-model APIs. The host did not have a complete AE environment by default; a temporary dependency path was used for `scalesim`, `seaborn`, and `pytz`. No repository source was changed by dependency setup.

Representative official A100 validation results:

| Operator | Shape | Official path | Result | Latency | Mapper result |
|---|---|---|---|---:|---|
| Matmul | 32 x 12288 x 12288 | `compile_and_simulate(..., heuristic-GPU)` | PASS | 0.150296 ms | 211,918 cycles |
| Softmax | 4096 x 1024 | `compile_and_simulate` | PASS | 0.014479 ms | 20,416 cycles |
| LayerNorm | 4096 x 1536 | `compile_and_simulate(..., heuristic-GPU)` | PASS | 0.013424 ms | 18,928 cycles |
| GeLU | 65536 elements | `compile_and_simulate(..., heuristic-GPU)` | PASS | 0.000174 ms | roofline/vector path; no mapper object |

This confirms that the official mapper and cycle-count path is functional. It does not validate the custom RTX 4000 Ada Qwen adapter, because that adapter currently calls selected roofline methods and then inserts the resulting durations into a separate HBFSim transaction graph.

## Reproduction differences

| Dimension | Official ISCA_AE | Current Qwen reproduction | Consequence |
|---|---|---|---|
| Workload | GPT-3 transformer-layer experiments, FP16, commonly A100/TP4 | Qwen2.5-1.5B, BF16-native SGLang reference, TP1, P32D2 matrix | The graphs and shapes are different |
| Operator path | `compile_and_simulate` invokes the operator mapper and cycle model | `LLMCompassCostModel` uses roofline calls for matrix/softmax operators and custom vector timing | Current results are not an official full-graph mapper reproduction |
| Hardware profile | Official configs, primarily GA100 | RTX 4000 Ada profile v2: 48 SM, 2.175 GHz observed runtime SM clock, 342 GB/s runtime GDDR6 rate | Profile identity is explicit, but not a native Ada microarchitecture model |
| Kernel fusion | Not represented as a framework-level fusion graph | SGLang has merged QKV, merged gate/up, `SiluAndMul`, residual-aware RMSNorm, and FlashInfer attention | The logical LLMCompass graph has more materialization boundaries |
| Launch overhead | AE scripts add fixed operator constants in some figure scripts | Ada profile sets operator overhead to zero; the adapter does not model host launch or runtime dispatch | No launch claim is made for the Qwen simulation |
| Memory | LLMCompass models hierarchical buffer bandwidth and mapping; no hardware cache counters | Phase 1 analytical cache accounting is `MODEL_ESTIMATE`; HBFSim uses generic HBM with a GDDR-inspired overlay | L1/L2/DRAM physical traffic is not claimed |
| Timing unit | Mapper cycles divided by configured device clock give seconds | `compute_ns` is a rounded analytical duration; HBFSim receives it as a barrier duration in nanoseconds | The final value is a serial operator-boundary completion clock |

## How HBFSim time enters an operator time

For each LLMCompass operator, the adapter creates:

1. HBFSim read transactions with `issue_ns=0` relative to the current batch.
2. One `BARRIER` transaction whose `duration_ns` equals `compute_ns`.
3. HBFSim write transactions depending on the barrier.
4. A blocking frontier containing the writes.

The adapter passes `session.run(transactions, frontier=write_ids)`. HBFSim returns a `BatchResult` in `batch_relative_ns`; its `blocking_finish_ns` becomes the next adapter batch origin and the operator's completion time. Therefore, the effective relation is:

```text
operator_finish_ns = HBFSim blocking frontier completion
                   = memory completion and queueing
                     plus the dependent analytical compute barrier
```

The adapter does not add a separate HBFSim service-time scalar, and it does not convert HBFSim nanoseconds through another frequency. HBFSim memory service and queue work are already represented in transaction completion timestamps. The adapter serializes operator batches by carrying the blocking frontier forward. The result is deterministic `MODEL_ESTIMATE` event time, not CUDA wall time, kernel time, or cycle-accurate GPU execution.

## LLMCompass-to-SGLang semantic mapping

The mapping below is source-based. The existing NCU collection has aggregate ROI rows and no kernel-name census (`expected_kernel_count` is unavailable), so a numeric hardware error for each individual kernel cannot be claimed.

| LLMCompass operator(s) | SGLang implementation/fusion group | Mapping status | Diagnostic traffic implication |
|---|---|---|---|
| `token_embedding` | `VocabParallelEmbedding` | Direct semantic match | Embedding weight read and hidden-state production |
| `input_rmsnorm` | `RMSNorm`; residual-aware on later layers | Conditional match | Later-layer residual input is represented differently |
| `q_proj`, `k_proj`, `v_proj` | `QKVParallelLinear` (`qkv_proj`) | Fused group | Separate logical reuse of the same normalized input |
| `rope` | `rotary_emb` after QKV split | Source-level match; launch identity unavailable | Q/K intermediate materialization is not independently measured |
| `kv_append`, attention score, softmax, attention value | `RadixAttention` with the FlashInfer backend | Fused attention group | Score/probability tensors are logical LLMCompass intermediates, not confirmed standalone SGLang tensors |
| `o_proj` | `RowParallelLinear` (`o_proj`) | Direct semantic match | Attention output remains an input to the projection |
| `attention_residual`, `post_attention_rmsnorm` | Residual-aware `RMSNorm(hidden_states, residual)` | Fused boundary | Explicit residual output and reload are not present in the SGLang source path |
| `gate_proj`, `up_proj` | `MergedColumnParallelLinear` (`gate_up_proj`) | Fused group | The normalized input is logically read twice by the current plan |
| `silu`, `gated_multiply` | `SiluAndMul` | Fused group | `gate_silu` is an explicit LLMCompass intermediate |
| `down_proj` | `RowParallelLinear` (`down_proj`) | Direct semantic match | Projection output is passed to the layer residual path |
| `mlp_residual`, next-layer `input_rmsnorm` | Residual is carried and combined by the next residual-aware RMSNorm | Semantic mismatch requiring caution | Current plan materializes an explicit MLP residual result; SGLang carries a separate residual state |
| `final_rmsnorm` | Final `RMSNorm` | Direct semantic match | Standalone final normalization |
| `tied_lm_head` | Tied `lm_head` / logits processor | Direct semantic match at graph level | Kernel-level launch identity unavailable |

The `mlp_residual` row is not a harmless fusion difference. In SGLang, the residual state is carried from the attention path and combined at the next layer's input normalization. The current adapter instead performs an explicit residual add into the hidden scratch slot. This is a semantic graph discrepancy and should remain a separately reported diagnostic; it must not be silently treated as a cache or timing calibration issue.

## Upper-bound diagnostic for unfused activation boundaries

The following is a source-derived upper-bound estimate, not a hardware measurement. It counts only candidate boundary bytes that can disappear when the confirmed SGLang fusion groups keep values within a fused execution path:

- two redundant normalized-input reads for separate Q/K/V projection;
- one redundant normalized-input read for separate gate/up projection;
- `silu` output write plus reload before gated multiply;
- attention score and probability write/read materialization;
- explicit attention residual write plus reload before residual-aware RMSNorm.

It does not claim that every byte is physically eliminated by a particular CUDA kernel. It excludes the unresolved MLP residual semantic mismatch and excludes cache effects.

| Case | LLMCompass total logical traffic after abstract remap | Candidate unfused activation bytes | Candidate fraction |
|---|---:|---:|---:|
| P32D2 | 9.477 GB | 1.846 MB | 0.0195% |
| P64D2 | 9.684 GB | 3.785 MB | 0.0391% |
| P128D2 | 10.115 GB | 8.254 MB | 0.0816% |
| P128D4 | 16.311 GB | 8.381 MB | 0.0514% |
| P128D8 | 28.703 GB | 8.638 MB | 0.0301% |
| P128D16 | 53.489 GB | 9.149 MB | 0.0171% |
| P256D2 | 11.042 GB | 19.550 MB | 0.1771% |
| P512D2 | 13.160 GB | 51.581 MB | 0.3920% |

The candidate fusion-boundary traffic is therefore far too small to explain the approximately 1.79--2.24x simulated-time/hardware-time gap or the approximately 2.17--2.27x model-bandwidth/hardware-bandwidth gap. Fusion is a useful independent diagnosis, but it is not a justified correction factor for the current model.

## Semantic traffic closure

| Semantic quantity | LLMCompass/HBFSim side | Hardware side | Closure status |
|---|---|---|---|
| Weight reads | `HBF_STATIC` accesses; abstract remap places them in HBM for the Ada comparison | NCU `dram__bytes_read.sum` | Aggregate comparison only; no cache filtering |
| Activation reads/writes | HBM scratch and KV accesses from the operator plan | NCU DRAM read/write counters | Aggregate comparison only; no kernel-level join |
| L1 traffic | Not modeled | Not collected as a physical L1 byte total in the acceptance table | `NOT_MODELED` / `NOT_AVAILABLE` |
| L2 traffic | L2 bandwidth is a roofline input only | L2 counters are diagnostic, not used as DRAM bytes | `NOT_MODELED` for LLMCompass |
| Simulated time | HBFSim `blocking_finish_ns` after the analytical barrier | CUDA-event median over the native ROI | Different semantic clocks; comparison only |
| Effective bandwidth | HBFSim returned bytes divided by serial analytical interval | NCU DRAM bytes divided by CUDA-event median | Trend comparison only |
| Per-kernel error | No kernel-level output in current adapter receipt | Current NCU raw CSV is aggregate; kernel-name census is absent | `NOT_AVAILABLE` |

## Per-operator error status

A numeric per-operator hardware error is intentionally not reported from the current evidence. The NCU receipt contains aggregate ROI counters and the native finish receipt records `expected_kernel_count` as unavailable. Assigning aggregate DRAM bytes or aggregate time to individual LLMCompass operators would create an unsupported denominator and would mix fused SGLang groups with unfused analytical operators.

The defensible per-operator result is therefore a semantic status:

- `DIRECT_MATCH`: embedding, output projection, down projection, final normalization, and graph-level tied head.
- `FUSED_GROUP`: QKV, gate/up, SiLU/multiply, and FlashInfer attention.
- `CONDITIONAL_MATCH`: RMSNorm because residual-aware calls change the state boundary.
- `SEMANTIC_MISMATCH`: current explicit MLP residual materialization versus SGLang's carried residual state.
- `NOT_MEASURABLE`: numeric hardware error for an individual kernel without a kernel census and a stable operator-to-kernel join key.

## Acceptance conclusion

The official LLMCompass mapper/operator path is functional and reproducible. The current Qwen integration is a separate analytical adapter with explicit HBFSim completion receipts. Its DRAM traffic comparison is useful as an aggregate `MODEL_ESTIMATE` versus `OBSERVED` report, but it is not a per-kernel or physical cache-traffic validation.

The fusion diagnostic should remain independent. The source-derived unfused activation upper bound is below 0.4% of total abstract-remapped traffic in the tested matrix, so it cannot explain the dominant time and bandwidth discrepancy. The dominant conclusion remains that the current serial analytical completion clock and aggregate HBFSim memory model are not the same execution-time semantics as native SGLang CUDA execution.

No cache model, hardware-fitting coefficient, launch model, or original Qwen model change is introduced by this audit.
