# RTX 4000 Ada Analytical Reproduction Checkpoint

Current milestone: **accepted for analytical reproduction and accounting; not accepted for physical DRAM or full-inference timing accuracy.** Frozen on 2026-10-09. This is an operator-level reproduction on a different GPU, not a reproduction of every end-to-end accuracy claim in the original paper.

Current extension: **static batched Qwen plan support passes regression; completed larger-batch timing results are not part of this support checkpoint.** The eight-case batch-one checkpoint below remains the previous frozen milestone.

## Definitions and experimental scope

LLMCompass is the official analytical accelerator model. Its mapper selects tiling and data movement through main memory, global buffer, and local buffer. These buffers are analytical hierarchy boundaries, not measured native NVIDIA L1/L2 cache counters. DRAM means dynamic random-access memory. HBFSim is an external memory-service simulator; GDDR6 means Graphics Double Data Rate 6. NVIDIA Nsight Compute (NCU) supplies observed aggregate DRAM byte counters.

The workload is **Qwen2.5-1.5B-Instruct**, not Qwen1.5-1.5B: 28 layers, hidden size 1536, feed-forward size 8960, 12 query heads, 2 key/value heads, batch size 1, one device. P128D4 means a 128-token Prefill followed by four one-token Decode steps. The eight cases are P32D2, P64D2, P128D2, P256D2, P512D2, P128D4, P128D8, and P128D16. The model uses two-byte FP16 storage as a size proxy for BF16 hardware storage; this does not establish numerical equivalence.

The full model window is Prefill plus every requested Decode step, with 507 modeled operators per phase: 23,322 operators, 46 phases, and 38 Decode steps across the matrix. CPU wall-clock and profiler replay duration are never GPU-bandwidth denominators.

## Accepted and excluded claims

| Check | Milestone decision |
|---|---|
| Official analytical implementation | Accepted: all 111 unique frozen shapes match unchanged official operator latencies; `software_model`, `hardware_model`, and `ae/figure5` match ISCA_AE commit `62321b1ee28ddbdba8a2eb7475d7caa30f75e8be`. |
| Full Qwen operator and phase accounting | Accepted: all eight cases close operator counts, integer-nanosecond time, classified Read/Write bytes, and separately disclosed unknown-direction IO. |
| Semantic traffic accounting | Accepted as MODEL_ESTIMATE: Weights, KV cache, Activation, Other close to mapper/adapter boundary bytes. |
| Offline visualization | Accepted as model reporting: 24 Read/Write/Total SVG donut charts, unchanged embedded aggregates, no external resources. |
| RTX 4000 Ada native timing and DRAM accuracy | Not accepted. Only 2/8 full windows meet a numerical 10% timing bound; 0/8 pass the additional every-phase anti-cancellation guard in the archived validation. No physical-counter equivalence is established. |
| New GDDR6 HBFSim integration | Functional/accounting extension only. 8 cases close, but only 7/8 full cases and 1/8 every-phase cases satisfy its no-regression check against the official baseline. It is not an accuracy upgrade accepted for all cases. |

The 10% rule means `abs(model_time / hardware_time - 1) * 100 <= 10` per case. The every-phase rule is a stricter project diagnostic, not a claimed original-paper criterion. A full-window pass caused by opposite-sign Prefill/Decode errors is not an accuracy acceptance.

## Timing and transfer contracts

Formal time is the sum of official `compile_and_simulate` operator bodies, including the official Decode shortcut and BatchedMatmul minimum of its two candidates. Projection-bias and Qwen-specific vector operations remain explicitly labeled adapters. The primary ledger uses integer nanoseconds; there is no maintained native GPU cycle clock. The official transformer-level overhead aggregation is not implemented in this Qwen evaluator. No A100 launch constants, Qwen-fitted multiplier, or separately added HBFSim memory time enters this result.

Formal traffic comes directly from selected mapper transfer decisions: actual tile extents times element size, with read/write direction and reuse as implemented in the source. It is not reconstructed from rounded IO cycles times bandwidth. Qwen vector-adapter accesses are counted separately. The official K-concatenated BatchedMatmul performance surrogate is retained, not treated as a numerically equivalent executable batch graph. Its unclassifiable extra IO remains unknown-direction; it is neither guessed nor dropped.

The retired roofline-barrier-plus-serial-memory path is rejected for nonzero operator barriers before a result directory is created. It must not generate new formal results. Historical request-deletion/fusion sensitivity experiments are excluded from this snapshot, and no cross-operator QKV residency or new cache replacement policy is assumed.

Formal bandwidth is `(known_read + known_write + unknown_direction_bytes) / official_model_nanoseconds`, numerically decimal GB/s. Hardware effective bandwidth is measured NCU DRAM bytes divided by a matched unprofiled CUDA-event workload duration. These are distinct boundary estimates and effective-window observations, not identical instantaneous controller bandwidth. Hardware cycle counters were not collected.

## Frozen hardware profile and external backend

`RTX4000Ada_xmu_profile_v3.json` is the only current profile: 48 streaming multiprocessors (SMs), 2175 MHz modeled SM clock, 160-bit GDDR6 interface, 17.1 Gb/s per pin, 342 GB/s raw interface rate. Dense tensor architectural throughput is 106.9056 TFLOPS after correcting the existing `mac_per_cycle` field to 0.5. This is architectural peak, not a claim that measured GEMM throughput reaches peak. Profile SHA-256: `d7c2659d32b14c4cf0f5da876be09375a810bcdbfdc7b03c4b820cf2fc4288b1`.

Frozen native hardware is RTX 4000 Ada GPU `GPU-18ace299-5348-e6e4-d48c-1ee5a602859b`. Later paired measurements usually sampled 2325 MHz SM clock at endpoints, while the analytical profile remains frozen at 2175 MHz. Endpoint samples are not continuous clock traces. This difference is disclosed, not fitted away.

The current external backend is `/home/xmu/nvidiagds/stable/bhbfsims/gddr6-ada-lightweight`, branch `feature/gddr6-ada-lightweight-20261008`, commit `117d017737369fa1cfdf130662594c3f411230d5`. Its overlay is `configs/overlays/dram/rtx4000-ada-gddr6.cfg`, preserved byte-for-byte here as `rtx4000ada-gddr6-aggregate.cfg`: 10 x 16-bit channels and 342 GB/s raw rate, with independently calibrated aggregate efficiency 0.961886648136 (328.965 GB/s effective service ceiling). This is GDDR6-aggregate-v1, not native bank/command/refresh modeling. It is pinned externally, not vendored into this repository.

The GDDR extension replaces the modeled main-memory service at dependent stages while retaining on-chip work; it does not add a second whole-operator memory cost. Operator templates are compiled in isolated backend sessions and aggregated analytically, not run as a continuous native GPU inference. It has zero contribution to the official primary time.

## Latest eight-case comparison

Numbers below use the latest paired collection (2026-10-08 r3), not the older hardware collection used by the archived official/GDDR gates. Read uses decimal GB, Write decimal MB, time ms, and bandwidth GB/s. M means official model estimate; H means native SGLang hardware. Unknown-direction IO is included in M bandwidth but not assigned to M Read or Write.

| Case | M / H time | Time delta | M / H Read | M / H Write | M / H effective bandwidth |
|---|---:|---:|---:|---:|---:|
| P32D2 | 28.009 / 33.379 | -16.09% | 9.424 / 9.243 | 128.54 / 71.58 | 341.16 / 279.05 |
| P64D2 | 28.891 / 33.736 | -14.36% | 9.581 / 9.252 | 251.90 / 151.90 | 340.60 / 278.74 |
| P128D2 | 30.711 / 34.643 | -11.35% | 9.904 / 9.280 | 503.14 / 278.91 | 339.52 / 275.92 |
| P256D2 | 34.987 / 37.346 | -6.31% | 10.582 / 9.364 | 1023.65 / 644.58 | 333.44 / 268.00 |
| P512D2 | 49.475 / 45.436 | +8.89% | 12.070 / 9.621 | 2136.09 / 1204.35 | 291.23 / 238.24 |
| P128D4 | 48.938 / 56.469 | -13.34% | 16.131 / 15.458 | 509.82 / 279.19 | 340.44 / 278.68 |
| P128D8 | 85.399 / 100.079 | -14.67% | 28.587 / 27.808 | 523.19 / 279.33 | 341.10 / 280.65 |
| P128D16 | 158.347 / 187.131 | -15.38% | 53.507 / 52.509 | 550.00 / 279.72 | 341.51 / 282.09 |

`checkpoint/paired-collection.json` also contains the explicit-operator hardware reference. Pairing/accounting passed 320 unprofiled workflows, 80 full counter ranges, and 120 warm group counter ranges. Strict elementwise logits match in 5/8 cases and strict cross-implementation KV tolerance in 0/8; this is not an exact numerical reference. Isolated warm groups cannot be extrapolated to full-inference traffic.

Confirmed mismatch sources include materialized attention/MLP activations, mapper partial-output transfers, different native kernels and workflow windows, and the optimistic official one-row Decode shortcut. Fusion alone does not explain all differences, and remaining causal attribution is incomplete. Changing GDDR timing parameters cannot correct source Write bytes: the extension preserves them exactly.

## Semantic categories

- Weights: learned projection, normalization, bias, and tied embedding/output-head accesses; repeated traffic, not unique parameter size; inference writes are zero.
- KV cache: persistent key/value cache appends and reads. Projected temporary K/V and rotary-transformed tensors belong to Activation.
- Activation: hidden/residual tensors, Q/K/V temporaries, attention score/probability materialization, MLP intermediates, logits, and mapper partial outputs.
- Other: unclassified official batch-surrogate IO. Excluded from Read/Write pies and counted exactly once in Total.

Hardware aggregate counters do not provide these object labels. No model proportions are applied to hardware. Open `checkpoint/qwen-semantic-traffic.html` directly; the figures need neither a server nor network access.

## Source organization

- Primary: `evaluate_official_inference.py`, `mapper_matmul.py`, `mapper_softmax.py`, `qwen_hbfsim_cosim.py` (plan/cost definitions), model JSON, and v3 profile.
- Accounting/reporting: `semantic_traffic_breakdown.py`, `verify_official_inference.py`, `verify_stage_checkpoint.py`, and their tests.
- Independent GDDR diagnostic: `evaluate_gddr_integration.py`, `mapper_event_coupling.py`, `mapper_tensor_addresses.py`, `mapper_operator_paths.py`, and GDDR tests. Full hardware evidence is external.
- Hardware diagnosis: `paired_qwen_reference.py`, `verify_paired_qwen_reference.py`, `validate_ae_ada.py`, and `diagnose_inference_timing.py`. These require the XMU SGLang/checkpoint/input package and NCU installation; they are not standalone portable GPU benchmarks.
- Legacy shell collectors/summarizers remain for provenance only. Their absolute lab paths and old snapshots are not current entrypoints.

## Static batched Qwen P128D8 stage

Batch size B is the integer number of simultaneous, synchronized requests. This stage uses Qwen2.5-1.5B-Instruct on the same frozen RTX 4000 Ada profile, with B in {2, 4, 8, 16, 32}; B=1 is a fresh regression baseline. P128D8 means 128 Prefill tokens per request followed by eight explicit one-token Decode calls per request. It does not mean seven Decode calls after Prefill emits a first output token.

All requests share one learned-weight allocation. Each request has distinct persistent key and value cache regions across all 28 layers. Linear layers use B times the token count as their row dimension; attention uses B times the query-head count as its batch dimension. The output head consumes each request's own last token. These changes enlarge the existing graph shapes and addresses; they do not introduce continuous batching, arrival queues, a scheduler, a new cache, numerical GPU execution, or latency fitting.

`evaluate_batch_inference.py` reuses official Matmul, BatchedMatmul two-candidate selection, Softmax and Decode rules, with the same labeled Qwen vector/bias adapters. Main-memory bytes come from existing mapper transfer decisions; batch-surrogate extra IO remains unknown-direction. Weight traffic is determined by selected mappings, not a blanket batch-one multiplier. HBFSim is not a prerequisite and adds zero time to this stage.

Each complete batch case has 4,563 modeled operators in nine phases, with 507 operators per phase. The baseline plus five target cases require 27,378 operators and 54 phases. Unique resident key/value payload is `2 * 28 * B * 136 * 256 * 2` bytes; repeated mapper reads are not unique capacity. Scratch arena offsets are plan allocations, not measured native peak memory usage.

Time uses integer nanoseconds summed across the whole batch. Every synchronized request's full completion latency is the complete batch interval. Interval divided by B is explicitly the amortized per-request throughput cost, not latency. Prefill throughput is `B * 128 / Prefill seconds`; Decode throughput is `B * 8 / sum of Decode seconds`; request throughput is `B / full-window seconds`. Effective model bandwidth is `(known Read + known Write + unknown-direction IO) / full-window nanoseconds`, in decimal GB/s. None uses CPU mapping/compilation wall-clock.

Validation completed for support: 65 unit tests; exact old/new plans across the original eight cases (23,322 operators, including access bytes and addresses); all six batch plans' request/KV isolation and last-token input checks; and negative controls for corrupt totals, wrong denominators, missing cases, KV aliasing and incomplete matrix acceptance. Fresh B1 P128D8 also matches frozen time and every phase's Read/Write/unknown IO exactly. This is support/regression validation, not completed larger-batch timing or hardware acceptance.

The fresh aggregate-only run is `/home/xmu/nvidiagds/simulators/wgslogs/llmcompass/qwen-static-batch-p128d8-20261009-r2`. Consult its `manifest.json` for live status; the support checkpoint does not assert completed larger-batch timing. An earlier empty preflight directory r1 is preserved; it is not an experiment result. The compiler writes only `manifest.json`, `comparison.json`, `shapes.json` and optional compact validation receipts, without raw transfer traces or per-operator streams. The independent verifier requires terminal PASS, all six cases, source/result hashes, official shape parity, semantic/phase/family closure and exact B1 regression before the matrix can be published as complete. `checkpoint/batch-support-validation.json` records the bounded support tests, separately from this full-matrix gate.

The verified runtime uses Python 3.10 with ScaleSim 2.0.2, PyTorch 2.5.1, NumPy 2.2.6 and pandas 2.3.3. The previous archival runtime receipt records PyTorch 2.7.1+cu126; this is a disclosed environment difference, not an identical-runtime claim. No CUDA kernel is executed by this analytical matrix, and fresh B1 arithmetic is unchanged. Existing ScaleSim/analytical dependencies are external, not copied into Git.

```sh
env PYTHONPATH=/tmp/llmcompass-scalesim202-20261007:/tmp/llmcompass-ae-deps-20261006 \
  PYTHONNOUSERSITE=1 /home/xmu/miniconda3/envs/deepspeed/bin/python3.10 -B \
  integration/hbfsim/evaluate_batch_inference.py --output-root /absolute/fresh-result-root
python -B integration/hbfsim/verify_batch_inference.py /absolute/fresh-result-root \
  --output /absolute/fresh-result-root/validation.json
cd integration/hbfsim
python -B -m unittest test_qwen_hbfsim_cosim test_semantic_traffic_breakdown \
  test_verify_official_inference test_paired_qwen_reference test_evaluate_gddr_integration \
  test_verify_stage_checkpoint test_batch_inference test_verify_batch_inference
```

The temporary dependency paths above describe the existing XMU environment, not portable installation instructions. Retain official geometry lookup tables. Use a new absolute result directory for each full run; the evaluator refuses existing result roots. No larger-batch hardware collection is included, and previous B1 hardware must not be reused as a B>1 accuracy denominator.

## Historical Llama semantic key totals

`checkpoint/llama-semantic-summary.json` publishes only eight-case Read/Write totals and their four semantic categories. Its model is Llama-3.1-8B with eight-bit matrix weights (W8) and two-byte brain floating-point activation/key-value storage (BF16). It uses the historical GA100/generic high-bandwidth-memory configuration and legacy adapter, not the current official Qwen/Ada path. The model section retains original default P32D2 conditions; each case row overrides Prefill/Decode length.

Five target cases reconcile to archived simulation totals; P64D2, P256D2 and P128D4 are plans only, explicitly labeled `PLAN_ONLY_NOT_SIMULATED`. All nine actual historical runs close 50,373 operator signatures and 87 phase windows with zero Read/Write residual. The current reconstruction adapter differs from the archived source hash, and three P512 address layouts differ. The summary preserves these provenance limitations and does not establish exact-address, official-mapper or physical-DRAM equivalence. Legacy timing is excluded from this publication. No raw operator/phase stream or hardware semantic attribution is published.

## Reproduction and verification

From the repository root, the portable aggregate check uses Python 3.10 or later and the standard library only:

```sh
python -B integration/hbfsim/verify_stage_checkpoint.py
cd integration/hbfsim
python -B -m unittest test_qwen_hbfsim_cosim test_semantic_traffic_breakdown test_verify_official_inference test_paired_qwen_reference test_evaluate_gddr_integration test_verify_stage_checkpoint
```

The model-loading Qwen test needs the official numerical dependencies. Exact frozen environment versions and commands are recorded in `checkpoint/manifest.json`; ScaleSim must be 2.0.2. Keep the tracked geometry lookup tables. The official `environment.yml` remains unchanged. GPU-dependent historical diagnostics need additional PyTorch/SGLang/NCU dependencies, not just the standard-library check.

Fresh archive validation passes 53 unit tests, including eight checkpoint negative/conservation controls. Four Matmul shapes (including a one-row shortcut and remainder tiles), three BatchedMatmul shapes, and one Softmax shape preserve official mapping/time and direct transfer-byte closure. The complete P32D2 tensor-bound audit passes 1,521 operators and 7,989 matrix views; this is an address-bound check, not continuous full-GPU simulation. Independent external-evidence rechecks also pass the complete frozen official, paired-hardware, and GDDR accounting ledgers. No new hardware measurements or new eight-case timing simulation was performed for this archive.

Recompile the official full matrix with an existing aggregate hardware collection and a **fresh absolute output directory** (this computes analytical model results, not new GPU measurements):

```sh
python -B integration/hbfsim/evaluate_official_inference.py \
  --hardware-root /path/to/collection-r2 \
  --output-root /path/to/fresh-official-matrix
python -B integration/hbfsim/verify_official_inference.py /path/to/fresh-official-matrix
python -B integration/hbfsim/semantic_traffic_breakdown.py analyze \
  /path/to/fresh-official-matrix /path/to/fresh-semantic-output
python -B integration/hbfsim/semantic_traffic_breakdown.py html \
  integration/hbfsim/checkpoint/semantic-breakdown.json integration/hbfsim/checkpoint/paired-collection.json \
  /path/to/fresh-html-output --report-date 2026-10-09
```

The independent official verifier additionally requires the original referenced hardware timing/counter aggregates. The eight-case GDDR runner currently consumes the historical frozen matrix and selected-operands package at its documented `results/` locations. Those detailed ledgers remain outside Git; do not silently substitute compact checkpoint files for them. The archive contains compact gate results, not every consumed measurement receipt.

## Preservation and publication

`checkpoint/manifest.json` binds published sources/configuration, upstream-source hashes, compact results, external evidence identities, and the fresh validation receipt. This proves portable aggregate conservation and a frozen code snapshot; independent full-ledger validation remains an explicitly identified external evidence check.

Publication continues the authorized historical branch `codex/integration-key-acceptance-20261001`, through checkpoint `25e75accc49ee4985a5d1906a8292ec456bef7f1`, descending from integration commit `613cbe73f489ad304b28b10d314c373324cb5dc4`. No new branch is created. Original dirty worktrees and historical evidence remain unchanged. Raw traces, NCU/NSYS binaries, per-operator streams, caches, builds, runtime logs and incomplete batch timing results are not published. Historical objects in parent Git commits remain in history; the earlier source-only archive is immutable evidence for its own checkpoint, not a snapshot of these later support changes.

Do not promote this milestone to physical DRAM accuracy, every-case timing within 10%, exact native-kernel execution, strict numerical equivalence, or continuous full-GPU/HBFSim simulation.
