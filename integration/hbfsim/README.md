# Qwen2.5-1.5B P32D2 LLMCompass and HBFSim Integration

This directory contains the curated implementation and acceptance evidence for the Qwen2.5-1.5B P32D2 analytical co-simulation path.

## Scope and contract

Qwen2.5-1.5B is modeled with 28 Transformer layers, hidden size 1536, gated feed-forward size 8960, 12 query heads, 2 key/value heads, and BF16-sized elements. P32D2 means batch size 1, a 32-token prompt prefill, and two one-token autoregressive decode steps. TP1 means one-device tensor parallelism. The three phase context lengths are 32, 33, and 34.

The adapter builds an operator-boundary analytical closed loop: LLMCompass supplies analytical operator duration, HBFSim executes the operator memory transaction DAG, and the HBFSim completion frontier becomes the next operator submission time. The loop ends at operator boundaries. It is not a CTA, warp, instruction, cache, pipeline, or GPU-cycle simulator, and it is not hardware calibration.

## Memory and claim boundary

Weights, activations, and KV cache use deterministic simulated logical addresses. These are not CUDA virtual addresses or physical hardware addresses. The `gddr_abstract_all_hbm` mode applies an uncalibrated GDDR-inspired numeric overlay to the generic HBFSim HBM device. It is not a native RTX 4000 Ada GDDR6 backend.

The current adapter does not model L1 cache, L2 cache, CTA access, warp access, instruction access, cache replacement, or cache hit/miss behavior. The configured L2 bandwidth is a roofline input only. Therefore L1 and L2 hardware counters must remain `NOT_MODELED` on the simulation side.

`traffic.json` reports `logical_bytes`, `physical_bytes`, transactions, phase intervals, and interval-average GB/s. These are HBFSim completion values, not measured hardware link bandwidth. Hardware DRAM rows come from NVIDIA Nsight Compute (NCU) counters and are kept as a separate evidence class.

## Main files

- `qwen_hbfsim_cosim.py`: plan generation, transaction DAG construction, HBFSim session control, and comparison receipts.
- `test_qwen_hbfsim_cosim.py`: model, address, operator-family, dependency, and claim-boundary tests.
- `qwen25_1p5b.json`: self-contained Qwen model and P32D2 contract.
- `RTX4000Ada_xmu_candidate_v0.json`: explicitly uncalibrated RTX 4000 Ada candidate configuration.
- `summarize_four_roi_ncu.py`, `summarize_solid_bandwidth.py`, and `summarize_solid_cache_counters.py`: read-only evidence summarizers.
- `CURATED_RESULTS.md`: inventory of retained evidence and explicit exclusions.

## Current acceptance checkpoint

The latest model receipt is `results/qwen-p32d2-rtx4000ada-gddr-abstract-aligned-full-r1`. The latest hardware summaries are `results/qwen-p32d2-solid-bandwidth-r1` and `results/qwen-p32d2-solid-cache-counters-r1`. Hardware bandwidth uses five counter repeats and twenty uninstrumented natural CUDA-event timing repeats. Model differences are directional screening values defined as `(model - hardware) / hardware`; they are not hardware accuracy errors.

All bytes below use decimal GB/MB and all bandwidth values use decimal GB/s.

| Phase | Hardware DRAM read | Model HBM proxy read | Read delta | Hardware DRAM write | Model HBM proxy write | Write delta |
|---|---:|---:|---:|---:|---:|---:|
| Prefill | 3.095 GB | 3.194 GB | +3.21% | 66.746 MB | 93.168 MB | +39.59% |
| Decode 1 | 3.060 GB | 3.092 GB | +1.03% | 3.066 MB | 3.207 MB | +4.60% |
| Decode 2 | 3.088 GB | 3.092 GB | +0.12% | 0.902 MB | 3.209 MB | +255.76% |

Phase bandwidth is listed as read/write/total:

| Phase | Hardware DRAM GB/s | Model HBM proxy GB/s |
|---|---:|---:|
| Prefill | 255.265 / 5.505 / 260.770 | 232.815 / 6.791 / 239.606 |
| Decode 1 | 280.746 / 0.281 / 281.027 | 237.214 / 0.246 / 237.460 |
| Decode 2 | 283.290 / 0.083 / 283.372 | 237.212 / 0.246 / 237.458 |

The directional screening result is `PASS_DIRECTIONAL_TRAFFIC_SCREENING_NOT_HARDWARE_ACCURACY`. Decode read traffic is within 1.03% and 0.12% for the two steps. Prefill write traffic remains 39.59% higher in the model, while Decode 2 write traffic has a small absolute denominator and therefore a large relative delta. Model Decode read bandwidth is approximately 15--16% below the hardware measurements, so the operator-boundary timing path is not calibrated.

## Validation and preservation rules

The curated branch retains source, configuration, documentation, summaries, traffic receipts, manifests, comparisons, and identity records. It excludes raw CSV/JSONL streams, `*.ncu-rep` captures, `operators.jsonl`, build directories, object files, binaries, CUDA/Triton/torchinductor caches, HTML wear reports, temporary files, and run logs.

The original evidence remains copy-only on the XMU server. Historical failures and raw evidence are not deleted or reclassified by this curated branch.

Validation performed for this snapshot:

- All retained JSON files parse successfully.
- The Qwen/HBFSim unit test suite passes 11 tests.
- The curated integration tree contains no trace, capture, cache, build, or log artifacts.
