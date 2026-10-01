# Curated integration snapshot

This directory contains the small, reviewable implementation and acceptance evidence selected for branch `codex/integration-key-acceptance-20261001`.

## Scope

- `qwen_hbfsim_cosim.py`, tests, summarizers, collection scripts, model description, candidate RTX 4000 Ada configuration, and the current `README.md` are retained as the implementation/design surface.
- The retained result files are summaries, traffic receipts, manifests, comparisons, and identity records for the current Qwen2.5-1.5B BF16 TP1 P32D2 acceptance checkpoint. P32D2 means 32-token prompt prefill followed by two one-token decode steps. TP1 means one-device tensor parallelism.
- `DRAM` means real-device Dynamic Random Access Memory traffic reported by NVIDIA Nsight Compute (NCU). `HBM proxy` means HBFSim generic High Bandwidth Memory completion traffic under the GDDR-inspired overlay; it is not native RTX 4000 Ada GDDR6 traffic.

## Retained acceptance evidence

- `qwen-p32d2-rtx4000ada-gddr-abstract-aligned-full-r1`: model `summary.json`, `traffic.json`, and `manifest.json`.
- `qwen-p32d2-rtx4000ada-gddr-abstract-aligned-hardware-diagnostic-r1`: model-versus-hardware timing diagnostic.
- `qwen-p32d2-rtx4000ada-gddr-abstract-aligned-ncu-compatibility-r1`: cache/DRAM compatibility receipt.
- `qwen-p32d2-rtx4000ada-gddr-abstract-aligned-ncu-bandwidth-proxy-r1`: bandwidth proxy receipt.
- `qwen-p32d2-hardware-gpu0-baseline-r2`: hardware baseline receipt.
- `qwen-p32d2-solid-bandwidth-r1` and `qwen-p32d2-solid-cache-counters-r1`: five-repeat counter summaries, timing identity, and workload identity.

## Explicit exclusions

The curated branch does not contain raw CSV/JSONL streams, `*.ncu-rep` captures, `operators.jsonl`, build directories, object files, binaries, CUDA cache, Triton/torchinductor cache, HTML wear reports, temporary files, or run logs. The original evidence remains copy-only on the XMU server at `/home/xmu/nvidiagds/simulators/LLMCompass/integration/hbfsim`; this branch is not a replacement for raw-data archival.
