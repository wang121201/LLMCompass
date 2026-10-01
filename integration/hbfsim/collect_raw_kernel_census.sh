#!/usr/bin/env bash
# Collect reproducible hardware DRAM bandwidth evidence for Qwen2.5-1.5B P32/D2.
# NCU contributes physical DRAM bytes in counter-only app-range runs. Unprofiled
# CUDA-event durations contribute the denominator. The two are never mixed with
# NCU replay duration. All new source/output starts below LLMCompass/integration.
set -euo pipefail

readonly PACKAGE=/home/xmu/nvidiagds/simulators/hyfiss/analysis/write-gap-fix-20260920-r1/kernel-complete-input-r1/p32d2-smoke-r1/ncu-qwen/package
readonly OUTPUT=/home/xmu/nvidiagds/simulators/LLMCompass/integration/hbfsim/results/qwen-p32d2-solid-bandwidth-r1
readonly ARCHIVE=/home/xmu/nvidiagds/simulators/wgslogs/qwen-ncu/runs/qwen-p32d2-native-bandwidth-solid-r1
readonly NCU=/usr/local/cuda-12.8/bin/ncu
readonly PYTHON=/home/xmu/sgl/bin/python
readonly GPU_UUID=GPU-18ace299-5348-e6e4-d48c-1ee5a602859b
readonly COUNTER_REPEATS=5
readonly TIMING_REPEATS=20
readonly DRAM_METRICS=dram__bytes_read.sum,dram__bytes_write.sum
readonly HERE=/home/xmu/nvidiagds/simulators/LLMCompass/integration/hbfsim

if [[ -z "${root_sudo:-}" ]]; then
    echo 'root_sudo is unavailable; source from the authorized interactive shell' >&2
    exit 2
fi
if [[ -e "$OUTPUT" || -e "$ARCHIVE" ]]; then
    echo "refusing to overwrite existing output or archive: $OUTPUT ; $ARCHIVE" >&2
    exit 2
fi

mkdir -p "$OUTPUT/cache/triton" "$OUTPUT/cache/cuda" "$OUTPUT/cache/xdg" \
         "$OUTPUT/cache/flashinfer" "$OUTPUT/cache/tmp" "$OUTPUT/counters" "$OUTPUT/timing"
printf '%s\n' \
  'schema=LLMCOMPASS_QWEN_P32D2_SOLID_BANDWIDTH_V1' \
  'definition=NCU DRAM bytes divided by unprofiled natural CUDA-event ROI duration; never divide by NCU replay duration' \
  'contract=Qwen2.5-1.5B-Instruct; BF16; TP1; P32/D2; FlashInfer; eager; no CUDA graph; no torch.compile; no overlap; no radix cache' \
  "package_manifest_sha256=$(sha256sum "$PACKAGE/manifest.json" | awk '{print $1}')" \
  "gpu_uuid=$GPU_UUID" \
  "ncu_version=$($NCU --version | tail -1)" \
  "counter_repeats=$COUNTER_REPEATS" \
  "timing_repeats=$TIMING_REPEATS" \
  "counter_metrics=$DRAM_METRICS" \
  'counter_replay_mode=app-range' \
  'counter_cache_control=none' \
  'counter_clock_control=none' \
  > "$OUTPUT/identity.txt"

gpu_snapshot() {
    nvidia-smi --query-gpu=index,uuid,name,pstate,temperature.gpu,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw \
        --format=csv,noheader,nounits > "$1"
}

for roi in full Prefill D1 D2; do
    for repeat in $(seq 1 "$COUNTER_REPEATS"); do
        destination="$OUTPUT/counters/$roi/repeat-$repeat"
        mkdir -p "$destination"
        gpu_snapshot "$destination/gpu-before.csv"
        printf '%s\n' "$root_sudo" | sudo -S -k -p '' env \
            CUDA_VISIBLE_DEVICES=0 NVIDIA_VISIBLE_DEVICES="$GPU_UUID" \
            CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin \
            TRITON_CACHE_DIR="$OUTPUT/cache/triton" CUDA_CACHE_PATH="$OUTPUT/cache/cuda" \
            XDG_CACHE_HOME="$OUTPUT/cache/xdg" FLASHINFER_WORKSPACE_BASE="$OUTPUT/cache/flashinfer" \
            TMPDIR="$OUTPUT/cache/tmp" \
            "$NCU" --config-file off --rename-kernels off --disable-extra-suffixes \
            --target-processes application-only --replay-mode app-range --cache-control none --clock-control none \
            --metrics "$DRAM_METRICS" --export "$destination/capture" \
            "$PYTHON" -B "$PACKAGE/native_host.py" --model qwen25_1p5b --prefill-length 32 \
            --decode-steps 2 --roi "$roi" --mode capture --output "$destination/host" \
            > "$destination/ncu.stdout.log" 2> "$destination/ncu.stderr.log"
        "$NCU" --import "$destination/capture.ncu-rep" --page raw --csv --print-units base > "$destination/raw.csv"
        gpu_snapshot "$destination/gpu-after.csv"
    done
done

# Validate mode does no counter profiling. One run supplies all three phases,
# which makes Full the sum of those phase events within the same execution.
for repeat in $(seq 1 "$TIMING_REPEATS"); do
    destination="$OUTPUT/timing/repeat-$repeat"
    mkdir -p "$destination"
    gpu_snapshot "$destination/gpu-before.csv"
    printf '%s\n' "$root_sudo" | sudo -S -k -p '' env \
        CUDA_VISIBLE_DEVICES=0 NVIDIA_VISIBLE_DEVICES="$GPU_UUID" \
        CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin \
        TRITON_CACHE_DIR="$OUTPUT/cache/triton" CUDA_CACHE_PATH="$OUTPUT/cache/cuda" \
        XDG_CACHE_HOME="$OUTPUT/cache/xdg" FLASHINFER_WORKSPACE_BASE="$OUTPUT/cache/flashinfer" \
        TMPDIR="$OUTPUT/cache/tmp" \
        "$PYTHON" -B "$PACKAGE/native_host.py" --model qwen25_1p5b --prefill-length 32 \
        --decode-steps 2 --roi full --mode validate --output "$destination/host" \
        > "$destination/validate.stdout.log" 2> "$destination/validate.stderr.log"
    gpu_snapshot "$destination/gpu-after.csv"
done

"$HERE/summarize_solid_bandwidth.py" "$OUTPUT" > "$OUTPUT/summary.json"
(
    cd "$OUTPUT"
    find . -path './cache' -prune -o -type f ! -name sha256.txt -print0 | sort -z | xargs -0 sha256sum > sha256.txt
)
mkdir "$ARCHIVE"
(
    cd "$OUTPUT"
    find . -path './cache' -prune -o -type f -print0 | while IFS= read -r -d '' path; do
        mkdir -p "$ARCHIVE/$(dirname "$path")"
        cp -a "$path" "$ARCHIVE/$path"
    done
)
(
    cd "$ARCHIVE"
    sha256sum -c sha256.txt
) > "$ARCHIVE/archive-verification.txt"
