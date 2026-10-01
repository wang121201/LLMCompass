#!/usr/bin/env bash
# Five-repeat, four-ROI NCU cache-counter evidence for Qwen2.5-1.5B P32/D2.
set -euo pipefail

readonly PACKAGE=/home/xmu/nvidiagds/simulators/hyfiss/analysis/write-gap-fix-20260920-r1/kernel-complete-input-r1/p32d2-smoke-r1/ncu-qwen/package
readonly OUTPUT=/home/xmu/nvidiagds/simulators/LLMCompass/integration/hbfsim/results/qwen-p32d2-solid-cache-counters-r1
readonly ARCHIVE=/home/xmu/nvidiagds/simulators/wgslogs/qwen-ncu/runs/qwen-p32d2-native-cache-solid-r1
readonly NCU=/usr/local/cuda-12.8/bin/ncu
readonly PYTHON=/home/xmu/sgl/bin/python
readonly GPU_UUID=GPU-18ace299-5348-e6e4-d48c-1ee5a602859b
readonly REPEATS=5
readonly HERE=/home/xmu/nvidiagds/simulators/LLMCompass/integration/hbfsim
readonly METRICS=l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum,lts__t_sectors_aperture_device_op_read.sum,lts__t_sectors_aperture_device_op_read_lookup_hit.sum,lts__t_sectors_aperture_device_op_read_lookup_miss.sum,lts__t_sectors_aperture_device_op_write.sum,lts__t_sectors_aperture_device_op_write_lookup_hit.sum,lts__t_sectors_aperture_device_op_write_lookup_miss.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed

if [[ -z "${root_sudo:-}" ]]; then
    echo 'root_sudo is unavailable; source from the authorized interactive shell' >&2
    exit 2
fi
if [[ -e "$OUTPUT" || -e "$ARCHIVE" ]]; then
    echo "refusing to overwrite existing output or archive: $OUTPUT ; $ARCHIVE" >&2
    exit 2
fi

mkdir -p "$OUTPUT/cache/triton" "$OUTPUT/cache/cuda" "$OUTPUT/cache/xdg" \
         "$OUTPUT/cache/flashinfer" "$OUTPUT/cache/tmp"
printf '%s\n' \
  'schema=LLMCOMPASS_QWEN_P32D2_SOLID_CACHE_COUNTERS_V1' \
  'contract=Qwen2.5-1.5B-Instruct; BF16; TP1; P32/D2; FlashInfer; eager; no CUDA graph; no torch.compile; no overlap; no radix cache' \
  'semantics=L1 sectors/hits/misses are raw NCU counters; L2 total/hit/miss are retained raw and are not used to calculate a hit rate' \
  'dram_throughput_semantics=NCU profiled pct_of_peak_sustained_elapsed only; not an absolute native workload GB/s' \
  "package_manifest_sha256=$(sha256sum "$PACKAGE/manifest.json" | awk '{print $1}')" \
  "gpu_uuid=$GPU_UUID" "ncu_version=$($NCU --version | tail -1)" \
  "repeats=$REPEATS" "metrics=$METRICS" > "$OUTPUT/identity.txt"

gpu_snapshot() {
    nvidia-smi --query-gpu=index,uuid,name,pstate,temperature.gpu,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw \
        --format=csv,noheader,nounits > "$1"
}

for roi in full Prefill D1 D2; do
    for repeat in $(seq 1 "$REPEATS"); do
        destination="$OUTPUT/$roi/repeat-$repeat"
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
            --metrics "$METRICS" --export "$destination/capture" \
            "$PYTHON" -B "$PACKAGE/native_host.py" --model qwen25_1p5b --prefill-length 32 \
            --decode-steps 2 --roi "$roi" --mode capture --output "$destination/host" \
            > "$destination/ncu.stdout.log" 2> "$destination/ncu.stderr.log"
        "$NCU" --import "$destination/capture.ncu-rep" --page raw --csv --print-units base > "$destination/raw.csv"
        gpu_snapshot "$destination/gpu-after.csv"
    done
done

"$HERE/summarize_solid_cache_counters.py" "$OUTPUT" > "$OUTPUT/summary.json"
(
    cd "$OUTPUT"
    find . -path './cache' -prune -o -type f ! -name sha256.txt -print0 | sort -z | xargs -0 sha256sum > sha256.txt
)
printf '%s\n' "$root_sudo" | sudo -S -k -p '' mkdir "$ARCHIVE"
{
    printf '%s\n' "$root_sudo"
    (
        cd "$OUTPUT"
        find . -path './cache' -prune -o -type f -print0 | tar --null --files-from=- -cf -
    )
} | sudo -S -k -p '' tar -C "$ARCHIVE" -xpf -
printf '%s\n' "$root_sudo" | sudo -S -k -p '' bash -c "cd '$ARCHIVE' && sha256sum -c sha256.txt > archive-verification.txt"
