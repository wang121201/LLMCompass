#!/usr/bin/env bash
# Copy-only archival of an already sealed solid-bandwidth evidence package.
set -euo pipefail

readonly OUTPUT=/home/xmu/nvidiagds/simulators/LLMCompass/integration/hbfsim/results/qwen-p32d2-solid-bandwidth-r1
readonly ARCHIVE=/home/xmu/nvidiagds/simulators/wgslogs/qwen-ncu/runs/qwen-p32d2-native-bandwidth-solid-r1

if [[ -z "${root_sudo:-}" ]]; then
    echo 'root_sudo is unavailable; source from the authorized interactive shell' >&2
    exit 2
fi
if [[ ! -s "$OUTPUT/summary.json" || ! -f "$OUTPUT/sha256.txt" ]]; then
    echo 'sealed summary or SHA-256 manifest is missing' >&2
    exit 2
fi
if [[ -e "$ARCHIVE" ]]; then
    echo "refusing to overwrite existing archive: $ARCHIVE" >&2
    exit 2
fi

# Cache and JIT intermediates are deliberately excluded. This host has a
# zero-length sudo credential cache, so each privileged operation receives the
# authorized credential through stdin; the tar payload follows the first line.
printf '%s\n' "$root_sudo" | sudo -S -k -p '' mkdir "$ARCHIVE"
{
    printf '%s\n' "$root_sudo"
    (
        cd "$OUTPUT"
        find . -path './cache' -prune -o -type f -print0 | tar --null --files-from=- -cf -
    )
} | sudo -S -k -p '' tar -C "$ARCHIVE" -xpf -
printf '%s\n' "$root_sudo" | sudo -S -k -p '' bash -c "cd '$ARCHIVE' && sha256sum -c sha256.txt > archive-verification.txt"
