#!/usr/bin/env bash
# Copy the sealed bandwidth evidence under the sealed cache archive; no overwrite.
set -euo pipefail
readonly SOURCE=/home/xmu/nvidiagds/simulators/wgslogs/qwen-ncu/runs/qwen-p32d2-native-bandwidth-solid-r1
readonly DESTINATION=/home/xmu/nvidiagds/simulators/wgslogs/qwen-ncu/runs/qwen-p32d2-native-cache-solid-r1/bandwidth-solid-r1
[[ -n "${root_sudo:-}" ]] || { echo 'root_sudo is unavailable' >&2; exit 2; }
[[ -f "$SOURCE/sha256.txt" && -f "$SOURCE/summary.json" ]] || { echo 'source evidence is incomplete' >&2; exit 2; }
[[ ! -e "$DESTINATION" ]] || { echo "refusing to overwrite: $DESTINATION" >&2; exit 2; }
printf '%s\n' "$root_sudo" | sudo -S -k -p '' mkdir "$DESTINATION"
{
    printf '%s\n' "$root_sudo"
    (cd "$SOURCE" && find . -type f -print0 | tar --null --files-from=- -cf -)
} | sudo -S -k -p '' tar -C "$DESTINATION" -xpf -
printf '%s\n' "$root_sudo" | sudo -S -k -p '' bash -c "cd '$DESTINATION' && sha256sum -c sha256.txt > copied-sha256-verification.txt"
