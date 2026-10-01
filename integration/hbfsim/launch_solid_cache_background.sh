#!/usr/bin/env bash
# Start the cache campaign after an interactive shell has supplied root_sudo.
set -euo pipefail
readonly OUTPUT=/home/xmu/nvidiagds/simulators/LLMCompass/integration/hbfsim/results/qwen-p32d2-solid-cache-counters-r1
readonly COLLECTOR=/home/xmu/nvidiagds/simulators/LLMCompass/integration/hbfsim/collect_solid_cache_counters.sh
[[ -n "${root_sudo:-}" ]] || { echo 'root_sudo is unavailable' >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "output already exists: $OUTPUT" >&2; exit 2; }
export root_sudo
setsid bash -c "source '$COLLECTOR'" > "${OUTPUT}.launcher.log" 2>&1 < /dev/null &
echo "$!"
