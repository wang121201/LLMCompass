#!/usr/bin/env python3
"""Summarize five-repeat raw NCU cache counters without inventing L2 rates."""
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rois = ("full", "Prefill", "D1", "D2")
metrics = (
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum",
    "lts__t_sectors_aperture_device_op_read.sum",
    "lts__t_sectors_aperture_device_op_read_lookup_hit.sum",
    "lts__t_sectors_aperture_device_op_read_lookup_miss.sum",
    "lts__t_sectors_aperture_device_op_write.sum",
    "lts__t_sectors_aperture_device_op_write_lookup_hit.sum",
    "lts__t_sectors_aperture_device_op_write_lookup_miss.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
)


def stats(values):
    values = sorted(values)
    mean = statistics.fmean(values)
    return {
        "samples": values,
        "count": len(values),
        "median": statistics.median(values),
        "p10": statistics.quantiles(values, n=10, method="inclusive")[0],
        "p90": statistics.quantiles(values, n=10, method="inclusive")[8],
        "coefficient_of_variation_pct": statistics.stdev(values) / mean * 100,
    }


result = {
    "schema": "LLMCOMPASS_QWEN_P32D2_SOLID_CACHE_COUNTERS_V1",
    "metric_semantics": {
        "sector": "32-byte sector on this Ada GPU",
        "l1": "L1 totals, lookup hits, lookup misses are raw NCU counters; total=hit+miss is verified per action",
        "l2": "L2 total, lookup hits, lookup misses are raw NCU counters with different aggregation semantics; no L2 hit rate is calculated",
        "dram_throughput": "NCU range-profile pct_of_peak_sustained_elapsed; diagnostic only, not absolute native GB/s",
    },
    "rois": {},
}
for roi in rois:
    values = {metric: [] for metric in metrics}
    actions = []
    for path in sorted((root / roi).glob("repeat-*/raw.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 2 or rows[1]["Kernel Name"] != "range":
            raise SystemExit(f"{path}: expected unit row plus one app-range action")
        row = rows[1]
        parsed = {}
        for metric in metrics:
            raw = row[metric].replace(",", "")
            parsed[metric] = float(raw) if metric.startswith("dram__throughput") else int(raw)
            values[metric].append(parsed[metric])
        if parsed[metrics[0]] != parsed[metrics[1]] + parsed[metrics[2]]:
            raise SystemExit(f"{path}: L1 load sector identity failed")
        if parsed[metrics[3]] != parsed[metrics[4]] + parsed[metrics[5]]:
            raise SystemExit(f"{path}: L1 store sector identity failed")
        actions.append({"path": str(path.relative_to(root)), "replay_passes": int(row["profiler__replayer_passes"])})
    if len(actions) != 5:
        raise SystemExit(f"{roi}: expected five raw reports")
    replay = {action["replay_passes"] for action in actions}
    if len(replay) != 1:
        raise SystemExit(f"{roi}: replay pass count changed across repeats: {sorted(replay)}")
    result["rois"][roi] = {
        "actions": actions,
        "replay_passes": replay.pop(),
        "metrics": {metric: stats(values[metric]) for metric in metrics},
    }
json.dump(result, sys.stdout, indent=2, sort_keys=True)
print()
