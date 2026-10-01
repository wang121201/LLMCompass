#!/usr/bin/env python3
"""Summarize counter-only NCU bytes and unprofiled CUDA-event timings."""
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rois = ("full", "Prefill", "D1", "D2")


def distribution(values):
    values = sorted(values)
    median = statistics.median(values)
    return {
        "samples": values,
        "count": len(values),
        "median": median,
        "mean": statistics.fmean(values),
        "p10": statistics.quantiles(values, n=10, method="inclusive")[0],
        "p90": statistics.quantiles(values, n=10, method="inclusive")[8],
        "stdev": statistics.stdev(values),
        "coefficient_of_variation_pct": statistics.stdev(values) / statistics.fmean(values) * 100,
    }


counter = {roi: {"read_bytes": [], "write_bytes": [], "actions": []} for roi in rois}
for roi in rois:
    for raw_path in sorted((root / "counters" / roi).glob("repeat-*/raw.csv")):
        with raw_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 2:
            raise SystemExit(f"{raw_path}: expected unit row plus one app-range action, got {len(rows)} rows")
        row = rows[1]
        # In an app-range report, NCU stores the aggregate action label in the
        # display column rather than the launch metadata column.
        if row["Kernel Name"] != "range":
            raise SystemExit(f"{raw_path}: expected app-range action, got {row['Kernel Name']!r}")
        counter[roi]["read_bytes"].append(int(row["dram__bytes_read.sum"].replace(",", "")))
        counter[roi]["write_bytes"].append(int(row["dram__bytes_write.sum"].replace(",", "")))
        counter[roi]["actions"].append({
            "path": str(raw_path.relative_to(root)),
            "replay_passes": int(row["profiler__replayer_passes"]),
        })

timing = {roi: [] for roi in rois}
wall = {roi: [] for roi in rois}
for finish_path in sorted((root / "timing").glob("repeat-*/host/process-*/finish.json")):
    with finish_path.open() as handle:
        payload = json.load(handle)
    if payload["status"] != "PASS_NATIVE_WORKFLOW_AND_ROI" or payload["mode"] != "validate":
        raise SystemExit(f"{finish_path}: not a successful unprofiled native validation")
    event = payload["natural_cuda_event_ms"]
    rows = [row for row in payload["aligned_execution"]["phases"] if row["stage"] == "Measured"]
    by_phase = {row["phase"]: row for row in rows}
    for roi, phase in (("Prefill", "Prefill"), ("D1", "Decode1"), ("D2", "Decode2")):
        timing[roi].append(float(event[phase]))
        wall[roi].append(float(by_phase[phase]["instrumented_wall_seconds"]) * 1000)
    timing["full"].append(sum(float(event[phase]) for phase in ("Prefill", "Decode1", "Decode2")))
    wall["full"].append(sum(float(by_phase[phase]["instrumented_wall_seconds"]) * 1000
                              for phase in ("Prefill", "Decode1", "Decode2")))

summary = {
    "schema": "LLMCOMPASS_QWEN_P32D2_SOLID_BANDWIDTH_V1",
    "definition": {
        "read_gbps": "median NCU dram__bytes_read.sum / median unprofiled natural CUDA-event duration",
        "write_gbps": "median NCU dram__bytes_write.sum / median unprofiled natural CUDA-event duration",
        "total_gbps": "read_gbps + write_gbps",
        "units": {"bytes": "byte", "duration": "ms", "bandwidth": "decimal GB/s"},
        "not_used_as_denominator": "NCU gpu__time_duration.sum or any NCU replay elapsed time",
    },
    "rois": {},
}
for roi in rois:
    read = distribution(counter[roi]["read_bytes"])
    write = distribution(counter[roi]["write_bytes"])
    duration = distribution(timing[roi])
    wall_duration = distribution(wall[roi])
    if read["count"] != 5 or duration["count"] != 20:
        raise SystemExit(f"{roi}: expected five counter and twenty timing samples")
    if any(action["replay_passes"] != 1 for action in counter[roi]["actions"]):
        raise SystemExit(f"{roi}: counter replay pass count is not one")
    summary["rois"][roi] = {
        "counter_actions": counter[roi]["actions"],
        "dram_read_bytes": read,
        "dram_write_bytes": write,
        "natural_cuda_event_ms": duration,
        "synchronized_wall_ms_crosscheck": wall_duration,
        "read_gbps": read["median"] / duration["median"] / 1_000_000,
        "write_gbps": write["median"] / duration["median"] / 1_000_000,
        "total_gbps": (read["median"] + write["median"]) / duration["median"] / 1_000_000,
    }
json.dump(summary, sys.stdout, indent=2, sort_keys=True)
print()
