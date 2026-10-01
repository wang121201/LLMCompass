#!/usr/bin/env python3
"""Emit the one aggregate app-range action from each four-ROI NCU CSV."""
import csv
import pathlib

ROOT = pathlib.Path(
    "/home/xmu/nvidiagds/simulators/LLMCompass/integration/hbfsim/results/"
    "qwen-p32d2-ncu-four-roi-cache-dram-r1"
)
METRICS = [
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed", "gpu__time_duration.sum",
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
]

writer = csv.DictWriter(__import__("sys").stdout, fieldnames=["roi", *METRICS])
writer.writeheader()
for roi in ("full", "Prefill", "D1", "D2"):
    with (ROOT / roi / "raw.csv").open(newline="") as raw:
        rows = list(csv.DictReader(raw))
    if len(rows) != 2:
        raise SystemExit(f"{roi}: expected unit row and exactly one aggregate action, got {len(rows)} rows")
    writer.writerow({"roi": roi, **{metric: rows[1][metric] for metric in METRICS}})
