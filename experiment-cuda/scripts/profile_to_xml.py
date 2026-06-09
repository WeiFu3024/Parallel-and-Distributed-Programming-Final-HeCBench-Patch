#!/usr/bin/env python3
"""
profile_to_xml.py — Convert Nsight Compute CSV output to structured XML.

Usage:
  # Generate feedback XML (baseline + your kernel), all kernels shown:
  python profile_to_xml.py feedback \
      --baseline baseline/nsight_raw.csv \
      --yours    cell_c/round_1/nsight_raw.csv \
      --round 1 \
      --output   cell_c/round_2/feedback.xml

  # Generate a full profile XML for all kernels in a single CSV:
  python profile_to_xml.py single \
      --input  baseline/nsight_raw.csv \
      --output baseline/nsight.xml

  # List all metric names found in a CSV (for debugging):
  python profile_to_xml.py discover \
      --input baseline/nsight_raw.csv

ncu command to produce the CSV:
  ncu --set full --csv --log-file nsight_raw.csv ./binary [args]
"""

import argparse
import csv
import sys
import io
from collections import defaultdict

# ============================================================
# Metric name mapping: our XML field → ncu metric name
#
# If your ncu version uses different names, update here.
# Run `python profile_to_xml.py discover --input file.csv`
# to see all available metric names.
# ============================================================

METRIC_MAP = {
    # Execution — ncu --set full --csv uses human-readable names, not internal counter names
    "kernel_time_ns":          "Duration",           # unit: nsecond
    "grid_size":               "Grid Size",
    "block_size":              "Block Size",

    # Occupancy
    "achieved_occupancy_pct":  "Achieved Occupancy",    # unit: %
    "theoretical_occupancy_pct": "Theoretical Occupancy",  # unit: %

    # Memory throughput — ncu CSV exposes combined "Memory Throughput" in byte/second
    # (appears twice in CSV: once as %, once as byte/s; the byte/s row overwrites)
    "global_load_throughput":  "Memory Throughput",   # byte/second (combined r+w proxy)
    "global_store_throughput": "__not_available__",   # no separate store metric in CSV
    "shared_mem_throughput":   "__not_available__",   # no shared-mem throughput in CSV

    # Hit rates
    "l1_hit_rate_pct":         "L1/TEX Hit Rate",     # unit: %
    "l2_hit_rate_pct":         "L2 Hit Rate",         # unit: %

    # Utilization
    "dram_utilization_pct":    "DRAM Throughput",     # unit: %
    "sm_utilization_pct":      "Compute (SM) Throughput",  # unit: %
    "issue_slots_busy_pct":    "Issue Slots Busy",    # % of cycles with ≥1 instruction issued
    # Warp execution efficiency is computed, not a direct CSV column (see compute_warp_exec_efficiency)

    # Bottleneck indicators used in heuristic stall analysis
    "warp_cycles_per_inst":    "Warp Cycles Per Issued Instruction",  # high → stalling
    "branch_efficiency_pct":   "Branch Efficiency",                   # low  → divergence
    "eligible_warps":          "Eligible Warps Per Scheduler",        # low  → latency not hidden
}

# ncu --set full --csv does not export per-reason stall percentages as individual rows.
# Stall analysis is approximated from available bottleneck indicators (see get_stall_indicators).
STALL_PREFIX = "__no_stall_metrics__"
STALL_SUFFIX = ".pct"


def parse_ncu_csv(filepath):
    """
    Parse ncu --set full --csv output.
    Returns: {kernel_name: {metric_name: (value_str, unit_str)}}

    Handles the standard ncu CSV format with columns:
    ID, Process ID, Process Name, Host Name, Kernel Name,
    Context, Stream, Section Name, Metric Name, Metric Unit, Metric Value
    """
    kernels = defaultdict(dict)

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if '"ID"' in line and '"Kernel Name"' in line and '"Metric Name"' in line:
            header_idx = i
            break
        if "ID," in line and "Kernel Name" in line and "Metric Name" in line:
            header_idx = i
            break

    if header_idx is None:
        print(f"Error: Could not find CSV header in {filepath}", file=sys.stderr)
        print("Expected columns: ID, Kernel Name, Metric Name, Metric Unit, Metric Value",
              file=sys.stderr)
        sys.exit(1)

    csv_text = "".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_text))

    for row in reader:
        kernel = row.get("Kernel Name", "").strip().strip('"')
        metric = row.get("Metric Name", "").strip().strip('"')
        value = row.get("Metric Value", "").strip().strip('"')
        unit = row.get("Metric Unit", "").strip().strip('"')

        if kernel and metric:
            kernels[kernel][metric] = (value, unit)

    return dict(kernels)


def sort_kernels_by_time(kernels_dict):
    """Return list of (name, metrics) sorted by Duration descending (slowest first)."""
    time_key = METRIC_MAP["kernel_time_ns"]

    def duration(item):
        _, metrics = item
        if time_key in metrics:
            try:
                return float(metrics[time_key][0].replace(",", ""))
            except (ValueError, TypeError):
                pass
        return 0.0

    return sorted(kernels_dict.items(), key=duration, reverse=True)


def get_metric(metrics, our_key, default="N/A"):
    """Get a metric value by our mapping key."""
    ncu_name = METRIC_MAP.get(our_key)
    if ncu_name and ncu_name in metrics:
        return metrics[ncu_name][0]
    return default


def compute_warp_exec_efficiency(metrics):
    """
    Warp execution efficiency = avg non-predicated-off threads per warp / 32.
    ncu does not expose this directly in --set full --csv; we derive it from
    'Avg. Not Predicated Off Threads Per Warp'.
    """
    key = "Avg. Not Predicated Off Threads Per Warp"
    if key in metrics:
        try:
            val = float(metrics[key][0].replace(",", ""))
            return f"{val / 32 * 100:.1f}%"
        except (ValueError, TypeError):
            pass
    return "N/A"


def get_stall_indicators(metrics):
    """
    Derive bottleneck indicators from available --set full metrics.
    Returns a list of (label, value_str) sorted by severity (worst first).

    ncu --set full --csv does not include per-reason warp-stall breakdown.
    We use three proxy metrics as heuristic bottleneck signals:
      - Warp Cycles Per Issued Instruction  (>20 → likely memory-bound)
      - Branch Efficiency                   (<90% → significant divergence)
      - Eligible Warps Per Scheduler        (<1.0 → insufficient latency hiding)
    """
    indicators = []

    cpi_raw = get_metric(metrics, "warp_cycles_per_inst")
    try:
        cpi = float(cpi_raw.replace(",", ""))
        indicators.append(("warp_cycles_per_issued_inst", cpi_raw + " cycles",
                           cpi))  # sort key: higher = worse
    except (ValueError, TypeError):
        pass

    be_raw = get_metric(metrics, "branch_efficiency_pct")
    try:
        be = float(be_raw.replace(",", ""))
        indicators.append(("branch_efficiency", f"{be:.1f}%",
                           100 - be))  # sort key: lower efficiency = worse
    except (ValueError, TypeError):
        pass

    ew_raw = get_metric(metrics, "eligible_warps")
    try:
        ew = float(ew_raw.replace(",", ""))
        indicators.append(("eligible_warps_per_scheduler", ew_raw,
                           max(0, 4 - ew)))  # sort key: fewer warps = worse
    except (ValueError, TypeError):
        pass

    indicators.sort(key=lambda x: x[2], reverse=True)
    return [(label, val) for label, val, _ in indicators]


def get_stall_top3(metrics):
    """Extract top 3 warp stall reasons by percentage."""
    stalls = []
    for metric_name, (value, unit) in metrics.items():
        if metric_name.startswith(STALL_PREFIX) and metric_name.endswith(STALL_SUFFIX):
            reason = metric_name[len(STALL_PREFIX):-len(STALL_SUFFIX)]
            try:
                pct = float(value.replace(",", ""))
                stalls.append((reason, pct))
            except ValueError:
                pass

    stalls.sort(key=lambda x: x[1], reverse=True)
    return stalls[:3]


def ns_to_ms(value_str):
    """Convert nanoseconds string to milliseconds string."""
    try:
        ns = float(value_str.replace(",", ""))
        return f"{ns / 1_000_000:.4f}"
    except (ValueError, TypeError):
        return value_str


def bytes_per_sec_to_gb(value_str):
    """Convert bytes/sec to GB/s."""
    try:
        bps = float(value_str.replace(",", ""))
        return f"{bps / 1e9:.2f}"
    except (ValueError, TypeError):
        return value_str


def occupancy_limiting_factor(metrics):
    """
    Heuristic to determine occupancy limiting factor.
    Check if achieved << theoretical, then look at register/shared mem usage.
    """
    reg_key = "Registers Per Thread"
    smem_key = "Dynamic Shared Memory Per Block"

    if reg_key in metrics:
        try:
            regs = int(float(metrics[reg_key][0].replace(",", "")))
            if regs > 64:
                return "registers"
        except (ValueError, TypeError):
            pass

    if smem_key in metrics:
        try:
            smem = float(metrics[smem_key][0].replace(",", ""))
            if smem > 32768:
                return "shared_memory"
        except (ValueError, TypeError):
            pass

    return "block_size"


def format_kernel_xml(kernel_name, metrics, indent="  ", rank=None):
    """Format a single kernel's profile as an XML <Kernel> block."""
    i = indent
    i2 = indent + "  "
    i3 = indent + "    "

    stalls = get_stall_top3(metrics)
    stall_indicators = get_stall_indicators(metrics)
    limiter = occupancy_limiting_factor(metrics)

    time_ms = ns_to_ms(get_metric(metrics, "kernel_time_ns", "0"))
    gl_load = bytes_per_sec_to_gb(get_metric(metrics, "global_load_throughput", "0"))
    gl_store = bytes_per_sec_to_gb(get_metric(metrics, "global_store_throughput", "0"))
    sh_thru = bytes_per_sec_to_gb(get_metric(metrics, "shared_mem_throughput", "0"))

    rank_attr = f' rank="{rank}"' if rank is not None else ""

    xml = []
    xml.append(f"{i}<Kernel{rank_attr}>")

    xml.append(f"{i2}<Execution>")
    xml.append(f"{i3}<Kernel_Name>{kernel_name}</Kernel_Name>")
    xml.append(f"{i3}<Kernel_Time_ms>{time_ms}</Kernel_Time_ms>")
    xml.append(f"{i3}<Grid_Size>{get_metric(metrics, 'grid_size')}</Grid_Size>")
    xml.append(f"{i3}<Block_Size>{get_metric(metrics, 'block_size')}</Block_Size>")
    xml.append(f"{i2}</Execution>")

    xml.append(f"{i2}<Occupancy>")
    xml.append(f"{i3}<Achieved>{get_metric(metrics, 'achieved_occupancy_pct')}%</Achieved>")
    xml.append(f"{i3}<Theoretical_Max>{get_metric(metrics, 'theoretical_occupancy_pct')}%</Theoretical_Max>")
    xml.append(f"{i3}<Limiting_Factor>{limiter}</Limiting_Factor>")
    xml.append(f"{i2}</Occupancy>")

    xml.append(f"{i2}<Memory>")
    xml.append(f"{i3}<Global_Load_Throughput>{gl_load} GB/s</Global_Load_Throughput>")
    xml.append(f"{i3}<Global_Store_Throughput>{gl_store} GB/s</Global_Store_Throughput>")
    xml.append(f"{i3}<Shared_Memory_Throughput>{sh_thru} GB/s</Shared_Memory_Throughput>")
    xml.append(f"{i3}<L1_Hit_Rate>{get_metric(metrics, 'l1_hit_rate_pct')}%</L1_Hit_Rate>")
    xml.append(f"{i3}<L2_Hit_Rate>{get_metric(metrics, 'l2_hit_rate_pct')}%</L2_Hit_Rate>")
    xml.append(f"{i3}<DRAM_Utilization>{get_metric(metrics, 'dram_utilization_pct')}%</DRAM_Utilization>")
    xml.append(f"{i2}</Memory>")

    xml.append(f"{i2}<Compute>")
    xml.append(f"{i3}<SM_Utilization>{get_metric(metrics, 'sm_utilization_pct')}%</SM_Utilization>")
    xml.append(f"{i3}<Issue_Slots_Busy>{get_metric(metrics, 'issue_slots_busy_pct')}%</Issue_Slots_Busy>")
    xml.append(f"{i3}<Warp_Execution_Efficiency>{compute_warp_exec_efficiency(metrics)}</Warp_Execution_Efficiency>")
    xml.append(f"{i2}</Compute>")

    xml.append(f"{i2}<Stall_Analysis>")
    xml.append(f"{i3}<Top_Stalls>")
    if stalls:
        # Per-reason stall data (only available when explicit ncu --metrics are added)
        for reason, pct in stalls:
            xml.append(f'{i3}  <Stall reason="{reason}" percentage="{pct:.1f}%"/>')
    elif stall_indicators:
        # Bottleneck indicators derived from --set full metrics (heuristic proxies)
        for label, val in stall_indicators:
            xml.append(f'{i3}  <Bottleneck name="{label}" value="{val}"/>')
    else:
        xml.append(f'{i3}  <Stall reason="unknown" percentage="N/A"/>')
    xml.append(f"{i3}</Top_Stalls>")
    xml.append(f"{i2}</Stall_Analysis>")

    xml.append(f"{i}</Kernel>")
    return "\n".join(xml)


def format_all_kernels_xml(kernels_dict, wrapper_tag, indent="  "):
    """Format all kernels from a CSV as XML blocks inside wrapper_tag, sorted by time."""
    sorted_kernels = sort_kernels_by_time(kernels_dict)
    lines = [f"{indent}<{wrapper_tag}>"]
    for rank, (name, metrics) in enumerate(sorted_kernels, start=1):
        lines.append("")
        lines.append(format_kernel_xml(name, metrics, indent=indent + "  ", rank=rank))
    lines.append("")
    lines.append(f"{indent}</{wrapper_tag}>")
    return "\n".join(lines)


def cmd_feedback(args):
    """Generate Iteration_Feedback XML from baseline + your kernel CSV (all kernels)."""
    baseline_kernels = parse_ncu_csv(args.baseline)
    yours_kernels = parse_ncu_csv(args.yours)

    xml = [f'<Iteration_Feedback round="{args.round}">']
    xml.append("")
    xml.append(format_all_kernels_xml(baseline_kernels, "Baseline_Profile"))
    xml.append("")
    xml.append(format_all_kernels_xml(yours_kernels, "Your_Profile"))
    xml.append("")
    xml.append("</Iteration_Feedback>")

    output = "\n".join(xml)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Wrote feedback XML to {args.output}")
    else:
        print(output)


def cmd_single(args):
    """Generate profile XML for all kernels in a CSV, sorted by execution time."""
    kernels = parse_ncu_csv(args.input)

    xml = ["<Nsight_Profile>"]
    xml.append("")
    xml.append(format_all_kernels_xml(kernels, "Kernel_Profile", indent=""))
    xml.append("")
    xml.append("</Nsight_Profile>")

    output = "\n".join(xml)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Wrote profile XML to {args.output}")
    else:
        print(output)


def cmd_discover(args):
    """List all metric names found in a CSV file."""
    kernels = parse_ncu_csv(args.input)

    for kernel_name, metrics in kernels.items():
        print(f"\n=== Kernel: {kernel_name} ({len(metrics)} metrics) ===\n")
        for metric_name in sorted(metrics.keys()):
            value, unit = metrics[metric_name]
            print(f"  {metric_name:70s}  {value:>15s}  {unit}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Nsight Compute CSV to structured XML (all kernels)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # feedback subcommand
    p_fb = subparsers.add_parser("feedback",
        help="Generate Iteration_Feedback XML from baseline + your kernel (all kernels)")
    p_fb.add_argument("--baseline", required=True, help="Path to baseline ncu CSV")
    p_fb.add_argument("--yours", required=True, help="Path to your kernel's ncu CSV")
    p_fb.add_argument("--round", type=int, required=True, help="Iteration round number")
    p_fb.add_argument("--output", "-o", help="Output XML path (default: stdout)")

    # single subcommand
    p_si = subparsers.add_parser("single",
        help="Generate profile XML for all kernels in a CSV, sorted by execution time")
    p_si.add_argument("--input", required=True, help="Path to ncu CSV")
    p_si.add_argument("--output", "-o", help="Output XML path (default: stdout)")

    # discover subcommand
    p_di = subparsers.add_parser("discover",
        help="List all metric names in a CSV (for debugging)")
    p_di.add_argument("--input", required=True, help="Path to ncu CSV")

    args = parser.parse_args()

    if args.command == "feedback":
        cmd_feedback(args)
    elif args.command == "single":
        cmd_single(args)
    elif args.command == "discover":
        cmd_discover(args)


if __name__ == "__main__":
    main()
