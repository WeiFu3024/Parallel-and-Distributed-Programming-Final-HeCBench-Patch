#!/usr/bin/env python3
"""
profile_to_xml.py — Convert Nsight Compute CSV output to Iteration_Feedback XML.

Usage:
  # Generate feedback XML (baseline + your kernel):
  python profile_to_xml.py feedback \
      --baseline baseline/nsight_raw.csv \
      --yours    cell_c/round_1/nsight_raw.csv \
      --round 1 \
      --output   cell_c/round_2/feedback.xml

  # Generate a single kernel profile XML (e.g., for baseline):
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
    "fp32_utilization_pct":    "__not_available__",   # no FP32-pipe metric in CSV
    "warp_exec_efficiency":    "__not_available__",   # no per-thread efficiency in CSV
}

# ncu --set full --csv does not export per-reason stall percentages as individual rows.
# Set to a prefix that will never match so get_stall_top3 returns an empty list.
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
        # Skip lines until we find the header row
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if '"ID"' in line and '"Kernel Name"' in line and '"Metric Name"' in line:
            header_idx = i
            break
        # Also try unquoted headers
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


def select_kernel(kernels_dict):
    """
    If multiple kernels were profiled, select the one with the longest duration.
    Returns: (kernel_name, metrics_dict)
    """
    if not kernels_dict:
        print("Error: No kernel data found in CSV.", file=sys.stderr)
        sys.exit(1)

    if len(kernels_dict) == 1:
        name = list(kernels_dict.keys())[0]
        return name, kernels_dict[name]

    # Pick kernel with longest Duration
    best_name, best_time = None, -1
    for name, metrics in kernels_dict.items():
        time_key = METRIC_MAP["kernel_time_ns"]
        if time_key in metrics:
            try:
                t = float(metrics[time_key][0].replace(",", ""))
                if t > best_time:
                    best_time = t
                    best_name = name
            except ValueError:
                pass

    if best_name is None:
        best_name = list(kernels_dict.keys())[0]

    return best_name, kernels_dict[best_name]


def find_kernel(kernels_dict, kernel_name=None):
    """
    Select a kernel from the profiled kernels dict.

    If kernel_name is given, return the first kernel whose full name contains
    kernel_name as a substring (case-sensitive).  Prints a warning and falls
    back to the longest-duration heuristic when no match is found.

    If kernel_name is None, delegates to select_kernel().

    Returns: (kernel_name_str, metrics_dict)
    """
    if not kernel_name:
        return select_kernel(kernels_dict)

    matches = [(name, metrics) for name, metrics in kernels_dict.items()
               if kernel_name in name]

    if matches:
        if len(matches) > 1:
            names = [m[0] for m in matches]
            print(f"Warning: '{kernel_name}' matches multiple kernels: {names}; "
                  "using the first match.", file=sys.stderr)
        return matches[0]

    available = list(kernels_dict.keys())
    print(f"Warning: kernel '{kernel_name}' not found in CSV. "
          f"Available kernels: {available}. "
          "Falling back to longest-duration heuristic.", file=sys.stderr)
    return select_kernel(kernels_dict)


def get_metric(metrics, our_key, default="N/A"):
    """Get a metric value by our mapping key."""
    ncu_name = METRIC_MAP.get(our_key)
    if ncu_name and ncu_name in metrics:
        return metrics[ncu_name][0]
    return default


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
    # This is a simplified heuristic; ncu's occupancy section has more detail
    # Users may want to customize this
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


def format_kernel_xml(tag_name, kernel_name, metrics, indent="  "):
    """Format a single kernel's profile as XML."""
    i = indent
    i2 = indent * 2
    i3 = indent * 3

    stalls = get_stall_top3(metrics)
    limiter = occupancy_limiting_factor(metrics)

    time_ms = ns_to_ms(get_metric(metrics, "kernel_time_ns", "0"))
    gl_load = bytes_per_sec_to_gb(get_metric(metrics, "global_load_throughput", "0"))
    gl_store = bytes_per_sec_to_gb(get_metric(metrics, "global_store_throughput", "0"))
    sh_thru = bytes_per_sec_to_gb(get_metric(metrics, "shared_mem_throughput", "0"))

    xml = []
    xml.append(f"{i}<{tag_name}>")

    # Execution
    xml.append(f"{i2}<Execution>")
    xml.append(f"{i3}<Kernel_Name>{kernel_name}</Kernel_Name>")
    xml.append(f"{i3}<Kernel_Time_ms>{time_ms}</Kernel_Time_ms>")
    xml.append(f"{i3}<Grid_Size>{get_metric(metrics, 'grid_size')}</Grid_Size>")
    xml.append(f"{i3}<Block_Size>{get_metric(metrics, 'block_size')}</Block_Size>")
    xml.append(f"{i2}</Execution>")

    # Occupancy
    xml.append(f"{i2}<Occupancy>")
    xml.append(f"{i3}<Achieved>{get_metric(metrics, 'achieved_occupancy_pct')}%</Achieved>")
    xml.append(f"{i3}<Theoretical_Max>{get_metric(metrics, 'theoretical_occupancy_pct')}%</Theoretical_Max>")
    xml.append(f"{i3}<Limiting_Factor>{limiter}</Limiting_Factor>")
    xml.append(f"{i2}</Occupancy>")

    # Memory
    xml.append(f"{i2}<Memory>")
    xml.append(f"{i3}<Global_Load_Throughput>{gl_load} GB/s</Global_Load_Throughput>")
    xml.append(f"{i3}<Global_Store_Throughput>{gl_store} GB/s</Global_Store_Throughput>")
    xml.append(f"{i3}<Shared_Memory_Throughput>{sh_thru} GB/s</Shared_Memory_Throughput>")
    xml.append(f"{i3}<L1_Hit_Rate>{get_metric(metrics, 'l1_hit_rate_pct')}%</L1_Hit_Rate>")
    xml.append(f"{i3}<L2_Hit_Rate>{get_metric(metrics, 'l2_hit_rate_pct')}%</L2_Hit_Rate>")
    xml.append(f"{i3}<DRAM_Utilization>{get_metric(metrics, 'dram_utilization_pct')}%</DRAM_Utilization>")
    xml.append(f"{i2}</Memory>")

    # Compute
    xml.append(f"{i2}<Compute>")
    xml.append(f"{i3}<SM_Utilization>{get_metric(metrics, 'sm_utilization_pct')}%</SM_Utilization>")
    xml.append(f"{i3}<FP32_Utilization>{get_metric(metrics, 'fp32_utilization_pct')}%</FP32_Utilization>")
    xml.append(f"{i3}<Warp_Execution_Efficiency>{get_metric(metrics, 'warp_exec_efficiency')}</Warp_Execution_Efficiency>")
    xml.append(f"{i2}</Compute>")

    # Stall Analysis
    xml.append(f"{i2}<Stall_Analysis>")
    xml.append(f"{i3}<Top_Stalls>")
    for reason, pct in stalls:
        xml.append(f'{i3}  <Stall reason="{reason}" percentage="{pct:.1f}%"/>')
    if not stalls:
        xml.append(f'{i3}  <Stall reason="unknown" percentage="N/A"/>')
    xml.append(f"{i3}</Top_Stalls>")
    xml.append(f"{i2}</Stall_Analysis>")

    xml.append(f"{i}</{tag_name}>")
    return "\n".join(xml)


def cmd_feedback(args):
    """Generate Iteration_Feedback XML from baseline + your kernel CSV."""
    baseline_kernels = parse_ncu_csv(args.baseline)
    yours_kernels = parse_ncu_csv(args.yours)

    base_name, base_metrics = find_kernel(baseline_kernels, args.kernel_name)
    your_name, your_metrics = find_kernel(yours_kernels, args.kernel_name)

    xml = []
    xml.append(f'<Iteration_Feedback round="{args.round}">')
    xml.append("")
    xml.append(format_kernel_xml("Baseline_Kernel", base_name, base_metrics))
    xml.append("")
    xml.append(format_kernel_xml("Your_Kernel", your_name, your_metrics))
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
    """Generate a single kernel profile XML."""
    kernels = parse_ncu_csv(args.input)
    name, metrics = find_kernel(kernels, args.kernel_name)

    xml = format_kernel_xml("Kernel_Profile", name, metrics, indent="  ")
    output = f"<Nsight_Profile>\n{xml}\n</Nsight_Profile>"

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
        description="Convert Nsight Compute CSV to Iteration_Feedback XML"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # feedback subcommand
    p_fb = subparsers.add_parser("feedback",
        help="Generate Iteration_Feedback XML from baseline + your kernel")
    p_fb.add_argument("--baseline", required=True, help="Path to baseline ncu CSV")
    p_fb.add_argument("--yours", required=True, help="Path to your kernel's ncu CSV")
    p_fb.add_argument("--round", type=int, required=True, help="Iteration round number")
    p_fb.add_argument("--output", "-o", help="Output XML path (default: stdout)")
    p_fb.add_argument("--kernel-name", "-k", default=None, metavar="NAME",
        help="Substring to match the target kernel name in both CSVs. "
             "Use this when a benchmark has multiple kernels so the correct one "
             "is selected from both --baseline and --yours. "
             "Falls back to longest-duration heuristic when omitted or no match found.")

    # single subcommand
    p_si = subparsers.add_parser("single",
        help="Generate single kernel profile XML")
    p_si.add_argument("--input", required=True, help="Path to ncu CSV")
    p_si.add_argument("--output", "-o", help="Output XML path (default: stdout)")
    p_si.add_argument("--kernel-name", "-k", default=None, metavar="NAME",
        help="Substring to match the target kernel name (default: longest-duration kernel).")

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
