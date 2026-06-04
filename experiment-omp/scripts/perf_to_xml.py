#!/usr/bin/env python3
"""
perf_to_xml.py — Convert `perf stat` output to OpenMP Iteration_Feedback XML.

Usage:
  # Generate feedback XML (baseline + your kernel):
  python perf_to_xml.py feedback \
      --baseline baseline/perf_raw.txt \
      --yours    cell_c/round_1/perf_raw.txt \
      --round 1 \
      --threads 16 \
      --peak-bw 50.0 \
      --output   cell_c/round_2/feedback.xml

  # Generate single kernel profile XML:
  python perf_to_xml.py single \
      --input  baseline/perf_raw.txt \
      --threads 16 \
      --peak-bw 50.0 \
      --output baseline/perf.xml

  # Discover all counters in a perf output file:
  python perf_to_xml.py discover --input perf_raw.txt

perf command to produce the input:
  perf stat -e task-clock,cycles,instructions,branches,branch-misses,\\
    L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,\\
    context-switches,cpu-migrations \\
    -o perf_raw.txt -- env OMP_NUM_THREADS=16 ./binary [args]

Notes:
  - L2 counters are architecture-specific. On Intel: l2_rqsts.references,
    l2_rqsts.miss. On AMD: check `perf list`. If unavailable, those fields
    will show N/A.
  - --peak-bw is peak memory bandwidth in GB/s (from hardware_spec.txt).
  - --threads is OMP_NUM_THREADS used during the run.
"""

import argparse
import re
import sys


# Cache line size in bytes (standard on x86)
CACHE_LINE_BYTES = 64


def parse_perf_output(filepath):
    """
    Parse `perf stat -o` output into a dict of counter_name → value.
    
    Handles formats like:
        12,345,678      instructions       #    1.88  insn per cycle
             0.456      task-clock (msec)
    """
    counters = {}
    wall_time_ms = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Extract wall time from "X seconds time elapsed"
            m = re.match(r"^\s*([\d,.]+)\s+seconds\s+time\s+elapsed", line)
            if m:
                wall_time_ms = float(m.group(1).replace(",", "")) * 1000
                continue

            # Extract counter: "12,345,678  counter-name  # ..."
            m = re.match(r"^\s*([\d,]+)\s+(\S+)", line)
            if m:
                value_str = m.group(1).replace(",", "")
                name = m.group(2)
                try:
                    counters[name] = int(value_str)
                except ValueError:
                    try:
                        counters[name] = float(value_str)
                    except ValueError:
                        pass
                continue

            # Extract task-clock (may have msec unit)
            m = re.match(r"^\s*([\d,.]+)\s+msec\s+task-clock", line)
            if m:
                counters["task-clock"] = float(m.group(1).replace(",", ""))
                continue

            # Alternative format: "12,345.67 msec task-clock"
            m = re.match(r"^\s*([\d,.]+)\s+task-clock", line)
            if m:
                counters["task-clock"] = float(m.group(1).replace(",", ""))

    # Derive wall time from task-clock if not found
    if wall_time_ms is None and "task-clock" in counters:
        # task-clock is in msec of CPU time; wall_time ≈ task-clock / thread_count
        # but we can't know thread count here, so store task-clock
        wall_time_ms = counters["task-clock"]

    counters["_wall_time_ms"] = wall_time_ms
    return counters


def safe_get(counters, key, default=None):
    return counters.get(key, default)


def safe_ratio(numerator, denominator, multiply=100, fmt=".1f"):
    if numerator is None or denominator is None or denominator == 0:
        return "N/A"
    return f"{(numerator / denominator) * multiply:{fmt}}"


def safe_fmt(value, fmt=",d"):
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:{fmt}}"
    return f"{value:.2f}"


def compute_metrics(counters, thread_count, peak_bw_gbs):
    """Compute derived metrics from raw perf counters."""
    m = {}

    # Wall time
    wall_ms = counters.get("_wall_time_ms")
    m["wall_time_ms"] = f"{wall_ms:.1f}" if wall_ms else "N/A"
    m["thread_count"] = str(thread_count)

    # Instruction profile
    instructions = safe_get(counters, "instructions")
    cycles = safe_get(counters, "cycles")
    branches = safe_get(counters, "branches")
    branch_misses = safe_get(counters, "branch-misses")

    m["instructions"] = safe_fmt(instructions)
    m["cycles"] = safe_fmt(cycles)

    if instructions and cycles and cycles > 0:
        m["ipc"] = f"{instructions / cycles:.2f}"
    else:
        m["ipc"] = "N/A"

    m["branch_miss_rate"] = safe_ratio(branch_misses, branches) + "%" \
        if safe_ratio(branch_misses, branches) != "N/A" else "N/A"

    # Cache — L1d
    l1_loads = safe_get(counters, "L1-dcache-loads")
    l1_misses = safe_get(counters, "L1-dcache-load-misses")
    m["l1d_miss_count"] = safe_fmt(l1_misses)
    if l1_loads and l1_misses is not None and l1_loads > 0:
        m["l1d_hit_rate"] = f"{(1 - l1_misses / l1_loads) * 100:.1f}%"
    else:
        m["l1d_hit_rate"] = "N/A"

    # Cache — L2 (Intel-specific names; may need adjustment)
    l2_refs = safe_get(counters, "l2_rqsts.references") or safe_get(counters, "L2-loads")
    l2_misses = safe_get(counters, "l2_rqsts.miss") or safe_get(counters, "L2-load-misses")
    m["l2_miss_count"] = safe_fmt(l2_misses)
    if l2_refs and l2_misses is not None and l2_refs > 0:
        m["l2_hit_rate"] = f"{(1 - l2_misses / l2_refs) * 100:.1f}%"
    else:
        m["l2_hit_rate"] = "N/A"

    # Cache — LLC
    llc_loads = safe_get(counters, "LLC-loads")
    llc_misses = safe_get(counters, "LLC-load-misses")
    m["llc_miss_count"] = safe_fmt(llc_misses)
    if llc_loads and llc_misses is not None and llc_loads > 0:
        m["llc_hit_rate"] = f"{(1 - llc_misses / llc_loads) * 100:.1f}%"
    else:
        m["llc_hit_rate"] = "N/A"

    # Memory bandwidth estimate
    if llc_misses is not None and wall_ms and wall_ms > 0:
        bw_gbs = (llc_misses * CACHE_LINE_BYTES) / (wall_ms / 1000) / 1e9
        m["est_bandwidth_gbs"] = f"{bw_gbs:.1f}"
        if peak_bw_gbs and peak_bw_gbs > 0:
            m["bw_utilization"] = f"{bw_gbs / peak_bw_gbs * 100:.1f}%"
        else:
            m["bw_utilization"] = "N/A"
    else:
        m["est_bandwidth_gbs"] = "N/A"
        m["bw_utilization"] = "N/A"

    # Threading
    m["context_switches"] = safe_fmt(safe_get(counters, "context-switches"))
    m["cpu_migrations"] = safe_fmt(safe_get(counters, "cpu-migrations"))

    task_clock = safe_get(counters, "task-clock")
    if task_clock and wall_ms and wall_ms > 0 and thread_count > 0:
        m["cpu_utilization"] = f"{task_clock / (wall_ms * thread_count) * 100:.1f}%"
    else:
        m["cpu_utilization"] = "N/A"

    return m


def format_kernel_xml(tag_name, metrics, source_label=None, indent="  "):
    i = indent
    i2 = indent * 2
    i3 = indent * 3

    lines = []
    lines.append(f"{i}<{tag_name}>")

    if source_label:
        lines.append(f"{i2}<Source>{source_label}</Source>")

    # Execution
    lines.append(f"{i2}<Execution>")
    lines.append(f"{i3}<Wall_Time_ms>{metrics['wall_time_ms']}</Wall_Time_ms>")
    lines.append(f"{i3}<Thread_Count>{metrics['thread_count']}</Thread_Count>")
    lines.append(f"{i2}</Execution>")

    # Instruction Profile
    lines.append(f"{i2}<Instruction_Profile>")
    lines.append(f"{i3}<Instructions>{metrics['instructions']}</Instructions>")
    lines.append(f"{i3}<Cycles>{metrics['cycles']}</Cycles>")
    lines.append(f"{i3}<IPC>{metrics['ipc']}</IPC>")
    lines.append(f"{i3}<Branch_Miss_Rate>{metrics['branch_miss_rate']}</Branch_Miss_Rate>")
    lines.append(f"{i2}</Instruction_Profile>")

    # Cache
    lines.append(f"{i2}<Cache>")
    lines.append(f"{i3}<L1d_Hit_Rate>{metrics['l1d_hit_rate']}</L1d_Hit_Rate>")
    lines.append(f"{i3}<L1d_Miss_Count>{metrics['l1d_miss_count']}</L1d_Miss_Count>")
    lines.append(f"{i3}<L2_Hit_Rate>{metrics['l2_hit_rate']}</L2_Hit_Rate>")
    lines.append(f"{i3}<L2_Miss_Count>{metrics['l2_miss_count']}</L2_Miss_Count>")
    lines.append(f"{i3}<LLC_Hit_Rate>{metrics['llc_hit_rate']}</LLC_Hit_Rate>")
    lines.append(f"{i3}<LLC_Miss_Count>{metrics['llc_miss_count']}</LLC_Miss_Count>")
    lines.append(f"{i2}</Cache>")

    # Memory
    lines.append(f"{i2}<Memory>")
    lines.append(f"{i3}<Estimated_Bandwidth_GBs>{metrics['est_bandwidth_gbs']}</Estimated_Bandwidth_GBs>")
    lines.append(f"{i3}<Bandwidth_Utilization>{metrics['bw_utilization']}</Bandwidth_Utilization>")
    lines.append(f"{i2}</Memory>")

    # Threading
    lines.append(f"{i2}<Threading>")
    lines.append(f"{i3}<Context_Switches>{metrics['context_switches']}</Context_Switches>")
    lines.append(f"{i3}<CPU_Migrations>{metrics['cpu_migrations']}</CPU_Migrations>")
    lines.append(f"{i3}<CPU_Utilization>{metrics['cpu_utilization']}</CPU_Utilization>")
    lines.append(f"{i2}</Threading>")

    lines.append(f"{i}</{tag_name}>")
    return "\n".join(lines)


def cmd_feedback(args):
    base_counters = parse_perf_output(args.baseline)
    your_counters = parse_perf_output(args.yours)

    base_m = compute_metrics(base_counters, args.threads, args.peak_bw)
    your_m = compute_metrics(your_counters, args.threads, args.peak_bw)

    xml = []
    xml.append(f'<Iteration_Feedback round="{args.round}">')
    xml.append("")
    xml.append(format_kernel_xml("Baseline_Kernel", base_m,
                                  source_label="HeCBench original OpenMP code"))
    xml.append("")
    xml.append(format_kernel_xml("Your_Kernel", your_m))
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
    counters = parse_perf_output(args.input)
    metrics = compute_metrics(counters, args.threads, args.peak_bw)

    xml = format_kernel_xml("Kernel_Profile", metrics, indent="  ")
    output = f"<Perf_Profile>\n{xml}\n</Perf_Profile>"

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Wrote profile XML to {args.output}")
    else:
        print(output)


def cmd_discover(args):
    counters = parse_perf_output(args.input)
    print(f"\n=== Counters found in {args.input} ({len(counters)} entries) ===\n")
    for name in sorted(counters.keys()):
        if name.startswith("_"):
            continue
        value = counters[name]
        print(f"  {name:45s}  {value:>20}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert perf stat output to OpenMP Iteration_Feedback XML"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared args
    def add_common(p):
        p.add_argument("--threads", type=int, required=True,
                        help="OMP_NUM_THREADS used during the run")
        p.add_argument("--peak-bw", type=float, default=None,
                        help="Peak memory bandwidth in GB/s (for utilization %)")

    # feedback
    p_fb = subparsers.add_parser("feedback",
        help="Generate Iteration_Feedback XML from baseline + your kernel")
    p_fb.add_argument("--baseline", required=True)
    p_fb.add_argument("--yours", required=True)
    p_fb.add_argument("--round", type=int, required=True)
    p_fb.add_argument("--output", "-o")
    add_common(p_fb)

    # single
    p_si = subparsers.add_parser("single",
        help="Generate single kernel profile XML")
    p_si.add_argument("--input", required=True)
    p_si.add_argument("--output", "-o")
    add_common(p_si)

    # discover
    p_di = subparsers.add_parser("discover",
        help="List all counters in a perf output file")
    p_di.add_argument("--input", required=True)

    args = parser.parse_args()

    if args.command == "feedback":
        cmd_feedback(args)
    elif args.command == "single":
        cmd_single(args)
    elif args.command == "discover":
        cmd_discover(args)


if __name__ == "__main__":
    main()
