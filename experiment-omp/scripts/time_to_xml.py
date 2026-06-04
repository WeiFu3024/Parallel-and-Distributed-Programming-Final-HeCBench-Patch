#!/usr/bin/env python3
"""
time_to_xml.py — Convert /usr/bin/time -v output (+ binary stdout) to
                 OpenMP Iteration_Feedback XML.

Used as fallback when perf_event_paranoid > 1 prevents perf stat.
Parses the combined time_raw.txt written by profile.sh.

Input format (time_raw.txt):
  === BINARY OUTPUT ===
  <stdout of the binary, including self-reported kernel time>
  === TIME OUTPUT ===
  <stderr from /usr/bin/time -v>

Usage:
  # Generate feedback XML (baseline vs your kernel):
  python time_to_xml.py feedback \\
      --baseline baseline/time_raw.txt \\
      --yours    cell_c/round_1/time_raw.txt \\
      --round 1 \\
      --benchmark accuracy \\
      --threads 16 \\
      --output   cell_c/round_2/feedback.xml

  # Single kernel profile XML:
  python time_to_xml.py single \\
      --input  baseline/time_raw.txt \\
      --benchmark accuracy \\
      --threads 16 \\
      --output baseline/time.xml

  # Show all parsed values:
  python time_to_xml.py discover --input time_raw.txt --benchmark accuracy

Supported benchmarks (for kernel-time parsing):
  accuracy  — "Average execution time of accuracy kernel: XXXXX (us)"
  aes       — "Average kernel execution time X.XXXXXX (s)"
  fft       — "Average kernel execution time X.XXXXXX (s)"
"""

import argparse
import re
import sys


def parse_time_raw(filepath, benchmark):
    """
    Parse a time_raw.txt file (written by profile.sh) into a dict of metrics.
    Returns a dict with keys: kernel_time_ms, wall_time_ms, user_time_s,
    sys_time_s, cpu_pct, voluntary_cs, involuntary_cs, max_rss_kb, thread_count.
    Missing values are None.
    """
    m = {}

    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"ERROR: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    # Split into sections
    binary_section = ""
    time_section = ""
    if "=== BINARY OUTPUT ===" in content and "=== TIME OUTPUT ===" in content:
        parts = content.split("=== TIME OUTPUT ===", 1)
        binary_section = parts[0].replace("=== BINARY OUTPUT ===", "").strip()
        time_section = parts[1].strip()
    else:
        # Fallback: treat entire content as time output
        time_section = content

    # --- Parse kernel execution time from binary stdout ---
    kernel_time_ms = None

    if benchmark == "accuracy":
        # "Average execution time of accuracy kernel: 3792604.750000 (us)"
        pat = re.search(
            r"Average execution time of accuracy kernel:\s*([\d.]+)\s*\(us\)",
            binary_section,
        )
        if pat:
            # Multiple grid sizes are reported; take the last (largest = N=8192)
            all_matches = re.findall(
                r"Average execution time of accuracy kernel:\s*([\d.]+)\s*\(us\)",
                binary_section,
            )
            if all_matches:
                kernel_time_ms = float(all_matches[-1]) / 1000.0

    elif benchmark in ("aes", "fft"):
        # "Average kernel execution time 0.123456 (s)"
        all_matches = re.findall(
            r"Average kernel execution time\s+([\d.eE+\-]+)\s*\(s\)",
            binary_section,
        )
        if all_matches:
            # Take last match (most meaningful for multi-pass)
            kernel_time_ms = float(all_matches[-1]) * 1000.0

    m["kernel_time_ms"] = kernel_time_ms

    # --- Parse /usr/bin/time -v output ---

    def _time_search(pattern, text=time_section):
        hit = re.search(pattern, text)
        return hit.group(1) if hit else None

    # Wall clock time: "0:01.23" or "1:23:45"
    wall_raw = _time_search(r"Elapsed \(wall clock\) time.*?:\s*([\d:\.]+)")
    wall_time_ms = None
    if wall_raw:
        parts = wall_raw.split(":")
        if len(parts) == 2:
            wall_time_ms = float(parts[0]) * 60000 + float(parts[1]) * 1000
        elif len(parts) == 3:
            wall_time_ms = (
                float(parts[0]) * 3600000
                + float(parts[1]) * 60000
                + float(parts[2]) * 1000
            )
    m["wall_time_ms"] = wall_time_ms

    # User and system time
    user_s = _time_search(r"User time \(seconds\):\s*([\d.]+)")
    sys_s = _time_search(r"System time \(seconds\):\s*([\d.]+)")
    m["user_time_s"] = float(user_s) if user_s else None
    m["sys_time_s"] = float(sys_s) if sys_s else None

    # Percent CPU
    cpu_pct = _time_search(r"Percent of CPU this job got:\s*(\d+)%")
    m["cpu_pct"] = int(cpu_pct) if cpu_pct else None

    # Context switches
    vol_cs = _time_search(r"Voluntary context switches:\s*(\d+)")
    invol_cs = _time_search(r"Involuntary context switches:\s*(\d+)")
    m["voluntary_cs"] = int(vol_cs) if vol_cs else None
    m["involuntary_cs"] = int(invol_cs) if invol_cs else None

    # Max RSS
    rss = _time_search(r"Maximum resident set size \(kbytes\):\s*(\d+)")
    m["max_rss_kb"] = int(rss) if rss else None

    return m


def _fmt(value, unit="", fmt=".1f"):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:{fmt}}{unit}"
    return f"{value}{unit}"


def compute_derived(raw, thread_count):
    """Compute display-ready values from raw metrics."""
    d = {}

    d["kernel_time_ms"] = _fmt(raw.get("kernel_time_ms"), fmt=".3f")
    d["wall_time_ms"] = _fmt(raw.get("wall_time_ms"), fmt=".1f")
    d["thread_count"] = str(thread_count)

    # CPU utilization: total CPU time / (wall_time × threads) × 100
    user_s = raw.get("user_time_s")
    sys_s = raw.get("sys_time_s")
    wall_ms = raw.get("wall_time_ms")
    if user_s is not None and sys_s is not None and wall_ms and wall_ms > 0 and thread_count > 0:
        total_cpu_s = user_s + sys_s
        wall_s = wall_ms / 1000.0
        d["cpu_utilization"] = f"{total_cpu_s / (wall_s * thread_count) * 100:.1f}%"
    else:
        d["cpu_utilization"] = "N/A"

    # Context switches (combine voluntary + involuntary)
    vol = raw.get("voluntary_cs")
    invol = raw.get("involuntary_cs")
    if vol is not None and invol is not None:
        d["context_switches"] = str(vol + invol)
    else:
        d["context_switches"] = "N/A"

    d["max_rss_mb"] = _fmt(
        raw.get("max_rss_kb", None) and raw["max_rss_kb"] / 1024,
        fmt=".1f"
    )

    return d


def format_kernel_xml(tag, derived, source_label=None, indent="  "):
    i = indent
    i2 = indent * 2
    i3 = indent * 3
    na = "N/A"

    lines = [f"{i}<{tag}>"]
    if source_label:
        lines.append(f"{i2}<Source>{source_label}</Source>")

    lines += [
        f"{i2}<Execution>",
        f"{i3}<Kernel_Time_ms>{derived['kernel_time_ms']}</Kernel_Time_ms>",
        f"{i3}<Wall_Time_ms>{derived['wall_time_ms']}</Wall_Time_ms>",
        f"{i3}<Thread_Count>{derived['thread_count']}</Thread_Count>",
        f"{i2}</Execution>",

        f"{i2}<Instruction_Profile>",
        f"{i3}<Instructions>{na}</Instructions>",
        f"{i3}<Cycles>{na}</Cycles>",
        f"{i3}<IPC>{na}</IPC>",
        f"{i3}<Branch_Miss_Rate>{na}</Branch_Miss_Rate>",
        f"{i2}</Instruction_Profile>",

        f"{i2}<Cache>",
        f"{i3}<L1d_Hit_Rate>{na}</L1d_Hit_Rate>",
        f"{i3}<L1d_Miss_Count>{na}</L1d_Miss_Count>",
        f"{i3}<L2_Hit_Rate>{na}</L2_Hit_Rate>",
        f"{i3}<L2_Miss_Count>{na}</L2_Miss_Count>",
        f"{i3}<LLC_Hit_Rate>{na}</LLC_Hit_Rate>",
        f"{i3}<LLC_Miss_Count>{na}</LLC_Miss_Count>",
        f"{i2}</Cache>",

        f"{i2}<Memory>",
        f"{i3}<Estimated_Bandwidth_GBs>{na}</Estimated_Bandwidth_GBs>",
        f"{i3}<Bandwidth_Utilization>{na}</Bandwidth_Utilization>",
        f"{i3}<Max_RSS_MB>{derived['max_rss_mb']}</Max_RSS_MB>",
        f"{i2}</Memory>",

        f"{i2}<Threading>",
        f"{i3}<Context_Switches>{derived['context_switches']}</Context_Switches>",
        f"{i3}<CPU_Migrations>{na}</CPU_Migrations>",
        f"{i3}<CPU_Utilization>{derived['cpu_utilization']}</CPU_Utilization>",
        f"{i2}</Threading>",

        f"{i}</{tag}>",
    ]
    return "\n".join(lines)


def write_output(text, path):
    if path:
        with open(path, "w") as f:
            f.write(text)
        print(f"Wrote to {path}")
    else:
        print(text)


def cmd_feedback(args):
    base_raw = parse_time_raw(args.baseline, args.benchmark)
    your_raw = parse_time_raw(args.yours, args.benchmark)

    base_d = compute_derived(base_raw, args.threads)
    your_d = compute_derived(your_raw, args.threads)

    note = (
        "<!-- Note: Instruction_Profile, Cache, and Bandwidth fields are N/A\n"
        "     because perf_event_paranoid blocked hardware counters.\n"
        "     Kernel_Time_ms is the binary's self-reported parallel region time.\n"
        "     Wall_Time_ms and CPU_Utilization are from /usr/bin/time -v. -->"
    )

    xml = "\n".join([
        f'<Iteration_Feedback round="{args.round}">',
        "",
        note,
        "",
        format_kernel_xml("Baseline_Kernel", base_d,
                          source_label="HeCBench original OpenMP code"),
        "",
        format_kernel_xml("Your_Kernel", your_d),
        "",
        "</Iteration_Feedback>",
    ])
    write_output(xml, args.output)


def cmd_single(args):
    raw = parse_time_raw(args.input, args.benchmark)
    d = compute_derived(raw, args.threads)
    xml = f"<Time_Profile>\n{format_kernel_xml('Kernel', d, indent='  ')}\n</Time_Profile>"
    write_output(xml, args.output)


def cmd_discover(args):
    raw = parse_time_raw(args.input, args.benchmark)
    print(f"\n=== Parsed values from {args.input} ===\n")
    for k, v in sorted(raw.items()):
        print(f"  {k:30s}  {v}")


def main():
    p = argparse.ArgumentParser(
        description="Convert /usr/bin/time -v output to OpenMP Iteration_Feedback XML"
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--benchmark", required=True,
                        choices=["accuracy", "aes", "fft"],
                        help="Benchmark name (for kernel-time parsing)")
        sp.add_argument("--threads", type=int, required=True,
                        help="OMP_NUM_THREADS used during the run")

    p_fb = sub.add_parser("feedback")
    p_fb.add_argument("--baseline", required=True)
    p_fb.add_argument("--yours", required=True)
    p_fb.add_argument("--round", type=int, required=True)
    p_fb.add_argument("--output", "-o")
    add_common(p_fb)

    p_si = sub.add_parser("single")
    p_si.add_argument("--input", required=True)
    p_si.add_argument("--output", "-o")
    add_common(p_si)

    p_di = sub.add_parser("discover")
    p_di.add_argument("--input", required=True)
    p_di.add_argument("--benchmark", required=True,
                      choices=["accuracy", "aes", "fft"])

    args = p.parse_args()

    if args.command == "feedback":
        cmd_feedback(args)
    elif args.command == "single":
        cmd_single(args)
    elif args.command == "discover":
        cmd_discover(args)


if __name__ == "__main__":
    main()
