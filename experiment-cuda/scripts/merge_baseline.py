#!/usr/bin/env python3
"""
merge_baseline.py — Inline local kernel/helper sources into baseline/main.cu.

Skips operator-assigned benchmarks and benchmarks whose kernels are already
fully contained in main.cu with no local .cu/.cuh kernel includes.
"""

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

# Operator-assigned: do not modify
SKIP_BENCHMARKS = {
    "blas-gemm", "maxpool3d", "binomial", "mcpr", "fluidSim",
    "bfs", "nw", "bscan", "sc", "lzss",
}

# Already hand-merged previously
ALREADY_MERGED = {"heartwall", "sssp", "binomial", "fluidSim", "mcpr", "sc"}

# No __global__ kernel to merge (Thrust library path)
NO_KERNEL_BENCHMARKS = {"segment-reduce"}

# Extra search roots per benchmark (for cross-package includes)
EXTRA_SEARCH = {
    "layernorm": ["rmsnorm-cuda"],
    "aes": ["include"],
    "bfs": ["bfs-sycl"],
}

SRC_NAME = {
    "transpose": "matrixT-cuda",
}


def bench_src_name(bench: str) -> str:
    return SRC_NAME.get(bench, f"{bench}-cuda")


def search_dirs(hecbench: Path, bench: str) -> list[Path]:
    dirs = [hecbench / "src" / bench_src_name(bench)]
    for extra in EXTRA_SEARCH.get(bench, []):
        p = hecbench / "src" / extra
        if p.is_dir():
            dirs.append(p)
    return dirs


def resolve_include(name: str, from_dir: Path, search: list[Path]) -> Path | None:
    candidate = from_dir / name
    if candidate.is_file():
        return candidate
    for root in search:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def inline_includes(
    text: str,
    search: list[Path],
    seen: set[str],
    from_dir: Path,
) -> str:
    pattern = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)

    def replace_include(match: re.Match) -> str:
        name = match.group(1)
        if name in seen:
            return f'// [merge] skipped duplicate include "{name}"'
        path = resolve_include(name, from_dir, search)
        if path is None:
            return match.group(0)
        seen.add(name)
        body = path.read_text(encoding="utf-8", errors="replace")
        body = inline_includes(body, search, seen, path.parent)
        return (
            f"// ---- INLINED: {name} (from {path}) ----\n"
            f"{body}\n"
            f"// ---- END INLINED: {name} ----\n"
        )

    return pattern.sub(replace_include, text)


def needs_merge(main_cu: Path) -> bool:
    text = main_cu.read_text(encoding="utf-8", errors="replace")
    if "__global__" not in text:
        return True
    if re.search(r'#include\s+"[^"]+\.(cu|cuh)"', text):
        return True
    if re.search(r'#include\s+"[^"]+\.h"', text) and any(
        k in text
        for k in ("kernels.", "kernel.h", "fft1D", "histogram_", "sort_", "reduce.cuh", "utils.cuh")
    ):
        return True
    return False


def merge_benchmark(hecbench: Path, bench: str, dry_run: bool = False) -> bool:
    if bench in SKIP_BENCHMARKS or bench in NO_KERNEL_BENCHMARKS:
        print(f"  SKIP {bench}: excluded")
        return False

    main_cu = RESULTS_DIR / bench / "baseline" / "main.cu"
    if not main_cu.is_file():
        print(f"  SKIP {bench}: no baseline/main.cu")
        return False

    if bench in ALREADY_MERGED and not needs_merge(main_cu):
        print(f"  SKIP {bench}: already merged")
        return False

    original = main_cu.read_text(encoding="utf-8", errors="replace")
    if not needs_merge(main_cu):
        print(f"  SKIP {bench}: kernel already in main.cu")
        return False

    search = search_dirs(hecbench, bench)
    merged = inline_includes(original, search, set(), search[0])

    if merged == original:
        print(f"  SKIP {bench}: no changes after merge pass")
        return False

    if dry_run:
        print(f"  DRY-RUN {bench}: would write {len(merged.splitlines())} lines")
        return True

    backup = main_cu.with_suffix(".cu.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    main_cu.write_text(merged, encoding="utf-8")
    print(f"  MERGED {bench}: {len(original.splitlines())} -> {len(merged.splitlines())} lines")
    return True


def main():
    parser = argparse.ArgumentParser(description="Inline kernel sources into baseline/main.cu")
    parser.add_argument("hecbench_path", type=Path)
    parser.add_argument("benchmarks", nargs="*", help="Benchmark names (default: all under results/)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hecbench = args.hecbench_path.resolve()
    if args.benchmarks:
        benches = args.benchmarks
    else:
        benches = sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir())

    changed = 0
    for bench in benches:
        if merge_benchmark(hecbench, bench, dry_run=args.dry_run):
            changed += 1

    print(f"\nDone. {'Would change' if args.dry_run else 'Changed'} {changed} benchmark(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
