#!/usr/bin/env python3
"""Run the SpGEMM test binary several times and compare latencies against a saved baseline.

Usage:
    scripts/bench.py                    # 3 runs, compare with bench/baseline.json
    scripts/bench.py --runs 5
    scripts/bench.py --save-baseline    # record this run as the new baseline
    scripts/bench.py --mtx matrices     # SuiteSparse matrices (A * A); baseline bench/baseline_suitesparse.json

Exit status: 0 if every test passed and no case regressed beyond --threshold, 1 if a test
failed (or the binary crashed), 2 if tests passed but performance regressed.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATENCY = re.compile(r"^\s+(\S+)\s+latency:\s+([\d.]+) ms")
REFERENCE = "cuSPARSE"   # the external baseline; every other implementation is ours
# Older outputs and baselines called the CSR implementation "spgemm".
RENAMED = {"spgemm": "CSR"}
RESULT = re.compile(r"^\s+\[\s*(OK|FAIL)\s*\]\s+(\S+)")


def parse(output):
    """Returns {case name: {implementation: ms, ..., "ok": bool}}."""
    cases, current = {}, None
    for line in output.splitlines():
        if line and not line[0].isspace() and not line.startswith(("All tests", "Some tests")):
            current = line.strip()
            cases[current] = {"ok": True}
        elif current and (m := LATENCY.match(line)):
            cases[current][RENAMED.get(m.group(1), m.group(1))] = float(m.group(2))
        elif current and (m := RESULT.match(line)):
            cases[current]["ok"] &= m.group(1) == "OK"
    return cases


def git_commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "."], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip().splitlines()[0]
    except (OSError, IndexError):
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default=str(ROOT / "build/bin/spgemm_nvcuda"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--baseline", help="default: bench/baseline.json, or bench/baseline_suitesparse.json with --mtx")
    ap.add_argument("--mtx", nargs="+", metavar="PATH", help="Matrix Market files or directories, passed to the binary")
    ap.add_argument("--hamiltonian", nargs="+", metavar="ARG",
                    help="cells orbitals cutoff [--shuffle]: one generated Hamiltonian, passed to the binary")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--threshold", type=float, default=10.0, help="regression threshold in percent (default 10)")
    ap.add_argument("--min-delta", type=float, default=0.05,
                    help="ignore changes smaller than this many ms; sub-millisecond cases are noisy (default 0.05)")
    args = ap.parse_args()
    if args.baseline is None:
        name = "baseline_suitesparse" if args.mtx else "baseline_hamiltonian" if args.hamiltonian else "baseline"
        args.baseline = str(ROOT / "bench" / f"{name}.json")
    command = [args.binary]
    if args.mtx:
        command += ["--mtx"] + args.mtx
    elif args.hamiltonian:
        command += ["--hamiltonian"] + args.hamiltonian

    runs = []
    for i in range(args.runs):
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode not in (0, 1) or not proc.stdout.strip():
            print(f"Run {i + 1}: binary exited with status {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
            return 1
        runs.append(parse(proc.stdout))

    names = list(runs[0])
    impls = []                                                       # in the order the binary prints them
    for name in names:
        impls += [k for k in runs[0][name] if k != "ok" and k not in impls]
    impls = [i for i in impls if i != REFERENCE] + [i for i in impls if i == REFERENCE]   # reference last
    ours = [k for k in impls if k != REFERENCE]
    current = {}
    for name in names:
        current[name] = {"ok": all(r.get(name, {}).get("ok", False) for r in runs)}
        for impl in impls:
            times = [r[name][impl] for r in runs if impl in r.get(name, {})]
            if times:                                                    # not every implementation runs every case
                current[name][impl] = statistics.median(times)

    baseline_path = Path(args.baseline)
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    base_cases = baseline["cases"] if baseline else {}
    base_cases = {case: {RENAMED.get(k, k): v for k, v in vals.items()} for case, vals in base_cases.items()}

    print(f"GPU: {gpu_name()}   commit: {git_commit()}   runs: {args.runs} (median)")
    if baseline:
        print(f"Baseline: commit {baseline['commit']}, recorded {baseline['date']}, "
              f"threshold {args.threshold:.0f}% and {args.min_delta} ms")
    else:
        print("Baseline: none (run with --save-baseline to record one)")
    print()
    header = ["Case", "Correct"] + [f"{i} (ms)" for i in impls] + [f"{i} vs {REFERENCE}" for i in ours]
    header += [f"{i} vs baseline" for i in ours]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))

    failed = regressed = False
    for name, c in current.items():
        failed |= not c["ok"]
        cells = [name, "yes" if c["ok"] else "NO"] + [f"{c[i]:.3f}" if i in c else "-" for i in impls]
        cells += [f"{c[REFERENCE] / c[i]:.2f}x" if REFERENCE in c and i in c else "-" for i in ours]
        for impl in ours:
            base = base_cases.get(name, {}).get(impl)
            if impl not in c:
                cells.append("-")
                continue
            if not base:
                cells.append("new")
                continue
            change = (c[impl] - base) / base * 100
            significant = abs(c[impl] - base) >= args.min_delta and abs(change) > args.threshold
            flag = (" REGRESSION" if change > 0 else " faster") if significant else ""
            regressed |= significant and change > 0
            cells.append(f"{change:+.1f}%{flag}")
        print("| " + " | ".join(cells) + " |")

    for name in base_cases:
        if name not in current:
            print(f"\nNote: baseline case missing from this run: {name}")

    if args.save_baseline:
        if failed:
            print("\nNot saving baseline: some tests failed.")
        else:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps({
                "commit": git_commit(), "gpu": gpu_name(), "date": datetime.now().isoformat(timespec="seconds"),
                "runs": args.runs, "cases": current}, indent=2) + "\n")
            print(f"\nSaved baseline to {baseline_path}")

    if failed:
        print("\nRESULT: FAIL (correctness)")
        return 1
    if regressed:
        print("\nRESULT: PASS with performance regressions")
        return 2
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
