#!/usr/bin/env python3
"""Run the SpGEMM test binary several times and compare latencies against a saved baseline.

Usage:
    scripts/bench.py                    # 3 runs, compare with bench/baseline.json
    scripts/bench.py --runs 5
    scripts/bench.py --save-baseline    # record this run as the new baseline

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
LATENCY = re.compile(r"^\s+(spgemm|cuSPARSE)\s+latency:\s+([\d.]+) ms")
RESULT = re.compile(r"^\s+\[\s*(OK|FAIL)\s*\]\s+(\S+)")


def parse(output):
    """Returns {case name: {"spgemm": ms, "cuSPARSE": ms, "ok": bool}}."""
    cases, current = {}, None
    for line in output.splitlines():
        if line and not line[0].isspace() and not line.startswith(("All tests", "Some tests")):
            current = line.strip()
            cases[current] = {"ok": True}
        elif current and (m := LATENCY.match(line)):
            cases[current][m.group(1)] = float(m.group(2))
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
    ap.add_argument("--baseline", default=str(ROOT / "bench/baseline.json"))
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--threshold", type=float, default=10.0, help="regression threshold in percent (default 10)")
    ap.add_argument("--min-delta", type=float, default=0.05,
                    help="ignore changes smaller than this many ms; sub-millisecond cases are noisy (default 0.05)")
    args = ap.parse_args()

    runs = []
    for i in range(args.runs):
        proc = subprocess.run([args.binary], capture_output=True, text=True)
        if proc.returncode not in (0, 1) or not proc.stdout.strip():
            print(f"Run {i + 1}: binary exited with status {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
            return 1
        runs.append(parse(proc.stdout))

    names = list(runs[0])
    current = {}
    for name in names:
        current[name] = {
            "ok": all(r.get(name, {}).get("ok", False) for r in runs),
            "spgemm": statistics.median(r[name]["spgemm"] for r in runs if "spgemm" in r.get(name, {})),
            "cuSPARSE": statistics.median(r[name]["cuSPARSE"] for r in runs if "cuSPARSE" in r.get(name, {})),
        }

    baseline_path = Path(args.baseline)
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    base_cases = baseline["cases"] if baseline else {}

    print(f"GPU: {gpu_name()}   commit: {git_commit()}   runs: {args.runs} (median)")
    if baseline:
        print(f"Baseline: commit {baseline['commit']}, recorded {baseline['date']}, "
              f"threshold {args.threshold:.0f}% and {args.min_delta} ms")
    else:
        print("Baseline: none (run with --save-baseline to record one)")
    print()
    print("| Case | Correct | spgemm (ms) | cuSPARSE (ms) | vs cuSPARSE | Baseline (ms) | Change |")
    print("|---|---|---|---|---|---|---|")

    failed = regressed = False
    for name, c in current.items():
        failed |= not c["ok"]
        speedup = c["cuSPARSE"] / c["spgemm"]
        base = base_cases.get(name, {}).get("spgemm")
        if base:
            change = (c["spgemm"] - base) / base * 100
            significant = abs(c["spgemm"] - base) >= args.min_delta and abs(change) > args.threshold
            flag = (" REGRESSION" if change > 0 else " faster") if significant else ""
            regressed |= significant and change > 0
            base_str, change_str = f"{base:.3f}", f"{change:+.1f}%{flag}"
        else:
            base_str, change_str = "-", "new case"
        print(f"| {name} | {'yes' if c['ok'] else 'NO'} | {c['spgemm']:.3f} | {c['cuSPARSE']:.3f} | "
              f"{speedup:.2f}x | {base_str} | {change_str} |")

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
