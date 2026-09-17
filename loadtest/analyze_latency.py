"""Exact latency percentiles from a LOADTEST_LATENCY_DUMP sweep.

locust's summary CSV only holds whole-run cumulative percentiles, so the
connection storm in the first seconds of a run leaks into p99. The latency
dump records every request with its elapsed time, so this script can drop the
warm-up window and report true steady-state percentiles.

    uv run loadtest/analyze_latency.py loadtest/results/ab-steady [--skip 5]
"""

import glob
import os
import re
import sys
from collections import defaultdict

GATEWAYS = ["direct", "proxy", "litellm"]


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def load(path, skip):
    """Return list of (elapsed, stat_name, latency) after `skip` seconds."""
    out = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 3:
                continue
            try:
                elapsed, latency = float(parts[0]), float(parts[2])
            except ValueError:
                continue
            if elapsed >= skip:
                out.append((elapsed, parts[1], latency))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    outdir = args[0] if args else "loadtest/results/ab-steady"
    skip = 5.0
    if "--skip" in sys.argv:
        skip = float(sys.argv[sys.argv.index("--skip") + 1])

    files = sorted(glob.glob(f"{outdir}/*.latency.csv"))
    if not files:
        sys.exit(f"no *.latency.csv in {outdir}")

    runs = []
    for path in files:
        base = os.path.basename(path)
        m = re.match(r"(?:r(\d+)-)?(direct|proxy|litellm)-u(\d+)\.latency\.csv$", base)
        if not m:
            continue
        rep = int(m.group(1) or 1)
        gw, users = m.group(2), int(m.group(3))
        samples = load(path, skip)
        if not samples:
            continue
        lats = sorted(s[2] for s in samples)
        span = max(s[0] for s in samples) - skip
        per = defaultdict(list)
        for _, name, lat in samples:
            per[name].append(lat)
        runs.append(
            {
                "rep": rep,
                "gateway": gw,
                "users": users,
                "n": len(lats),
                "span": span,
                "rps": len(lats) / span if span > 0 else 0.0,
                "p50": percentile(lats, 50),
                "p95": percentile(lats, 95),
                "p99": percentile(lats, 99),
                "p999": percentile(lats, 99.9),
                "max": lats[-1],
                "per": {k: sorted(v) for k, v in per.items()},
            }
        )

    levels = sorted({r["users"] for r in runs})
    print(f"\n# Steady-state (warm-up window of first {skip:g}s dropped)\n")
    print(
        "| users | gateway | RPS | p50 (ms) | p95 (ms) | p99 (ms) | p99.9 (ms) | max (ms) | reqs |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for users in levels:
        for gw in GATEWAYS:
            sel = [r for r in runs if r["users"] == users and r["gateway"] == gw]
            if not sel:
                continue
            n = len(sel)
            print(
                f"| {users} | {gw} | {sum(r['rps'] for r in sel) / n:.0f} | "
                f"{sum(r['p50'] for r in sel) / n:.1f} | {sum(r['p95'] for r in sel) / n:.1f} | "
                f"{sum(r['p99'] for r in sel) / n:.1f} | {sum(r['p999'] for r in sel) / n:.1f} | "
                f"{sum(r['max'] for r in sel) / n:.0f} | {sum(r['n'] for r in sel) / n:.0f} |"
            )

    print("\n# Steady-state per scenario\n")
    print("| users | gateway | scenario | RPS | p50 | p95 | p99 |")
    print("|---|---|---|---|---|---|---|")
    for users in levels:
        for gw in GATEWAYS:
            sel = [r for r in runs if r["users"] == users and r["gateway"] == gw]
            if not sel:
                continue
            names = sorted(sel[0]["per"])
            for name in names:
                vals = sorted(v for r in sel for v in r["per"].get(name, []))
                if not vals:
                    continue
                rps = sum(
                    len(r["per"].get(name, [])) / r["span"] for r in sel if r["span"] > 0
                ) / len(sel)
                print(
                    f"| {users} | {gw} | {name} | {rps:.0f} | {percentile(vals, 50):.1f} | "
                    f"{percentile(vals, 95):.1f} | {percentile(vals, 99):.1f} |"
                )


if __name__ == "__main__":
    main()
