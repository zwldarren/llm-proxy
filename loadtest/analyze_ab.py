"""Analyse a gateway A/B sweep produced by loadtest/gateway-ab.sh.

Reads the locust per-run CSVs plus the resource samples and prints markdown
tables: throughput, latency percentiles, failures, and container CPU/memory.

Headline latency is a **request-count-weighted mean of the per-scenario
percentiles** — locust's summary CSV has no raw sample stream, so an exact
overall percentile cannot be recovered. Exact per-scenario numbers are printed
alongside. Throughput and failure counts are exact sums.

    uv run loadtest/analyze_ab.py  # defaults to loadtest/results/ab-$(date -u +%F)
"""

import csv
import glob
import os
import re
import sys
from datetime import UTC, datetime

PCTS = ["50%", "95%", "99%"]


def fnum(value) -> float:
    try:
        return float(value)
    except TypeError, ValueError:
        return 0.0


def read_stats(path):
    """Return (post_rows, custom_rows) from a locust *_stats.csv."""
    posts, customs = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["Type"] == "POST":
                posts.append(row)
            elif row["Type"] == "Custom":
                customs.append(row)
    return posts, customs


def stat_name_to_key(name: str) -> str:
    """Map a locust stat name back to a scenarios.py key.

    'chat:openai_compat [stream]' -> 'chat_openai_compat_stream'
    'embeddings:openai_compat'    -> 'embeddings_openai_compat_nonstream'
    """
    stream = name.endswith(" [stream]")
    base = name.removesuffix(" [stream]")
    proto, provider = base.split(":", 1)
    return f"{proto}_{provider}_{'stream' if stream else 'nonstream'}"


def weighted_pct(rows, pct):
    total = sum(fnum(r["Request Count"]) for r in rows)
    if total <= 0:
        return 0.0
    return sum(fnum(r[pct]) * fnum(r["Request Count"]) for r in rows) / total


def load_run(outdir, path):
    base = os.path.basename(path)
    m = re.match(r"r(\d+)-(direct|proxy|litellm)-u(\d+)_stats\.csv$", base)
    if not m:
        return None
    rep, gateway, users = int(m.group(1)), m.group(2), int(m.group(3))
    posts, customs = read_stats(path)

    counts = sum(fnum(r["Request Count"]) for r in posts)
    failures = sum(fnum(r["Failure Count"]) for r in posts)
    rps = sum(fnum(r["Requests/s"]) for r in posts)

    per_scenario = {
        r["Name"]: {
            "rps": fnum(r["Requests/s"]),
            "count": fnum(r["Request Count"]),
            "fail": fnum(r["Failure Count"]),
            "p50": fnum(r["50%"]),
            "p95": fnum(r["95%"]),
            "p99": fnum(r["99%"]),
            "avg": fnum(r["Average Response Time"]),
        }
        for r in posts
    }
    overhead = {}
    ttft = {}
    for r in customs:
        name = r["Name"]
        if name.startswith("overhead:"):
            overhead[name.split(":", 1)[1]] = fnum(r["Average Response Time"])
        elif name.startswith("ttft:"):
            ttft[name.split(":", 1)[1]] = fnum(r["Average Response Time"])

    return {
        "rep": rep,
        "gateway": gateway,
        "users": users,
        "rps": rps,
        "count": counts,
        "failures": failures,
        "fail_rate": (failures / counts * 100) if counts else 0.0,
        "p50": weighted_pct(posts, "50%"),
        "p95": weighted_pct(posts, "95%"),
        "p99": weighted_pct(posts, "99%"),
        "max": max((fnum(r["100%"]) for r in posts), default=0.0),
        "per_scenario": per_scenario,
        "overhead": overhead,
        "ttft": ttft,
    }


def read_resources(path):
    """Return (mean_cpu, max_cpu, max_mem_mib, mean_load) from a samples csv."""
    if not os.path.exists(path):
        return None
    cpus, mems, loads = [], [], []
    with open(path, newline="") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                loads.append(float(parts[1]))
                cpus.append(float(parts[2].rstrip("%")))
            except ValueError:
                continue
            # "1.23GiB / 12.9GiB"
            mem = parts[3].split("/")[0].strip()
            for unit, scale in (("GiB", 1024), ("MiB", 1), ("KiB", 1 / 1024)):
                if mem.endswith(unit):
                    mems.append(float(mem[: -len(unit)]) * scale)
                    break
    if not cpus:
        return None
    return {
        "mean_cpu": sum(cpus) / len(cpus),
        "max_cpu": max(cpus),
        "max_mem_mib": max(mems) if mems else 0.0,
        "mean_load": sum(loads) / len(loads) if loads else 0.0,
        "max_load": max(loads) if loads else 0.0,
    }


def main():
    outdir = (
        sys.argv[1] if len(sys.argv) > 1 else f"loadtest/results/ab-{datetime.now(UTC):%Y-%m-%d}"
    )
    runs = [r for p in sorted(glob.glob(f"{outdir}/r*-u*_stats.csv")) if (r := load_run(outdir, p))]
    if not runs:
        sys.exit(f"no runs found in {outdir}")

    levels = sorted({r["users"] for r in runs})
    gateways = ["direct", "proxy", "litellm"]

    print(f"\n# Raw runs ({len(runs)})\n")
    print(
        "| rep | gateway | users | RPS | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | reqs | fails |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: (r["rep"], r["users"], gateways.index(r["gateway"]))):
        print(
            f"| {r['rep']} | {r['gateway']} | {r['users']} | {r['rps']:.0f} | "
            f"{r['p50']:.1f} | {r['p95']:.1f} | {r['p99']:.1f} | {r['max']:.0f} | "
            f"{r['count']:.0f} | {r['failures']:.0f} |"
        )

    print("\n# Per-level means over reps\n")
    print(
        "| users | gateway | RPS | p50 | p95 | p99 | fail% | mean CPU% | max CPU% "
        "| max mem MiB | host load |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for users in levels:
        for gw in gateways:
            sel = [r for r in runs if r["users"] == users and r["gateway"] == gw]
            if not sel:
                continue
            n = len(sel)
            res = [read_resources(f"{outdir}/r{r['rep']}-{gw}-u{users}.resources.csv") for r in sel]
            res = [x for x in res if x]
            cpu = sum(x["mean_cpu"] for x in res) / len(res) if res else 0.0
            cpumax = max((x["max_cpu"] for x in res), default=0.0)
            mem = max((x["max_mem_mib"] for x in res), default=0.0)
            load = sum(x["mean_load"] for x in res) / len(res) if res else 0.0
            print(
                f"| {users} | {gw} | {sum(r['rps'] for r in sel) / n:.0f} | "
                f"{sum(r['p50'] for r in sel) / n:.1f} | {sum(r['p95'] for r in sel) / n:.1f} | "
                f"{sum(r['p99'] for r in sel) / n:.1f} | "
                f"{sum(r['fail_rate'] for r in sel) / n:.2f} | {cpu:.0f} | {cpumax:.0f} | "
                f"{mem:.0f} | {load:.2f} |"
            )

    print("\n# Per-scenario detail (mean over reps)\n")
    print(
        "| users | gateway | scenario | RPS | p50 | p95 | p99 | fails "
        "| gateway overhead (mean ms) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for users in levels:
        for gw in gateways:
            sel = [r for r in runs if r["users"] == users and r["gateway"] == gw]
            if not sel:
                continue
            names = sorted(sel[0]["per_scenario"])
            for name in names:
                n = len(sel)
                rps = (
                    sum(r["per_scenario"][name]["rps"] for r in sel if name in r["per_scenario"])
                    / n
                )
                p50 = (
                    sum(r["per_scenario"][name]["p50"] for r in sel if name in r["per_scenario"])
                    / n
                )
                p95 = (
                    sum(r["per_scenario"][name]["p95"] for r in sel if name in r["per_scenario"])
                    / n
                )
                p99 = (
                    sum(r["per_scenario"][name]["p99"] for r in sel if name in r["per_scenario"])
                    / n
                )
                fail = (
                    sum(r["per_scenario"][name]["fail"] for r in sel if name in r["per_scenario"])
                    / n
                )
                key = stat_name_to_key(name)
                ovh = [r["overhead"][key] for r in sel if key in r["overhead"]]
                ovh_s = f"{sum(ovh) / len(ovh):.1f}" if ovh else "-"
                print(
                    f"| {users} | {gw} | {name} | {rps:.0f} | {p50:.1f} | {p95:.1f} | "
                    f"{p99:.1f} | {fail:.0f} | {ovh_s} |"
                )


if __name__ == "__main__":
    main()
