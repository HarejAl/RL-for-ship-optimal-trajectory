"""
Rebuild the staleness-study CSV and summary from a run's stdout log.

The study prints every case as it completes, so a run that later dies (a rate-limited
region, a killed process) still leaves the finished cases recoverable. This parses those
lines back into the same rows the study would have written, with no API calls.

    python recover_study.py output/logs/staleness_full.log
"""

import argparse
import os
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from staleness_study import write_csv, summarise  # noqa: E402

RE_REGION = re.compile(r"^===\s+(.+?)\s+lat=")
RE_SCALE = re.compile(r"^\s*([\d.]+) km/unit, ([\d.]+) \(m/s\)/unit, 1 model time unit = ([\d.]+) h")
RE_CASE = re.compile(
    r"^\s*dep\+\s*(\d+)h route(\d+):\s*"
    r"oracle (ok|FAIL) J=\s*([\d.]+) \(\s*([\d.]+)h\)\s*\|\s*"
    r"stale (ok|FAIL) J=\s*([\d.]+) \(\s*([-+][\d.]+)%\)\s*\|\s*"
    r"policy (ok|FAIL) J=\s*([\d.]+) \(\s*([-+][\d.]+)%\)")


def parse(path):
    rows, region, kmu, hpu = [], None, np.nan, np.nan
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_REGION.match(line.strip())
            if m:
                region = m.group(1).strip()
                continue
            m = RE_SCALE.match(line)
            if m:
                kmu, hpu = float(m.group(1)), float(m.group(3))
                continue
            m = RE_CASE.match(line)
            if m:
                (dep, route, o_ok, o_J, o_h, s_ok, s_J, s_pct,
                 p_ok, p_J, p_pct) = m.groups()
                rows.append(dict(
                    region=region, departure_h=int(dep), route=int(route),
                    km_per_unit=kmu, hours_per_unit=hpu, dp_solve_s=np.nan,
                    oracle_ok=int(o_ok == "ok"), oracle_J=float(o_J), oracle_h=float(o_h),
                    stale_ok=int(s_ok == "ok"), stale_J=float(s_J), stale_h=np.nan,
                    pol_ok=int(p_ok == "ok"), pol_J=float(p_J), pol_h=np.nan,
                    stale_cost=float(s_pct) / 100.0, policy_gap=float(p_pct) / 100.0))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    args = ap.parse_args()
    rows = []
    seen = set()
    for p in args.logs:
        got = parse(p)
        for r in got:
            key = (r["region"], r["departure_h"], r["route"])
            if key not in seen:       # first log wins if runs overlap
                seen.add(key)
                rows.append(r)
        print(f"{os.path.basename(p)}: parsed {len(got)} cases")
    if not rows:
        print("no cases found")
        return
    regions = sorted({r["region"] for r in rows})
    print(f"recovered {len(rows)} unique cases across {len(regions)} regions: {', '.join(regions)}")
    print("saved", write_csv(rows))
    summarise(rows, args)


if __name__ == "__main__":
    main()
