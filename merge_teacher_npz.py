"""
Concatenate DP-teacher datasets produced by parallel `dp_dataset_prefs.py` workers.

Each worker covers a disjoint range of training fields (`--first-field`), so merging is a
plain row-wise concatenation; the `meta` row (solves, VI seconds, grid) is summed for the
first two entries and taken from the first file for the grid.

    python merge_teacher_npz.py --prefs fast balanced eco \
        --inputs dp_teacher_pref dp_teacher_prefB --out dp_teacher_all

reads data/<input>_<pref>.npz for every input and writes data/<out>_<pref>.npz.
"""

import argparse
import os

import numpy as np

# Anchor all paths to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

ROW_KEYS = ("field_seed", "goal", "state", "action", "value", "source",
            "next_state", "reward", "terminated")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="file prefixes under data/")
    ap.add_argument("--prefs", nargs="+", default=["fast", "balanced", "eco"])
    ap.add_argument("--out", default="dp_teacher_all", help="output prefix under data/")
    return ap.parse_args()


def main():
    args = parse_args()
    for pref in args.prefs:
        parts = []
        for prefix in args.inputs:
            path = os.path.join(DATA_DIR, f"{prefix}_{pref}.npz")
            if not os.path.exists(path):
                print(f"  skip missing {path}")
                continue
            with np.load(path, allow_pickle=True) as f:
                parts.append({k: f[k] for k in f.files})
            print(f"  {os.path.basename(path)}: {len(parts[-1]['state']):,} rows, "
                  f"{len(np.unique(parts[-1]['field_seed']))} fields")
        if not parts:
            raise SystemExit(f"no inputs found for preference {pref}")
        merged = {k: np.concatenate([p[k] for p in parts]) for k in ROW_KEYS}
        meta = np.array(parts[0]["meta"], dtype=float)
        meta[0] = sum(float(p["meta"][0]) for p in parts)   # solves
        meta[1] = sum(float(p["meta"][1]) for p in parts)   # VI seconds
        out = os.path.join(DATA_DIR, f"{args.out}_{pref}.npz")
        np.savez_compressed(out, **merged, meta=meta,
                            cost_weights=parts[0]["cost_weights"], pref=np.array(pref))
        print(f"{pref:<9s} -> {out}: {len(merged['state']):,} rows, "
              f"{len(np.unique(merged['field_seed']))} fields, {int(meta[0])} DP solves\n")


if __name__ == "__main__":
    main()
