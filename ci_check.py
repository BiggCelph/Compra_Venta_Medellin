"""
Quality gate for the automated refresh. Run AFTER train.py, BEFORE committing.

Compares the freshly trained artifacts/ with the copy taken before training (artifacts_prev/).
Exits with code 1 (so the workflow stops and nothing is committed) if the scrape or the
training looks broken: missing files, far fewer listings than before, or a worse model.
"""
import pickle
import sys
from pathlib import Path

import pandas as pd

NEW, OLD = Path("artifacts"), Path("artifacts_prev")

MIN_ROWS = 500            # absolute floor per dataset
MIN_ROWS_RATIO = 0.60     # new rows must be >= 60% of the previous run
MIN_R2 = 0.50             # absolute floor (log-space R2, as saved by train.py)
MAX_R2_DROP = 0.05        # allowed drop versus the previous run

REQUIRED = [
    "arr_mede_final.csv", "ven_mede_final.csv", "model_r2.pkl", "list_barrios.pkl", "metro_stations.pkl",
    "stack_arr.pkl", "stack_ven.pkl", "q10_arr.pkl", "q90_arr.pkl", "q10_ven.pkl", "q90_ven.pkl",
    "preprocessor_arr.pkl", "preprocessor_ven.pkl", "best_features_arr.pkl", "best_features_ven.pkl",
    "barrio_te_arr.pkl", "barrio_te_ven.pkl",
    "price_per_m2_arr.pkl", "price_per_m2_ven.pkl", "price_per_space_arr.pkl", "price_per_space_ven.pkl",
    "price_per_parking_arr.pkl", "price_per_parking_ven.pkl",
]


def n_rows(folder: Path, name: str):
    p = folder / name
    return len(pd.read_csv(p, usecols=[0])) if p.exists() else None


def r2(folder: Path):
    p = folder / "model_r2.pkl"
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)      # our own file, written by train.py


def main() -> int:
    problems = []

    missing = [n for n in REQUIRED if not (NEW / n).exists()]
    if missing:
        problems.append(f"missing files: {missing}")

    new_r2, old_r2 = r2(NEW), r2(OLD)
    print(f"{'':10}{'rows new':>10}{'rows old':>10}{'R2 new':>9}{'R2 old':>9}")
    for kind, csv in (("arr", "arr_mede_final.csv"), ("ven", "ven_mede_final.csv")):
        rn, ro = n_rows(NEW, csv), n_rows(OLD, csv)
        a, b = new_r2.get(kind), old_r2.get(kind)
        print(f"{kind:10}{str(rn):>10}{str(ro):>10}{(f'{a:.4f}' if a is not None else '-'):>9}"
              f"{(f'{b:.4f}' if b is not None else '-'):>9}")

        if rn is None or rn < MIN_ROWS:
            problems.append(f"{kind}: only {rn} rows (minimum {MIN_ROWS})")
        elif ro and rn < MIN_ROWS_RATIO * ro:
            problems.append(
                f"{kind}: rows dropped from {ro} to {rn} (< {MIN_ROWS_RATIO:.0%}) - scrape probably incomplete")
        if not a or a < MIN_R2:
            problems.append(f"{kind}: R2 {a} is below {MIN_R2}")
        elif b and a < b - MAX_R2_DROP:
            problems.append(
                f"{kind}: R2 fell from {b:.4f} to {a:.4f} (more than {MAX_R2_DROP})")

    if problems:
        print("\nQUALITY GATE FAILED - nothing will be committed:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
