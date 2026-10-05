"""
Weekly history store for the Medellín real-estate app.

Two append-only CSVs in ./artifacts/ (one row-block per ISO week, Monday date):

    history_prices.csv   week, kind, tipo, barrio, n, median_ppm2, p25_ppm2, p75_ppm2
    history_metrics.csv  week, kind, r2, oof_r2, mae, mape, coverage80, n_rows

Re-running inside the same week REPLACES that week's rows (idempotent), so
running the job twice never duplicates points.

Used by train.py (metrics every training run, prices when freshly scraped),
snapshot.py (lightweight weekly price job) and read by app.py.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path("artifacts")
PRICES_CSV = ARTIFACTS_DIR / "history_prices.csv"
METRICS_CSV = ARTIFACTS_DIR / "history_metrics.csv"
CITY = "Medellín (total)"      # citywide row; must match app.py

PRICE_COLS = ["week", "kind", "tipo", "barrio", "n",
              "median_ppm2", "p25_ppm2", "p75_ppm2"]
METRIC_COLS = ["week", "kind", "r2", "oof_r2", "mae", "mape",
               "coverage80", "n_rows"]


def week_start(d: dt.date | str | None = None) -> str:
    """Monday of the ISO week containing `d` (default: today), as YYYY-MM-DD."""
    if d is None:
        d = dt.date.today()
    elif isinstance(d, str):
        d = dt.date.fromisoformat(d)
    return (d - dt.timedelta(days=d.weekday())).isoformat()


def _write(path: Path, new: pd.DataFrame, cols: list[str]) -> None:
    """Append `new`, replacing any existing rows with the same (week, kind)."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        keys = new[["week", "kind"]].drop_duplicates()
        m = old.merge(keys, on=["week", "kind"], how="left", indicator=True)
        old = old[(m["_merge"] == "left_only").values]
        new = pd.concat([old, new], ignore_index=True)
    new[cols].sort_values(["week", "kind"]).to_csv(path, index=False)


def _agg(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (d.groupby(by)["ppm2"]
             .agg(n="count", median_ppm2="median",
                  p25_ppm2=lambda s: s.quantile(0.25),
                  p75_ppm2=lambda s: s.quantile(0.75))
             .reset_index())


def snapshot_prices(frames: dict[str, pd.DataFrame],
                    week: str | None = None) -> None:
    """
    frames: {"arr": df, "ven": df} with columns nombre, tipo, precio, area.
    Stores median / P25 / P75 price per m² for every barrio, for each tipo
    and for "all" types, plus a citywide row.
    """
    wk = week_start(week)
    parts = []
    for kind, df in frames.items():
        if df is None or df.empty:
            continue
        d = df[["nombre", "tipo", "precio", "area"]].dropna().copy()
        d = d[d["area"] > 0]
        d["ppm2"] = d["precio"] / d["area"]
        d = pd.concat([d, d.assign(nombre=CITY)], ignore_index=True)
        by_all = _agg(d, ["nombre"]).assign(tipo="all")
        by_tipo = _agg(d, ["nombre", "tipo"])
        out = pd.concat([by_all, by_tipo], ignore_index=True)
        out = out.rename(columns={"nombre": "barrio"})
        out["kind"], out["week"] = kind, wk
        parts.append(out)
    if not parts:
        return
    _write(PRICES_CSV, pd.concat(parts, ignore_index=True), PRICE_COLS)
    print(f"  History: price snapshot for week {wk} → {PRICES_CSV}")


def log_metrics(metrics: dict[str, dict], week: str | None = None) -> None:
    """metrics: {"arr": {r2, oof_r2, mae, mape, coverage80, n_rows}, "ven": …}"""
    wk = week_start(week)
    rows = []
    for kind, m in metrics.items():
        rows.append({"week": wk, "kind": kind,
                     **{c: m.get(c) for c in METRIC_COLS[2:]}})
    if not rows:
        return
    _write(METRICS_CSV, pd.DataFrame(rows), METRIC_COLS)
    print(f"  History: model metrics for week {wk} → {METRICS_CSV}")
