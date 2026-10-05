"""
Weekly price snapshot (no model training).

Scrapes metrocuadrado, cleans with the same pipeline as train.py and appends
the median / P25 / P75 price per m² per barrio to artifacts/history_prices.csv.

    export MC_PUBLIC_KEY=...
    python snapshot.py                         # scrape + snapshot (this week)
    python snapshot.py --week-of 2026-09-28    # label it with another week
    python snapshot.py --from-final            # seed from artifacts/*_final.csv
                                               # (no scraping; use --week-of
                                               #  with the date that data is from)

Schedule it weekly, e.g. cron:   0 6 * * 1  cd /path/to/app && python snapshot.py
"""
import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

import history
from train import ARTIFACTS_DIR, build_url_map, clean, scrape_all


def _final(name: str) -> pd.DataFrame:
    p = ARTIFACTS_DIR / name
    return pd.read_csv(p if p.exists() else Path(name))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--week-of", default=None,
                    help="any date in the week, YYYY-MM-DD")
    ap.add_argument("--from-final", action="store_true",
                    help="use the existing *_mede_final.csv instead of scraping")
    a = ap.parse_args()

    if a.from_final:
        arr, ven = _final("arr_mede_final.csv"), _final("ven_mede_final.csv")
    else:
        geo = gpd.read_file("med.shp")
        arr_raw, ven_raw = scrape_all("arriendo"), scrape_all("venta")
        url_map = build_url_map(arr_raw, ven_raw, geo)
        arr = clean(arr_raw, geo, "arriendo", url_map)
        ven = clean(ven_raw, geo, "venta", url_map)

    history.snapshot_prices({"arr": arr, "ven": ven}, a.week_of)
