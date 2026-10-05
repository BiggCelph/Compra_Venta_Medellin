"""
Rentabilidad por barrio / Rental yield by barrio  —  self-contained Streamlit page.

Put this file in a folder called `pages/` next to app.py. Streamlit adds it to the
sidebar automatically: NO changes to app.py, no imports, no variable names to match.
It loads artifacts/arr_mede_final.csv, artifacts/ven_mede_final.csv and medellin.geojson
by itself.

gross yield % = (median rent per m2 * 12) / median sale price per m2 * 100
"""
import json
import math
import re
import traceback
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rentabilidad · Medellín",
                   page_icon="💰", layout="wide")

T = {
    "en": dict(
        title="Rental yield by barrio",
        intro=("**What we're doing:** using the rent and sale listings scraped from metrocuadrado to find in which "
               "Medellín barrios buying to rent pays off most. For each barrio we compare the median price per m² of "
               "rent vs sale and turn it into a gross annual yield (annual rent ÷ purchase price).  \n"
               "**How to use it:** raise or lower the minimum listings, pick a barrio to zoom the map into it, "
               "and use the calculator to compare a mortgage payment against the rent."),
        poly_sum="Polygons: {ex} official by name · {sp} official by location · {fz} similar name · {ap} approximate",
        poly_odd="Barrios not matched by exact name",
        gray_note=("Gray polygons have too few listings (fewer than {n} for rent or for sale) to compute a reliable "
                   "yield. Hover to see their counts, or lower the minimum above."),
        few="not enough listings",
        map_file="Map file: {f} · {n} polygons · names read from column '{c}'",
        map_fix="{d} polygons with empty geometry were ignored and {x} invalid ones repaired",
        map_warn=("med.shp was not found, so medellin.geojson is being used and it may be missing barrios. "
                  "Copy med.shp (with its .dbf, .shx and .prj files) into the project folder."),
        poly_title="Where each polygon comes from and how big it is",
        poly_cols=dict(barrio="Barrio", polygon="Polygon", km2="Area (km²)"),
        poly_note=("official by name = from the map file · by location = the official polygon that contains most of that barrio's listings · similar name = matched to a polygon with a slightly different name · "
                   "approximate = area drawn around that barrio's listings, ignoring far-away outliers (not the official boundary). "
                   "Table: the 15 largest areas — check these first."),
        focus="Zoom into a barrio", all_b="All barrios", rank="Rank", calc_for="Calculating for",
        min_n="Minimum listings per barrio (rent and sale)", tipo="Property type", all="All",
        best="Highest yield", best_y="Top yield", median="City median yield", n_b="Barrios shown",
        top="Top barrios by gross yield", scatter="Price per m² vs yield", map="Yield map",
        table="All barrios", calc="Rent vs buy calculator", barrio="Barrio", area="Area (m²)",
        down="Down payment (%)", rate="Mortgage rate (% effective annual)", years="Term (years)",
        price="Estimated price", downamt="Down payment", rent="Estimated rent / month",
        pay="Mortgage / month", ratio="Mortgage vs rent",
        more="Buying costs about {p:.0f}% more per month than renting here.",
        less="The mortgage payment is about {p:.0f}% below the rent here.",
        caveat=("Based on asking prices and asking rents, not closed deals. Gross yield ignores vacancy, "
                "taxes, admin fees, maintenance and price changes. The mortgage rate is an assumption — set your own."),
        no_data="Not enough listings with these filters. Lower the minimum.",
        cols=dict(barrio="Barrio", yield_pct="Gross yield %", payback_yrs="Payback (yrs)",
                  ppm2_ven="Sale $/m²", ppm2_arr="Rent $/m²", n_ven="# sale", n_arr="# rent"),
    ),
    "es": dict(
        title="Rentabilidad por barrio",
        intro=("**Qué estamos haciendo:** usamos los anuncios de arriendo y venta de metrocuadrado para ver en qué "
               "barrios de Medellín rinde más comprar para arrendar. Para cada barrio comparamos la mediana de precio "
               "por m² de arriendo y de venta y la convertimos en rentabilidad bruta anual (arriendo anual ÷ precio de compra).  \n"
               "**Cómo usarla:** sube o baja el mínimo de anuncios, elige un barrio para acercar el mapa y usa la "
               "calculadora para comparar una cuota hipotecaria con el arriendo."),
        poly_sum="Polígonos: {ex} oficiales por nombre · {sp} oficiales por ubicación · {fz} por nombre parecido · {ap} aproximados",
        poly_odd="Barrios sin coincidencia exacta de nombre",
        gray_note=("Los polígonos grises tienen muy pocos anuncios (menos de {n} en arriendo o en venta) para calcular "
                   "una rentabilidad confiable. Pasa el mouse para ver cuántos hay, o baja el mínimo de arriba."),
        few="pocos anuncios",
        map_file="Archivo del mapa: {f} · {n} polígonos · nombres leídos de la columna '{c}'",
        map_fix="{d} polígonos con geometría vacía se ignoraron y {x} inválidos se repararon",
        map_warn=("No se encontró med.shp, así que se usa medellin.geojson, que puede tener barrios faltantes. "
                  "Copia med.shp (con sus archivos .dbf, .shx y .prj) a la carpeta del proyecto."),
        poly_title="De dónde sale cada polígono y qué tan grande es",
        poly_cols=dict(barrio="Barrio", polygon="Polígono", km2="Área (km²)"),
        poly_note=("oficial por nombre = del archivo del mapa · por ubicación = el polígono oficial que contiene la mayoría de los anuncios del barrio · nombre parecido = emparejado con un polígono de nombre casi igual · "
                   "aproximado = área dibujada alrededor de los anuncios del barrio, sin contar los que quedan lejos (no es el límite oficial). "
                   "Tabla: las 15 áreas más grandes — revisa esas primero."),
        focus="Acercar un barrio", all_b="Todos los barrios", rank="Puesto", calc_for="Calculando para",
        min_n="Mínimo de anuncios por barrio (arriendo y venta)", tipo="Tipo de inmueble", all="Todos",
        best="Mayor rentabilidad", best_y="Rentabilidad máx.", median="Mediana de la ciudad", n_b="Barrios mostrados",
        top="Barrios con mayor rentabilidad bruta", scatter="Precio por m² vs rentabilidad", map="Mapa de rentabilidad",
        table="Todos los barrios", calc="Calculadora arriendo vs compra", barrio="Barrio", area="Área (m²)",
        down="Cuota inicial (%)", rate="Tasa hipotecaria (% efectiva anual)", years="Plazo (años)",
        price="Precio estimado", downamt="Cuota inicial", rent="Arriendo estimado / mes",
        pay="Cuota hipotecaria / mes", ratio="Cuota vs arriendo",
        more="Comprar cuesta cerca de {p:.0f}% más al mes que arrendar aquí.",
        less="La cuota queda cerca de {p:.0f}% por debajo del arriendo aquí.",
        caveat=("Basado en precios y arriendos publicados, no en cierres de negocio. La rentabilidad bruta no descuenta "
                "vacancia, impuestos, administración, mantenimiento ni valorización. La tasa hipotecaria es un supuesto: ajústala."),
        no_data="No hay suficientes anuncios con estos filtros. Baja el mínimo.",
        cols=dict(barrio="Barrio", yield_pct="Rentab. bruta %", payback_yrs="Retorno (años)",
                  ppm2_ven="Venta $/m²", ppm2_arr="Arriendo $/m²", n_ven="# venta", n_arr="# arriendo"),
    ),
}

NAME_COLS = ["nombre", "NOMBRE", "name", "NAME", "barrio", "BARRIO"]


# ── helpers ──────────────────────────────────────────────────────────────────
def norm(s) -> str:
    s = unicodedata.normalize("NFD", str(s).lower().strip())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def fmt(v) -> str:
    if v is None or v != v:
        return "—"
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"


def mortgage_payment(principal, ea_rate, years):
    n = int(years * 12)
    if principal <= 0 or n <= 0:
        return 0.0
    i = (1 + ea_rate) ** (1 / 12) - 1
    return principal / n if i == 0 else principal * i / (1 - (1 + i) ** -n)


def find_file(name):
    here = Path(__file__).resolve().parent
    for root in (Path.cwd(), here.parent, here):
        for sub in ("artifacts", "."):
            p = root / sub / name
            if p.exists():
                return p
    return None


@st.cache_data(ttl=3600)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_geo(path: str):
    import geopandas as gpd
    return gpd.read_file(path)


def compute_yield(arr, ven, min_listings=10, tipo=None) -> pd.DataFrame:
    def agg(df):
        d = df.copy()
        if tipo and "tipo" in d.columns:
            d = d[d["tipo"].astype(str).str.lower() == tipo.lower()]
        d = d[(d["area"] > 0) & (d["precio"] > 0)]
        d["ppm2"] = d["precio"] / d["area"]
        return d.groupby("nombre").agg(ppm2=("ppm2", "median"), precio=("precio", "median"), n=("ppm2", "size"))

    out = agg(arr).join(agg(ven), lsuffix="_arr", rsuffix="_ven", how="inner")
    out = out[(out["n_arr"] >= min_listings) & (
        out["n_ven"] >= min_listings)].copy()
    out["yield_pct"] = out["ppm2_arr"] * 12 / out["ppm2_ven"] * 100
    out = out[out["yield_pct"].between(0.01, 100)].copy()
    out["payback_yrs"] = 100 / out["yield_pct"]
    out = out.reset_index().rename(columns={"nombre": "barrio"})
    return out.sort_values("yield_pct", ascending=False).reset_index(drop=True)


def clean_nombre(s) -> str:
    """Same cleanup train.py applies to the shapefile names."""
    s = str(s).strip()
    for pfx in ("Área de Expansión", "Area de Expansion"):
        s = s.replace(pfx, "").strip(" -–—")
    return s


def key(s) -> str:
    """Accent-free, lowercase, punctuation and repeated spaces removed."""
    return re.sub(r"[^a-z0-9]+", " ", norm(clean_nombre(s))).strip()


def build_polygons(gdf, barrios, pts=None):
    """
    One polygon set per barrio, from the SAME shapefile train.py used (med.shp), so names match exactly.
      1. exact   - polygon(s) whose cleaned, accent-free name matches. If several polygons share the name
                   (e.g. a barrio and its 'Área de Expansión'), keep only those that really contain its listings.
      2. fuzzy   - an unused polygon with an almost identical name
      3. approx  - SMALL circle at the median of the barrio's listings (never a convex hull: a few
                   mis-located listings used to blow the hull up to a huge area)
    Returns (geojson, bounds {key: (minx, miny, maxx, maxy)}, report {barrio: {"how", "km2"}}).
    """
    import difflib

    import geopandas as gpd
    import numpy as np
    from shapely.geometry import Point

    from shapely.ops import unary_union

    wanted = {key(b): b for b in barrios}
    how, parts, g = {}, [], None
    meta = {"n": 0 if gdf is None else len(
        gdf), "dropped": 0, "fixed": 0, "name_col": None}

    # listing coordinates per wanted barrio
    pts_by_key = {}
    if pts is not None:
        q = pts.assign(_k=pts["nombre"].map(key)).dropna(
            subset=["latitud", "longitud"])
        pts_by_key = {k: grp[["longitud", "latitud"]]
                      for k, grp in q.groupby("_k") if k in wanted}

    if gdf is not None:
        best = detect_name_col(gdf, set(wanted))
        g = gdf[[gdf.geometry.name] + ([best] if best else [])].copy()
        if g.geometry.name != "geometry":
            g = g.rename_geometry("geometry")
        if g.crs is not None:
            g = g.to_crs("EPSG:4326")
        meta["name_col"] = best
        # null / empty polygons can't be drawn
        bad_geom = g.geometry.isna() | g.geometry.is_empty
        meta["dropped"] = int(bad_geom.sum())
        g = g[~bad_geom].copy()
        # e.g. self-intersecting outlines
        invalid = ~g.geometry.is_valid
        meta["fixed"] = int(invalid.sum())
        if invalid.any():
            g.loc[invalid, "geometry"] = g.loc[invalid].geometry.buffer(0)
            g = g[~g.geometry.is_empty].copy()
        g["_key"] = g[best].map(key) if best else ""
        g["_name"] = g[best].astype(str) if best else ""
        g["_row"] = range(len(g))

    def pick(rows, k):
        """Several polygons with the same name -> keep the ones that contain its listings."""
        q = pts_by_key.get(k)
        if len(rows) == 1 or q is None or q.empty:
            return rows
        P = gpd.GeoSeries(gpd.points_from_xy(
            q["longitud"], q["latitud"]), crs="EPSG:4326")
        counts = [int(P.within(geom).sum()) for geom in rows.geometry]
        total = sum(counts)
        if total == 0:
            return rows
        keep = [i for i, c in enumerate(counts) if c / total >= 0.10]
        return rows.iloc[keep]

    have = set(g["_key"]) if g is not None else set()
    for k, b in wanted.items():
        if k in have:
            parts.append(pick(g[g["_key"] == k][["_key", "geometry"]], k))
            how[b] = "exact"

    missing = [k for k in wanted if k not in have]

    # no name match -> the official polygon that contains most of the barrio's listings
    if g is not None and missing and pts_by_key:
        frames = [gpd.GeoDataFrame({"_k": k}, index=range(len(pts_by_key[k])),
                                   geometry=gpd.points_from_xy(
                                       pts_by_key[k]["longitud"], pts_by_key[k]["latitud"]),
                                   crs="EPSG:4326")
                  for k in missing if k in pts_by_key and len(pts_by_key[k])]
        if frames:
            P = gpd.GeoDataFrame(
                pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")
            J = gpd.sjoin(P, g[["_row", "geometry"]],
                          how="left").dropna(subset=["_row"])
            cnt = J.groupby(["_k", "_row"]).size()
            done = set(cnt.index.get_level_values(0))
            for k in list(missing):
                if k not in done:
                    continue
                sub = cnt.loc[k]
                if sub.max() / len(pts_by_key[k]) >= 0.5:
                    row = g[g["_row"] == sub.idxmax()]
                    parts.append(row[["_key", "geometry"]].assign(_key=k))
                    how[wanted[k]] = f"spatial -> {row['_name'].iloc[0]}"
                    missing.remove(k)

    if g is not None and missing:
        # never steal an already matched polygon
        unused = sorted(have - set(wanted))
        for k in list(missing):
            close = difflib.get_close_matches(k, unused, n=1, cutoff=0.82)
            if close:
                part = pick(g[g["_key"] == close[0]]
                            [["_key", "geometry"]].copy(), k)
                part = part.assign(_key=k)
                parts.append(part)
                how[wanted[k]] = f"fuzzy -> {close[0]}"
                missing.remove(k)

    def blob(q):
        """Shape of the listings themselves; far-away (mis-located) ones are ignored."""
        q = q.head(300)
        cx, cy = q["longitud"].median(), q["latitud"].median()
        d = np.hypot((q["longitud"] - cx) *
                     np.cos(np.radians(cy)), q["latitud"] - cy)   # degrees
        core = q[d <= min(max(np.percentile(d, 80), 0.002), 0.012)
                 ]                       # <= ~1.3 km
        if len(core) < 3:
            return Point(cx, cy).buffer(0.0015)
        shape = unary_union([Point(x, y).buffer(0.0009)
                            for x, y in zip(core["longitud"], core["latitud"])])
        # close small gaps
        return shape.buffer(0.0003).buffer(-0.0003)

    rows = []
    for k in missing:
        q = pts_by_key.get(k)
        poly = blob(q) if q is not None and len(q) else None
        how[wanted[k]] = "approx" if poly is not None else "none"
        if poly is not None:
            rows.append({"_key": k, "geometry": poly})
    if rows:
        parts.append(gpd.GeoDataFrame(
            rows, geometry="geometry", crs="EPSG:4326"))

    if not parts:
        return None, {}, {b: {"how": h, "km2": None} for b, h in how.items()}, meta
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                           geometry="geometry", crs="EPSG:4326")
    # ONE feature per barrio (duplicate ids draw badly)
    out = out.dissolve(by="_key", as_index=False)
    out["geometry"] = out.geometry.simplify(
        0.0001, preserve_topology=True)      # ~11 m, lighter map

    try:                                             # area per barrio, in km2
        km2 = (out.to_crs("EPSG:3116").geometry.area.groupby(
            out["_key"].values).sum() / 1e6).to_dict()
    except Exception:
        km2 = {}
    bb = out.bounds.assign(_key=out["_key"].values).groupby("_key").agg(
        minx=("minx", "min"), miny=("miny", "min"), maxx=("maxx", "max"), maxy=("maxy", "max"))
    bounds = {k: (r.minx, r.miny, r.maxx, r.maxy) for k, r in bb.iterrows()}
    report = {b: {"how": h, "km2": (round(km2[key(b)], 3) if key(
        b) in km2 else None)} for b, h in how.items()}
    return json.loads(out.to_json()), bounds, report, meta


@st.cache_data(ttl=3600, show_spinner=False)
def polygons(geo_path: str, arr_path: str, ven_path: str, barrios: tuple):
    gdf = None
    if geo_path:
        try:
            gdf = load_geo(geo_path)
        except Exception:
            gdf = None
    frames = []
    for pth in (arr_path, ven_path):
        d = load_csv(pth)
        if {"nombre", "latitud", "longitud"} <= set(d.columns):
            frames.append(d[["nombre", "latitud", "longitud"]])
    return build_polygons(gdf, list(barrios), pd.concat(frames) if frames else None)


def detect_name_col(gdf, keys):
    """The text column of the map file whose values match the most of our barrio names."""
    best, best_n = None, 0
    for c in gdf.columns:
        if c == gdf.geometry.name or not (pd.api.types.is_string_dtype(gdf[c])
                                          or pd.api.types.is_object_dtype(gdf[c])):
            continue
        n = len(set(gdf[c].dropna().map(key)) & set(keys))
        if n > best_n:
            best, best_n = c, n
    return best


def coverage(arr, ven, tipo=None) -> pd.DataFrame:
    """Listing counts (rent / sale) for EVERY barrio, keyed by key(nombre)."""
    def cnt(df):
        d = df
        if tipo and "tipo" in d.columns:
            d = d[d["tipo"].astype(str).str.lower() == tipo.lower()]
        return d[(d["area"] > 0) & (d["precio"] > 0)].groupby("nombre").size()

    c = pd.concat({"n_arr": cnt(arr), "n_ven": cnt(ven)},
                  axis=1).fillna(0).astype(int)
    c.index = c.index.map(key)
    return c.groupby(level=0).sum()


@st.cache_data(ttl=3600, show_spinner=False)
def base_polygons(geo_path: str, barrios: tuple):
    """Every polygon of the map file (one feature each, unique _id) for the gray background layer."""
    if not geo_path:
        return None
    try:
        gdf = load_geo(geo_path)
    except Exception:
        return None
    col = detect_name_col(gdf, {key(b) for b in barrios})
    g = gdf[[gdf.geometry.name] + ([col] if col else [])].copy()
    if g.geometry.name != "geometry":
        g = g.rename_geometry("geometry")
    if g.crs is not None:
        g = g.to_crs("EPSG:4326")
    g = g[~(g.geometry.isna() | g.geometry.is_empty)].copy()
    invalid = ~g.geometry.is_valid
    if invalid.any():
        g.loc[invalid, "geometry"] = g.loc[invalid].geometry.buffer(0)
    g["_name"] = g[col].astype(str) if col else ""
    g["_key"] = g["_name"].map(key)
    g["_id"] = [str(i) for i in range(len(g))]
    g["geometry"] = g.geometry.simplify(0.0001, preserve_topology=True)
    return json.loads(g[["_id", "_key", "_name", "geometry"]].to_json())


def view_for(b, width_px=900, height_px=520):
    """Map centre + zoom that frames a bounding box (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = b
    dlon = max((maxx - minx) * 1.35, 1e-4)          # 35% margin
    dlat = max((maxy - miny) * 1.35, 1e-4)
    z = min(math.log2(width_px * 360 / (512 * dlon)),
            math.log2(height_px * 360 / (512 * dlat)))
    return dict(lat=(miny + maxy) / 2, lon=(minx + maxx) / 2), max(10.3, min(z, 16.0))


# ── page ─────────────────────────────────────────────────────────────────────
def main():
    import plotly.express as px
    import plotly.graph_objects as go

    default_lang = st.session_state.get("lang", "es")
    lang = st.radio("Idioma / Language", ["es", "en"], horizontal=True,
                    index=0 if default_lang != "en" else 1, key="yld_lang")
    t = T[lang]
    st.markdown(f"## {t['title']}")
    st.info(t["intro"])

    p_arr, p_ven = find_file(
        "arr_mede_final.csv"), find_file("ven_mede_final.csv")
    if not p_arr or not p_ven:
        st.error("No encuentro arr_mede_final.csv / ven_mede_final.csv. "
                 f"Busqué en: {Path.cwd()} y en la carpeta del proyecto (artifacts/).")
        return
    arr, ven = load_csv(str(p_arr)), load_csv(str(p_ven))
    for name, d in (("arriendo", arr), ("venta", ven)):
        missing = {"nombre", "precio", "area"} - set(d.columns)
        if missing:
            st.error(
                f"'{name}' no tiene las columnas {sorted(missing)}. Columnas: {list(d.columns)[:15]}")
            return

    f1, f2 = st.columns(2)
    with f1:
        min_n = st.slider(t["min_n"], 3, 40, 10, key="yld_min_n")
    with f2:
        tipos = [t["all"]] + (sorted(arr["tipo"].dropna().astype(str).str.lower().unique())
                              if "tipo" in arr.columns else [])
        tipo_sel = st.selectbox(t["tipo"], tipos, key="yld_tipo")
    tipo_f = None if tipo_sel == t["all"] else tipo_sel
    df = compute_yield(arr, ven, min_n, tipo_f)
    if df.empty:
        st.warning(t["no_data"])
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t["best"], str(df.loc[0, "barrio"]))
    k2.metric(t["best_y"], f"{df.loc[0, 'yield_pct']:.1f}%")
    k3.metric(t["median"], f"{df['yield_pct'].median():.1f}%")
    k4.metric(t["n_b"], f"{len(df)}")

    left, right = st.columns(2)
    with left:
        st.markdown(f"**{t['top']}**")
        fig = px.bar(df.head(15), x="yield_pct", y="barrio", orientation="h",
                     color_discrete_sequence=["#14b8a6"], labels={"yield_pct": "%", "barrio": ""})
        fig.update_layout(yaxis=dict(autorange="reversed"),
                          height=460, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown(f"**{t['scatter']}**")
        sc = df.assign(n_total=df["n_arr"] + df["n_ven"])
        fig2 = px.scatter(sc, x="ppm2_ven", y="yield_pct", size="n_total", hover_name="barrio",
                          color_discrete_sequence=["#14b8a6"],
                          labels={"ppm2_ven": t["cols"]["ppm2_ven"], "yield_pct": t["cols"]["yield_pct"]})
        fig2.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # focus on one barrio: stats + zoom the map into it
    focus = st.selectbox(t["focus"], [t["all_b"]] +
                         sorted(df["barrio"]), key="yld_focus")
    focus_key = None if focus == t["all_b"] else key(focus)
    if focus_key:
        fr = df[df["barrio"] == focus].iloc[0]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric(t["cols"]["yield_pct"], f"{fr['yield_pct']:.2f}%")
        s2.metric(
            t["rank"], f"{int(df.index[df['barrio'] == focus][0]) + 1} / {len(df)}")
        s3.metric(t["cols"]["ppm2_ven"], fmt(fr["ppm2_ven"]))
        s4.metric(t["cols"]["ppm2_arr"], fmt(fr["ppm2_arr"]))

    # map - every barrio gets a polygon (official, name-matched or approximate)
    p_geo = find_file("med.shp") or find_file(
        "medellin.geojson")   # med.shp = what train.py used
    try:
        gj, bounds, report, meta = polygons(str(p_geo) if p_geo else "", str(p_arr), str(p_ven),
                                            tuple(sorted(df["barrio"])))
        if gj:
            st.markdown(f"**{t['map']}**")
            m = df.assign(_key=df["barrio"].map(key))
            m["label"] = m["barrio"] + m["barrio"].map(
                lambda x: " (aprox.)" if report.get(x, {}).get("how") == "approx" else "")
            new_api = hasattr(go, "Choroplethmap")
            Trace = go.Choroplethmap if new_api else go.Choroplethmapbox
            common = dict(geojson=gj, featureidkey="properties._key", colorscale="Teal",
                          zmin=float(m["yield_pct"].min()), zmax=float(m["yield_pct"].max()))
            fig3 = go.Figure()
            base = None
            if p_geo:      # gray background: EVERY polygon of the map file, incl. barrios with too few listings
                base = base_polygons(str(p_geo), tuple(sorted(
                    pd.concat([arr["nombre"], ven["nombre"]]).dropna().unique())))
            if base:
                cov, ok_keys, ids, texts = coverage(
                    arr, ven, tipo_f), set(m["_key"]), [], []
                for f in base["features"]:
                    pr = f["properties"]
                    ids.append(pr["_id"])
                    if pr["_key"] in ok_keys:
                        texts.append(pr["_name"])
                    else:
                        a_, v_ = ((int(cov.loc[pr["_key"], "n_arr"]), int(cov.loc[pr["_key"], "n_ven"]))
                                  if pr["_key"] in cov.index else (0, 0))
                        texts.append(f"{pr['_name']}<br>{t['cols']['n_arr']}: {a_} · "
                                     f"{t['cols']['n_ven']}: {v_}<br>{t['few']}")
                fig3.add_trace(Trace(
                    geojson=base, featureidkey="properties._id", locations=ids, z=[0] * len(ids),
                    colorscale=[[0, "#d5dae1"], [1, "#d5dae1"]], zmin=0, zmax=1, showscale=False,
                    marker_opacity=0.55, marker_line_width=0.4, marker_line_color="#8b95a5",
                    text=texts, hovertemplate="%{text}<extra></extra>"))
            fig3.add_trace(Trace(
                locations=m["_key"], z=m["yield_pct"], text=m["label"],
                marker_opacity=0.3 if focus_key else 0.75, marker_line_width=0.5,
                hovertemplate="%{text}<br>%{z:.2f}%<extra></extra>", colorbar=dict(title="%"), **common))
            center, zoom = dict(lat=6.25, lon=-75.575), 10.3
            if focus_key:
                one = m[m["_key"] == focus_key]
                if focus_key in bounds and not one.empty:
                    fig3.add_trace(Trace(
                        locations=one["_key"], z=one["yield_pct"], text=one["label"],
                        marker_opacity=0.95, marker_line_width=3, marker_line_color="#ffffff",
                        showscale=False, hovertemplate="%{text}<br>%{z:.2f}%<extra></extra>", **common))
                    center, zoom = view_for(bounds[focus_key])
                else:
                    st.caption(f"{focus}: sin polígono / no polygon")
            fig3.update_layout(
                **{"map" if new_api else "mapbox": dict(style="carto-positron", zoom=zoom, center=center)},
                margin=dict(l=0, r=0, t=0, b=0), height=520)
            st.plotly_chart(fig3, use_container_width=True)
            if base:
                st.caption(t["gray_note"].format(n=min_n))

            rep_df = pd.DataFrame([{"barrio": b, "polygon": v["how"], "km2": v["km2"]}
                                   for b, v in report.items()])
            how = rep_df["polygon"]
            st.caption(t["poly_sum"].format(ex=int((how == "exact").sum()),
                                            sp=int(how.str.startswith(
                                                "spatial").sum()),
                                            fz=int(how.str.startswith(
                                                "fuzzy").sum()),
                                            ap=int((how == "approx").sum())))
            if p_geo and p_geo.name != "med.shp":
                st.warning(t["map_warn"])
            with st.expander(t["poly_title"]):
                st.caption(t["map_file"].format(f=p_geo.name if p_geo else "—", n=meta["n"],
                                                c=meta["name_col"] or "—"))
                if meta["dropped"] or meta["fixed"]:
                    st.caption(t["map_fix"].format(
                        d=meta["dropped"], x=meta["fixed"]))
                st.caption(t["poly_note"])
                odd = rep_df[rep_df["polygon"] != "exact"]
                if len(odd):
                    st.markdown(f"**{t['poly_odd']}**")
                    st.dataframe(odd.rename(
                        columns=t["poly_cols"]), use_container_width=True, hide_index=True)
                st.dataframe(rep_df.sort_values("km2", ascending=False).head(15).rename(columns=t["poly_cols"]),
                             use_container_width=True, hide_index=True)
    except Exception as e:
        st.caption(f"Map unavailable: {type(e).__name__}: {e}")

    st.markdown(f"**{t['table']}**")
    show = df[["barrio", "yield_pct", "payback_yrs",
               "ppm2_ven", "ppm2_arr", "n_ven", "n_arr"]].copy()
    show["yield_pct"], show["payback_yrs"] = show["yield_pct"].round(
        2), show["payback_yrs"].round(1)
    show["ppm2_ven"], show["ppm2_arr"] = show["ppm2_ven"].map(
        fmt), show["ppm2_arr"].map(fmt)
    st.dataframe(show.rename(
        columns=t["cols"]), use_container_width=True, hide_index=True, height=320)

    st.markdown(f"### {t['calc']}")
    q1, q2, q3, q4, q5 = st.columns(5)
    with q1:
        if focus_key:
            b = focus
            st.markdown(f"**{t['calc_for']}**  \n{focus}")
        else:
            b = st.selectbox(t["barrio"], sorted(
                df["barrio"]), key="yld_calc_barrio")
    with q2:
        area = st.number_input(t["area"], 20, 500, 80, key="yld_calc_area")
    with q3:
        down = st.slider(t["down"], 10, 60, 30, key="yld_calc_down")
    with q4:
        rate = st.slider(t["rate"], 6.0, 20.0, 12.0, 0.25, key="yld_calc_rate")
    with q5:
        years = st.slider(t["years"], 5, 30, 20, key="yld_calc_years")
    row = df[df["barrio"] == b].iloc[0]
    price, rent = row["ppm2_ven"] * area, row["ppm2_arr"] * area
    pay = mortgage_payment(price * (1 - down / 100), rate / 100, years)
    ratio = pay / rent if rent else float("nan")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric(t["price"], fmt(price))
    r2.metric(t["downamt"], fmt(price * down / 100))
    r3.metric(t["rent"], fmt(rent))
    r4.metric(t["pay"], fmt(pay))
    r5.metric(t["ratio"], f"{ratio * 100:.0f}%" if ratio == ratio else "—")
    if ratio == ratio:
        st.info(t["more"].format(p=(ratio - 1) * 100) if ratio >
                1 else t["less"].format(p=(1 - ratio) * 100))
    st.caption(t["caveat"])


try:
    main()
except Exception:
    st.error(
        "Error en la página de rentabilidad / Yield page error — copia este texto:")
    st.code(traceback.format_exc())
