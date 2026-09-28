"""
Medellín Real Estate — Opportunities · Price check · Predictor
Author: Roman Alejandro Correa

Sections
    1. Opportunities   listings priced >X% below the model, with a simple map
                       (selected property + metro stations, malls, supermarkets)
    2. Price check     type a barrio -> rent and sale price per m²
    3. Predictor       model estimate for a property

Needs (from train.py, in ./artifacts/ or ./):
    arr_mede_final.csv, ven_mede_final.csv, preprocessor_*.pkl, stack_*.pkl,
    best_features_*.pkl, barrio_te_*.pkl, price_per_*.pkl, list_barrios.pkl,
    metro_stations.pkl (optional), q10/q90_*.pkl (optional), model_r2.pkl (optional)
"""
from __future__ import annotations

import datetime
import difflib
import json
import pickle
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:  # optional: per-property explanations
    import shap
except Exception:
    shap = None

st.set_page_config(page_title="Propiedades Medellín", page_icon="🏙️",
                   layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Translations
# ─────────────────────────────────────────────────────────────────────────────
TR = {
    "es": {
        "nav_opp": "🎯 Oportunidades",
        "nav_price": "💰 Precio por barrio",
        "nav_pred": "🔮 Predictor",
        "kind_arr": "Arriendo",
        "kind_ven": "Venta",
        "all": "Todos",
        "barrio": "Barrio",
        "tipo": "Tipo",
        "tipo_apartamento": "Apartamento",
        "tipo_casa": "Casa",
        "disclaimer": "Los barrios se asignan por ubicación geográfica y pueden no "
                      "coincidir con el barrio publicado en el anuncio. Verifica "
                      "siempre en el link del anuncio.",
        "data_updated": "Datos actualizados",
        "model_r2": "R² del modelo",
        "load_error": "Error cargando datos o modelos",
        # opportunities
        "opp_title": "Oportunidades",
        "opp_intro": "Propiedades cuyo precio está por debajo del valor estimado por el modelo.",
        "oos_note": "El precio del modelo es una predicción fuera de muestra: el modelo nunca vio ese anuncio al entrenar.",
        "min_disc": "Descuento mínimo (%)",
        "min_price": "Precio mínimo (COP)",
        "max_price": "Precio máximo (COP)",
        "zero_no_limit": "0 = sin límite",
        "opp_found": "{n:,} propiedades encontradas",
        "no_opp": "No hay propiedades con estos filtros.",
        "col_barrio": "Barrio",
        "col_tipo": "Tipo",
        "col_price": "Precio real",
        "col_model": "Precio modelo",
        "col_disc": "Descuento",
        "col_area": "Área m²",
        "col_ppm2": "Precio/m²",
        "col_link": "Link",
        "link_text": "Ver →",
        "open_listing": "Abrir anuncio",
        "select_row": "Selecciona una fila para ver la ubicación de la propiedad.",
        "map_prop": "Propiedad",
        "map_opps": "Oportunidades",
        "map_metro": "Estaciones (metro/tranvía/cable)",
        "map_mall": "Centros comerciales",
        "map_super": "Supermercados",
        "nearest_metro": "Estación más cercana: {name} ({d:.2f} km)",
        "poi_error": "No se pudieron cargar centros comerciales y supermercados (OpenStreetMap no respondió).",
        "poi_none": "Sin centros comerciales ni supermercados mapeados en 1.5 km.",
        "model_short": "modelo",
        # price check
        "price_title": "Precio por m² por barrio",
        "price_intro": "Escribe un barrio y obtén el precio por m² en arriendo y en venta.",
        "price_input": "Barrio",
        "price_placeholder": "Ej: Laureles, Poblado, Belén…",
        "price_prompt": "Escribe el nombre de un barrio.",
        "price_pick": "Varias coincidencias, elige una",
        "price_no_match": "No encontré ese barrio.",
        "price_suggest": "¿Quisiste decir? ",
        "per_m2_arr": "COP por m² al mes",
        "per_m2_ven": "COP por m²",
        "range_line": "Rango típico (P25–P75): {lo} – {hi} · {n} anuncios",
        "few": "Pocos anuncios, toma el dato con cautela.",
        "no_data": "Sin anuncios en este barrio.",
        # predictor
        "pred_title": "Predictor de precios",
        "pred_intro": "Estima el precio de arriendo o venta de una propiedad.",
        "transaction": "Transacción",
        "area": "Área (m²)",
        "rooms": "Habitaciones",
        "baths": "Baños",
        "parking": "Parqueaderos",
        "estrato": "Estrato",
        "admin": "Administración mensual (COP)",
        "metro_km": "Distancia al metro (km)",
        "predict_btn": "Predecir precio →",
        "est_price": "Precio estimado",
        "range": "Rango probable: {lo} — {hi}",
        "ppm2_yours": "Precio/m² estimado",
        "ppm2_barrio": "Mediana del barrio/m²",
        "comparables": "Propiedades similares en el barrio",
        "no_comps": "No hay comparables con estos filtros.",
        "no_barrio_data": "No hay datos de este barrio para {kind}.",
        "pred_prompt": "Completa el formulario y presiona «Predecir precio».",
        "app_title": "Propiedades Medellín",
        "app_tagline": "Precios de arriendo y venta, oportunidades y estimaciones con machine learning",
        "kpi_listings": "Anuncios analizados",
        "kpi_barrios": "Barrios",
        "kpi_rent": "Arriendo mediano/m²",
        "kpi_sale": "Venta mediana/m²",
        "choro_title": "Mapa de precio mediano por m²",
        "choro_note": "Solo barrios con al menos {n} anuncios. El barrio buscado está resaltado en rojo.",
        "why_title": "Factores principales del precio del modelo",
        "why_caption": "Efecto aproximado de cada factor frente a un anuncio promedio (SHAP). "
                       "Explica el precio del modelo, no por qué el anuncio está barato.",
        "median_error": "Error mediano",
        "model_error": "Error mediano del modelo: {pct:.1f}%",
    },
    "en": {
        "nav_opp": "🎯 Opportunities",
        "nav_price": "💰 Price by neighbourhood",
        "nav_pred": "🔮 Predictor",
        "kind_arr": "Rent",
        "kind_ven": "Sale",
        "all": "All",
        "barrio": "Neighbourhood",
        "tipo": "Type",
        "tipo_apartamento": "Apartment",
        "tipo_casa": "House",
        "disclaimer": "Neighbourhoods are assigned from geographic location and may "
                      "differ from the one advertised in the listing. Always "
                      "double-check on the listing link.",
        "data_updated": "Data updated",
        "model_r2": "Model R²",
        "load_error": "Error loading data or models",
        "opp_title": "Opportunities",
        "opp_intro": "Listings priced below the value estimated by the model.",
        "oos_note": "The model price is an out-of-sample prediction: the model never saw that listing during training.",
        "min_disc": "Minimum discount (%)",
        "min_price": "Min price (COP)",
        "max_price": "Max price (COP)",
        "zero_no_limit": "0 = no limit",
        "opp_found": "{n:,} properties found",
        "no_opp": "No properties match these filters.",
        "col_barrio": "Neighbourhood",
        "col_tipo": "Type",
        "col_price": "Actual price",
        "col_model": "Model price",
        "col_disc": "Discount",
        "col_area": "Area m²",
        "col_ppm2": "Price/m²",
        "col_link": "Link",
        "link_text": "View →",
        "open_listing": "Open listing",
        "select_row": "Select a row to see the property location.",
        "map_prop": "Property",
        "map_opps": "Opportunities",
        "map_metro": "Stations (metro/tram/cable)",
        "map_mall": "Shopping centers",
        "map_super": "Supermarkets",
        "nearest_metro": "Nearest station: {name} ({d:.2f} km)",
        "poi_error": "Couldn't load shopping centers and supermarkets (OpenStreetMap didn't respond).",
        "poi_none": "No mapped shopping centers or supermarkets within 1.5 km.",
        "model_short": "model",
        "price_title": "Price per m² by neighbourhood",
        "price_intro": "Type a neighbourhood to get the rent and sale price per m².",
        "price_input": "Neighbourhood",
        "price_placeholder": "e.g. Laureles, Poblado, Belén…",
        "price_prompt": "Type a neighbourhood name.",
        "price_pick": "Several matches, pick one",
        "price_no_match": "I couldn't find that neighbourhood.",
        "price_suggest": "Did you mean? ",
        "per_m2_arr": "COP per m² per month",
        "per_m2_ven": "COP per m²",
        "range_line": "Typical range (P25–P75): {lo} – {hi} · {n} listings",
        "few": "Few listings, take this figure with caution.",
        "no_data": "No listings in this neighbourhood.",
        "pred_title": "Price predictor",
        "pred_intro": "Estimate the rent or sale price of a property.",
        "transaction": "Transaction",
        "area": "Area (m²)",
        "rooms": "Bedrooms",
        "baths": "Bathrooms",
        "parking": "Parking spots",
        "estrato": "Estrato",
        "admin": "Monthly admin fee (COP)",
        "metro_km": "Distance to metro (km)",
        "predict_btn": "Predict price →",
        "est_price": "Estimated price",
        "range": "Likely range: {lo} — {hi}",
        "ppm2_yours": "Estimated price/m²",
        "ppm2_barrio": "Neighbourhood median/m²",
        "comparables": "Similar properties in the neighbourhood",
        "no_comps": "No comparables with these filters.",
        "no_barrio_data": "No data for this neighbourhood in {kind}.",
        "pred_prompt": "Fill in the form and press “Predict price”.",
        "app_title": "Medellín Real Estate",
        "app_tagline": "Rent and sale prices, opportunities and machine-learning estimates",
        "kpi_listings": "Listings analysed",
        "kpi_barrios": "Neighbourhoods",
        "kpi_rent": "Median rent/m²",
        "kpi_sale": "Median sale/m²",
        "choro_title": "Median price per m² map",
        "choro_note": "Only neighbourhoods with at least {n} listings. The searched one is outlined in red.",
        "why_title": "Main factors behind the model price",
        "why_caption": "Approximate effect of each factor versus an average listing (SHAP). "
                       "It explains the model price, not why the listing is cheap.",
        "median_error": "Median error",
        "model_error": "Model median error: {pct:.1f}%",
    },
}


def t(key: str, **kw) -> str:
    lang = st.session_state.get("lang", "es")
    s = TR.get(lang, TR["es"]).get(key) or TR["es"].get(key, key)
    return s.format(**kw) if kw else s


def tipo_label(x) -> str:
    key = f"tipo_{str(x).strip().lower()}"
    lang = st.session_state.get("lang", "es")
    return TR[lang].get(key, str(x).title())


FEAT_LABELS = {
    "es": {
        "area": "Área", "habitaciones": "Habitaciones", "baños": "Baños",
        "parqueaderos": "Parqueaderos", "espacios": "Número de espacios",
        "axe": "Área por espacio", "axh": "Área por habitación", "axa": "Área (efecto no lineal)",
        "parq2": "Parqueaderos (no lineal)", "garaje_bin": "Tiene garaje",
        "estrato": "Estrato", "administracion": "Administración",
        "dist_metro_km": "Distancia al metro", "ppmc": "Precio/m² del barrio",
        "pppz": "Precio por espacio del barrio", "pppp": "Precio por parqueadero del barrio",
        "new_index": "Índice de precio del barrio", "barrio_count": "Oferta en el barrio",
        "pppp/pppz": "Perfil de precios del barrio", "pppp/ppmc": "Perfil de precios del barrio",
        "pppz/ppmc": "Perfil de precios del barrio", "barrio_te": "Nivel de precios del barrio",
        "tipo": "Tipo de inmueble",
    },
    "en": {
        "area": "Area", "habitaciones": "Bedrooms", "baños": "Bathrooms",
        "parqueaderos": "Parking", "espacios": "Number of spaces",
        "axe": "Area per space", "axh": "Area per bedroom", "axa": "Area (non-linear)",
        "parq2": "Parking (non-linear)", "garaje_bin": "Has garage",
        "estrato": "Estrato", "administracion": "Admin fee",
        "dist_metro_km": "Distance to metro", "ppmc": "Neighbourhood price/m²",
        "pppz": "Neighbourhood price per space", "pppp": "Neighbourhood price per parking",
        "new_index": "Neighbourhood price index", "barrio_count": "Neighbourhood supply",
        "pppp/pppz": "Neighbourhood price profile", "pppp/ppmc": "Neighbourhood price profile",
        "pppz/ppmc": "Neighbourhood price profile", "barrio_te": "Neighbourhood price level",
        "tipo": "Property type",
    },
}


def feat_label(col: str) -> str:
    """'num__area' -> 'Área'; 'cat__tipo_casa' -> 'Tipo de inmueble'."""
    lang = st.session_state.get("lang", "es")
    base = col.split("__", 1)[-1]
    if col.startswith("cat__"):
        base = "tipo"
    return FEAT_LABELS[lang].get(base, base)


# ─────────────────────────────────────────────────────────────────────────────
# Constants (must match train.py)
# ─────────────────────────────────────────────────────────────────────────────
NUM_ATTRIBS = [
    "area", "habitaciones", "baños", "parqueaderos", "espacios",
    "axe", "axh", "axa", "parq2", "garaje_bin",
    "estrato", "administracion", "dist_metro_km",
    "ppmc", "pppz", "pppp", "new_index", "barrio_count",
    "pppp/pppz", "pppp/ppmc", "pppz/ppmc",
    "barrio_te",
]
CAT_ATTRIBS = ["tipo"]
KINDS = ("arr", "ven")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _norm(s) -> str:
    s = unicodedata.normalize("NFD", str(s).lower().strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


def fmt_price(v) -> str:
    """Compact: $850K style is avoided, we use $12.3M (millones)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if v >= 1_000_000:
        return f"${v / 1_000_000:,.1f}M"
    return f"${v:,.0f}"


def fmt_full(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"${v:,.0f}"


def _km(lat1, lon1, lat2, lon2):
    """Haversine distance in km (works with numpy arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _artifact(name: str) -> str:
    p = Path("artifacts") / name
    return str(p) if p.exists() else name


def disclaimer():
    with st.container(border=True):
        st.caption("ℹ️ " + t("disclaimer"))


# ─────────────────────────────────────────────────────────────────────────────
# StackedEnsemble — must exist here so pickle can deserialise it
# ─────────────────────────────────────────────────────────────────────────────
class StackedEnsemble:
    """XGBoost + LightGBM → Ridge meta-learner."""

    def __init__(self, xgb_p=None, lgb_p=None, cat_p=None):
        self.xgb_p = xgb_p or {}
        self.lgb_p = lgb_p or {}
        self.base_ = []
        self.meta_ = None

    def predict(self, X) -> np.ndarray:
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median().fillna(0))
        return self.meta_.predict(
            np.column_stack([m.predict(X) for m in self.base_])
        )


# ─────────────────────────────────────────────────────────────────────────────
# Loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
def _load(name: str, required: bool = True):
    p = Path(_artifact(name))
    if not p.exists():
        if required:
            raise FileNotFoundError(
                f"Artifact not found: {name} — run train.py first.")
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_artifacts() -> dict:
    A = {
        "barrios": _load("list_barrios.pkl"),
        "r2": _load("model_r2.pkl", required=False),
        "metrics": _load("model_metrics.pkl", required=False),
    }
    for kind in KINDS:
        for key, fname in [
            ("feats", "best_features"), ("te", "barrio_te"),
            ("pp", "preprocessor"), ("model", "stack"),
            ("q10", "q10"), ("q90", "q90"),
            ("ppmc", "price_per_m2"), ("pppz", "price_per_space"),
            ("pppp", "price_per_parking"),
        ]:
            optional = key in ("q10", "q90")
            A.setdefault(key, {})[kind] = _load(
                f"{fname}_{kind}.pkl", required=not optional)

    # Metro / tram / cable stations, de-duplicated by coordinates
    raw = _load("metro_stations.pkl", required=False) or []
    seen, stations = set(), []
    for name, la, lo in raw:
        k = (round(la, 5), round(lo, 5))
        if k not in seen:
            seen.add(k)
            stations.append((name, la, lo))
    A["stations"] = stations
    return A


def _to_frame(X, pp) -> pd.DataFrame:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pd.DataFrame(X, columns=pp.get_feature_names_out())


@st.cache_data(ttl=3600, show_spinner=False)
def load_tables() -> dict:
    out = {}
    for kind, csv in [("arr", "arr_mede_final.csv"), ("ven", "ven_mede_final.csv")]:
        df = pd.read_csv(_artifact(csv))
        if "cat_pred" not in df.columns or df["cat_pred"].isna().all():
            raise ValueError(
                f"{csv} has no out-of-sample predictions (cat_pred). "
                "Re-run train.py (leak-free version) to regenerate it.")
        df["barrio_norm"] = df["nombre"].astype(str).map(_norm)
        df["ppm2"] = df["precio"] / df["area"].replace(0, np.nan)
        # cat_pred is out-of-fold (train rows) or held-out (test rows):
        # the model never saw the listing it is pricing.
        df["pct_underpriced"] = (
            df["cat_pred"] - df["precio"]) / df["cat_pred"] * 100
        out[kind] = df
    return out


def data_date() -> str:
    try:
        ts = Path(_artifact("arr_mede_final.csv")).stat().st_mtime
        return datetime.datetime.fromtimestamp(ts).strftime("%d %b %Y")
    except Exception:
        return "—"


# ─────────────────────────────────────────────────────────────────────────────
# Map
# ─────────────────────────────────────────────────────────────────────────────
_MapTrace = getattr(go, "Scattermap", None) or go.Scattermapbox
_MAP_KEY = "map" if hasattr(go, "Scattermap") else "mapbox"
_ChoroTrace = getattr(go, "Choroplethmap", None) or go.Choroplethmapbox
MIN_CHORO_N = 3


def _clean_nombre(s) -> str:
    """Same cleaning train.py applies to shapefile barrio names."""
    s = str(s).strip()
    for pfx in ["Área de Expansión", "Area de Expansion"]:
        s = s.replace(pfx, "").strip(" -–—")
    return s


def _first_coord(c):
    while isinstance(c, list) and c and isinstance(c[0], list):
        c = c[0]
    return c


@st.cache_resource(show_spinner=False)
def load_barrio_geojson():
    """Barrio polygons keyed by normalised name; None if unavailable."""
    try:
        p = Path(_artifact("medellin.geojson"))
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            gj = json.load(f)
        for feat in gj["features"]:
            props = feat.get("properties", {})
            key = next((k for k in props if k.lower() == "nombre"), None)
            if key is None:
                return None
            feat["id"] = _norm(_clean_nombre(props[key]))
        x = _first_coord(gj["features"][0]["geometry"]["coordinates"])[0]
        if abs(x) > 180:      # projected CRS, not lat/lon
            return None
        return gj
    except Exception:
        return None


def price_choropleth(chosen_norm: str, tipo: str):
    gj = load_barrio_geojson()
    if gj is None:
        return
    st.subheader(t("choro_title"))
    kind = st.radio(t("transaction"), ["arr", "ven"],
                    format_func=lambda k: t(f"kind_{k}"), horizontal=True,
                    key="choro_kind")
    df = T[kind]
    if tipo != "all":
        df = df[df["tipo"] == tipo]
    g = (df.groupby("barrio_norm")
           .agg(z=("ppm2", "median"), n=("ppm2", "count"), name=("nombre", "first"))
           .reset_index())
    ids = {f["id"] for f in gj["features"]}
    g = g[(g["n"] >= MIN_CHORO_N) & g["barrio_norm"].isin(ids)]
    if g.empty:
        return
    lo, hi = g["z"].quantile([0.05, 0.95])
    fig = go.Figure(_ChoroTrace(
        geojson=gj, locations=g["barrio_norm"], z=g["z"], text=g["name"],
        zmin=lo, zmax=hi, colorscale="Teal",
        marker=dict(line=dict(width=0.5, color="#ffffff")),
        hovertemplate="<b>%{text}</b><br>$%{z:,.0f}<extra></extra>",
        colorbar=dict(title=t(f"per_m2_{kind}"))))
    if chosen_norm in ids:      # outline the searched barrio
        fig.add_trace(_ChoroTrace(
            geojson=gj, locations=[chosen_norm], z=[0], showscale=False,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            marker=dict(line=dict(width=3, color="#dc2626")),
            hoverinfo="skip"))
    fig.update_layout(
        **{_MAP_KEY: dict(style="carto-positron",
                          center=dict(lat=6.25, lon=-75.575), zoom=10.6)},
        margin=dict(r=0, t=0, l=0, b=0), height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(t("choro_note", n=MIN_CHORO_N))


# ── Per-property explanation (SHAP through the stacked ensemble) ─────────────
def _prop_X(kind: str, prop: pd.Series):
    """Model features for a listing (same path as the predictor). None if the
    barrio has no statistics (e.g. it only appears in the test split)."""
    if prop["nombre"] not in A["ppmc"][kind].index:
        return None
    inp = build_input(prop["area"], int(prop["habitaciones"]), int(prop["baños"]),
                      int(prop["parqueaderos"]
                          ), prop["nombre"], prop["tipo"], kind,
                      prop["estrato"], prop["administracion"], prop["dist_metro_km"])
    pp, cats = A["pp"][kind], A["feats"][kind]
    return _to_frame(pp.transform(inp[NUM_ATTRIBS + CAT_ATTRIBS]), pp)[cats]


@st.cache_resource(show_spinner=False)
def _explainers(kind: str):
    return [shap.TreeExplainer(m) for m in load_artifacts()["model"][kind].base_]


def explain(kind: str, X_row: pd.DataFrame, top: int = 3) -> list[tuple[str, float]]:
    """Top factors as (label, approx % effect on price vs an average listing)."""
    model = A["model"][kind]
    if shap is None or model is None:
        return []
    try:
        Xs = X_row.replace([np.inf, -np.inf], np.nan).fillna(0)
        contrib = np.zeros(Xs.shape[1])
        for coef, ex in zip(model.meta_.coef_, _explainers(kind)):
            contrib += coef * np.asarray(ex.shap_values(Xs))[0]
        by_label: dict[str, float] = {}
        for col, c in zip(Xs.columns, contrib):
            by_label[feat_label(col)] = by_label.get(
                feat_label(col), 0.0) + float(c)
        best = sorted(by_label.items(), key=lambda kv: abs(
            kv[1]), reverse=True)[:top]
        return [(k, float(np.expm1(v) * 100)) for k, v in best]
    except Exception:
        return []


def show_factors(kind: str, X_row: pd.DataFrame | None):
    if X_row is None:
        return
    factors = explain(kind, X_row)
    if not factors:
        return
    with st.container(border=True):
        st.markdown(f"**{t('why_title')}**")
        for label, pct in factors:
            st.markdown(
                f"{'🟢 ⬆' if pct > 0 else '🔴 ⬇'} **{label}** &nbsp; {pct:+.0f}%")
        st.caption(t("why_caption"))


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_pois(lat: float, lon: float, radius_m: int = 1500) -> pd.DataFrame:
    """Supermarkets + malls around a point (OpenStreetMap / Overpass)."""
    q = f"""
    [out:json][timeout:20];
    (
      nwr[shop=supermarket](around:{radius_m},{lat},{lon});
      nwr[shop=mall](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """
    r = requests.post("https://overpass-api.de/api/interpreter",
                      data={"data": q},
                      headers={"User-Agent": "medellin-re-app/1.0"},
                      timeout=25)
    r.raise_for_status()
    rows = []
    for e in r.json().get("elements", []):
        la = e.get("lat", e.get("center", {}).get("lat"))
        lo = e.get("lon", e.get("center", {}).get("lon"))
        if la is None or lo is None:
            continue
        tags = e.get("tags", {})
        rows.append({
            "kind": "mall" if tags.get("shop") == "mall" else "super",
            "name": tags.get("name", ""),
            "lat": la, "lon": lo,
        })
    return pd.DataFrame(rows, columns=["kind", "name", "lat", "lon"])


def _finish_map(fig, lat, lon, zoom):
    fig.update_layout(
        **{_MAP_KEY: dict(style="carto-positron",
                          center=dict(lat=lat, lon=lon), zoom=zoom)},
        margin=dict(r=0, t=0, l=0, b=0),
        height=560,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ccc", borderwidth=1, font=dict(color="#222")),
    )
    return fig


def overview_map(opp: pd.DataFrame):
    pts = opp.sample(min(2000, len(opp)), random_state=1)
    fig = go.Figure(_MapTrace(
        lat=pts["latitud"], lon=pts["longitud"], mode="markers",
        marker=dict(size=7, color="#0d9488", opacity=0.7),
        text=[f"{n} · {fmt_price(p)} · -{d:.0f}%" for n, p, d in
              zip(pts["nombre"], pts["precio"], pts["pct_underpriced"])],
        hoverinfo="text", name=t("map_opps")))
    return _finish_map(fig, pts["latitud"].mean(), pts["longitud"].mean(), 11.5)


def property_map(prop: pd.Series, stations: list, pois: pd.DataFrame | None):
    lat, lon = float(prop["latitud"]), float(prop["longitud"])
    fig = go.Figure()

    # Stations within 2 km (or the single nearest one if none)
    if stations:
        s = pd.DataFrame(stations, columns=["name", "lat", "lon"])
        s["d"] = _km(lat, lon, s["lat"].values, s["lon"].values)
        near = s[s["d"] <= 2.0]
        if near.empty:
            near = s.nsmallest(1, "d")
        fig.add_trace(_MapTrace(
            lat=near["lat"], lon=near["lon"], mode="markers",
            marker=dict(size=11, color="#2563eb"),
            text=[f"{n} · {d:.2f} km" for n,
                  d in zip(near["name"], near["d"])],
            hoverinfo="text", name=t("map_metro")))

    if pois is not None and len(pois):
        for kind, color, label in [("mall", "#9333ea", t("map_mall")),
                                   ("super", "#16a34a", t("map_super"))]:
            p = pois[pois["kind"] == kind]
            if p.empty:
                continue
            d = _km(lat, lon, p["lat"].values, p["lon"].values)
            fig.add_trace(_MapTrace(
                lat=p["lat"], lon=p["lon"], mode="markers",
                marker=dict(size=11, color=color),
                text=[f"{(n or label)} · {x:.2f} km" for n,
                      x in zip(p["name"], d)],
                hoverinfo="text", name=label))

    # Property last so it sits on top
    fig.add_trace(_MapTrace(
        lat=[lat], lon=[lon], mode="markers",
        marker=dict(size=20, color="#dc2626"),
        text=[f"{prop['nombre']} · {fmt_price(prop['precio'])}"],
        hoverinfo="text", name=t("map_prop")))
    return _finish_map(fig, lat, lon, 14)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar (language first: everything else uses t())
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    lang_choice = st.radio("Idioma / Language", ["Español", "English"],
                           horizontal=True, key="lang_choice")
    st.session_state["lang"] = "es" if lang_choice == "Español" else "en"

try:
    A = load_artifacts()
    T = load_tables()
except Exception as e:
    st.error(f"{t('load_error')}: {type(e).__name__}: {e}")
    st.stop()

TIPOS = sorted(set(T["arr"]["tipo"].dropna().unique())
               | set(T["ven"]["tipo"].dropna().unique()))

with st.sidebar:
    section = st.radio("nav", ["opp", "price", "pred"],
                       format_func=lambda k: t(f"nav_{k}"),
                       label_visibility="collapsed", key="section")
    st.divider()
    st.caption(f"📅 {t('data_updated')}: {data_date()}")
    _m = A.get("metrics") or {}
    if _m:
        st.caption(f"🎯 {t('median_error')} — {t('kind_arr')}: "
                   f"{_m.get('arr', {}).get('mape', float('nan')):.1f}% · "
                   f"{t('kind_ven')}: {_m.get('ven', {}).get('mape', float('nan')):.1f}%")
    if A.get("r2"):
        st.caption(f"📐 {t('model_r2')} — {t('kind_arr')}: {A['r2'].get('arr', 0):.3f} · "
                   f"{t('kind_ven')}: {A['r2'].get('ven', 0):.3f}")
    st.caption("Roman Alejandro Correa")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Opportunities
# ─────────────────────────────────────────────────────────────────────────────
def page_opportunities():
    st.header(t("opp_title"))
    st.caption(t("opp_intro"))
    st.caption("🔎 " + t("oos_note"))
    disclaimer()

    f1, f2, f3, f4, f5 = st.columns([1.2, 1.6, 1.4, 1.2, 1.2])
    kind = f1.radio(t("transaction"), ["ven", "arr"],
                    format_func=lambda k: t(f"kind_{k}"), horizontal=True,
                    key="opp_kind")
    df = T[kind]
    barrios = [None] + sorted(df["nombre"].dropna().unique().tolist())
    barrio = f2.selectbox(t("barrio"), barrios,
                          format_func=lambda b: t("all") if b is None else b,
                          key="opp_barrio")
    thr = f3.slider(t("min_disc"), 10, 60, 20, key="opp_thr")
    min_p = f4.number_input(t("min_price"), 0, value=0, step=1_000_000,
                            help=t("zero_no_limit"), key="opp_min")
    max_p = f5.number_input(t("max_price"), 0, value=0, step=1_000_000,
                            help=t("zero_no_limit"), key="opp_max")

    opp = df[df["pct_underpriced"] > thr]
    if barrio:
        opp = opp[opp["nombre"] == barrio]
    if min_p:
        opp = opp[opp["precio"] >= min_p]
    if max_p:
        opp = opp[opp["precio"] <= max_p]
    opp = opp.sort_values(
        "pct_underpriced", ascending=False).reset_index(drop=True)

    if opp.empty:
        st.info(t("no_opp"))
        return

    col_tbl, col_map = st.columns([1, 1], gap="medium")

    with col_tbl:
        st.subheader(t("opp_found", n=len(opp)))
        disp = pd.DataFrame({
            t("col_barrio"): opp["nombre"],
            t("col_tipo"): opp["tipo"].map(tipo_label),
            t("col_price"): opp["precio"].map(fmt_price),
            t("col_model"): opp["cat_pred"].map(fmt_price),
            t("col_disc"): opp["pct_underpriced"].map(lambda x: f"{x:.0f}%"),
            t("col_area"): opp["area"].round(0).astype(int),
            t("col_ppm2"): opp["ppm2"].map(fmt_full),
            t("col_link"): opp["url"] if "url" in opp.columns else None,
        })
        cfg = {t("col_link"): st.column_config.LinkColumn(
            t("col_link"), display_text=t("link_text"))}
        event = st.dataframe(disp, hide_index=True, height=520,
                             use_container_width=True, column_config=cfg,
                             on_select="rerun", selection_mode="single-row",
                             key="opp_table")
        rows = event.selection.rows if event else []
        prop = opp.iloc[rows[0]] if rows and rows[0] < len(opp) else None

    with col_map:
        if prop is None:
            st.caption(t("select_row"))
            st.plotly_chart(overview_map(opp), use_container_width=True)
        else:
            st.markdown(
                f"**{prop['nombre']}** · {tipo_label(prop['tipo'])} · "
                f"{fmt_price(prop['precio'])} "
                f"({t('model_short')} {fmt_price(prop['cat_pred'])}, "
                f"-{prop['pct_underpriced']:.0f}%)")

            lat, lon = float(prop["latitud"]), float(prop["longitud"])
            if A["stations"]:
                st_arr = np.array([(s[1], s[2]) for s in A["stations"]])
                d = _km(lat, lon, st_arr[:, 0], st_arr[:, 1])
                i = int(np.argmin(d))
                st.caption(t("nearest_metro",
                             name=A["stations"][i][0], d=float(d[i])))

            pois = None
            try:
                with st.spinner("…"):
                    pois = fetch_pois(round(lat, 4), round(lon, 4))
                if pois.empty:
                    st.caption(t("poi_none"))
            except Exception:
                st.caption("⚠️ " + t("poi_error"))

            st.plotly_chart(property_map(prop, A["stations"], pois),
                            use_container_width=True)
            if isinstance(prop.get("url"), str) and prop["url"]:
                st.link_button(t("open_listing"), prop["url"])
            show_factors(kind, _prop_X(kind, prop))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Price check by neighbourhood
# ─────────────────────────────────────────────────────────────────────────────
def _ppm2_stats(df: pd.DataFrame, barrio_norm: str, tipo: str):
    d = df[df["barrio_norm"] == barrio_norm]
    if tipo != "all":
        d = d[d["tipo"] == tipo]
    s = d["ppm2"].dropna()
    if s.empty:
        return 0, None, None, None
    return len(s), s.median(), s.quantile(0.25), s.quantile(0.75)


def page_price_check():
    st.header(t("price_title"))
    st.caption(t("price_intro"))
    disclaimer()

    names: dict[str, str] = {}
    for k in KINDS:
        for n in T[k]["nombre"].dropna().unique():
            names.setdefault(_norm(n), n)

    c1, c2 = st.columns([3, 1])
    q = c1.text_input(t("price_input"), placeholder=t("price_placeholder"),
                      key="pc_query")
    tipo = c2.selectbox(t("tipo"), ["all"] + TIPOS,
                        format_func=lambda x: t(
                            "all") if x == "all" else tipo_label(x),
                        key="pc_tipo")

    qn = _norm(q)
    if not qn:
        st.info(t("price_prompt"))
        return

    if qn in names:
        chosen = qn
    else:
        cands = sorted(k for k in names if qn in k)
        if len(cands) == 1:
            chosen = cands[0]
        elif len(cands) > 1:
            chosen = st.selectbox(t("price_pick"), cands,
                                  format_func=lambda k: names[k], key="pc_pick")
        else:
            st.warning(t("price_no_match"))
            sug = difflib.get_close_matches(qn, list(names), n=5, cutoff=0.6)
            if sug:
                st.write(t("price_suggest") + ", ".join(names[s] for s in sug))
            return

    st.subheader(names[chosen])
    col_arr, col_ven = st.columns(2)
    for col, kind in ((col_arr, "arr"), (col_ven, "ven")):
        n, med, lo, hi = _ppm2_stats(T[kind], chosen, tipo)
        with col.container(border=True):
            st.caption(t(f"per_m2_{kind}"))
            st.metric(t(f"kind_{kind}"), fmt_full(med))
            if n == 0:
                st.caption(t("no_data"))
            else:
                st.caption(
                    t("range_line", lo=fmt_full(lo), hi=fmt_full(hi), n=n))
                if n < 5:
                    st.caption("⚠️ " + t("few"))

    price_choropleth(chosen, tipo)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Predictor
# ─────────────────────────────────────────────────────────────────────────────
def _ratio(a, b) -> float:
    try:
        r = float(a) / float(b)
        return r if np.isfinite(r) else 1.0
    except Exception:
        return 1.0


def build_input(area, hab, banos, parq, barrio, tipo, kind,
                estrato, admin, dist_metro) -> pd.DataFrame:
    ppmc, pppz, pppp = A["ppmc"][kind], A["pppz"][kind], A["pppp"][kind]
    te = A["te"][kind]
    espacios = hab + parq + banos
    b_ppmc, b_pppz, b_pppp = (ppmc.get(barrio, np.nan),
                              pppz.get(barrio, np.nan),
                              pppp.get(barrio, np.nan))
    ppmc_max = float(ppmc.max())
    row = {
        "area": area, "habitaciones": hab, "baños": banos,
        "parqueaderos": parq, "espacios": espacios,
        "axe": area if espacios == 0 else area / espacios,
        "axh": area if hab == 0 else area / hab,
        "axa": area ** 2, "parq2": parq ** 2,
        "garaje_bin": 1 if parq > 0 else 0,
        "estrato": estrato, "administracion": admin,
        "dist_metro_km": dist_metro,
        "ppmc": b_ppmc, "pppz": b_pppz, "pppp": b_pppp,
        "new_index": b_ppmc / ppmc_max * 100,
        "barrio_count": float((T[kind]["nombre"] == barrio).sum()),
        "pppp/pppz": _ratio(b_pppp, b_pppz),
        "pppp/ppmc": _ratio(b_pppp, b_ppmc),
        "pppz/ppmc": _ratio(b_pppz, b_ppmc),
        "barrio_te": float(te.get(barrio, te.mean())),
        "tipo": tipo,
    }
    return pd.DataFrame([row])


def predict(kind, barrio, tipo, **f):
    if barrio not in A["ppmc"][kind].index or A["model"][kind] is None:
        return None
    pp, cats, model = A["pp"][kind], A["feats"][kind], A["model"][kind]
    inp = build_input(f["area"], f["hab"], f["banos"], f["parq"], barrio, tipo,
                      kind, f["estrato"], f["admin"], f["metro"])
    X = _to_frame(pp.transform(inp[NUM_ATTRIBS + CAT_ATTRIBS]), pp)[cats]
    price = float(np.expm1(model.predict(X))[0])
    q10, q90 = A["q10"][kind], A["q90"][kind]
    if q10 is not None and q90 is not None:
        low = float(np.expm1(q10.predict(X))[0])
        high = float(np.expm1(q90.predict(X))[0])
    else:
        low, high = price * 0.88, price * 1.12
    return price, min(low, price), max(high, price), X


def page_predictor():
    st.header(t("pred_title"))
    st.caption(t("pred_intro"))
    disclaimer()

    kind = st.radio(t("transaction"), ["arr", "ven"],
                    format_func=lambda k: t(f"kind_{k}"), horizontal=True,
                    key="pred_kind")
    df = T[kind]

    c_form, c_res = st.columns(2, gap="large")
    with c_form:
        barrio = st.selectbox(t("barrio"), sorted(
            A["barrios"]), key="pred_barrio")
        tipo = st.selectbox(t("tipo"), TIPOS, format_func=tipo_label,
                            key="pred_tipo")
        area = st.slider(t("area"), 20, 500, 80, key="pred_area")

        # Sensible defaults from the barrio's own listings
        b_df = df[df["nombre"] == barrio]
        e_def = b_df["estrato"].median() if len(b_df) else np.nan
        m_def = b_df["dist_metro_km"].median() if len(b_df) else np.nan
        e_def = int(np.clip(round(e_def), 1, 6)) if pd.notna(e_def) else 3
        m_def = round(float(min(m_def, 15.0)), 1) if pd.notna(m_def) else 1.0

        a, b = st.columns(2)
        hab = a.number_input(t("rooms"), 0, 10, 2, key="pred_hab")
        banos = b.number_input(t("baths"), 0, 10, 2, key="pred_banos")
        parq = a.number_input(t("parking"), 0, 5, 1, key="pred_parq")
        estrato = b.number_input(t("estrato"), 1, 6, e_def,
                                 key=f"pred_estrato_{kind}_{barrio}")
        c, d = st.columns(2)
        admin = c.number_input(t("admin"), 0, 5_000_000, 0, step=50_000,
                               key="pred_admin")
        metro = d.number_input(t("metro_km"), 0.0, 15.0, m_def, step=0.1,
                               key=f"pred_metro_{kind}_{barrio}")
        go_btn = st.button(t("predict_btn"), type="primary",
                           use_container_width=True)

    with c_res:
        if not go_btn:
            st.info(t("pred_prompt"))
            return
        res = predict(kind, barrio, tipo, area=area, hab=hab, banos=banos,
                      parq=parq, estrato=estrato, admin=admin, metro=metro)
        if res is None:
            st.error(t("no_barrio_data", kind=t(f"kind_{kind}")))
            return
        price, low, high, X = res

        with st.container(border=True):
            st.metric(f"{t('est_price')} · {t(f'kind_{kind}')}",
                      fmt_full(price))
            st.caption(t("range", lo=fmt_price(low), hi=fmt_price(high)))
            _mape = ((A.get("metrics") or {}).get(kind) or {}).get("mape")
            if _mape:
                st.caption(t("model_error", pct=_mape))

        m1, m2 = st.columns(2)
        m1.metric(t("ppm2_yours"), fmt_full(price / area))
        b_ppm2 = b_df["ppm2"].median() if len(b_df) else np.nan
        m2.metric(t("ppm2_barrio"), fmt_full(b_ppm2))
        show_factors(kind, X)

        st.subheader(t("comparables"))
        comp = df[(df["nombre"] == barrio)
                  & (df["tipo"] == tipo)
                  & (df["area"].between(area * 0.75, area * 1.25))].head(5)
        if comp.empty:
            st.caption(t("no_comps"))
        else:
            show = pd.DataFrame({
                t("col_tipo"): comp["tipo"].map(tipo_label),
                t("col_price"): comp["precio"].map(fmt_price),
                t("col_area"): comp["area"].round(0).astype(int),
                t("rooms"): comp["habitaciones"],
                t("baths"): comp["baños"],
                t("parking"): comp["parqueaderos"],
            })
            st.dataframe(show, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
def kpi_strip():
    st.markdown(f"### 🏙️ {t('app_title')}")
    st.caption(t("app_tagline"))
    n = len(T["arr"]) + len(T["ven"])
    nb = len(set(T["arr"]["barrio_norm"]) | set(T["ven"]["barrio_norm"]))
    c = st.columns(4)
    c[0].metric(t("kpi_listings"), f"{n:,}")
    c[1].metric(t("kpi_barrios"), f"{nb:,}")
    c[2].metric(t("kpi_rent"), fmt_full(T["arr"]["ppm2"].median()))
    c[3].metric(t("kpi_sale"), fmt_full(T["ven"]["ppm2"].median()))
    st.divider()


kpi_strip()
{"opp": page_opportunities,
 "price": page_price_check,
 "pred": page_predictor}[section]()
