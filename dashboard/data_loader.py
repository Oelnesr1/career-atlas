"""
Data-loading utilities for the Career Atlas Dash dashboard.

This file is intentionally responsible only for:
- reading project outputs
- standardizing key columns
- downloading/creating CBSA centroids if missing
- validating required files/columns
- producing dropdown option lists

It should not contain dashboard callbacks or figure-building logic.
"""

from __future__ import annotations

import io
import zipfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from . import config
except ImportError:
    import config


# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------

@dataclass
class DashboardData:
    panel: pd.DataFrame
    dashboard: pd.DataFrame
    scores: pd.DataFrame
    predictions: pd.DataFrame
    metro_profiles: pd.DataFrame
    nearest_metros: pd.DataFrame
    cluster_summary: pd.DataFrame
    regression_metrics: pd.DataFrame
    rf_importance_rent_burden: pd.DataFrame
    rf_importance_real_income: pd.DataFrame
    cbsa_centroids: pd.DataFrame

    occupation_options: list[dict[str, Any]]
    metro_options: list[dict[str, Any]]
    city_first_options: list[dict[str, Any]]
    state_options: list[dict[str, Any]]
    cluster_options: list[dict[str, Any]]
    latest_year: int | None


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _empty_df(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=columns or [])


def _read_csv(
    path: Path,
    *,
    required: bool = False,
    dtype: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        message = f"Missing data file: {path}"

        if required:
            raise FileNotFoundError(message)

        warnings.warn(message)
        return _empty_df()

    df = pd.read_csv(path, dtype=dtype, low_memory=False)
    return df


def _standardize_cbsa_code(df: pd.DataFrame, column: str = "cbsa_code") -> pd.DataFrame:
    if column in df.columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .fillna("")
            .str.zfill(5)
        )

    return df


def _standardize_occ_code(df: pd.DataFrame, column: str = "occ_code") -> pd.DataFrame:
    if column in df.columns:
        df[column] = df[column].astype(str)

    return df


def _standardize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_cbsa_code(df)
    df = _standardize_occ_code(df)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    skip_cols = {
        "cbsa_code",
        "occ_code",
        "occ_title",
        "bls_msa_name",
        "acs_area_name",
        "bea_area_name",
        "prim_state",
        "source_metro",
        "neighbor_metro",
        "target",
        "model",
        "split",
        "feature",
    }

    for col in df.columns:
        if col in skip_cols:
            continue

        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            df[col] = converted

    return df


def _validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    *,
    label: str,
    strict: bool = False,
) -> None:
    missing = [col for col in required_columns if col not in df.columns]

    if not missing:
        return

    message = f"{label} is missing required columns: {missing}"

    if strict:
        raise ValueError(message)

    warnings.warn(message)


def _sort_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(options, key=lambda item: str(item["label"]).lower())


def _make_options_from_series(series: pd.Series) -> list[dict[str, Any]]:
    values = sorted(v for v in series.dropna().unique())

    return [{"label": str(v), "value": v} for v in values]


# -----------------------------------------------------------------------------
# CBSA centroids
# -----------------------------------------------------------------------------

def ensure_cbsa_centroids(
    *,
    force_download: bool = False,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Ensure a clean CBSA centroid file exists.

    If data/static/cbsa_centroids.csv exists, read it.
    Otherwise, download the Census CBSA Gazetteer ZIP, extract the centroid file,
    and save a clean CSV with:

        cbsa_code
        cbsa_name
        latitude
        longitude

    If download fails, return an empty dataframe. The dashboard can still run,
    but map charts should show a user-facing warning/fallback.
    """

    if config.CBSA_CENTROIDS_PATH.exists() and not force_download:
        centroids = pd.read_csv(config.CBSA_CENTROIDS_PATH, dtype={"cbsa_code": str})
        centroids = _standardize_cbsa_code(centroids)
        return centroids

    try:
        response = requests.get(config.CENSUS_CBSA_GAZETTEER_URL, timeout=timeout)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            text_members = [
                name for name in zf.namelist()
                if name.lower().endswith((".txt", ".csv"))
            ]

            if not text_members:
                raise FileNotFoundError("No TXT/CSV file found inside Census Gazetteer ZIP.")

            member = text_members[0]

            with zf.open(member) as f:
                raw = pd.read_csv(f, sep="\t", dtype=str)

        raw.columns = [str(c).strip() for c in raw.columns]

        column_map = {
            "GEOID": "cbsa_code",
            "NAME": "cbsa_name",
            "INTPTLAT": "latitude",
            "INTPTLONG": "longitude",
        }

        missing_cols = [col for col in column_map if col not in raw.columns]

        if missing_cols:
            raise ValueError(
                f"Census Gazetteer file missing expected columns: {missing_cols}. "
                f"Available columns: {list(raw.columns)}"
            )

        centroids = (
            raw[list(column_map.keys())]
            .rename(columns=column_map)
            .copy()
        )

        centroids = _standardize_cbsa_code(centroids)
        centroids["latitude"] = pd.to_numeric(centroids["latitude"], errors="coerce")
        centroids["longitude"] = pd.to_numeric(centroids["longitude"], errors="coerce")

        centroids = (
            centroids
            .dropna(subset=["cbsa_code", "latitude", "longitude"])
            .drop_duplicates(subset=["cbsa_code"])
            .sort_values("cbsa_code")
            .reset_index(drop=True)
        )

        config.CBSA_CENTROIDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        centroids.to_csv(config.CBSA_CENTROIDS_PATH, index=False)

        return centroids

    except Exception as exc:
        warnings.warn(
            "Could not create CBSA centroid file. "
            "Map figures will not be available until data/static/cbsa_centroids.csv exists. "
            f"Reason: {exc}"
        )

        return _empty_df(["cbsa_code", "cbsa_name", "latitude", "longitude"])


def add_centroids(
    df: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    cbsa_col: str = "cbsa_code",
    prefix: str = "",
) -> pd.DataFrame:
    if df.empty or centroids.empty or cbsa_col not in df.columns:
        return df

    df = df.copy()
    df[cbsa_col] = df[cbsa_col].astype(str).str.zfill(5)

    centroid_cols = ["cbsa_code", "latitude", "longitude"]

    if "cbsa_name" in centroids.columns:
        centroid_cols.append("cbsa_name")

    centroid_lookup = centroids[centroid_cols].copy()

    if prefix:
        rename_map = {
            "latitude": f"{prefix}latitude",
            "longitude": f"{prefix}longitude",
            "cbsa_name": f"{prefix}cbsa_name",
        }
        centroid_lookup = centroid_lookup.rename(columns=rename_map)

    merged = df.merge(
        centroid_lookup,
        left_on=cbsa_col,
        right_on="cbsa_code",
        how="left",
        suffixes=("", "_centroid"),
    )

    if cbsa_col != "cbsa_code" and "cbsa_code_centroid" in merged.columns:
        merged = merged.drop(columns=["cbsa_code_centroid"])

    return merged


# -----------------------------------------------------------------------------
# Main data-loading functions
# -----------------------------------------------------------------------------

def load_panel_data() -> pd.DataFrame:
    df = _read_csv(
        config.PANEL_DATA_PATH,
        required=True,
        dtype={"cbsa_code": str, "occ_code": str},
    )

    df = _standardize_common_columns(df)
    df = _coerce_numeric_columns(df)

    _validate_columns(
        df,
        config.REQUIRED_PANEL_COLUMNS,
        label="Panel dataset",
        strict=False,
    )

    return df


def load_dashboard_output() -> pd.DataFrame:
    df = _read_csv(
        config.DASHBOARD_OUTPUT_PATH,
        required=True,
        dtype={"cbsa_code": str, "occ_code": str},
    )

    df = _standardize_common_columns(df)
    df = _coerce_numeric_columns(df)

    _validate_columns(
        df,
        config.REQUIRED_DASHBOARD_COLUMNS,
        label="Dashboard modeling output",
        strict=False,
    )

    return df


def load_scores() -> pd.DataFrame:
    df = _read_csv(
        config.CAREER_CITY_SCORES_PATH,
        required=False,
        dtype={"cbsa_code": str, "occ_code": str},
    )

    df = _standardize_common_columns(df)
    df = _coerce_numeric_columns(df)

    return df


def load_predictions() -> pd.DataFrame:
    df = _read_csv(
        config.NEXT_YEAR_PREDICTIONS_PATH,
        required=False,
        dtype={"cbsa_code": str, "occ_code": str},
    )

    df = _standardize_common_columns(df)
    df = _coerce_numeric_columns(df)

    return df


def load_metro_profiles() -> pd.DataFrame:
    df = _read_csv(
        config.METRO_PROFILES_PATH,
        required=False,
        dtype={"cbsa_code": str},
    )

    df = _standardize_cbsa_code(df)
    df = _coerce_numeric_columns(df)

    return df


def load_nearest_metros() -> pd.DataFrame:
    df = _read_csv(
        config.NEAREST_METROS_PATH,
        required=False,
        dtype={"source_cbsa_code": str, "neighbor_cbsa_code": str},
    )

    if "source_cbsa_code" in df.columns:
        df["source_cbsa_code"] = df["source_cbsa_code"].astype(str).str.zfill(5)

    if "neighbor_cbsa_code" in df.columns:
        df["neighbor_cbsa_code"] = df["neighbor_cbsa_code"].astype(str).str.zfill(5)

    df = _coerce_numeric_columns(df)

    return df


def load_cluster_summary() -> pd.DataFrame:
    df = _read_csv(config.METRO_CLUSTER_SUMMARY_PATH, required=False)
    df = _coerce_numeric_columns(df)
    return df


def load_regression_metrics() -> pd.DataFrame:
    df = _read_csv(config.REGRESSION_METRICS_PATH, required=False)
    df = _coerce_numeric_columns(df)
    return df


def load_rf_importance_rent_burden() -> pd.DataFrame:
    df = _read_csv(config.RF_IMPORTANCE_RENT_BURDEN_PATH, required=False)
    df = _coerce_numeric_columns(df)
    return df


def load_rf_importance_real_income() -> pd.DataFrame:
    df = _read_csv(config.RF_IMPORTANCE_REAL_INCOME_PATH, required=False)
    df = _coerce_numeric_columns(df)
    return df


# -----------------------------------------------------------------------------
# Dropdown options
# -----------------------------------------------------------------------------

def build_occupation_options(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "occ_title" not in df.columns:
        return []

    cols = ["occ_title"]

    if "occ_code" in df.columns:
        cols.append("occ_code")

    occ_df = (
        df[cols]
        .dropna(subset=["occ_title"])
        .drop_duplicates()
        .sort_values("occ_title")
    )

    options = []

    for _, row in occ_df.iterrows():
        occ_title = row["occ_title"]
        occ_code = row["occ_code"] if "occ_code" in occ_df.columns else ""

        label = f"{occ_title} ({occ_code})" if occ_code else str(occ_title)
        value = str(occ_title)

        options.append({"label": label, "value": value})

    return _sort_options(options)


def build_metro_options(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "bls_msa_name" not in df.columns:
        return []

    cols = ["bls_msa_name"]

    if "cbsa_code" in df.columns:
        cols.append("cbsa_code")

    metro_df = (
        df[cols]
        .dropna(subset=["bls_msa_name"])
        .drop_duplicates()
        .sort_values("bls_msa_name")
    )

    options = []

    for _, row in metro_df.iterrows():
        metro_name = row["bls_msa_name"]
        cbsa_code = row["cbsa_code"] if "cbsa_code" in metro_df.columns else ""

        label = f"{metro_name} ({cbsa_code})" if cbsa_code else str(metro_name)
        value = str(metro_name)

        options.append({"label": label, "value": value})

    return _sort_options(options)


def build_state_options(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "prim_state" not in df.columns:
        return []

    return _make_options_from_series(df["prim_state"])


def build_cluster_options(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "metro_cluster" not in df.columns:
        return []

    cluster_values = (
        pd.to_numeric(df["metro_cluster"], errors="coerce")
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
    )

    return [
        {"label": f"Cluster {cluster_id}", "value": int(cluster_id)}
        for cluster_id in cluster_values
    ]


def choose_default_occupation(options: list[dict[str, Any]]) -> str | None:
    if not options:
        return None

    labels = [str(option["label"]) for option in options]

    for keyword in config.DEFAULT_OCCUPATION_KEYWORDS:
        for option, label in zip(options, labels):
            if keyword.lower() in label.lower():
                return option["value"]

    return options[0]["value"]


def choose_default_metro(options: list[dict[str, Any]]) -> str | None:
    if not options:
        return None

    labels = [str(option["label"]) for option in options]

    for keyword in config.DEFAULT_METRO_KEYWORDS:
        for option, label in zip(options, labels):
            if keyword.lower() in label.lower():
                return option["value"]

    return options[0]["value"]


# -----------------------------------------------------------------------------
# Top-level load function
# -----------------------------------------------------------------------------

def load_all_data(*, force_centroid_download: bool = False) -> DashboardData:
    """
    Load every dataset needed by the dashboard.

    This function is intended to run once at app startup.
    """

    panel = load_panel_data()
    dashboard = load_dashboard_output()
    scores = load_scores()
    predictions = load_predictions()
    metro_profiles = load_metro_profiles()
    nearest_metros = load_nearest_metros()
    cluster_summary = load_cluster_summary()
    regression_metrics = load_regression_metrics()
    rf_importance_rent_burden = load_rf_importance_rent_burden()
    rf_importance_real_income = load_rf_importance_real_income()

    cbsa_centroids = ensure_cbsa_centroids(force_download=force_centroid_download)

    # Enrich the key tables with map coordinates when possible.
    dashboard = add_centroids(dashboard, cbsa_centroids, cbsa_col="cbsa_code")
    scores = add_centroids(scores, cbsa_centroids, cbsa_col="cbsa_code")
    predictions = add_centroids(predictions, cbsa_centroids, cbsa_col="cbsa_code")
    metro_profiles = add_centroids(metro_profiles, cbsa_centroids, cbsa_col="cbsa_code")

    # Add neighbor coordinates to nearest_metros for similar-city maps.
    if not nearest_metros.empty and not cbsa_centroids.empty:
        source_centroids = cbsa_centroids.rename(
            columns={
                "cbsa_code": "source_cbsa_code",
                "latitude": "source_latitude",
                "longitude": "source_longitude",
                "cbsa_name": "source_cbsa_name",
            }
        )

        neighbor_centroids = cbsa_centroids.rename(
            columns={
                "cbsa_code": "neighbor_cbsa_code",
                "latitude": "neighbor_latitude",
                "longitude": "neighbor_longitude",
                "cbsa_name": "neighbor_cbsa_name",
            }
        )

        nearest_metros = nearest_metros.merge(
            source_centroids,
            on="source_cbsa_code",
            how="left",
        )

        nearest_metros = nearest_metros.merge(
            neighbor_centroids,
            on="neighbor_cbsa_code",
            how="left",
        )

    occupation_options = build_occupation_options(dashboard)
    metro_options = build_metro_options(dashboard)
    city_first_options = build_metro_options(panel)
    state_options = build_state_options(dashboard)
    cluster_options = build_cluster_options(dashboard)

    latest_year = None

    if "year" in panel.columns and not panel["year"].dropna().empty:
        latest_year = int(panel["year"].dropna().max())

    return DashboardData(
        panel=panel,
        dashboard=dashboard,
        scores=scores,
        predictions=predictions,
        metro_profiles=metro_profiles,
        nearest_metros=nearest_metros,
        cluster_summary=cluster_summary,
        regression_metrics=regression_metrics,
        rf_importance_rent_burden=rf_importance_rent_burden,
        rf_importance_real_income=rf_importance_real_income,
        cbsa_centroids=cbsa_centroids,
        occupation_options=occupation_options,
        metro_options=metro_options,
        city_first_options=city_first_options,
        state_options=state_options,
        cluster_options=cluster_options,
        latest_year=latest_year,
    )


# -----------------------------------------------------------------------------
# Debug / smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_all_data()

    print("Loaded dashboard data successfully.")
    print(f"Panel rows: {len(data.panel):,}")
    print(f"Dashboard rows: {len(data.dashboard):,}")
    print(f"Metro profiles: {len(data.metro_profiles):,}")
    print(f"Nearest-metro rows: {len(data.nearest_metros):,}")
    print(f"Centroids: {len(data.cbsa_centroids):,}")
    print(f"Occupations: {len(data.occupation_options):,}")
    print(f"Metros: {len(data.metro_options):,}")
    print(f"Latest year: {data.latest_year}")