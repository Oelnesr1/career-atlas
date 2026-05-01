"""
Analytics utilities for the Career Atlas dashboard.

This file contains reusable computations for:
- custom weighted scoring
- best cities for an occupation
- best occupations for a city
- similar cities
- similar occupations
- city/occupation trend extraction
- KPI summaries
- table formatting helpers

DuckDB is used where SQL-style querying is cleaner, especially for nearest-neighbor
lookups and dashboard verification-style queries.

This file should not contain Dash callbacks or layout code.
"""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from . import config
except ImportError:
    import config


# -----------------------------------------------------------------------------
# Dataframe helpers
# -----------------------------------------------------------------------------

def to_polars(df: Any) -> pl.DataFrame:
    """
    Convert supported dataframe-like objects to Polars.

    The dashboard data loader may return either Polars or Pandas depending on
    which version you are using. This keeps the analytics layer robust.
    """

    if df is None:
        return pl.DataFrame()

    if isinstance(df, pl.DataFrame):
        return df.clone()

    if hasattr(df, "to_dict") and hasattr(df, "columns"):
        return pl.from_pandas(df)

    if isinstance(df, list):
        return pl.DataFrame(df)

    return pl.DataFrame()


def is_empty(df: Any) -> bool:
    df_pl = to_polars(df)
    return df_pl.height == 0


def has_columns(df: pl.DataFrame, columns: list[str]) -> bool:
    return all(col in df.columns for col in columns)


def clean_string(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def ensure_numeric_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    expressions = []

    for col in columns:
        if col in df.columns:
            expressions.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))

    if not expressions:
        return df

    return df.with_columns(expressions)


def first_available_column(df: pl.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    return None


def get_metric_label(metric: str) -> str:
    return config.METRIC_LABELS.get(metric, metric)


def get_metric_format(metric: str) -> str:
    return config.METRIC_FORMATS.get(metric, "{}")


def format_metric_value(value: Any, metric: str) -> str:
    if value is None:
        return "N/A"

    try:
        if value != value:  # NaN check
            return "N/A"
    except Exception:
        pass

    fmt = get_metric_format(metric)

    try:
        return fmt.format(value)
    except Exception:
        return str(value)


def normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """
    Normalize a raw weight dictionary so values sum to 1.

    If all supplied weights are zero or missing, use config.DEFAULT_WEIGHTS.
    """

    if not weights:
        weights = config.DEFAULT_WEIGHTS.copy()

    clean_weights = {}

    for key in config.DEFAULT_WEIGHTS:
        raw_value = weights.get(key, config.DEFAULT_WEIGHTS.get(key, 0))
        clean_weights[key] = max(float(raw_value or 0), 0.0)

    total = sum(clean_weights.values())

    if total <= 0:
        clean_weights = config.DEFAULT_WEIGHTS.copy()
        total = sum(clean_weights.values())

    if total <= 0:
        n = len(clean_weights)
        return {key: 1 / n for key in clean_weights}

    return {key: value / total for key, value in clean_weights.items()}


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


# -----------------------------------------------------------------------------
# DuckDB helpers
# -----------------------------------------------------------------------------

def duckdb_query(df: pl.DataFrame, query: str, params: list[Any] | None = None) -> pl.DataFrame:
    """
    Run a DuckDB SQL query against a Polars dataframe and return Polars output.
    """

    con = duckdb.connect(database=":memory:")
    con.register("df", df)

    try:
        result = con.execute(query, params or []).pl()
    finally:
        con.close()

    return result


def duckdb_query_many(
    tables: dict[str, pl.DataFrame],
    query: str,
    params: list[Any] | None = None,
) -> pl.DataFrame:
    """
    Run a DuckDB SQL query against multiple Polars dataframe tables.
    """

    con = duckdb.connect(database=":memory:")

    try:
        for name, table in tables.items():
            con.register(name, table)

        result = con.execute(query, params or []).pl()
    finally:
        con.close()

    return result


# -----------------------------------------------------------------------------
# Option helpers
# -----------------------------------------------------------------------------

def make_options_from_column(df: Any, column: str) -> list[dict[str, Any]]:
    df_pl = to_polars(df)

    if df_pl.height == 0 or column not in df_pl.columns:
        return []

    values = (
        df_pl
        .select(pl.col(column).drop_nulls().unique().sort())
        .to_series()
        .to_list()
    )

    return [{"label": str(value), "value": value} for value in values]


def get_occupations_for_city(
    panel_df: Any,
    city_name: str | None,
) -> list[dict[str, str]]:
    df = to_polars(panel_df)

    if df.height == 0 or not city_name or "bls_msa_name" not in df.columns or "occ_title" not in df.columns:
        return []

    cols = ["occ_title"]

    if "occ_code" in df.columns:
        cols.append("occ_code")

    occ_df = (
        df
        .filter(pl.col("bls_msa_name") == city_name)
        .select(cols)
        .drop_nulls("occ_title")
        .unique()
        .sort("occ_title")
    )

    options = []

    for row in occ_df.iter_rows(named=True):
        occ_title = row.get("occ_title")
        occ_code = row.get("occ_code", "")

        label = f"{occ_title} ({occ_code})" if occ_code else str(occ_title)
        options.append({"label": label, "value": occ_title})

    return options


def choose_default_occupation_for_city(
    panel_df: Any,
    city_name: str | None,
) -> str | None:
    options = get_occupations_for_city(panel_df, city_name)

    if not options:
        return None

    labels = [option["label"] for option in options]

    for keyword in config.DEFAULT_OCCUPATION_KEYWORDS:
        for option, label in zip(options, labels):
            if keyword.lower() in label.lower():
                return option["value"]

    return options[0]["value"]


# -----------------------------------------------------------------------------
# Filtering helpers
# -----------------------------------------------------------------------------

def filter_dashboard_rows(
    df: Any,
    *,
    occupation: str | None = None,
    city: str | None = None,
    states: list[str] | None = None,
    clusters: list[int] | None = None,
    min_employment: float | None = None,
    min_lq: float | None = None,
    max_rpp: float | None = None,
    require_coordinates: bool = False,
) -> pl.DataFrame:
    result = to_polars(df)

    if result.height == 0:
        return result

    filters = []

    if occupation and "occ_title" in result.columns:
        filters.append(pl.col("occ_title") == occupation)

    if city and "bls_msa_name" in result.columns:
        filters.append(pl.col("bls_msa_name") == city)

    state_values = ensure_list(states)

    if state_values and "prim_state" in result.columns:
        filters.append(pl.col("prim_state").is_in(state_values))

    cluster_values = ensure_list(clusters)

    if cluster_values and "metro_cluster" in result.columns:
        filters.append(
            pl.col("metro_cluster").cast(pl.Int64, strict=False).is_in([int(c) for c in cluster_values])
        )

    if min_employment is not None and "tot_emp" in result.columns:
        filters.append(pl.col("tot_emp").cast(pl.Float64, strict=False).fill_null(0) >= float(min_employment))

    if min_lq is not None and "loc_quotient" in result.columns:
        filters.append(pl.col("loc_quotient").cast(pl.Float64, strict=False).fill_null(0) >= float(min_lq))

    if max_rpp is not None and "rpp_all_items" in result.columns:
        filters.append(pl.col("rpp_all_items").cast(pl.Float64, strict=False).fill_null(float("inf")) <= float(max_rpp))

    if require_coordinates:
        if "latitude" in result.columns and "longitude" in result.columns:
            filters.append(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
        else:
            return result.head(0)

    if filters:
        combined = filters[0]

        for next_filter in filters[1:]:
            combined = combined & next_filter

        result = result.filter(combined)

    return result


# -----------------------------------------------------------------------------
# Custom scoring
# -----------------------------------------------------------------------------

def _component_expr(
    metric: str,
    alias: str,
    *,
    higher_is_better: bool = True,
    group_col: str | None = None,
) -> pl.Expr:
    """
    Create a percentile component expression where higher output means better.

    For lower-is-better metrics, descending=True makes smaller values receive
    larger percentile ranks.
    """

    descending = not higher_is_better

    value_expr = pl.col(metric).cast(pl.Float64, strict=False)

    rank_expr = value_expr.rank(method="average", descending=descending)
    count_expr = value_expr.count()

    if group_col:
        rank_expr = rank_expr.over(group_col)
        count_expr = count_expr.over(group_col)

    return (
        pl.when(count_expr > 0)
        .then(rank_expr / count_expr)
        .otherwise(0.5)
        .fill_null(0.5)
        .alias(alias)
    )


def _metric_has_values(df: pl.DataFrame, metric: str) -> bool:
    if metric not in df.columns:
        return False

    try:
        return df.select(pl.col(metric).is_not_null().sum()).item() > 0
    except Exception:
        return False


def _component_from_metric(
    df: pl.DataFrame,
    metric: str,
    alias: str,
    *,
    higher_is_better: bool = True,
    fallback_metric: str | None = None,
    fallback_higher_is_better: bool | None = None,
    group_col: str | None = None,
) -> tuple[pl.DataFrame, str]:
    if _metric_has_values(df, metric):
        source_metric = metric
        source_higher_is_better = higher_is_better
    elif fallback_metric and _metric_has_values(df, fallback_metric):
        source_metric = fallback_metric
        source_higher_is_better = (
            higher_is_better
            if fallback_higher_is_better is None
            else fallback_higher_is_better
        )
    else:
        return df.with_columns(pl.lit(0.5).alias(alias)), alias

    return df.with_columns(
        _component_expr(
            source_metric,
            alias,
            higher_is_better=source_higher_is_better,
            group_col=group_col,
        )
    ), alias


def add_custom_score(
    df: Any,
    *,
    weights: dict[str, float] | None = None,
    group_col: str | None = None,
) -> pl.DataFrame:
    """
    Add custom score components and a 0-100 custom score.

    All components are oriented so higher means better.
    """

    result = to_polars(df)

    if result.height == 0:
        return result.with_columns([
            pl.lit(None).cast(pl.Float64).alias("custom_score"),
            pl.lit(None).cast(pl.Float64).alias("custom_score_0_100"),
        ])

    normalized_weights = normalize_weights(weights)

    if "tot_emp" in result.columns:
        result = result.with_columns(
            pl.when(pl.col("tot_emp").cast(pl.Float64, strict=False).fill_null(0) > 0)
            .then(pl.col("tot_emp").cast(pl.Float64, strict=False).fill_null(0).log1p())
            .otherwise(0.0)
            .alias("_log_tot_emp_for_score")
        )
    else:
        result = result.with_columns(pl.lit(0.0).alias("_log_tot_emp_for_score"))

    result, _ = _component_from_metric(
        result,
        "real_income_after_rent_median",
        "score_real_income_after_rent",
        higher_is_better=True,
        group_col=group_col,
    )

    result, _ = _component_from_metric(
        result,
        "median_wage_to_rent_ratio",
        "score_rent_affordability",
        higher_is_better=True,
        fallback_metric="rent_to_median_wage_ratio",
        fallback_higher_is_better=False,
        group_col=group_col,
    )

    result, _ = _component_from_metric(
        result,
        "real_wage_annual_median",
        "score_real_wage",
        higher_is_better=True,
        fallback_metric="wage_annual_median",
        fallback_higher_is_better=True,
        group_col=group_col,
    )

    result, _ = _component_from_metric(
        result,
        "loc_quotient",
        "score_location_quotient",
        higher_is_better=True,
        group_col=group_col,
    )

    result, _ = _component_from_metric(
        result,
        "_log_tot_emp_for_score",
        "score_employment",
        higher_is_better=True,
        group_col=group_col,
    )

    result, _ = _component_from_metric(
        result,
        "predicted_change_in_real_income_after_rent",
        "score_predicted_real_income_growth",
        higher_is_better=True,
        group_col=group_col,
    )

    result, _ = _component_from_metric(
        result,
        "rpp_all_items",
        "score_low_cost",
        higher_is_better=False,
        group_col=group_col,
    )

    result = result.with_columns(
        (
            normalized_weights["real_income_after_rent"] * pl.col("score_real_income_after_rent")
            + normalized_weights["rent_affordability"] * pl.col("score_rent_affordability")
            + normalized_weights["real_wage"] * pl.col("score_real_wage")
            + normalized_weights["location_quotient"] * pl.col("score_location_quotient")
            + normalized_weights["employment"] * pl.col("score_employment")
            + normalized_weights["predicted_real_income_growth"] * pl.col("score_predicted_real_income_growth")
            + normalized_weights["low_cost"] * pl.col("score_low_cost")
        ).alias("custom_score")
    )

    result = result.with_columns(
        (100 * pl.col("custom_score")).alias("custom_score_0_100")
    )

    return result.drop("_log_tot_emp_for_score")


def get_best_cities_for_occupation(
    dashboard_df: Any,
    *,
    occupation: str,
    weights: dict[str, float] | None = None,
    states: list[str] | None = None,
    clusters: list[int] | None = None,
    min_employment: float = config.DEFAULT_MIN_EMPLOYMENT,
    min_lq: float = config.DEFAULT_MIN_LOCATION_QUOTIENT,
    max_rpp: float | None = None,
    top_n: int = config.DEFAULT_TOP_N,
) -> pl.DataFrame:
    filtered = filter_dashboard_rows(
        dashboard_df,
        occupation=occupation,
        states=states,
        clusters=clusters,
        min_employment=min_employment,
        min_lq=min_lq,
        max_rpp=max_rpp,
    )

    scored = add_custom_score(filtered, weights=weights)

    if scored.height == 0:
        return scored

    return scored.sort("custom_score_0_100", descending=True).head(top_n)


def get_best_occupations_for_city(
    dashboard_df: Any,
    *,
    city: str,
    weights: dict[str, float] | None = None,
    min_employment: float = config.DEFAULT_MIN_EMPLOYMENT,
    min_lq: float = 0.0,
    top_n: int = config.DEFAULT_TOP_N,
) -> pl.DataFrame:
    filtered = filter_dashboard_rows(
        dashboard_df,
        city=city,
        min_employment=min_employment,
        min_lq=min_lq,
    )

    scored = add_custom_score(filtered, weights=weights)

    if scored.height == 0:
        return scored

    return scored.sort("custom_score_0_100", descending=True).head(top_n)

def get_best_worst_combinations_for_metric(
    dashboard_df: Any,
    *,
    metric: str,
    weights: dict[str, float] | None = None,
    states: list[str] | None = None,
    clusters: list[int] | None = None,
    min_employment: float = config.DEFAULT_MIN_EMPLOYMENT,
    min_lq: float = config.DEFAULT_MIN_LOCATION_QUOTIENT,
    max_rpp: float | None = None,
    top_n: int = config.DEFAULT_TOP_N,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Return best and worst occupation-metro combinations for a selected metric.

    The function respects the dashboard's current filters and uses metric direction
    from config.METRIC_HIGHER_IS_BETTER.

    For example:
    - real_income_after_rent_median: best = highest values
    - rent_to_median_wage_ratio: best = lowest values
    - annual_rent: best = lowest values
    - loc_quotient: best = highest values
    """

    df = filter_dashboard_rows(
        dashboard_df,
        states=states,
        clusters=clusters,
        min_employment=min_employment,
        min_lq=min_lq,
        max_rpp=max_rpp,
    )

    if df.height == 0:
        return pl.DataFrame(), pl.DataFrame()

    # Always compute custom score so users can choose "custom_score_0_100"
    # as the selected metric. Group by occupation so the custom score is
    # percentile-based within each occupation.
    score_group = "occ_code" if "occ_code" in df.columns else None
    df = add_custom_score(df, weights=weights, group_col=score_group)

    if metric not in df.columns:
        return pl.DataFrame(), pl.DataFrame()

    higher_is_better = config.METRIC_HIGHER_IS_BETTER.get(metric, True)

    df = (
        df
        .with_columns([
            pl.col(metric).cast(pl.Float64, strict=False).alias("metric_value"),
            pl.concat_str(
                [
                    pl.col("occ_title").cast(pl.Utf8),
                    pl.lit(" — "),
                    pl.col("bls_msa_name").cast(pl.Utf8),
                ],
                ignore_nulls=True,
            ).alias("career_city_combo"),
        ])
        .drop_nulls("metric_value")
    )

    if df.height == 0:
        return pl.DataFrame(), pl.DataFrame()

    # Best sorting direction depends on metric meaning.
    best = (
        df
        .sort("metric_value", descending=higher_is_better)
        .head(top_n)
        .with_row_index("rank_position", offset=1)
        .with_columns(pl.lit("Best").alias("extreme_type"))
    )

    # Worst is the opposite direction.
    worst = (
        df
        .sort("metric_value", descending=not higher_is_better)
        .head(top_n)
        .with_row_index("rank_position", offset=1)
        .with_columns(pl.lit("Worst").alias("extreme_type"))
    )

    return best, worst

# -----------------------------------------------------------------------------
# City / occupation summaries
# -----------------------------------------------------------------------------

def get_city_occupation_latest_row(
    dashboard_df: Any,
    *,
    city: str,
    occupation: str,
) -> dict[str, Any] | None:
    filtered = filter_dashboard_rows(
        dashboard_df,
        city=city,
        occupation=occupation,
    )

    if filtered.height == 0:
        return None

    return filtered.row(0, named=True)


def get_city_occupation_history(
    panel_df: Any,
    *,
    city: str,
    occupation: str,
) -> pl.DataFrame:
    df = to_polars(panel_df)

    if df.height == 0 or not has_columns(df, ["year", "bls_msa_name", "occ_title"]):
        return pl.DataFrame()

    return (
        df
        .filter(
            (pl.col("bls_msa_name") == city)
            & (pl.col("occ_title") == occupation)
        )
        .with_columns(pl.col("year").cast(pl.Int64, strict=False))
        .sort("year")
    )


def get_metric_distribution_for_occupation(
    panel_df: Any,
    *,
    occupation: str,
    year: int | None = None,
    metric: str = "real_income_after_rent_median",
    min_employment: float | None = None,
) -> pl.DataFrame:
    df = to_polars(panel_df)

    if df.height == 0 or metric not in df.columns or "occ_title" not in df.columns:
        return pl.DataFrame()

    filters = [pl.col("occ_title") == occupation]

    if year is not None and "year" in df.columns:
        filters.append(pl.col("year").cast(pl.Int64, strict=False) == int(year))

    if min_employment is not None and "tot_emp" in df.columns:
        filters.append(pl.col("tot_emp").cast(pl.Float64, strict=False).fill_null(0) >= min_employment)

    expr = filters[0]

    for next_expr in filters[1:]:
        expr = expr & next_expr

    return df.filter(expr)


def make_kpi_summary(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if row is None:
        return []

    kpis = [
        ("real_income_after_rent_median", "Purchasing Power After Rent"),
        ("rent_to_median_wage_ratio", "Rent Burden"),
        ("real_wage_annual_median", "RPP-Adjusted Wage"),
        ("annual_rent", "Annual Rent"),
        ("rpp_all_items", "RPP All-Items"),
        ("tot_emp", "Employment"),
        ("loc_quotient", "Location Quotient"),
        ("predicted_change_in_real_income_after_rent", "Predicted Change"),
    ]

    output = []

    for metric, label in kpis:
        if metric not in row:
            continue

        output.append({
            "metric": label,
            "value": format_metric_value(row.get(metric), metric),
            "raw_metric": metric,
            "raw_value": row.get(metric),
        })

    return output


def make_warning_messages(row: dict[str, Any] | None) -> list[str]:
    if row is None:
        return ["No data available for this city and occupation."]

    warnings_list = []

    if bool(row.get("low_employment_warning", False)):
        warnings_list.append("Low local employment estimate. Interpret this ranking cautiously.")

    if bool(row.get("low_lq_warning", False)):
        warnings_list.append("This occupation is not highly concentrated in this metro.")

    if bool(row.get("high_cost_warning", False)):
        warnings_list.append("This is a high-cost metro based on BEA RPP.")

    if "predicted_improving_real_income_after_rent" in row:
        if bool(row.get("predicted_improving_real_income_after_rent", False)):
            warnings_list.append("Model predicts improving RPP-adjusted income after rent.")
        else:
            warnings_list.append("Model does not predict improving RPP-adjusted income after rent.")

    return warnings_list


# -----------------------------------------------------------------------------
# Similar cities
# -----------------------------------------------------------------------------

def get_similar_cities(
    nearest_metros_df: Any,
    *,
    city: str,
    top_n: int = 10,
) -> pl.DataFrame:
    df = to_polars(nearest_metros_df)

    if df.height == 0 or "source_metro" not in df.columns:
        return pl.DataFrame()

    query = """
        SELECT *
        FROM df
        WHERE source_metro = ?
        ORDER BY neighbor_rank
        LIMIT ?
    """

    result = duckdb_query(df, query, [city, top_n])

    if result.height > 0:
        return result

    query_fuzzy = """
        SELECT *
        FROM df
        WHERE LOWER(source_metro) LIKE ?
        ORDER BY neighbor_rank
        LIMIT ?
    """

    return duckdb_query(df, query_fuzzy, [f"%{city.lower()}%", top_n])


# -----------------------------------------------------------------------------
# Similar occupations
# -----------------------------------------------------------------------------

def build_occupation_profiles(
    dashboard_df: Any,
    *,
    min_metros: int = 10,
) -> pl.DataFrame:
    df = to_polars(dashboard_df)

    if df.height == 0 or "occ_title" not in df.columns:
        return pl.DataFrame()

    numeric_cols = [
        "wage_annual_median",
        "real_wage_annual_median",
        "annual_rent",
        "rpp_all_items",
        "rent_to_median_wage_ratio",
        "real_income_after_rent_median",
        "tot_emp",
        "loc_quotient",
        "predicted_change_in_real_income_after_rent",
        "career_city_score_0_100",
    ]

    df = ensure_numeric_columns(df, [c for c in numeric_cols if c in df.columns])

    group_cols = ["occ_title"]

    if "occ_code" in df.columns:
        group_cols = ["occ_code", "occ_title"]

    aggs = []

    if "cbsa_code" in df.columns:
        aggs.append(pl.col("cbsa_code").n_unique().alias("n_metros"))
    else:
        aggs.append(pl.len().alias("n_metros"))

    agg_map = {
        "wage_annual_median": [
            pl.mean("wage_annual_median").alias("mean_wage"),
            pl.median("wage_annual_median").alias("median_wage"),
            pl.std("wage_annual_median").alias("wage_dispersion"),
        ],
        "real_wage_annual_median": [
            pl.mean("real_wage_annual_median").alias("mean_real_wage"),
            pl.median("real_wage_annual_median").alias("median_real_wage"),
        ],
        "rent_to_median_wage_ratio": [
            pl.mean("rent_to_median_wage_ratio").alias("mean_rent_burden"),
            pl.median("rent_to_median_wage_ratio").alias("median_rent_burden"),
        ],
        "real_income_after_rent_median": [
            pl.mean("real_income_after_rent_median").alias("mean_real_income_after_rent"),
            pl.median("real_income_after_rent_median").alias("median_real_income_after_rent"),
            pl.std("real_income_after_rent_median").alias("real_income_dispersion"),
        ],
        "tot_emp": [
            pl.sum("tot_emp").alias("total_employment"),
        ],
        "loc_quotient": [
            pl.median("loc_quotient").alias("median_lq"),
            pl.mean("loc_quotient").alias("mean_lq"),
        ],
        "predicted_change_in_real_income_after_rent": [
            pl.mean("predicted_change_in_real_income_after_rent").alias("mean_predicted_change"),
        ],
        "career_city_score_0_100": [
            pl.mean("career_city_score_0_100").alias("mean_score"),
        ],
    }

    for source_col, expressions in agg_map.items():
        if source_col in df.columns:
            aggs.extend(expressions)

    profiles = df.group_by(group_cols).agg(aggs)

    profiles = profiles.filter(pl.col("n_metros") >= min_metros)

    if "total_employment" in profiles.columns:
        profiles = profiles.with_columns(
            pl.when(pl.col("total_employment").fill_null(0) > 0)
            .then(pl.col("total_employment").fill_null(0).log1p())
            .otherwise(0.0)
            .alias("log_total_employment")
        )

    return profiles.sort("occ_title")


def get_similar_occupations(
    dashboard_df: Any,
    *,
    occupation: str,
    top_n: int = 10,
) -> pl.DataFrame:
    profiles = build_occupation_profiles(dashboard_df)

    if profiles.height == 0 or "occ_title" not in profiles.columns:
        return pl.DataFrame()

    occ_titles = profiles.get_column("occ_title").to_list()

    if occupation not in occ_titles:
        return pl.DataFrame()

    feature_cols = [
        "mean_wage",
        "mean_real_wage",
        "mean_rent_burden",
        "mean_real_income_after_rent",
        "log_total_employment",
        "median_lq",
        "mean_predicted_change",
        "wage_dispersion",
        "real_income_dispersion",
        "mean_score",
    ]

    feature_cols = [c for c in feature_cols if c in profiles.columns]

    if not feature_cols:
        return pl.DataFrame()

    X = profiles.select(feature_cols).to_numpy()

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    X_scaled = pipeline.fit_transform(X)

    nn = NearestNeighbors(
        n_neighbors=min(top_n + 1, profiles.height),
        metric="euclidean",
    )

    nn.fit(X_scaled)

    source_idx = occ_titles.index(occupation)
    distances, indices = nn.kneighbors(X_scaled[source_idx].reshape(1, -1))

    rows = []

    for rank_position, idx in enumerate(indices[0], start=0):
        if idx == source_idx:
            continue

        row = profiles.row(int(idx), named=True)
        row["neighbor_rank"] = len(rows) + 1
        row["distance"] = float(distances[0][rank_position])
        rows.append(row)

        if len(rows) >= top_n:
            break

    return pl.DataFrame(rows)


# -----------------------------------------------------------------------------
# Model and cluster helpers
# -----------------------------------------------------------------------------

def get_model_metrics_for_target(
    metrics_df: Any,
    target_label_contains: str | None = None,
) -> pl.DataFrame:
    df = to_polars(metrics_df)

    if df.height == 0:
        return df

    if target_label_contains and "target" in df.columns:
        df = df.filter(pl.col("target").str.to_lowercase().str.contains(target_label_contains.lower()))

    return df


def get_feature_importance(
    importance_df: Any,
    *,
    top_n: int = 20,
) -> pl.DataFrame:
    df = to_polars(importance_df)

    if df.height == 0 or "importance" not in df.columns:
        return pl.DataFrame()

    return (
        df
        .with_columns(pl.col("importance").cast(pl.Float64, strict=False))
        .drop_nulls("importance")
        .sort("importance", descending=True)
        .head(top_n)
    )


def get_cluster_members(
    metro_profiles_df: Any,
    *,
    cluster_id: int | None = None,
    top_n: int = 25,
) -> pl.DataFrame:
    df = to_polars(metro_profiles_df)

    if df.height == 0:
        return df

    if cluster_id is not None and "metro_cluster" in df.columns:
        df = df.filter(pl.col("metro_cluster").cast(pl.Int64, strict=False) == int(cluster_id))

    sort_col = first_available_column(
        df,
        [
            "mean_real_income_after_rent",
            "median_real_income_after_rent",
            "mean_real_occ_wage",
            "population",
        ],
    )

    # if sort_col:
    #     df = df.sort(sort_col, descending=True)

    return df.head(top_n)


# -----------------------------------------------------------------------------
# SQL helpers for dashboard verification / summaries
# -----------------------------------------------------------------------------

def sql_top_cities_for_occupation(
    dashboard_df: Any,
    *,
    occupation: str,
    top_n: int = config.DEFAULT_TOP_N,
    min_employment: float = config.DEFAULT_MIN_EMPLOYMENT,
) -> pl.DataFrame:
    df = to_polars(dashboard_df)

    query = """
        SELECT
            bls_msa_name,
            occ_title,
            custom_score_0_100,
            career_city_score_0_100,
            wage_annual_median,
            real_wage_annual_median,
            annual_rent,
            rpp_all_items,
            real_income_after_rent_median,
            rent_to_median_wage_ratio,
            tot_emp,
            loc_quotient,
            predicted_change_in_real_income_after_rent
        FROM df
        WHERE occ_title = ?
          AND COALESCE(tot_emp, 0) >= ?
        ORDER BY COALESCE(custom_score_0_100, career_city_score_0_100) DESC
        LIMIT ?
    """

    return duckdb_query(df, query, [occupation, min_employment, top_n])


def sql_cluster_summary(dashboard_df: Any) -> pl.DataFrame:
    df = to_polars(dashboard_df)

    query = """
        SELECT
            metro_cluster,
            COUNT(DISTINCT cbsa_code) AS n_metros,
            ROUND(AVG(rpp_all_items), 2) AS avg_rpp_all_items,
            ROUND(AVG(annual_rent), 0) AS avg_annual_rent,
            ROUND(AVG(real_income_after_rent_median), 0) AS avg_real_income_after_rent,
            ROUND(AVG(career_city_score_0_100), 2) AS avg_career_city_score
        FROM df
        WHERE metro_cluster IS NOT NULL
        GROUP BY metro_cluster
        ORDER BY metro_cluster
    """

    return duckdb_query(df, query)


# -----------------------------------------------------------------------------
# Table helpers
# -----------------------------------------------------------------------------

def select_display_columns(
    df: Any,
    columns: list[str],
) -> pl.DataFrame:
    df_pl = to_polars(df)

    if df_pl.height == 0:
        return df_pl

    available = [col for col in columns if col in df_pl.columns]

    if not available:
        return df_pl

    return df_pl.select(available)


def prepare_dash_table(
    df: Any,
    *,
    columns: list[str] | None = None,
    max_rows: int = config.MAX_TABLE_ROWS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df_pl = to_polars(df)

    if df_pl.height == 0:
        return [], []

    display_df = select_display_columns(df_pl, columns) if columns else df_pl.clone()
    display_df = display_df.head(max_rows)

    table_columns = [
        {
            "name": config.METRIC_LABELS.get(col, col.replace("_", " ").title()),
            "id": col,
        }
        for col in display_df.columns
    ]

    return display_df.to_dicts(), table_columns


def summarize_weight_settings(weights: dict[str, float] | None) -> pl.DataFrame:
    normalized = normalize_weights(weights)

    rows = []

    for key, normalized_value in normalized.items():
        rows.append({
            "weight_key": key,
            "label": config.WEIGHT_LABELS.get(key, key),
            "normalized_weight": normalized_value,
            "percent": 100 * normalized_value,
            "description": config.WEIGHT_DESCRIPTIONS.get(key, ""),
        })

    return pl.DataFrame(rows)