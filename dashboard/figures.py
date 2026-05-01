"""
Plotly figure utilities for the Career Atlas dashboard.

This file contains reusable functions for:
- maps
- rankings
- scatterplots
- trend charts
- boxplots
- PCA cluster charts
- model diagnostics
- feature importance plots

All filtering, ranking, sorting, and transformation work is done with Polars.
Small plotting frames are converted to Pandas only at the final Plotly Express
boundary for maximum Plotly compatibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

try:
    from . import analytics, config
except ImportError:
    import analytics
    import config


# -----------------------------------------------------------------------------
# General figure helpers
# -----------------------------------------------------------------------------

def to_polars(df: Any) -> pl.DataFrame:
    return analytics.to_polars(df)


def empty_figure(
    message: str = "No data available for this selection.",
    *,
    title: str | None = None,
    height: int = 420,
) -> go.Figure:
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color=config.COLOR_MUTED),
        align="center",
    )

    fig.update_layout(
        title=title or "",
        template=config.PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=30, r=30, t=60, b=30),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def apply_standard_layout(
    fig: go.Figure,
    *,
    height: int = 440,
    title: str | None = None,
) -> go.Figure:
    fig.update_layout(
        template=config.PLOTLY_TEMPLATE,
        height=height,
        title=title if title is not None else fig.layout.title.text,
        margin=dict(l=30, r=30, t=60, b=35),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", color=config.COLOR_TEXT),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )

    return fig


def _metric_label(metric: str) -> str:
    return config.METRIC_LABELS.get(metric, metric.replace("_", " ").title())

def _color_scale_for_metric(metric: str | None):
    """
    Return an intuitive red-yellow-green scale for a metric.

    If higher values are better:
        low = red, high = green

    If lower values are better:
        low = green, high = red
    """

    if metric is None:
        return config.COLOR_SCALE_GOOD_HIGH

    higher_is_better = config.METRIC_HIGHER_IS_BETTER.get(metric)

    if higher_is_better is True:
        return config.COLOR_SCALE_GOOD_HIGH

    if higher_is_better is False:
        return config.COLOR_SCALE_GOOD_LOW

    return config.COLOR_SCALE_NEUTRAL

def _has_coordinates(df: pl.DataFrame) -> bool:
    return (
        df.height > 0
        and "latitude" in df.columns
        and "longitude" in df.columns
        and df.select((pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()).any()).item()
    )


def _add_safe_size_column(
    df: pl.DataFrame,
    size_col: str,
    *,
    output_col: str = "_plot_size",
    minimum: float = 5,
    maximum: float = 35,
) -> pl.DataFrame:
    if df.height == 0 or size_col not in df.columns:
        return df.with_columns(pl.lit(minimum).alias(output_col))

    df = df.with_columns(
        pl.when(pl.col(size_col).cast(pl.Float64, strict=False).fill_null(0) > 0)
        .then(pl.col(size_col).cast(pl.Float64, strict=False).fill_null(0))
        .otherwise(0.0)
        .alias("_raw_size_metric")
    )

    max_value = df.select(pl.col("_raw_size_metric").max()).item()

    if not max_value or max_value <= 0:
        return df.with_columns(pl.lit(minimum).alias(output_col)).drop("_raw_size_metric")

    return (
        df
        .with_columns(
            (
                minimum
                + (maximum - minimum)
                * (pl.col("_raw_size_metric") / max_value).sqrt()
            ).alias(output_col)
        )
        .drop("_raw_size_metric")
    )


def _available_hover_cols(df: pl.DataFrame, candidates: list[str]) -> list[str]:
    return [col for col in candidates if col in df.columns]


# -----------------------------------------------------------------------------
# Ranking figures
# -----------------------------------------------------------------------------

def make_city_ranking_bar(
    df: Any,
    *,
    score_col: str = "custom_score_0_100",
    top_n: int = config.DEFAULT_TOP_N,
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(df)

    if df_pl.height == 0 or "bls_msa_name" not in df_pl.columns or score_col not in df_pl.columns:
        return empty_figure("No ranked cities available.", title=title)

    plot_df = (
        df_pl
        .drop_nulls([score_col, "bls_msa_name"])
        .sort(score_col, descending=True)
        .head(top_n)
        .sort(score_col)
    )

    if plot_df.height == 0:
        return empty_figure("No ranked cities available.", title=title)

    hover_cols = _available_hover_cols(
        plot_df,
        [
            "real_income_after_rent_median",
            "real_wage_annual_median",
            "annual_rent",
            "rpp_all_items",
            "tot_emp",
            "loc_quotient",
            "predicted_change_in_real_income_after_rent",
        ],
    )

    fig = px.bar(
        plot_df.to_pandas(),
        x=score_col,
        y="bls_msa_name",
        orientation="h",
        hover_data=hover_cols,
        labels={
            score_col: _metric_label(score_col),
            "bls_msa_name": "Metro",
        },
        title=title or f"Top {top_n} Cities by {_metric_label(score_col)}",
    )

    fig.update_traces(marker_color=config.COLOR_SECONDARY, opacity=0.9)

    return apply_standard_layout(fig, height=520)


def make_occupation_ranking_bar(
    df: Any,
    *,
    score_col: str = "custom_score_0_100",
    top_n: int = config.DEFAULT_TOP_N,
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(df)

    if df_pl.height == 0 or "occ_title" not in df_pl.columns or score_col not in df_pl.columns:
        return empty_figure("No ranked occupations available.", title=title)

    plot_df = (
        df_pl
        .drop_nulls([score_col, "occ_title"])
        .sort(score_col, descending=True)
        .head(top_n)
        .sort(score_col)
    )

    if plot_df.height == 0:
        return empty_figure("No ranked occupations available.", title=title)

    hover_cols = _available_hover_cols(
        plot_df,
        [
            "real_income_after_rent_median",
            "real_wage_annual_median",
            "rent_to_median_wage_ratio",
            "tot_emp",
            "loc_quotient",
            "predicted_change_in_real_income_after_rent",
        ],
    )

    fig = px.bar(
        plot_df.to_pandas(),
        x=score_col,
        y="occ_title",
        orientation="h",
        hover_data=hover_cols,
        labels={
            score_col: _metric_label(score_col),
            "occ_title": "Occupation",
        },
        title=title or f"Top {top_n} Occupations by {_metric_label(score_col)}",
    )

    fig.update_traces(marker_color=config.COLOR_ACCENT, opacity=0.9)

    return apply_standard_layout(fig, height=560)

def make_metric_extremes_bar(
    best_df: Any,
    worst_df: Any,
    *,
    metric: str,
    title: str | None = None,
) -> go.Figure:
    """
    Visualize best and worst occupation-metro combinations for a selected metric.
    """

    best_pl = to_polars(best_df)
    worst_pl = to_polars(worst_df)

    if best_pl.height == 0 and worst_pl.height == 0:
        return empty_figure("No best/worst combinations available for this metric.", title=title)

    frames = []

    if best_pl.height > 0:
        frames.append(best_pl)

    if worst_pl.height > 0:
        frames.append(worst_pl)

    plot_df = pl.concat(frames, how="diagonal_relaxed")

    required = ["career_city_combo", "metric_value", "extreme_type"]

    if not all(col in plot_df.columns for col in required):
        return empty_figure("Best/worst metric data is missing required columns.", title=title)

    plot_df = plot_df.with_columns(
        pl.concat_str(
            [
                pl.col("extreme_type"),
                pl.lit(" #"),
                pl.col("rank_position").cast(pl.Utf8),
                pl.lit(": "),
                pl.col("career_city_combo"),
            ],
            ignore_nulls=True,
        ).alias("display_label")
    )

    # Put worst and best in a readable order.
    plot_df = plot_df.sort(["extreme_type", "rank_position"], descending=[True, True])

    fig = px.bar(
        plot_df.to_pandas(),
        x="metric_value",
        y="display_label",
        color="extreme_type",
        orientation="h",
        hover_data=[
            col for col in [
                "occ_title",
                "bls_msa_name",
                "custom_score_0_100",
                "career_city_score_0_100",
                "real_income_after_rent_median",
                "real_wage_annual_median",
                "annual_rent",
                "rpp_all_items",
                "rent_to_median_wage_ratio",
                "tot_emp",
                "loc_quotient",
                "predicted_change_in_real_income_after_rent",
            ]
            if col in plot_df.columns
        ],
        labels={
            "metric_value": _metric_label(metric),
            "display_label": "Career + City",
            "extreme_type": "Type",
        },
        title=title or f"Best and Worst Career-City Combinations by {_metric_label(metric)}",
        color_discrete_map={
            "Best": config.COLOR_ACCENT,
            "Worst": config.COLOR_DANGER,
        },
    )

    return apply_standard_layout(fig, height=720)

# -----------------------------------------------------------------------------
# Maps
# -----------------------------------------------------------------------------

def make_city_score_map(
    df: Any,
    *,
    color_col: str = "custom_score_0_100",
    size_col: str = "tot_emp",
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(df)

    if not _has_coordinates(df_pl):
        return empty_figure(
            "Map unavailable because CBSA centroid coordinates are missing.",
            title=title or "City Map",
        )

    plot_df = df_pl.drop_nulls(["latitude", "longitude"])

    if plot_df.height == 0:
        return empty_figure("No city coordinates available.", title=title or "City Map")

    if color_col not in plot_df.columns:
        color_col = "career_city_score_0_100" if "career_city_score_0_100" in plot_df.columns else None

    plot_df = _add_safe_size_column(plot_df, size_col, output_col="_map_size")

    hover_cols = _available_hover_cols(
        plot_df,
        [
            "occ_title",
            "custom_score_0_100",
            "career_city_score_0_100",
            "real_income_after_rent_median",
            "real_wage_annual_median",
            "annual_rent",
            "rpp_all_items",
            "rent_to_median_wage_ratio",
            "tot_emp",
            "loc_quotient",
            "predicted_change_in_real_income_after_rent",
        ],
    )

    fig = px.scatter_mapbox(
        plot_df.to_pandas(),
        lat="latitude",
        lon="longitude",
        color=color_col,
        size="_map_size",
        hover_name="bls_msa_name" if "bls_msa_name" in plot_df.columns else None,
        hover_data=hover_cols,
        color_continuous_scale="Viridis",
        # color_continuous_scale=_color_scale_for_metric(color_col),
        mapbox_style=config.MAP_STYLE,
        zoom=config.MAP_DEFAULT_ZOOM,
        center=config.MAP_DEFAULT_CENTER,
        title=title or "Career-City Map",
    )

    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", color=config.COLOR_TEXT),
    )

    return fig


def make_similar_cities_map(
    nearest_df: Any,
    *,
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(nearest_df)

    required = [
        "source_latitude",
        "source_longitude",
        "neighbor_latitude",
        "neighbor_longitude",
        "source_metro",
        "neighbor_metro",
    ]

    if df_pl.height == 0 or not all(col in df_pl.columns for col in required):
        return empty_figure(
            "Similar-city map unavailable because coordinates are missing.",
            title=title or "Similar Cities Map",
            height=560,
        )

    plot_df = df_pl.drop_nulls([
        "source_latitude",
        "source_longitude",
        "neighbor_latitude",
        "neighbor_longitude",
    ])

    if plot_df.height == 0:
        return empty_figure("No similar-city coordinates available.", title=title or "Similar Cities Map")

    source = plot_df.row(0, named=True)

    fig = go.Figure()

    for row in plot_df.iter_rows(named=True):
        fig.add_trace(
            go.Scattermapbox(
                lat=[row["source_latitude"], row["neighbor_latitude"]],
                lon=[row["source_longitude"], row["neighbor_longitude"]],
                mode="lines",
                line=dict(width=1.5, color="rgba(31, 59, 87, 0.35)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scattermapbox(
            lat=[source["source_latitude"]],
            lon=[source["source_longitude"]],
            mode="markers",
            marker=dict(size=18, color=config.COLOR_DANGER),
            text=[source["source_metro"]],
            name="Selected metro",
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

    neighbor_rows = plot_df.to_dicts()

    fig.add_trace(
        go.Scattermapbox(
            lat=[row["neighbor_latitude"] for row in neighbor_rows],
            lon=[row["neighbor_longitude"] for row in neighbor_rows],
            mode="markers",
            marker=dict(
                size=12,
                color=[row.get("neighbor_rank") for row in neighbor_rows],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Rank"),
            ),
            text=[row["neighbor_metro"] for row in neighbor_rows],
            customdata=[
                [row.get("neighbor_rank"), row.get("distance")]
                for row in neighbor_rows
            ],
            name="Similar metros",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Rank: %{customdata[0]}<br>"
                "Distance: %{customdata[1]:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or f"Cities Similar to {source['source_metro']}",
        mapbox=dict(
            style=config.MAP_STYLE,
            center=dict(
                lat=float(source["source_latitude"]),
                lon=float(source["source_longitude"]),
            ),
            zoom=3,
        ),
        height=580,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", color=config.COLOR_TEXT),
    )

    return fig


# -----------------------------------------------------------------------------
# Scatterplots and tradeoff charts
# -----------------------------------------------------------------------------

def make_tradeoff_scatter(
    df: Any,
    *,
    selected_city: str | None = None,
    x_col: str = "rent_to_median_wage_ratio",
    y_col: str = "real_income_after_rent_median",
    color_col: str = "loc_quotient",
    size_col: str = "tot_emp",
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(df)

    if df_pl.height == 0 or x_col not in df_pl.columns or y_col not in df_pl.columns:
        return empty_figure("No data available for this tradeoff chart.", title=title)

    plot_df = df_pl.drop_nulls([x_col, y_col])

    if plot_df.height == 0:
        return empty_figure("No valid rows for this tradeoff chart.", title=title)

    plot_df = _add_safe_size_column(plot_df, size_col)

    hover_cols = _available_hover_cols(
        plot_df,
        [
            "bls_msa_name",
            "occ_title",
            "custom_score_0_100",
            "real_wage_annual_median",
            "annual_rent",
            "rpp_all_items",
            "tot_emp",
            "loc_quotient",
            "predicted_change_in_real_income_after_rent",
        ],
    )

    fig = px.scatter(
        plot_df.to_pandas(),
        x=x_col,
        y=y_col,
        color=color_col if color_col in plot_df.columns else None,
        size="_plot_size",
        hover_name="bls_msa_name" if "bls_msa_name" in plot_df.columns else None,
        hover_data=hover_cols,
        labels={
            x_col: _metric_label(x_col),
            y_col: _metric_label(y_col),
            color_col: _metric_label(color_col),
        },
        title=title or f"{_metric_label(y_col)} vs. {_metric_label(x_col)}",
        color_continuous_scale="Viridis",
        # color_continuous_scale=_color_scale_for_metric(color_col),
    )

    if selected_city and "bls_msa_name" in plot_df.columns:
        selected = plot_df.filter(pl.col("bls_msa_name") == selected_city)

        if selected.height > 0:
            fig.add_trace(
                go.Scatter(
                    x=selected.get_column(x_col).to_list(),
                    y=selected.get_column(y_col).to_list(),
                    mode="markers+text",
                    marker=dict(
                        size=18,
                        color=config.COLOR_DANGER,
                        line=dict(color="white", width=2),
                        symbol="star",
                    ),
                    text=selected.get_column("bls_msa_name").to_list(),
                    textposition="top center",
                    name="Selected city",
                    hovertemplate="<b>%{text}</b><extra></extra>",
                )
            )

    return apply_standard_layout(fig, height=560)


# -----------------------------------------------------------------------------
# Trends and distributions
# -----------------------------------------------------------------------------

def make_trend_chart(
    history_df: Any,
    *,
    metrics: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(history_df)

    if df_pl.height == 0 or "year" not in df_pl.columns:
        return empty_figure("No historical data available.", title=title or "Trend")

    metrics = metrics or [
        "rent_to_median_wage_ratio",
        "real_income_after_rent_median",
    ]

    available_metrics = [metric for metric in metrics if metric in df_pl.columns]

    if not available_metrics:
        return empty_figure("No requested trend metrics are available.", title=title or "Trend")

    plot_df = (
        df_pl
        .select(["year"] + available_metrics)
        .with_columns(pl.col("year").cast(pl.Int64, strict=False))
        .drop_nulls("year")
    )

    long_df = (
        plot_df
        .melt(
            id_vars=["year"],
            value_vars=available_metrics,
            variable_name="metric",
            value_name="value",
        )
        .drop_nulls("value")
        .with_columns(
            pl.col("metric")
            .replace({metric: _metric_label(metric) for metric in available_metrics})
            .alias("metric_label")
        )
    )

    if long_df.height == 0:
        return empty_figure("No non-missing trend values are available.", title=title or "Trend")

    fig = px.line(
        long_df.to_pandas(),
        x="year",
        y="value",
        color="metric_label",
        markers=True,
        labels={
            "year": "Year",
            "value": "Metric value",
            "metric_label": "Metric",
        },
        title=title or "City–Occupation Trend",
    )

    return apply_standard_layout(fig, height=450)


def make_metric_boxplot(
    distribution_df: Any,
    *,
    metric: str,
    selected_city: str | None = None,
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(distribution_df)

    if df_pl.height == 0 or metric not in df_pl.columns:
        return empty_figure("No distribution data available.", title=title)

    plot_df = df_pl.drop_nulls(metric)

    if plot_df.height == 0:
        return empty_figure("No valid metric values available.", title=title)

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=plot_df.get_column(metric).to_list(),
            name=_metric_label(metric),
            boxpoints="outliers",
            marker_color=config.COLOR_SECONDARY,
        )
    )

    if selected_city and "bls_msa_name" in plot_df.columns:
        selected = plot_df.filter(pl.col("bls_msa_name") == selected_city)

        if selected.height > 0:
            fig.add_trace(
                go.Scatter(
                    x=[_metric_label(metric)] * selected.height,
                    y=selected.get_column(metric).to_list(),
                    mode="markers+text",
                    marker=dict(size=14, color=config.COLOR_DANGER, symbol="star"),
                    text=selected.get_column("bls_msa_name").to_list(),
                    textposition="top center",
                    name="Selected city",
                )
            )

    fig.update_layout(
        title=title or f"Distribution of {_metric_label(metric)} Across Metros",
        yaxis_title=_metric_label(metric),
        showlegend=True,
    )

    return apply_standard_layout(fig, height=460)


def make_metric_histogram(
    df: Any,
    *,
    metric: str,
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(df)

    if df_pl.height == 0 or metric not in df_pl.columns:
        return empty_figure("No data available for histogram.", title=title)

    plot_df = df_pl.drop_nulls(metric)

    if plot_df.height == 0:
        return empty_figure("No valid metric values available.", title=title)

    fig = px.histogram(
        plot_df.to_pandas(),
        x=metric,
        nbins=40,
        labels={metric: _metric_label(metric)},
        title=title or f"Distribution of {_metric_label(metric)}",
    )

    fig.update_traces(marker_color=config.COLOR_SECONDARY, opacity=0.85)

    return apply_standard_layout(fig, height=420)


# -----------------------------------------------------------------------------
# Clusters and PCA
# -----------------------------------------------------------------------------

def make_pca_cluster_plot(
    metro_profiles_df: Any,
    *,
    selected_city: str | None = None,
    cluster_id: int | None = None,
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(metro_profiles_df)

    required = ["pca_1", "pca_2", "metro_cluster", "bls_msa_name"]

    if df_pl.height == 0 or not all(col in df_pl.columns for col in required):
        return empty_figure("PCA cluster data is unavailable.", title=title or "Metro Clusters")

    plot_df = df_pl.clone()

    if cluster_id is not None:
        plot_df = plot_df.filter(pl.col("metro_cluster").cast(pl.Int64, strict=False) == int(cluster_id))

    if plot_df.height == 0:
        return empty_figure("No metros available for the selected cluster.", title=title or "Metro Clusters")

    hover_cols = _available_hover_cols(
        plot_df,
        [
            "annual_rent",
            "rpp_all_items",
            "mean_real_occ_wage",
            "mean_real_income_after_rent",
            "mean_rent_burden",
            "share_high_lq",
            "population",
        ],
    )

    fig = px.scatter(
        plot_df.to_pandas(),
        x="pca_1",
        y="pca_2",
        color="metro_cluster",
        hover_name="bls_msa_name",
        hover_data=hover_cols,
        labels={
            "pca_1": "PCA Component 1",
            "pca_2": "PCA Component 2",
            "metro_cluster": "Cluster",
        },
        title=title or "Metro Clusters",
        color_continuous_scale="Viridis",
        # color_continuous_scale=config.COLOR_SCALE_NEUTRAL,
    )

    if selected_city:
        selected = plot_df.filter(pl.col("bls_msa_name") == selected_city)

        if selected.height > 0:
            fig.add_trace(
                go.Scatter(
                    x=selected.get_column("pca_1").to_list(),
                    y=selected.get_column("pca_2").to_list(),
                    mode="markers+text",
                    marker=dict(
                        size=18,
                        color=config.COLOR_DANGER,
                        symbol="star",
                        line=dict(color="white", width=2),
                    ),
                    text=selected.get_column("bls_msa_name").to_list(),
                    textposition="top center",
                    name="Selected metro",
                )
            )

    return apply_standard_layout(fig, height=560)


def make_cluster_summary_bar(
    cluster_summary_df: Any,
    *,
    metric: str = "mean_real_income_after_rent",
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(cluster_summary_df)

    if df_pl.height == 0 or metric not in df_pl.columns:
        return empty_figure("Cluster summary data is unavailable.", title=title)

    plot_df = df_pl.drop_nulls(metric)

    if plot_df.height == 0:
        return empty_figure("No valid cluster summary values.", title=title)

    fig = px.bar(
        plot_df.to_pandas(),
        x="metro_cluster",
        y=metric,
        labels={
            "metro_cluster": "Metro Cluster",
            metric: _metric_label(metric),
        },
        title=title or f"Cluster Summary: {_metric_label(metric)}",
    )

    fig.update_traces(marker_color=config.COLOR_SECONDARY, opacity=0.9)

    return apply_standard_layout(fig, height=420)


# -----------------------------------------------------------------------------
# Similarity figures
# -----------------------------------------------------------------------------

def make_similar_city_comparison_bar(
    nearest_df: Any,
    *,
    metric: str = "neighbor_mean_real_income_after_rent",
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(nearest_df)

    if df_pl.height == 0 or metric not in df_pl.columns or "neighbor_metro" not in df_pl.columns:
        return empty_figure("No similar-city comparison data available.", title=title)

    sort_col = "neighbor_rank" if "neighbor_rank" in df_pl.columns else metric

    plot_df = (
        df_pl
        .drop_nulls([metric, "neighbor_metro"])
        .sort(sort_col)
        .sort(sort_col, descending=True)
    )

    if plot_df.height == 0:
        return empty_figure("No valid similar-city comparison values.", title=title)

    fig = px.bar(
        plot_df.to_pandas(),
        x=metric,
        y="neighbor_metro",
        orientation="h",
        color="neighbor_cluster" if "neighbor_cluster" in plot_df.columns else None,
        labels={
            metric: _metric_label(metric),
            "neighbor_metro": "Similar Metro",
            "neighbor_cluster": "Cluster",
        },
        title=title or f"Similar Cities by {_metric_label(metric)}",
    )

    return apply_standard_layout(fig, height=480)


def make_similar_occupation_bar(
    similar_occ_df: Any,
    *,
    metric: str = "mean_real_income_after_rent",
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(similar_occ_df)

    if df_pl.height == 0 or metric not in df_pl.columns or "occ_title" not in df_pl.columns:
        return empty_figure("No similar occupations available.", title=title)

    sort_col = "neighbor_rank" if "neighbor_rank" in df_pl.columns else metric

    plot_df = (
        df_pl
        .drop_nulls([metric, "occ_title"])
        .sort(sort_col)
        .sort(sort_col, descending=True)
    )

    if plot_df.height == 0:
        return empty_figure("No similar occupations available.", title=title)

    fig = px.bar(
        plot_df.to_pandas(),
        x=metric,
        y="occ_title",
        orientation="h",
        color="distance" if "distance" in plot_df.columns else None,
        labels={
            metric: _metric_label(metric),
            "occ_title": "Similar Occupation",
            "distance": "Similarity Distance",
        },
        title=title or "Similar Occupations",
        color_continuous_scale="Viridis_r",
        # color_continuous_scale=config.COLOR_SCALE_GOOD_LOW,
    )

    return apply_standard_layout(fig, height=520)


# -----------------------------------------------------------------------------
# Predictive modeling figures
# -----------------------------------------------------------------------------

def make_model_metrics_chart(
    metrics_df: Any,
    *,
    target_contains: str | None = None,
    metric: str = "mae",
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(metrics_df)

    if df_pl.height == 0 or metric not in df_pl.columns:
        return empty_figure("No model metrics available.", title=title)

    plot_df = df_pl.clone()

    if target_contains and "target" in plot_df.columns:
        plot_df = plot_df.filter(
            pl.col("target").str.to_lowercase().str.contains(target_contains.lower())
        )

    if plot_df.height == 0:
        return empty_figure("No model metrics available for the selected target.", title=title)

    fig = px.bar(
        plot_df.to_pandas(),
        x="model",
        y=metric,
        color="split" if "split" in plot_df.columns else None,
        barmode="group",
        labels={
            "model": "Model",
            metric: metric.upper(),
            "split": "Split",
        },
        title=title or f"Model Comparison by {metric.upper()}",
    )

    fig.update_layout(xaxis_tickangle=35)

    return apply_standard_layout(fig, height=500)


def make_feature_importance_chart(
    importance_df: Any,
    *,
    top_n: int = 20,
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(importance_df)

    if df_pl.height == 0 or "feature" not in df_pl.columns or "importance" not in df_pl.columns:
        return empty_figure("Feature-importance data is unavailable.", title=title)

    plot_df = (
        df_pl
        .with_columns(pl.col("importance").cast(pl.Float64, strict=False))
        .drop_nulls(["feature", "importance"])
        .sort("importance", descending=True)
        .head(top_n)
        .sort("importance")
    )

    if plot_df.height == 0:
        return empty_figure("No feature-importance values available.", title=title)

    fig = px.bar(
        plot_df.to_pandas(),
        x="importance",
        y="feature",
        orientation="h",
        labels={
            "importance": "Importance",
            "feature": "Feature",
        },
        title=title or "Random Forest Feature Importance",
    )

    fig.update_traces(marker_color=config.COLOR_PRIMARY, opacity=0.9)

    return apply_standard_layout(fig, height=560)


def make_prediction_change_bar(
    df: Any,
    *,
    metric: str = "predicted_change_in_real_income_after_rent",
    label_col: str = "bls_msa_name",
    top_n: int = config.DEFAULT_TOP_N,
    ascending: bool = False,
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(df)

    if df_pl.height == 0 or metric not in df_pl.columns or label_col not in df_pl.columns:
        return empty_figure("No prediction data available.", title=title)

    plot_df = (
        df_pl
        .drop_nulls([metric, label_col])
        .sort(metric, descending=not ascending)
        .head(top_n)
        .sort(metric, descending=ascending)
    )

    if plot_df.height == 0:
        return empty_figure("No valid prediction values available.", title=title)

    fig = px.bar(
        plot_df.to_pandas(),
        x=metric,
        y=label_col,
        orientation="h",
        labels={
            metric: _metric_label(metric),
            label_col: "Metro" if label_col == "bls_msa_name" else "Label",
        },
        title=title or f"Top Predicted Changes: {_metric_label(metric)}",
    )

    fig.update_traces(marker_color=config.COLOR_ACCENT, opacity=0.9)

    return apply_standard_layout(fig, height=520)


# -----------------------------------------------------------------------------
# Detail / comparison charts
# -----------------------------------------------------------------------------

def make_city_occupation_metric_cards_data(
    row: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    return analytics.make_kpi_summary(row)


def make_two_city_comparison_bar(
    comparison_df: Any,
    *,
    city_col: str = "bls_msa_name",
    metrics: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    df_pl = to_polars(comparison_df)

    if df_pl.height == 0 or city_col not in df_pl.columns:
        return empty_figure("No comparison data available.", title=title)

    metrics = metrics or [
        "real_income_after_rent_median",
        "real_wage_annual_median",
        "annual_rent",
        "rent_to_median_wage_ratio",
        "loc_quotient",
    ]

    available_metrics = [metric for metric in metrics if metric in df_pl.columns]

    if not available_metrics:
        return empty_figure("No comparison metrics available.", title=title)

    long_df = (
        df_pl
        .select([city_col] + available_metrics)
        .melt(
            id_vars=[city_col],
            value_vars=available_metrics,
            variable_name="metric",
            value_name="value",
        )
        .drop_nulls("value")
        .with_columns(
            pl.col("metric")
            .replace({metric: _metric_label(metric) for metric in available_metrics})
            .alias("metric_label")
        )
    )

    if long_df.height == 0:
        return empty_figure("No comparison values available.", title=title)

    fig = px.bar(
        long_df.to_pandas(),
        x="metric_label",
        y="value",
        color=city_col,
        barmode="group",
        labels={
            "metric_label": "Metric",
            "value": "Value",
            city_col: "City",
        },
        title=title or "City Comparison",
    )

    fig.update_layout(xaxis_tickangle=30)

    return apply_standard_layout(fig, height=500)