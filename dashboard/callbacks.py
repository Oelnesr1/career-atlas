"""
Dash callbacks for the Caree Atlas dashboard.

This file wires the UI to the analytics and figure utilities.

It expects:
- app: a Dash instance
- data: DashboardData from dashboard.data_loader.load_all_data()
"""

from __future__ import annotations

from typing import Any

import polars as pl
from dash import Input, Output, State, html

try:
    from . import analytics, config, figures
except ImportError:
    import analytics
    import config
    import figures


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _weight_inputs() -> list[Input]:
    return [
        Input(f"weight-{key}", "value")
        for key in config.DEFAULT_WEIGHTS
    ]


def _weights_from_values(values: list[float]) -> dict[str, float]:
    return {
        key: float(value or 0)
        for key, value in zip(config.DEFAULT_WEIGHTS.keys(), values)
    }


def _slider_value_outputs() -> list[Output]:
    return [
        Output("min-employment-slider-value", "children"),
        Output("min-lq-slider-value", "children"),
        Output("max-rpp-slider-value", "children"),
        Output("top-n-slider-value", "children"),
    ]


def _format_slider_value(value: Any, suffix: str = "") -> str:
    if value is None:
        return ""

    try:
        if isinstance(value, float):
            return f"{value:,.2f}{suffix}"
        return f"{value:,}{suffix}"
    except Exception:
        return str(value)


def _kpi_cards_from_rows(rows: list[dict[str, Any]]) -> list[html.Div]:
    if not rows:
        return [
            html.Div(
                [
                    html.Div("No Selection", className="kpi-title"),
                    html.Div("N/A", className="kpi-value"),
                ],
                className="kpi-card",
            )
        ]

    cards = []

    for item in rows:
        cards.append(
            html.Div(
                [
                    html.Div(item.get("metric", ""), className="kpi-title"),
                    html.Div(item.get("value", "N/A"), className="kpi-value"),
                ],
                className="kpi-card",
            )
        )

    return cards


def _warning_children(messages: list[str]) -> list[html.Div]:
    if not messages:
        return []

    return [
        html.Div(message, className="warning-badge")
        for message in messages
    ]

def _top_result_banner(
    *,
    label: str,
    value: str | None,
    score: float | None = None,
    subtitle: str | None = None,
) -> html.Div:
    if not value:
        value = "No matching result"

    score_text = ""

    if score is not None:
        try:
            score_text = f"Custom Score: {float(score):,.1f}"
        except Exception:
            score_text = ""

    return html.Div(
        [
            html.Div(label, className="top-result-label"),
            html.Div(value, className="top-result-value"),
            html.Div(
                " • ".join(part for part in [score_text, subtitle] if part),
                className="top-result-subtitle",
            ),
        ],
        className="top-result-banner",
    )


def _safe_table(df: Any, columns: list[str] | None = None, max_rows: int = config.MAX_TABLE_ROWS):
    return analytics.prepare_dash_table(df, columns=columns, max_rows=max_rows)


def _model_target_to_contains(target_value: str) -> str:
    if target_value == "rent_burden":
        return "rent burden"

    return "income after rent"


def _importance_for_target(data, target_value: str):
    if target_value == "rent_burden":
        return data.rf_importance_rent_burden

    return data.rf_importance_real_income


def _prediction_metric_for_target(target_value: str) -> tuple[str, bool]:
    """
    Return metric and sort direction.

    For real income after rent, larger predicted change is better.
    For rent burden, smaller predicted change is better.
    """

    if target_value == "rent_burden":
        return "predicted_change_in_rent_burden", True

    return "predicted_change_in_real_income_after_rent", False


# -----------------------------------------------------------------------------
# Main callback registration
# -----------------------------------------------------------------------------

def register_callbacks(app, data) -> None:
    """
    Register every callback used by the dashboard.
    """

    # -------------------------------------------------------------------------
    # Slider display values
    # -------------------------------------------------------------------------

    @app.callback(
        _slider_value_outputs(),
        Input("min-employment-slider", "value"),
        Input("min-lq-slider", "value"),
        Input("max-rpp-slider", "value"),
        Input("top-n-slider", "value"),
    )
    def update_slider_labels(min_emp, min_lq, max_rpp, top_n):
        return (
            _format_slider_value(min_emp),
            _format_slider_value(min_lq),
            _format_slider_value(max_rpp),
            _format_slider_value(top_n),
        )

    # -------------------------------------------------------------------------
    # Weight summary
    # -------------------------------------------------------------------------

    @app.callback(
        Output("weight-summary", "children"),
        *_weight_inputs(),
    )
    def update_weight_summary(*weight_values):
        weights = _weights_from_values(list(weight_values))
        weight_df = analytics.summarize_weight_settings(weights)

        if weight_df.height == 0:
            return ""

        rows = weight_df.select(["label", "percent"]).to_dicts()

        return html.Div(
            [
                html.Div("Normalized Weights", className="weight-summary-title"),
                html.Ul(
                    [
                        html.Li(f"{row['label']}: {row['percent']:.1f}%")
                        for row in rows
                        if row["percent"] > 0
                    ],
                    className="weight-summary-list",
                ),
            ]
        )

    # -------------------------------------------------------------------------
    # Career → Cities tab
    # -------------------------------------------------------------------------

    @app.callback(
        Output("career-kpis", "children"),
        Output("city-ranking-table", "data"),
        Output("city-ranking-table", "columns"),
        Output("city-ranking-bar", "figure"),
        Output("city-score-map", "figure"),
        Output("career-tradeoff-scatter", "figure"),
        Input("occupation-dropdown", "value"),
        Input("state-dropdown", "value"),
        Input("cluster-dropdown", "value"),
        Input("min-employment-slider", "value"),
        Input("min-lq-slider", "value"),
        Input("max-rpp-slider", "value"),
        Input("top-n-slider", "value"),
        *_weight_inputs(),
    )
    def update_career_tab(
        occupation,
        states,
        clusters,
        min_employment,
        min_lq,
        max_rpp,
        top_n,
        *weight_values,
    ):
        weights = _weights_from_values(list(weight_values))

        if not occupation:
            empty_fig = figures.empty_figure("Select an occupation to begin.")
            return [], [], [], empty_fig, empty_fig, empty_fig

        ranked = analytics.get_best_cities_for_occupation(
            data.dashboard,
            occupation=occupation,
            weights=weights,
            states=states or [],
            clusters=clusters or [],
            min_employment=min_employment,
            min_lq=min_lq,
            max_rpp=max_rpp,
            top_n=top_n or config.DEFAULT_TOP_N,
        )

        all_filtered = analytics.filter_dashboard_rows(
            data.dashboard,
            occupation=occupation,
            states=states or [],
            clusters=clusters or [],
            min_employment=min_employment,
            min_lq=min_lq,
            max_rpp=max_rpp,
        )
        all_scored = analytics.add_custom_score(all_filtered, weights=weights)

        top_row = ranked.row(0, named=True) if ranked.height > 0 else None

        top_location = top_row.get("bls_msa_name") if top_row else None
        top_score = top_row.get("custom_score_0_100") if top_row else None

        top_banner = _top_result_banner(
            label="Top Location",
            value=top_location,
            score=top_score,
            subtitle=f"Best metro for {occupation} under the current filters and score weights.",
        )

        kpis = [top_banner] + _kpi_cards_from_rows(analytics.make_kpi_summary(top_row))

        table_data, table_cols = _safe_table(
            ranked,
            columns=config.CITY_RANKING_COLUMNS,
            max_rows=top_n or config.DEFAULT_TOP_N,
        )

        bar_fig = figures.make_city_ranking_bar(
            ranked,
            score_col="custom_score_0_100",
            top_n=top_n or config.DEFAULT_TOP_N,
            title=f"Best Cities for {occupation}",
        )

        map_fig = figures.make_city_score_map(
            all_scored,
            color_col="custom_score_0_100",
            size_col="tot_emp",
            title=f"Career-City Fit Map: {occupation}",
        )

        selected_city = top_row.get("bls_msa_name") if top_row else None

        scatter_fig = figures.make_tradeoff_scatter(
            all_scored,
            selected_city=selected_city,
            x_col="rent_to_median_wage_ratio",
            y_col="real_income_after_rent_median",
            color_col="loc_quotient",
            size_col="tot_emp",
            title=f"Tradeoffs Across Metros: {occupation}",
        )

        return kpis, table_data, table_cols, bar_fig, map_fig, scatter_fig

    # -------------------------------------------------------------------------
    # City → Occupations tab
    # -------------------------------------------------------------------------

    @app.callback(
        Output("city-kpis", "children"),
        Output("city-occupation-table", "data"),
        Output("city-occupation-table", "columns"),
        Output("city-occupation-bar", "figure"),
        Output("city-occupation-prediction-bar", "figure"),
        Input("city-dropdown", "value"),
        Input("min-employment-slider", "value"),
        Input("min-lq-slider", "value"),
        Input("top-n-slider", "value"),
        *_weight_inputs(),
    )
    def update_city_tab(
        city,
        min_employment,
        min_lq,
        top_n,
        *weight_values,
    ):
        weights = _weights_from_values(list(weight_values))

        if not city:
            empty_fig = figures.empty_figure("Select a city to begin.")
            return [], [], [], empty_fig, empty_fig

        ranked = analytics.get_best_occupations_for_city(
            data.dashboard,
            city=city,
            weights=weights,
            min_employment=min_employment,
            min_lq=min_lq,
            top_n=top_n or config.DEFAULT_TOP_N,
        )

        top_row = ranked.row(0, named=True) if ranked.height > 0 else None

        top_career = top_row.get("occ_title") if top_row else None
        top_score = top_row.get("custom_score_0_100") if top_row else None

        top_banner = _top_result_banner(
            label="Top Career",
            value=top_career,
            score=top_score,
            subtitle=f"Best occupation in {city} under the current filters and score weights.",
        )

        kpis = [top_banner] + _kpi_cards_from_rows(analytics.make_kpi_summary(top_row))

        table_data, table_cols = _safe_table(
            ranked,
            columns=config.OCCUPATION_RANKING_COLUMNS,
            max_rows=top_n or config.DEFAULT_TOP_N,
        )

        bar_fig = figures.make_occupation_ranking_bar(
            ranked,
            score_col="custom_score_0_100",
            top_n=top_n or config.DEFAULT_TOP_N,
            title=f"Best Occupations in {city}",
        )

        prediction_fig = figures.make_prediction_change_bar(
            ranked,
            metric="predicted_change_in_real_income_after_rent",
            label_col="occ_title",
            top_n=top_n or config.DEFAULT_TOP_N,
            ascending=False,
            title=f"Predicted Improvement by Occupation: {city}",
        )

        return kpis, table_data, table_cols, bar_fig, prediction_fig

    # -------------------------------------------------------------------------
    # Explorer occupation options: choose city first, occupation second
    # -------------------------------------------------------------------------

    @app.callback(
        Output("explorer-occupation-dropdown", "options"),
        Output("explorer-occupation-dropdown", "value"),
        Input("explorer-city-dropdown", "value"),
    )
    def update_explorer_occupation_options(city):
        options = analytics.get_occupations_for_city(data.panel, city)
        value = analytics.choose_default_occupation_for_city(data.panel, city)
        return options, value

    # -------------------------------------------------------------------------
    # Explorer tab
    # -------------------------------------------------------------------------

    @app.callback(
        Output("explorer-kpis", "children"),
        Output("explorer-warnings", "children"),
        Output("explorer-trend-chart", "figure"),
        Output("explorer-boxplot", "figure"),
        Output("explorer-tradeoff-scatter", "figure"),
        Output("explorer-history-table", "data"),
        Output("explorer-history-table", "columns"),
        Input("explorer-city-dropdown", "value"),
        Input("explorer-occupation-dropdown", "value"),
        Input("explorer-metric-dropdown", "value"),
        Input("min-employment-slider", "value"),
    )
    def update_explorer_tab(city, occupation, metric, min_employment):
        if not city or not occupation:
            empty_fig = figures.empty_figure("Select a city and occupation.")
            return [], [], empty_fig, empty_fig, empty_fig, [], []

        latest_row = analytics.get_city_occupation_latest_row(
            data.dashboard,
            city=city,
            occupation=occupation,
        )

        kpis = _kpi_cards_from_rows(analytics.make_kpi_summary(latest_row))
        warnings_children = _warning_children(analytics.make_warning_messages(latest_row))

        history = analytics.get_city_occupation_history(
            data.panel,
            city=city,
            occupation=occupation,
        )

        trend_fig = figures.make_trend_chart(
            history,
            metrics=[
                "rent_to_median_wage_ratio",
                "real_income_after_rent_median",
                "real_wage_annual_median",
                "annual_rent",
            ],
            title=f"Historical Trend: {occupation} in {city}",
        )

        distribution = analytics.get_metric_distribution_for_occupation(
            data.panel,
            occupation=occupation,
            year=data.latest_year,
            metric=metric,
            min_employment=min_employment,
        )

        box_fig = figures.make_metric_boxplot(
            distribution,
            metric=metric,
            selected_city=city,
            title=f"{config.METRIC_LABELS.get(metric, metric)} Distribution for {occupation}",
        )

        latest_distribution = analytics.filter_dashboard_rows(
            data.dashboard,
            occupation=occupation,
            min_employment=min_employment,
        )

        scatter_fig = figures.make_tradeoff_scatter(
            latest_distribution,
            selected_city=city,
            x_col="rent_to_median_wage_ratio",
            y_col="real_income_after_rent_median",
            color_col="loc_quotient",
            size_col="tot_emp",
            title=f"Metro Tradeoff View: {occupation}",
        )

        table_data, table_cols = _safe_table(
            history,
            columns=[
                "year",
                "wage_annual_median",
                "real_wage_annual_median",
                "annual_rent",
                "rpp_all_items",
                "rent_to_median_wage_ratio",
                "real_income_after_rent_median",
                "tot_emp",
                "loc_quotient",
            ],
        )

        return kpis, warnings_children, trend_fig, box_fig, scatter_fig, table_data, table_cols
    
    # -------------------------------------------------------------------------
    # Explorer: best / worst career-city combinations for selected metric
    # -------------------------------------------------------------------------

    @app.callback(
        Output("explorer-best-combinations-table", "data"),
        Output("explorer-best-combinations-table", "columns"),
        Output("explorer-worst-combinations-table", "data"),
        Output("explorer-worst-combinations-table", "columns"),
        Output("explorer-extremes-bar", "figure"),
        Input("explorer-extreme-metric-dropdown", "value"),
        Input("state-dropdown", "value"),
        Input("cluster-dropdown", "value"),
        Input("min-employment-slider", "value"),
        Input("min-lq-slider", "value"),
        Input("max-rpp-slider", "value"),
        Input("top-n-slider", "value"),
        *_weight_inputs(),
    )
    def update_explorer_metric_extremes(
        metric,
        states,
        clusters,
        min_employment,
        min_lq,
        max_rpp,
        top_n,
        *weight_values,
    ):
        weights = _weights_from_values(list(weight_values))

        if not metric:
            empty_fig = figures.empty_figure("Select a metric.")
            return [], [], [], [], empty_fig

        best, worst = analytics.get_best_worst_combinations_for_metric(
            data.dashboard,
            metric=metric,
            weights=weights,
            states=states or [],
            clusters=clusters or [],
            min_employment=min_employment,
            min_lq=min_lq,
            max_rpp=max_rpp,
            top_n=top_n or config.DEFAULT_TOP_N,
        )

        best_table_data, best_table_cols = _safe_table(
            best,
            columns=config.COMBINATION_EXTREME_COLUMNS,
            max_rows=top_n or config.DEFAULT_TOP_N,
        )

        worst_table_data, worst_table_cols = _safe_table(
            worst,
            columns=config.COMBINATION_EXTREME_COLUMNS,
            max_rows=top_n or config.DEFAULT_TOP_N,
        )

        metric_label = config.METRIC_LABELS.get(metric, metric.replace("_", " ").title())

        extremes_fig = figures.make_metric_extremes_bar(
            best,
            worst,
            metric=metric,
            title=f"Best and Worst Career-City Combinations by {metric_label}",
        )

        return best_table_data, best_table_cols, worst_table_data, worst_table_cols, extremes_fig

    # -------------------------------------------------------------------------
    # Similarity tab
    # -------------------------------------------------------------------------

    @app.callback(
        Output("similar-cities-table", "data"),
        Output("similar-cities-table", "columns"),
        Output("similar-cities-map", "figure"),
        Output("similar-cities-bar", "figure"),
        Input("similarity-city-dropdown", "value"),
    )
    def update_similar_cities(city):
        if not city:
            empty_fig = figures.empty_figure("Select a city.")
            return [], [], empty_fig, empty_fig

        similar = analytics.get_similar_cities(
            data.nearest_metros,
            city=city,
            top_n=10,
        )

        table_data, table_cols = _safe_table(
            similar,
            columns=config.SIMILAR_CITY_COLUMNS,
            max_rows=10,
        )

        map_fig = figures.make_similar_cities_map(
            similar,
            title=f"Cities Similar to {city}",
        )

        bar_fig = figures.make_similar_city_comparison_bar(
            similar,
            metric="neighbor_mean_real_income_after_rent",
            title=f"Similar Cities: Purchasing Power Comparison",
        )

        return table_data, table_cols, map_fig, bar_fig

    @app.callback(
        Output("similar-occupations-table", "data"),
        Output("similar-occupations-table", "columns"),
        Output("similar-occupations-bar", "figure"),
        Input("similarity-occupation-dropdown", "value"),
    )
    def update_similar_occupations(occupation):
        if not occupation:
            empty_fig = figures.empty_figure("Select an occupation.")
            return [], [], empty_fig

        similar = analytics.get_similar_occupations(
            data.dashboard,
            occupation=occupation,
            top_n=10,
        )

        table_data, table_cols = _safe_table(
            similar,
            columns=[
                "neighbor_rank",
                "occ_title",
                "distance",
                "n_metros",
                "mean_real_wage",
                "mean_rent_burden",
                "mean_real_income_after_rent",
                "total_employment",
                "median_lq",
                "mean_predicted_change",
            ],
            max_rows=10,
        )

        bar_fig = figures.make_similar_occupation_bar(
            similar,
            metric="mean_real_income_after_rent",
            title=f"Occupations Similar to {occupation}",
        )

        return table_data, table_cols, bar_fig

    # -------------------------------------------------------------------------
    # Clusters tab
    # -------------------------------------------------------------------------

    @app.callback(
        Output("cluster-pca-plot", "figure"),
        Output("cluster-summary-bar", "figure"),
        Output("cluster-members-table", "data"),
        Output("cluster-members-table", "columns"),
        Input("cluster-dropdown", "value"),
        Input("city-dropdown", "value"),
    )
    def update_clusters_tab(clusters, selected_city):
        cluster_id = None

        if clusters:
            cluster_id = int(clusters[0]) if isinstance(clusters, list) else int(clusters)

        pca_fig = figures.make_pca_cluster_plot(
            data.metro_profiles,
            selected_city=selected_city,
            cluster_id=cluster_id,
            title="RPP-Aware Metro Clusters",
        )

        summary_fig = figures.make_cluster_summary_bar(
            data.cluster_summary,
            metric="mean_real_income_after_rent",
            title="Average Purchasing Power by Cluster",
        )

        members = analytics.get_cluster_members(
            data.metro_profiles,
            cluster_id=cluster_id,
            top_n=50,
        )

        table_data, table_cols = _safe_table(
            members,
            columns=[
                "bls_msa_name",
                "metro_cluster",
                "annual_rent",
                "rpp_all_items",
                "mean_real_occ_wage",
                "mean_real_income_after_rent",
                "mean_rent_burden",
                "share_high_lq",
                "population",
            ],
            max_rows=50,
        )

        return pca_fig, summary_fig, table_data, table_cols

    # -------------------------------------------------------------------------
    # Modeling tab
    # -------------------------------------------------------------------------

    @app.callback(
        Output("model-metrics-chart", "figure"),
        Output("feature-importance-chart", "figure"),
        Output("prediction-change-bar", "figure"),
        Output("model-metrics-table", "data"),
        Output("model-metrics-table", "columns"),
        Input("model-target-dropdown", "value"),
        Input("model-metric-dropdown", "value"),
        Input("occupation-dropdown", "value"),
        Input("min-employment-slider", "value"),
        Input("top-n-slider", "value"),
    )
    def update_modeling_tab(target_value, metric_value, occupation, min_employment, top_n):
        target_contains = _model_target_to_contains(target_value)

        metrics_df = analytics.get_model_metrics_for_target(
            data.regression_metrics,
            target_label_contains=target_contains,
        )

        metrics_fig = figures.make_model_metrics_chart(
            metrics_df,
            metric=metric_value or "mae",
            title="Model Performance by Split",
        )

        importance_df = _importance_for_target(data, target_value)

        importance_fig = figures.make_feature_importance_chart(
            importance_df,
            top_n=20,
            title="Random Forest Feature Importance",
        )

        filtered = analytics.filter_dashboard_rows(
            data.dashboard,
            occupation=occupation,
            min_employment=min_employment,
        )

        prediction_metric, ascending = _prediction_metric_for_target(target_value)

        prediction_fig = figures.make_prediction_change_bar(
            filtered,
            metric=prediction_metric,
            label_col="bls_msa_name",
            top_n=top_n or config.DEFAULT_TOP_N,
            ascending=ascending,
            title="Largest Predicted Changes",
        )

        table_data, table_cols = _safe_table(
            metrics_df,
            columns=config.MODEL_METRIC_COLUMNS,
            max_rows=100,
        )

        return metrics_fig, importance_fig, prediction_fig, table_data, table_cols