"""
Dash UI components for the Career Atlas dashboard.

This file defines:
- reusable cards
- sliders
- dropdown blocks
- table wrappers
- tab layouts
- full app layout

It should not contain data-processing logic or callbacks.
"""

from __future__ import annotations

from typing import Any

from dash import dash_table, dcc, html

try:
    from . import config
except ImportError:
    import config


# -----------------------------------------------------------------------------
# Small reusable components
# -----------------------------------------------------------------------------

def section_header(title: str, subtitle: str | None = None) -> html.Div:
    children = [html.H2(title, className="section-title")]

    if subtitle:
        children.append(html.P(subtitle, className="section-subtitle"))

    return html.Div(children, className="section-header")


def info_card(title: str, value: str = "N/A", subtitle: str | None = None, card_id: str | None = None) -> html.Div:
    children = [
        html.Div(title, className="kpi-title"),
        html.Div(value, className="kpi-value"),
    ]

    if subtitle:
        children.append(html.Div(subtitle, className="kpi-subtitle"))

    return html.Div(children, className="kpi-card", id=card_id)


def card(children: list[Any] | Any, className: str = "card", **kwargs) -> html.Div:
    if not isinstance(children, list):
        children = [children]

    return html.Div(children, className=className, **kwargs)


def graph_card(graph_id: str, title: str | None = None, className: str = "card graph-card") -> html.Div:
    children = []

    if title:
        children.append(html.H3(title, className="card-title"))

    children.append(
        dcc.Loading(
            dcc.Graph(id=graph_id, config={"displayModeBar": True}),
            type="circle",
        )
    )

    return html.Div(children, className=className)


def data_table(
    table_id: str,
    *,
    page_size: int = 15,
    height: str = "420px",
) -> dash_table.DataTable:
    return dash_table.DataTable(
        id=table_id,
        data=[],
        columns=[],
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        fixed_rows={"headers": True},
        style_table={
            "height": height,
            "overflowY": "auto",
            "overflowX": "auto",
            "border": "1px solid #e5e7eb",
            "borderRadius": "12px",
        },
        style_cell={
            "fontFamily": "Inter, Arial, sans-serif",
            "fontSize": "13px",
            "padding": "10px",
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
            "minWidth": "100px",
            "maxWidth": "320px",
        },
        style_header={
            "backgroundColor": "#f3f4f6",
            "fontWeight": "700",
            "border": "none",
            "color": "#111827",
        },
        style_data={
            "border": "none",
            "borderBottom": "1px solid #f1f5f9",
        },
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#fbfdff",
            }
        ],
    )


def table_card(
    table_id: str,
    title: str,
    subtitle: str | None = None,
    *,
    page_size: int = 15,
    height: str = "420px",
) -> html.Div:
    children = [html.H3(title, className="card-title")]

    if subtitle:
        children.append(html.P(subtitle, className="card-subtitle"))

    children.append(
        dcc.Loading(
            data_table(table_id, page_size=page_size, height=height),
            type="circle",
        )
    )

    return html.Div(children, className="card table-card")


def dropdown_block(
    label: str,
    dropdown_id: str,
    options: list[dict[str, Any]],
    value: Any = None,
    *,
    multi: bool = False,
    placeholder: str | None = None,
    clearable: bool = True,
) -> html.Div:
    return html.Div(
        [
            html.Label(label, className="control-label"),
            dcc.Dropdown(
                id=dropdown_id,
                options=options,
                value=value,
                multi=multi,
                placeholder=placeholder or f"Select {label.lower()}",
                clearable=clearable,
                className="control-dropdown",
            ),
        ],
        className="control-block",
    )


def slider_block(
    label: str,
    slider_id: str,
    *,
    min_value: float,
    max_value: float,
    step: float,
    value: float,
    marks: dict | None = None,
    tooltip: bool = True,
) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Label(label, className="control-label"),
                    html.Span(id=f"{slider_id}-value", className="slider-value"),
                ],
                className="slider-label-row",
            ),
            dcc.Slider(
                id=slider_id,
                min=min_value,
                max=max_value,
                step=step,
                value=value,
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": tooltip},
                className="control-slider",
            ),
        ],
        className="control-block slider-block",
    )


def weight_slider(weight_key: str, default_value: float) -> html.Div:
    return html.Div(
        [
            html.Label(config.WEIGHT_LABELS.get(weight_key, weight_key), className="weight-label"),
            dcc.Slider(
                id=f"weight-{weight_key}",
                min=0,
                max=100,
                step=5,
                value=default_value,
                tooltip={"placement": "bottom", "always_visible": False},
                marks={0: "0", 50: "50", 100: "100"},
                className="weight-slider",
            ),
            html.Div(
                config.WEIGHT_DESCRIPTIONS.get(weight_key, ""),
                className="weight-description",
            ),
        ],
        className="weight-slider-block",
    )


def warning_panel(panel_id: str) -> html.Div:
    return html.Div(id=panel_id, className="warning-panel")


def kpi_grid(grid_id: str) -> html.Div:
    return html.Div(id=grid_id, className="kpi-grid")


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

def build_sidebar(data) -> html.Div:
    default_occupation = data.occupation_options[0]["value"] if data.occupation_options else None
    default_metro = data.metro_options[0]["value"] if data.metro_options else None

    return html.Div(
        [
            html.Div(
                [
                    html.Div("Career Atlas", className="sidebar-title"),
                    html.Div("Controls", className="sidebar-subtitle"),
                ],
                className="sidebar-brand",
            ),

            dropdown_block(
                "Occupation",
                "occupation-dropdown",
                data.occupation_options,
                value=default_occupation,
                placeholder="Search occupations",
                clearable=False,
            ),

            dropdown_block(
                "City / Metro",
                "city-dropdown",
                data.metro_options,
                value=default_metro,
                placeholder="Search metros",
                clearable=False,
            ),

            dropdown_block(
                "State",
                "state-dropdown",
                data.state_options,
                value=[],
                multi=True,
                placeholder="Optional state filter",
            ),

            dropdown_block(
                "Metro Cluster",
                "cluster-dropdown",
                data.cluster_options,
                value=[],
                multi=True,
                placeholder="Optional cluster filter",
            ),

            html.Hr(className="sidebar-divider"),

            slider_block(
                "Minimum Employment",
                "min-employment-slider",
                min_value=0,
                max_value=5000,
                step=50,
                value=config.DEFAULT_MIN_EMPLOYMENT,
                marks={0: "0", 1000: "1k", 2500: "2.5k", 5000: "5k"},
                tooltip=False,
            ),

            slider_block(
                "Minimum Location Quotient",
                "min-lq-slider",
                min_value=0,
                max_value=3,
                step=0.05,
                value=config.DEFAULT_MIN_LOCATION_QUOTIENT,
                marks={0: "0", 1: "1", 2: "2", 3: "3"},
                tooltip=False,
            ),

            slider_block(
                "Maximum BEA RPP",
                "max-rpp-slider",
                min_value=80,
                max_value=140,
                step=1,
                value=140,
                marks={80: "80", 100: "100", 120: "120", 140: "140"},
                tooltip=False,
            ),

            slider_block(
                "Top N",
                "top-n-slider",
                min_value=5,
                max_value=50,
                step=5,
                value=config.DEFAULT_TOP_N,
                marks={5: "5", 15: "15", 30: "30", 50: "50"},
                tooltip=False,
            ),

            html.Hr(className="sidebar-divider"),

            html.Div("Custom Score Weights", className="sidebar-section-title"),

            html.Div(
                [
                    weight_slider(key, value)
                    for key, value in config.DEFAULT_WEIGHTS.items()
                ],
                className="weight-slider-container",
            ),

            html.Div(id="weight-summary", className="weight-summary"),
        ],
        className="sidebar",
    )


# -----------------------------------------------------------------------------
# Main tab layouts
# -----------------------------------------------------------------------------

def career_tab() -> html.Div:
    return html.Div(
        [
            section_header(
                "Best Cities for an Occupation",
                "Rank metros using your custom priorities: purchasing power, rent affordability, real wage, employment, and occupation concentration.",
            ),

            kpi_grid("career-kpis"),

            html.Div(
                [
                    graph_card("city-ranking-bar", "Top Cities"),
                    graph_card("city-score-map", "Map of Career-City Fit"),
                ],
                className="two-column-grid",
            ),

            html.Div(
                [
                    graph_card("career-tradeoff-scatter", "Affordability vs. Purchasing Power"),
                    table_card(
                        "city-ranking-table",
                        "City Ranking Table",
                        "Sortable and filterable ranking for the selected occupation.",
                    ),
                ],
                className="two-column-grid",
            ),
        ],
        className="tab-content",
    )


def city_tab() -> html.Div:
    return html.Div(
        [
            section_header(
                "Best Occupations for a City",
                "Reverse the question: given a city, which occupations look strongest on wage, affordability, employment, and predicted improvement?",
            ),

            kpi_grid("city-kpis"),

            html.Div(
                [
                    graph_card("city-occupation-bar", "Top Occupations"),
                    graph_card("city-occupation-prediction-bar", "Predicted Improvement by Occupation"),
                ],
                className="two-column-grid",
            ),

            table_card(
                "city-occupation-table",
                "Occupation Ranking Table",
                "Best occupations in the selected city under the current filters and weights.",
                height="520px",
            ),
        ],
        className="tab-content",
    )

def explorer_tab(data) -> html.Div:
    default_city = data.metro_options[0]["value"] if data.metro_options else None

    extreme_metric_options = [
        {
            "label": config.METRIC_LABELS.get(metric, metric.replace("_", " ").title()),
            "value": metric,
        }
        for metric in config.EXPLORER_EXTREME_METRICS
    ]

    return html.Div(
        [
            section_header(
                "City + Occupation Explorer",
                "Choose a city first, then choose an occupation available in that city. Explore historical trends, metro-wide distributions, and best/worst career-city combinations.",
            ),

            html.Div(
                [
                    dropdown_block(
                        "Explorer City",
                        "explorer-city-dropdown",
                        data.metro_options,
                        value=default_city,
                        clearable=False,
                    ),
                    dropdown_block(
                        "Explorer Occupation",
                        "explorer-occupation-dropdown",
                        [],
                        value=None,
                        clearable=False,
                    ),
                    dropdown_block(
                        "Distribution Metric",
                        "explorer-metric-dropdown",
                        [
                            {"label": config.METRIC_LABELS.get(metric, metric), "value": metric}
                            for metric in [
                                "real_income_after_rent_median",
                                "rent_to_median_wage_ratio",
                                "real_wage_annual_median",
                                "annual_rent",
                                "loc_quotient",
                            ]
                        ],
                        value="real_income_after_rent_median",
                        clearable=False,
                    ),
                ],
                className="three-column-grid",
            ),

            kpi_grid("explorer-kpis"),
            warning_panel("explorer-warnings"),

            html.Div(
                [
                    graph_card("explorer-trend-chart", "Historical Trend"),
                    graph_card("explorer-boxplot", "Selected City vs. Metro Distribution"),
                ],
                className="two-column-grid",
            ),

            html.Div(
                [
                    graph_card("explorer-tradeoff-scatter", "Metro Tradeoff View"),
                    table_card(
                        "explorer-history-table",
                        "Historical Data Table",
                        "Year-by-year values for the selected city and occupation.",
                    ),
                ],
                className="two-column-grid",
            ),

            section_header(
                "Best / Worst Career-City Combinations",
                "Pick any metric and find the strongest and weakest occupation-metro combinations under the current sidebar filters.",
            ),

            html.Div(
                [
                    dropdown_block(
                        "Best/Worst Metric",
                        "explorer-extreme-metric-dropdown",
                        extreme_metric_options,
                        value="real_income_after_rent_median",
                        clearable=False,
                    ),
                ],
                className="single-control-row",
            ),

            graph_card(
                "explorer-extremes-bar",
                "Best and Worst Career-City Combinations",
            ),

            html.Div(
                [
                    table_card(
                        "explorer-best-combinations-table",
                        "Best Career-City Combinations",
                        "Rows where the selected metric is strongest. Direction depends on the metric.",
                        height="420px",
                    ),
                    table_card(
                        "explorer-worst-combinations-table",
                        "Worst Career-City Combinations",
                        "Rows where the selected metric is weakest. Direction depends on the metric.",
                        height="420px",
                    ),
                ],
                className="two-column-grid",
            ),
        ],
        className="tab-content",
    )

def similarity_tab(data) -> html.Div:
    default_city = data.metro_options[0]["value"] if data.metro_options else None
    default_occ = data.occupation_options[0]["value"] if data.occupation_options else None

    return html.Div(
        [
            section_header(
                "Similar Cities and Similar Occupations",
                "Find metros with similar cost, wage, price, and occupational profiles. Also explore occupations with similar wage and geographic patterns.",
            ),

            html.Div(
                [
                    dropdown_block(
                        "Similarity City",
                        "similarity-city-dropdown",
                        data.metro_options,
                        value=default_city,
                        clearable=False,
                    ),
                    dropdown_block(
                        "Similarity Occupation",
                        "similarity-occupation-dropdown",
                        data.occupation_options,
                        value=default_occ,
                        clearable=False,
                    ),
                ],
                className="two-column-grid",
            ),

            html.Div(
                [
                    graph_card("similar-cities-map", "Similar City Map"),
                    graph_card("similar-cities-bar", "Similar City Comparison"),
                ],
                className="two-column-grid",
            ),

            table_card(
                "similar-cities-table",
                "Similar Cities",
                "Nearest metros based on the RPP-aware similarity model.",
                height="360px",
            ),

            html.Div(
                [
                    graph_card("similar-occupations-bar", "Similar Occupations"),
                    table_card(
                        "similar-occupations-table",
                        "Similar Occupations",
                        "Nearest occupations based on wage, affordability, employment, and geographic concentration profiles.",
                    ),
                ],
                className="two-column-grid",
            ),
        ],
        className="tab-content",
    )


def clusters_tab() -> html.Div:
    return html.Div(
        [
            section_header(
                "Metro Clusters",
                "Explore K-Means clusters created from rent, RPP, real wages, purchasing power, labor-market concentration, and occupational mix.",
            ),

            html.Div(
                [
                    graph_card("cluster-pca-plot", "PCA View of Metro Clusters"),
                    graph_card("cluster-summary-bar", "Cluster Summary"),
                ],
                className="two-column-grid",
            ),

            table_card(
                "cluster-members-table",
                "Cluster Members",
                "Representative metros in the selected cluster.",
                height="520px",
            ),
        ],
        className="tab-content",
    )


def modeling_tab() -> html.Div:
    return html.Div(
        [
            section_header(
                "Predictive Modeling",
                "Compare baselines, linear models, ridge regression, and random forests for rent burden and RPP-adjusted income after rent.",
            ),

            html.Div(
                [
                    dropdown_block(
                        "Model Target",
                        "model-target-dropdown",
                        [
                            {"label": "Rent Burden", "value": "rent_burden"},
                            {"label": "RPP-Adjusted Income After Rent", "value": "real_income"},
                        ],
                        value="real_income",
                        clearable=False,
                    ),
                    dropdown_block(
                        "Model Metric",
                        "model-metric-dropdown",
                        [
                            {"label": "MAE", "value": "mae"},
                            {"label": "RMSE", "value": "rmse"},
                            {"label": "R²", "value": "r2"},
                        ],
                        value="mae",
                        clearable=False,
                    ),
                ],
                className="two-column-grid",
            ),

            html.Div(
                [
                    graph_card("model-metrics-chart", "Model Comparison"),
                    graph_card("feature-importance-chart", "Random Forest Feature Importance"),
                ],
                className="two-column-grid",
            ),

            html.Div(
                [
                    graph_card("prediction-change-bar", "Predicted Change"),
                    table_card(
                        "model-metrics-table",
                        "Model Metrics Table",
                        "Train, validation, and test performance for each model.",
                    ),
                ],
                className="two-column-grid",
            ),
        ],
        className="tab-content",
    )


def methodology_tab() -> html.Div:
    return html.Div(
        [
            section_header(
                "Methodology",
                "How to interpret the data, scoring, clusters, and predictions.",
            ),

            html.Div(
                [
                    html.H3("Data Sources", className="card-title"),
                    html.Ul(
                        [
                            html.Li("BLS OEWS: occupation-specific wages, employment, jobs per 1,000, and location quotient."),
                            html.Li("Census ACS 5-year: metro-level rent, household income, and population."),
                            html.Li("BEA Regional Price Parities: metro-level overall and housing price levels."),
                        ],
                        className="methodology-list",
                    ),
                ],
                className="card",
            ),

            html.Div(
                [
                    html.H3("Default Score Formula", className="card-title"),
                    html.P(
                        "The default score combines RPP-adjusted income after rent, rent affordability, real wage, occupation concentration, and employment scale. Users can change these weights in the sidebar.",
                        className="card-subtitle",
                    ),
                    html.Pre(
                        "Score = 35% purchasing power after rent\n"
                        "      + 25% rent affordability\n"
                        "      + 15% RPP-adjusted wage\n"
                        "      + 15% location quotient\n"
                        "      + 10% employment scale",
                        className="formula-box",
                    ),
                ],
                className="card",
            ),

            html.Div(
                [
                    html.H3("Modeling Notes", className="card-title"),
                    html.Ul(
                        [
                            html.Li("Supervised models use time-based train/validation/test splits."),
                            html.Li("Predictions are short-term signals, not guarantees."),
                            html.Li("K-Means clusters and nearest-neighbor similarity are exploratory tools."),
                            html.Li("Low-employment metro-occupation pairs should be interpreted cautiously."),
                        ],
                        className="methodology-list",
                    ),
                ],
                className="card",
            ),
        ],
        className="tab-content methodology-tab",
    )


# -----------------------------------------------------------------------------
# Main layout
# -----------------------------------------------------------------------------

def build_layout(data) -> html.Div:
    latest_year_text = f"Latest year: {data.latest_year}" if data.latest_year else "Latest year unavailable"

    header = html.Div(
        [
            html.Div(
                [
                    html.H1(config.APP_TITLE, className="app-title"),
                    html.P(config.APP_SUBTITLE, className="app-subtitle"),
                ],
                className="app-title-block",
            ),
            html.Div(latest_year_text, className="year-badge"),
        ],
        className="app-header",
    )

    tabs = dcc.Tabs(
        id="main-tabs",
        value="career-tab",
        className="tabs",
        children=[
            dcc.Tab(label="Career → Cities", value="career-tab", children=career_tab()),
            dcc.Tab(label="City → Occupations", value="city-tab", children=city_tab()),
            dcc.Tab(label="Explorer", value="explorer-tab", children=explorer_tab(data)),
            dcc.Tab(label="Similarity", value="similarity-tab", children=similarity_tab(data)),
            dcc.Tab(label="Clusters", value="clusters-tab", children=clusters_tab()),
            dcc.Tab(label="Modeling", value="modeling-tab", children=modeling_tab()),
            dcc.Tab(label="Methodology", value="methodology-tab", children=methodology_tab()),
        ],
    )

    return html.Div(
        [
            header,
            html.Div(
                [
                    build_sidebar(data),
                    html.Main(tabs, className="main-content"),
                ],
                className="app-shell",
            ),
        ],
        className="app-root",
    )