"""
Career Atlas Dash Dashboard.

Run from the project root:

    python dashboard/app.py

This dashboard reads the processed/modeling outputs created by the notebooks and
provides an interactive interface for:

- best cities for an occupation
- best occupations for a city
- custom weighted scoring
- city + occupation trends
- similar cities
- similar occupations
- metro clusters
- predictive modeling results
"""

from __future__ import annotations

import sys
from pathlib import Path

from dash import Dash, html

# Make same-folder imports work when running:
#     python dashboard/app.py
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from . import callbacks, components, config, data_loader
except ImportError:
    import callbacks
    import components
    import config
    import data_loader


def _build_error_layout(error: Exception) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Career Atlas", className="app-title"),
                    html.P("Dashboard startup error", className="app-subtitle"),
                ],
                className="app-title-block",
            ),
            html.Div(
                [
                    html.H2("The dashboard could not load the required data files."),
                    html.P(
                        "Make sure you have already run the data wrangling/EDA notebook "
                        "and the modeling notebook so that the processed dashboard CSVs exist."
                    ),
                    html.Pre(str(error), className="error-box"),
                    html.H3("Expected key files"),
                    html.Ul(
                        [
                            html.Li(str(config.PANEL_DATA_PATH)),
                            html.Li(str(config.DASHBOARD_OUTPUT_PATH)),
                            html.Li(str(config.METRO_PROFILES_PATH)),
                            html.Li(str(config.NEAREST_METROS_PATH)),
                            html.Li(str(config.REGRESSION_METRICS_PATH)),
                        ]
                    ),
                ],
                className="card startup-error-card",
            ),
        ],
        className="app-root startup-error-root",
    )


def create_app() -> Dash:
    app = Dash(
        __name__,
        title=config.APP_TITLE,
        suppress_callback_exceptions=True,
        assets_folder=str(DASHBOARD_DIR / "assets"),
    )

    try:
        data = data_loader.load_all_data()
        app.layout = components.build_layout(data)
        callbacks.register_callbacks(app, data)

    except Exception as exc:
        app.layout = _build_error_layout(exc)

    return app


server = create_app().server


if __name__ == "__main__":
    app = create_app()
    app.run(
        debug=True,
        port=8050,
        dev_tools_ui=False,
        dev_tools_props_check=False,
    )