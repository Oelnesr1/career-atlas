"""
Dashboard configuration for Career Atlas.

This file centralizes:
- project paths
- input/output data file locations
- metric labels and formatting rules
- default scoring weights
- dashboard style constants
"""

from __future__ import annotations

from pathlib import Path


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELING_DIR = DATA_DIR / "modeling"
STATIC_DATA_DIR = DATA_DIR / "static"

FIGURES_DIR = PROJECT_ROOT / "figures"
DASHBOARD_ASSETS_DIR = DASHBOARD_DIR / "assets"

STATIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_ASSETS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Core processed data files
# -----------------------------------------------------------------------------

PANEL_DATA_PATH = PROCESSED_DIR / "career_city_fit_2020_2024.csv"
DASHBOARD_OUTPUT_PATH = PROCESSED_DIR / "dashboard_modeling_output_latest.csv"
CAREER_CITY_SCORES_PATH = PROCESSED_DIR / "career_city_scores_latest.csv"
NEXT_YEAR_PREDICTIONS_PATH = PROCESSED_DIR / "career_city_next_year_predictions_latest.csv"

# Modeling outputs
METRO_PROFILES_PATH = MODELING_DIR / "metro_profiles_clustered_latest.csv"
NEAREST_METROS_PATH = MODELING_DIR / "nearest_metros_latest.csv"
METRO_CLUSTER_SUMMARY_PATH = MODELING_DIR / "metro_cluster_summary.csv"
REGRESSION_METRICS_PATH = MODELING_DIR / "regression_model_metrics_all_targets.csv"

RF_IMPORTANCE_RENT_BURDEN_PATH = MODELING_DIR / "random_forest_feature_importance_rent_burden.csv"
RF_IMPORTANCE_REAL_INCOME_PATH = MODELING_DIR / "random_forest_feature_importance_real_income_after_rent.csv"

# Static dashboard helper files
CBSA_CENTROIDS_PATH = STATIC_DATA_DIR / "cbsa_centroids.csv"


# -----------------------------------------------------------------------------
# Census Gazetteer source for CBSA centroids
# -----------------------------------------------------------------------------

CENSUS_CBSA_GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2024_Gazetteer/2024_Gaz_cbsa_national.zip"
)


# -----------------------------------------------------------------------------
# Required columns
# -----------------------------------------------------------------------------

REQUIRED_PANEL_COLUMNS = [
    "year",
    "cbsa_code",
    "bls_msa_name",
    "occ_code",
    "occ_title",
    "wage_annual_median",
    "real_wage_annual_median",
    "annual_rent",
    "rpp_all_items",
    "rent_to_median_wage_ratio",
    "real_income_after_rent_median",
    "tot_emp",
    "loc_quotient",
]

REQUIRED_DASHBOARD_COLUMNS = [
    "cbsa_code",
    "bls_msa_name",
    "occ_code",
    "occ_title",
    "career_city_score_0_100",
    "wage_annual_median",
    "real_wage_annual_median",
    "annual_rent",
    "rpp_all_items",
    "real_income_after_rent_median",
    "rent_to_median_wage_ratio",
    "tot_emp",
    "loc_quotient",
]


# -----------------------------------------------------------------------------
# User-facing metric definitions
# -----------------------------------------------------------------------------

METRIC_LABELS = {
    "career_city_score_0_100": "Career Atlas Score",
    "custom_score_0_100": "Custom Score",
    "wage_annual_median": "Median Annual Wage",
    "real_wage_annual_median": "RPP-Adjusted Median Wage",
    "annual_rent": "Annual Rent",
    "median_gross_rent": "Monthly Gross Rent",
    "rpp_all_items": "BEA RPP: All Items",
    "rpp_services_housing": "BEA RPP: Housing",
    "rent_to_median_wage_ratio": "Rent Burden",
    "median_wage_to_rent_ratio": "Wage-to-Rent Ratio",
    "cost_adjusted_rent_burden_median": "Cost-Adjusted Rent Burden",
    "real_income_after_rent_median": "RPP-Adjusted Income After Rent",
    "tot_emp": "Estimated Employment",
    "jobs_1000": "Jobs per 1,000",
    "loc_quotient": "Location Quotient",
    "predicted_next_year_rent_to_median_wage_ratio": "Predicted Next-Year Rent Burden",
    "predicted_change_in_rent_burden": "Predicted Change in Rent Burden",
    "predicted_next_year_real_income_after_rent_median": "Predicted Next-Year Income After Rent",
    "predicted_change_in_real_income_after_rent": "Predicted Change in Income After Rent",
    "metro_cluster": "Metro Cluster",
    "pca_1": "PCA Component 1",
    "pca_2": "PCA Component 2",
}

METRIC_FORMATS = {
    "career_city_score_0_100": "{:,.1f}",
    "custom_score_0_100": "{:,.1f}",
    "wage_annual_median": "${:,.0f}",
    "real_wage_annual_median": "${:,.0f}",
    "annual_rent": "${:,.0f}",
    "median_gross_rent": "${:,.0f}",
    "rpp_all_items": "{:,.1f}",
    "rpp_services_housing": "{:,.1f}",
    "rent_to_median_wage_ratio": "{:.3f}",
    "median_wage_to_rent_ratio": "{:,.2f}",
    "cost_adjusted_rent_burden_median": "{:.3f}",
    "real_income_after_rent_median": "${:,.0f}",
    "tot_emp": "{:,.0f}",
    "jobs_1000": "{:,.2f}",
    "loc_quotient": "{:,.2f}",
    "predicted_next_year_rent_to_median_wage_ratio": "{:.3f}",
    "predicted_change_in_rent_burden": "{:+.3f}",
    "predicted_next_year_real_income_after_rent_median": "${:,.0f}",
    "predicted_change_in_real_income_after_rent": "${:+,.0f}",
}

# Whether a higher raw metric value is better.
# This is important for percentiles and custom weighted scores.
METRIC_HIGHER_IS_BETTER = {
    "career_city_score_0_100": True,
    "custom_score_0_100": True,
    "wage_annual_median": True,
    "real_wage_annual_median": True,
    "annual_rent": False,
    "median_gross_rent": False,
    "rpp_all_items": False,
    "rpp_services_housing": False,
    "rent_to_median_wage_ratio": False,
    "median_wage_to_rent_ratio": True,
    "cost_adjusted_rent_burden_median": False,
    "real_income_after_rent_median": True,
    "tot_emp": True,
    "jobs_1000": True,
    "loc_quotient": True,
    "predicted_next_year_rent_to_median_wage_ratio": False,
    "predicted_change_in_rent_burden": False,
    "predicted_next_year_real_income_after_rent_median": True,
    "predicted_change_in_real_income_after_rent": True,
}


# -----------------------------------------------------------------------------
# Default dashboard scoring weights
# -----------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "real_income_after_rent": 35,
    "rent_affordability": 25,
    "real_wage": 15,
    "location_quotient": 15,
    "employment": 10,
    "predicted_real_income_growth": 0,
    "low_cost": 0,
}

WEIGHT_LABELS = {
    "real_income_after_rent": "Purchasing Power After Rent",
    "rent_affordability": "Rent Affordability",
    "real_wage": "RPP-Adjusted Wage",
    "location_quotient": "Occupation Concentration",
    "employment": "Employment Scale",
    "predicted_real_income_growth": "Predicted Improvement",
    "low_cost": "Low Cost of Living",
}

WEIGHT_DESCRIPTIONS = {
    "real_income_after_rent": "Rewards metros where a worker has more purchasing power after paying rent.",
    "rent_affordability": "Rewards metros where annual rent is low relative to occupation-specific wages.",
    "real_wage": "Rewards metros with higher wages after adjusting for regional price levels.",
    "location_quotient": "Rewards metros where the occupation is locally concentrated.",
    "employment": "Rewards metros with larger estimated employment in the selected occupation.",
    "predicted_real_income_growth": "Rewards metros predicted to improve in purchasing power after rent.",
    "low_cost": "Rewards metros with lower BEA all-items regional price parity.",
}


# -----------------------------------------------------------------------------
# Dashboard options
# -----------------------------------------------------------------------------

DEFAULT_TOP_N = 15
MAX_TABLE_ROWS = 500
DEFAULT_MIN_EMPLOYMENT = 100
DEFAULT_MIN_LOCATION_QUOTIENT = 0.0

DEFAULT_OCCUPATION_KEYWORDS = [
    "Software Developers",
    "Registered Nurses",
    "Accountants and Auditors",
    "Data Scientists",
    "Financial and Investment Analysts",
]

DEFAULT_METRO_KEYWORDS = [
    "New York-Newark-Jersey City",
    "Los Angeles-Long Beach-Anaheim",
    "Chicago-Naperville-Elgin",
    "Dallas-Fort Worth-Arlington",
    "Pittsburgh",
    "Austin",
    "Philadelphia",
]


# -----------------------------------------------------------------------------
# Styling constants
# -----------------------------------------------------------------------------

APP_TITLE = "Career Atlas"
APP_SUBTITLE = "Occupation-specific affordability, purchasing power, and labor-market opportunity across U.S. metros"

COLOR_PRIMARY = "#1f3b57"
COLOR_SECONDARY = "#2f6f9f"
COLOR_ACCENT = "#2fbf71"
COLOR_WARNING = "#f59e0b"
COLOR_DANGER = "#dc2626"
COLOR_BACKGROUND = "#f5f7fb"
COLOR_CARD = "#ffffff"
COLOR_TEXT = "#1f2937"
COLOR_MUTED = "#6b7280"

PLOTLY_TEMPLATE = "plotly_white"

MAP_STYLE = "carto-positron"
MAP_DEFAULT_ZOOM = 3
MAP_DEFAULT_CENTER = {
    "lat": 39.5,
    "lon": -98.35,
}

# -----------------------------------------------------------------------------
# Plotly color scales
# -----------------------------------------------------------------------------

# Higher value = better. Example: score, real income, real wage, LQ.
COLOR_SCALE_GOOD_HIGH = [
    [0.0, COLOR_DANGER],
    [0.5, COLOR_WARNING],
    [1.0, COLOR_ACCENT],
]

# Lower value = better. Example: rent burden, annual rent, RPP.
COLOR_SCALE_GOOD_LOW = [
    [0.0, COLOR_ACCENT],
    [0.5, COLOR_WARNING],
    [1.0, COLOR_DANGER],
]

# For categorical or neutral continuous values where direction is not meaningful.
COLOR_SCALE_NEUTRAL = [
    [0.0, "#dbeafe"],
    [0.5, COLOR_SECONDARY],
    [1.0, COLOR_PRIMARY],
]


# -----------------------------------------------------------------------------
# Table display columns
# -----------------------------------------------------------------------------

CITY_RANKING_COLUMNS = [
    "bls_msa_name",
    "custom_score_0_100",
    "career_city_score_0_100",
    "wage_annual_median",
    "real_wage_annual_median",
    "annual_rent",
    "rpp_all_items",
    "real_income_after_rent_median",
    "rent_to_median_wage_ratio",
    "tot_emp",
    "loc_quotient",
    "predicted_change_in_real_income_after_rent",
]

OCCUPATION_RANKING_COLUMNS = [
    "occ_title",
    "custom_score_0_100",
    "career_city_score_0_100",
    "wage_annual_median",
    "real_wage_annual_median",
    "real_income_after_rent_median",
    "rent_to_median_wage_ratio",
    "tot_emp",
    "loc_quotient",
    "predicted_change_in_real_income_after_rent",
]

SIMILAR_CITY_COLUMNS = [
    "neighbor_rank",
    "neighbor_metro",
    "neighbor_cluster",
    "distance",
    "neighbor_annual_rent",
    "neighbor_rpp_all_items",
    "neighbor_mean_real_income_after_rent",
    "neighbor_share_high_lq",
]

EXPLORER_EXTREME_METRICS = [
    "custom_score_0_100",
    "career_city_score_0_100",
    "real_income_after_rent_median",
    "real_wage_annual_median",
    "wage_annual_median",
    "rent_to_median_wage_ratio",
    "median_wage_to_rent_ratio",
    "annual_rent",
    "rpp_all_items",
    "rpp_services_housing",
    "tot_emp",
    "jobs_1000",
    "loc_quotient",
    "predicted_change_in_real_income_after_rent",
    "predicted_change_in_rent_burden",
]

COMBINATION_EXTREME_COLUMNS = [
    "rank_position",
    "occ_title",
    "bls_msa_name",
    "metric_value",
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

MODEL_METRIC_COLUMNS = [
    "target",
    "model",
    "split",
    "mae",
    "rmse",
    "r2",
]

