
# Career Atlas: Occupation-Specific Affordability and Opportunity in U.S. Metros

 
## Project Overview


**Career Atlas** is an interactive data science project that helps answer:  

> For my career, which cities are actually good places to live and work?


The project combines occupation-specific labor-market data, local rent and income data, and regional cost-of-living data to compare U.S. metropolitan areas by career. It produces a cleaned occupation × metro × year dataset, exploratory analysis, predictive models, clustering outputs, and an interactive Dash dashboard.



The final product lets users:


- rank the best cities for a selected occupation,
- rank the best occupations for a selected city,
- adjust custom scoring weights,
- compare affordability, wages, cost of living, employment, and location quotient,
- explore similar cities and similar occupations,
- view metro clusters,
- inspect historical trends and distributions,
- view predictive modeling results.

## Final Files to Grade

The main final deliverables are:

```
src/data_wrangling_and_eda FINAL.ipynb
src/modeling FINAL.ipynb
dashboard/
README.md
```  
The `src/old/` folder contains earlier versions and should not be graded as the final version.

## Data Sources  

This project uses three primary public U.S. government datasets.

### 1. BLS Occupational Employment and Wage Statistics

Used for occupation-specific metro-level labor-market data:

* median annual wage,
* mean annual wage,
* hourly wage,
* employment estimate,
* jobs per 1,000 jobs,
* location quotient.

### 2. Census ACS 5-Year

Used for metro-level socioeconomic and housing context:

* median gross rent,
* annualized rent,
* median household income,
* population.

ACS 5-year data is used for consistency across 2020–2024.

### 3. BEA Regional Price Parities

Used for metro-level cost-of-living adjustment:

* all-items RPP,
* goods RPP,
* housing/services RPP,
* other services RPP.

BEA RPP enables cost-of-living-adjusted metrics such as real wage and real income after rent.

## Core Dataset

The main processed dataset is:

```
data/processed/career_city_fit_2020_2024.csv
```
Each row represents:
```
one occupation × one metropolitan area × one year
```
such as:
```
Software Developers × Pittsburgh, PA × 2024
Registered Nurses × Dallas-Fort Worth, TX × 2024
```
Key columns include:

| Column | Meaning |
| -------------------------------------------- | --------------------------------------------------------- |
| `year` | Data year |
| `cbsa_code` | Metro area identifier |
| `bls_msa_name` | Metro name |
| `occ_code` | Occupation code |
| `occ_title` | Occupation title |
| `wage_annual_median` | Occupation-specific median wage |
| `real_wage_annual_median` | Wage adjusted by BEA RPP |
| `annual_rent` | ACS median gross rent × 12 |
| `rpp_all_items` | BEA all-items regional price parity |
| `rent_to_median_wage_ratio` | Annual rent divided by annual median wage |
| `real_income_after_rent_median` | RPP-adjusted income after annual rent |
| `tot_emp` | Estimated local occupation employment |
| `loc_quotient` | Occupation concentration in the metro |
| `career_city_score_0_100` | Default dashboard ranking score |
| `predicted_change_in_real_income_after_rent` | Predicted next-year change in purchasing power after rent |
  

## Key Engineered Metrics

  
### Rent Burden

```
rent_to_median_wage_ratio = annual_rent / wage_annual_median
```

This measures how much of an occupation’s median annual wage would be absorbed by typical annual rent in a metro. Lower is better.
  

### Wage-to-Rent Ratio

```
median_wage_to_rent_ratio = wage_annual_median / annual_rent
```
This is the inverse of rent burden and is more intuitive for scoring. Higher is better.

### RPP-Adjusted Wage

```
real_wage_annual_median = wage_annual_median / (rpp_all_items / 100)
```
This estimates the wage’s purchasing power after adjusting for regional price differences. Higher is better.

### RPP-Adjusted Income After Rent

```
real_income_after_rent_median = (wage_annual_median - annual_rent) / (rpp_all_items / 100)
```
This is one of the project’s central metrics. It approximates how much cost-adjusted income remains after paying typical annual rent. Higher is better.

### Location Quotient

Location quotient measures whether an occupation is more concentrated in a metro than it is nationally. It helps distinguish affordable places from places that are actually strong labor markets for a career. Higher is generally better.

## Default Dashboard Score


The dashboard creates a transparent career-city score, dubbed the Career Atlas score. Users can also change the weights interactively.

The default formula is:

```
Career-City Score =

35% purchasing power after rent
+ 25% rent affordability
+ 15% RPP-adjusted wage
+ 15% location quotient
+ 10% employment scale
```

All components are converted into percentile-style scores so that higher is always better.

The dashboard also includes warnings for:

* low employment estimates,
* low occupation concentration,
* high-cost metros.

## Methodology


### Data Wrangling

The data wrangling notebook:

```
src/data_wrangling_and_eda FINAL.ipynb
```

performs the following steps:

1. Downloads raw BLS, ACS, and BEA data.
2. Cleans and standardizes columns.
3. Standardizes geographic identifiers using `cbsa_code`.
4. Merges BLS occupation-metro data with ACS metro context and BEA RPP data.
5. Creates affordability and cost-adjusted metrics.
6. Saves year-level merged files.
7. Saves the final 2020–2024 occupation-metro-year dataset.
8. Runs EDA, SQL checks, and hypothesis tests.

This notebook has several important outputs that are used in both the modeling pipeline and in the final dashboard:

```
data/processed/career_city_fit_2020_2024.csv
data/processed/career_city_fit_2020_2024.parquet
data/interim/yearly_build_summary.csv
data/interim/hypothesis_tests_2024.csv
```


## Exploratory Data Analysis


The EDA investigates:  

* row coverage by year,
* wage distribution,
* BEA RPP distribution,
* ACS rent vs. BEA housing RPP,
* annual rent vs. occupation wage,
* top metros by rent burden,
* top metros by RPP-adjusted income after rent,
* location quotient vs. real wage,
* nominal wage ranking vs. RPP-adjusted wage ranking,
* rent-burden distributions by occupation,
* selected occupation trends over time.


Key EDA insight:

> High nominal salary does not always imply strong career-city fit. Rent, regional price levels, employment scale, and occupation concentration meaningfully change the ranking.
  

## SQL Analysis


The project includes SQL-based analysis using DuckDB.  

SQL is used to:
* verify yearly dataset coverage,
* rank metros by RPP-adjusted income after rent,
* inspect ranking shifts after BEA RPP adjustment,
* verify dashboard outputs,
* summarize clusters.

This demonstrates that the final processed dataset can be queried relationally as a single analytical table.

## Hypothesis Testing

  
The EDA notebook includes exploratory statistical testing.

  
Tests include:

* Welch’s t-test,
* Mann-Whitney U test,
* Cohen’s d effect size.

Example questions tested:

1. Do high-RPP metros have significantly higher annual rent than low-RPP metros?
2. For a selected occupation, do high-RPP metros differ in RPP-adjusted income after rent?
3. For a selected occupation, do high-RPP metros differ in rent burden?

These tests are exploratory and are not interpreted causally.

## Modeling

The modeling notebook `src/modeling FINAL.ipynb` builds both unsupervised and supervised modeling outputs.

### Unsupervised Modeling

The notebook creates metro-level profiles and applies:

* K-Means clustering,
* PCA visualization,
* nearest-neighbor metro similarity.

Metro profiles include:
* rent,
* RPP,
* nominal wages,
* real wages,
* real income after rent,
* rent burden,
* population,
* employment,
* location quotient,
* occupational mix.


The important outputs here are:
```
data/modeling/metro_profiles_clustered_latest.csv
data/modeling/metro_cluster_summary.csv
data/modeling/nearest_metros_latest.csv
```

### Similar Cities

Similar cities are computed using nearest neighbors on standardized metro profile features.
The dashboard uses this to answer:

> What cities are most similar to this one in terms of cost, wages, price levels, and occupational structure?

### Similar Occupations

The dashboard also computes similar occupations using occupation-level profiles built from wage, affordability, employment, and geographic concentration patterns.

## Supervised Modeling

The project trains predictive models for two targets.

### Target 1: Rent Burden

```
target = rent_to_median_wage_ratio
```

This predicts next-year housing affordability for an occupation-metro pair. Lower is better.

### Target 2: RPP-Adjusted Income After Rent
 
```
target = real_income_after_rent_median
```
This predicts next-year purchasing power after rent. Higher is better.

### Model Families

The modeling notebook compares:

* persistence baseline,
* train-mean baseline,
* linear regression,
* ridge regression,
* random forest,
* tuned ridge regression,
* tuned random forest.

 
### Train / Validation / Test Split

Because this is a forecasting problem, the split is time-based:

```
Train: earliest target years
Validation: second-most-recent target year
Test: most recent target year
```
This avoids leakage from random row splitting.

### Hyperparameter Tuning

Validation-based tuning is used for:

* Ridge `alpha`,
* Random Forest depth,
* Random Forest leaf size,
* number of trees.

 The test set is not used for model selection.

### Model Evaluation

Models are assessed using:

* MAE,
* RMSE,
* R²,
* actual vs. predicted plots,
* residual distributions,
* feature importance.

Important outputs:

```
data/modeling/regression_model_metrics_all_targets.csv
data/modeling/regression_model_metrics_rent_burden.csv
data/modeling/regression_model_metrics_real_income_after_rent.csv
data/modeling/random_forest_feature_importance_rent_burden.csv
data/modeling/random_forest_feature_importance_real_income_after_rent.csv
data/processed/career_city_next_year_predictions_latest.csv
```  

## Dashboard

The interactive dashboard is located in the `dashboard` directory.

### Dashboard Features

  
The dashboard includes separate pages for each important functionality:

1.  **Career → Cities**

* Select an occupation.
* Rank the best cities using default or custom weights.
* View map, bar chart, scatterplot, and ranking table.
* Shows the current top location.

2.  **City → Occupations**

* Select a city.
* Rank the best occupations in that city.
* Shows the current top career.

3.  **Explorer**

* Choose a city first, then an occupation.
* View historical trends.
* View boxplots and tradeoff charts.
* Rank best/worst career-city combinations for any metric.
 
4.  **Similarity**

* Find cities similar to a selected metro.
* Find occupations similar to a selected occupation.
  
5.  **Clusters**

* View metro clusters using PCA.
* Inspect cluster summaries and member metros.
  
6.  **Modeling**

* Compare model metrics.
* View feature importance.
* Inspect predicted changes.

7.  **Methodology**

* Explains data sources, score formula, modeling methods, and limitations.
 
## Dashboard File Structure

```
dashboard/
	├── app.py
	├── config.py
	├── data_loader.py
	├── analytics.py
	├── figures.py
	├── components.py
	├── callbacks.py
	└── assets
		└── styles.css
```


### File Roles

  
| File | Purpose |
| ------------------- | ------------------------------------------------------------------------ |
| `app.py` | Main Dash entrypoint |
| `config.py` | Paths, metric labels, default weights, styling constants |
| `data_loader.py` | Loads processed/modeling outputs and CBSA centroids |
| `analytics.py` | Polars/DuckDB computations for rankings, scoring, similarity, and tables |
| `figures.py` | Plotly figures and maps |
| `components.py` | Reusable Dash UI components and page layouts |
| `callbacks.py` | Interactive Dash callbacks |
| `assets/styles.css` | Dashboard styling |
  
  

### Important Data Folders


| Folder | Contents |
| ----------------- | -------------------------------------------------------------- |
| `data/raw/` | Raw BLS, ACS, and BEA files |
| `data/interim/` | Cleaned source-level tables and diagnostics |
| `data/processed/` | Final merged datasets and dashboard-ready outputs |
| `data/modeling/` | Model metrics, clusters, feature importance, nearest neighbors |
| `data/static/` | Static dashboard helper files such as CBSA centroids |
| `figures/` | EDA and modeling figures |

Not all data used in the project may be in the GitHub repository, but the script will automatically download any data that is required that is not already locally available.

## How to Run the Project

  

### 1. Install dependencies

At minimum:

```bash
pip  install  polars  pandas  numpy  matplotlib  scikit-learn  joblib  pyarrow  requests  openpyxl  duckdb  scipy  dash  plotly
```

### 2. Run the notebooks end-to-end

  Run `'src/data_wrangling_and_eda FINAL.ipynb'` and `'src/modeling FINAL.ipynb'` to create all data and models required for the dashboard.
 
### 3. Run the dashboard

Run from the project root: `python3  dashboard/app.py`. Then open `http://127.0.0.1:8050`.


## Course Topics Used


This project applies many course topics:
  
| Topic | Application |
| --------------------- | ---------------------------------------------------------------- |
| Polars | Data wrangling and dashboard analytics |
| SQL | DuckDB queries for EDA and dashboard verification |
| Joins | BLS + ACS + BEA merged by year and CBSA |
| Record linking | CBSA standardization across datasets |
| Feature engineering | Affordability, RPP-adjusted wage, lags, growth, score components |
| Hypothesis testing | Welch’s t-test, Mann-Whitney U, Cohen’s d |
| Supervised learning | Regression models for future affordability and purchasing power |
| Unsupervised learning | K-Means, PCA, nearest neighbors |
| Time series | Lagged panel features and time-based split |
| Hyperparameter tuning | Validation-based Ridge and Random Forest tuning |
| Visualization | EDA figures and interactive Plotly/Dash dashboard |

  
## Major Insights


1.  **Nominal wage is not enough.**

High-wage metros often also have high rent and high regional price levels.

2.  **Rent burden and purchasing power are different.**

A city can have reasonable rent burden but still weaker overall cost-adjusted purchasing power.

3.  **BEA RPP changes rankings.**

Some metros fall after cost-of-living adjustment, while some mid-cost metros become more attractive.

4.  **Affordability and opportunity must be evaluated together.**

Cheap metros are not always strong labor markets for a given occupation. Location quotient and employment scale are important.

5.  **Career-city fit is occupation-specific.**

A city may be excellent for one occupation and poor for another.

6.  **Predictions are useful as signals.**

Forecasts should be interpreted as short-term indicators of improvement or decline, not exact guarantees.

## Limitations

  
* BEA RPP adjusts regional price levels but does not capture individual household spending patterns.
* ACS rent is metro-level and not occupation-specific.
* BLS occupation estimates can be noisy for small metro-occupation pairs.
* Some occupations have missing or estimated wage values.
* The time series covers 2020–2024, which limits long-term forecasting.
* K-Means clusters are exploratory and should not be treated as definitive metro categories.
* Feature importance is predictive, not causal.
* The dashboard score is a transparent weighted ranking system, not a causal model.  

## Future Work

Potential extensions include:

* adding LAUS unemployment data,
* adding O*NET occupation skill/education metadata,
* adding HUD Fair Market Rents,
* adding QCEW employment momentum data,
* adding classification models for “improving” vs. “declining” metro-occupation pairs,
* adding confidence intervals or uncertainty flags,
* supporting user-entered salary and rent assumptions,
* improving maps with CBSA polygon boundaries instead of centroids,
* deploying the dashboard publicly.
  

## Project Value Proposition

 
This project supports students, graduates, early-career workers, and relocating professionals by helping them compare cities for a chosen occupation using more than salary alone.

It combines:

```
occupation-specific wage
+ local rent
+ regional cost of living
+ employment scale
+ occupation concentration
+ historical trends
+ predictive signals
```

to provide a more complete view of where a career may offer strong affordability and opportunity.