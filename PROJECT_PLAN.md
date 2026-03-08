# ECE 1513 Project Plan: Toronto Bikeshare Demand Forecasting

## Context
ECE 1513 course project requiring an ML solution with a baseline → improved pipeline. The goal is to predict station-level bikeshare demand (pickups per time window) using Toronto Bikeshare ridership data, enriched with weather data. This combines the user's data engineering interests (ETL pipeline) with classical ML methods covered in the course.

## Problem Statement
**Predict the number of bike pickups at each station for a given hour**, given temporal features (hour, day-of-week, month, holiday) and weather conditions (temperature, precipitation, wind speed).

- **Evaluation metrics**: RMSE, MAE, and R² on a held-out test set
- **Business motivation**: Enables proactive bike rebalancing to avoid empty/full stations
- **Secondary output**: Capacity gap analysis — flag stations where predicted demand exceeds station capacity, signaling need for infrastructure expansion or more frequent rebalancing

## Course-Aligned ML Technique Progression
Based on ECE 1513 syllabus (topics explicitly taught in course marked with *):

| Step | Model | ML Technique | Course Link |
|------|-------|-------------|-------------|
| Baseline | Linear Regression (station features only) | *Linear Regression* | Part II: Classification & Regression |
| Improvement 1 | Ridge Regression (all features) | *Regularization* | Part II: Regularization and Ensemble Learning |
| Improvement 2 | Random Forest (all features) | *Ensemble Learning* | Part II: Regularization and Ensemble Learning |
| Improvement 3 | XGBoost (all features, tuned) | Gradient Boosting (outside course) | Extension beyond course content |

This progression satisfies the course requirement: *"A primary solution can be an idea discussed in the course... You should refine this solution via ML either via a tool taught in the course or something out of the course content."*

**Important distinction:**
- The **ML improvement** (Linear → Regularized → Ensemble → Boosting) satisfies the course rubric
- The **data enrichment** (weather API) is the data engineering contribution — evaluated separately via an ablation study

## Project Structure

```
ECE 1513/
├── data/
│   ├── raw/              # Raw bikeshare CSVs + weather data
│   └── processed/        # Cleaned, merged, feature-engineered data
├── notebooks/
│   ├── 01_data_ingestion.ipynb      # Download & explore raw data
│   ├── 02_data_cleaning.ipynb       # Clean, handle missing values, outliers
│   ├── 03_feature_engineering.ipynb  # Temporal features + weather merge
│   ├── 04_eda.ipynb                 # Exploratory data analysis & visualizations
│   ├── 05_baseline_model.ipynb      # Linear Regression baseline
│   ├── 06_improved_model.ipynb      # Ridge, RF, XGBoost
│   └── 07_evaluation.ipynb          # Comparison, ablation, capacity gap analysis
├── .gitignore
├── requirements.txt
└── README.md
```

## Implementation Steps

### Step 1: Repository Setup
- Create project directory structure (`data/raw/`, `data/processed/`, `notebooks/`)
- Create `.gitignore` (ignore `data/raw/`, large files, `.ipynb_checkpoints`)
- Create `requirements.txt` with: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `requests`, `jupyter`

### Step 2: Data Ingestion (`01_data_ingestion.ipynb`)
- Load Toronto Bikeshare ridership data (2025, monthly CSVs already downloaded, ~200K-1.1M rows per month)
  - Data fields: Trip ID, Trip Duration, Start Station ID/Name, End Station ID/Name, Start Time, End Time, User Type
  - No schema evolution needed — single year, consistent format
- Download historical weather data from **Open-Meteo API** (free, no key required)
  - Endpoint: `https://archive-api.open-meteo.com/v1/archive` with Toronto coordinates (43.65, -79.38)
  - Fields: temperature_2m, precipitation, wind_speed_10m, weather_code
- Save raw data to `data/raw/`

### Step 3: Data Cleaning (`02_data_cleaning.ipynb`)
- Parse datetime columns, standardize to EST timezone
- Remove trips with invalid/missing station IDs or extreme durations (< 1 min or > 24 hrs)
- Aggregate trip-level data into **hourly pickup counts per station** (GROUP BY + count)
- Fill implicit zeros: generate full (station × hour) grid, fill missing entries with 0
- Clean weather data (handle missing values via interpolation)
- Save cleaned dataframes to `data/processed/`

### Step 4: Feature Engineering (`03_feature_engineering.ipynb`)
- **Station features**: station capacity, historical average demand (station popularity)
- **Temporal features**: hour_of_day, day_of_week, month, is_weekend, is_holiday, is_rush_hour
- **Weather features**: temperature, precipitation, wind_speed, is_rainy (binary)
- **Lag features**: demand at same station in previous hour, same hour yesterday, same hour last week
- Merge all features into a single modeling-ready dataframe
- **Train/test split**: Chronological — Jan-Aug 2025 for training (~80%), Sep-Oct 2025 for testing (~20%). Model has seen both cold (Jan-Mar) and warm (Jun-Aug) months during training, so it can generalize to fall test data via weather + month features

### Step 5: EDA (`04_eda.ipynb`)
- Distribution of pickups across stations, hours, days
- Seasonal and weekly patterns (heatmaps)
- Correlation between weather and demand
- Station maps (lat/lon scatter colored by average demand)
- These visualizations will be used in the report

### Step 6: Baseline Model (`05_baseline_model.ipynb`)
- **Baseline**: Linear Regression on station features only (capacity, historical popularity)
  - Model learns a static popularity score per station — same prediction regardless of time/weather
- Evaluate on test set: RMSE, MAE, R²

### Step 7: Improved Models (`06_improved_model.ipynb`)
- **Model 1 — Linear Regression + temporal features**: Same model, but now knows *when* (hour, day, month). Shows value of temporal context
- **Model 2 — Linear Regression + all features (incl. weather)**: Shows value of data enrichment
- **Model 3 — Ridge Regression (all features)**: Adds *regularization* (course-taught technique) to handle feature correlations
- **Model 4 — Random Forest (all features)**: *Ensemble learning* (course-taught technique) captures non-linear interactions
- **Model 5 — XGBoost (all features, tuned)**: Gradient boosting (beyond course) with hyperparameter tuning via cross-validation
- Feature importance analysis on best model

### Step 8: Evaluation & Report Figures (`07_evaluation.ipynb`)
- **Model comparison table**: Baseline → each improvement step (RMSE, MAE, R²)
- **Ablation table**: Performance with station-only → +temporal → +weather features
- **Capacity gap analysis**: For best model, compute `predicted_pickups - station_capacity` per station per hour. Rank stations most likely to exceed capacity
- Plot: Predicted vs actual demand for sample stations/days
- Plot: Feature importance bar chart
- Error analysis: which stations/times are hardest to predict?

## Data Engineering Highlights (for resume)
- **Multi-source ingestion**: Ingest monthly ridership CSVs (~5M+ total rows) + external weather API data
- **API enrichment**: Pull historical weather data via Open-Meteo REST API, join on temporal keys
- **Data quality**: Handle nulls, implicit zeros, timezone alignment, outlier filtering
- **Aggregation pipeline**: Transform raw trip-level events into hourly station-level pickup counts

## Key Design Justifications (for the report)
- **Why demand forecasting?** Direct operational value for bike rebalancing
- **Why hourly granularity?** Balances signal quality vs. data sparsity
- **Why weather enrichment?** Weather is an intuitive driver of cycling behavior; ablation study proves its value
- **Why Ridge → RF → XGBoost?** Regularization and ensemble learning are taught in course; boosting extends beyond course
- **Why chronological split?** Prevents data leakage from future → past

## Risk Assessment

| Category | Items | Risk |
|----------|-------|------|
| Easy / quick | Repo setup, weather API, EDA, model training, evaluation | Low |
| Moderate effort | Aggregation, lag features, timezone alignment | Medium |
| Potential time sinks | Implicit zero filling (full station×hour grid), data volume (~5M+ raw rows) | Medium-High |

**Key mitigation**: Aggregate to hourly counts early to reduce data size. Use dtype optimizations if pandas is slow.

## Verification / Testing
1. Run notebooks 01-03 end-to-end, verify processed data has expected shape and no nulls
2. Verify baseline RMSE is reasonable (not zero, not absurdly high)
3. Verify each improved model beats the previous step on all metrics
4. Verify ablation shows weather features contribute meaningfully
5. Spot-check predictions on known days (rainy weekday vs sunny weekend)
6. Verify capacity gap analysis produces sensible station rankings

## Timeline Considerations
- **Presentation**: April 6-10, 2026
- **Report**: Max 5 pages, mandatory LaTeX template, PDF only
- **Code submission**: Compressed .zip with documentation and README
