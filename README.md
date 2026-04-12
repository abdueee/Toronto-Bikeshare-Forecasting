# Toronto Bikeshare Demand Forecasting

Predicts hourly bike pickup demand at the cluster level across Toronto using tabular ML models. Ridership, station metadata, and weather data are combined into a shared pipeline comparing Linear Regression, Random Forest, XGBoost, and a PyTorch MLP.

## Results (February 2026 test set)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 11.18 | 33.01 | 0.708 |
| Random Forest (tuned) | 7.48 | 23.44 | 0.853 |
| XGBoost (tuned) | **7.15** | **22.03** | **0.870** |
| PyTorch MLP | 7.61 | 23.52 | 0.852 |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/abdueee/Toronto-Bikeshare-Forecasting.git
cd Toronto-Bikeshare-Forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch is not included in `requirements.txt` because the install command depends on your hardware. Get the correct command for your system at [pytorch.org/get-started](https://pytorch.org/get-started/locally/) — for CPU-only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Add ridership data

Download the monthly ridership CSVs from the [Toronto Open Data portal](https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/) and place them in `toronto-bikeshare-data/`:

```
toronto-bikeshare-data/
  bikeshare_2025_01.csv
  bikeshare_2025_02.csv
  ...
  bikeshare_2025_10.csv
  bikeshare_2026_01.csv
  bikeshare_2026_02.csv
```

Weather and station metadata are fetched automatically at runtime (no manual download needed).

---

## Usage

Run notebooks in order. Each notebook saves its outputs to `data/processed/` or `results/` for the next step to consume.

### 1. `notebooks/data_ingestion.ipynb`

Loads all ridership CSVs, cleans trips, aggregates to hourly pickups per station, fetches station metadata from the Toronto GBFS API and weather from Open-Meteo (Jan 2025 to Feb 2026).

**Outputs:** `data/raw/rides_raw.csv`, `data/raw/stations.csv`, `data/processed/hourly_pickups.csv`, `data/processed/weather_hourly.csv`

### 2. `notebooks/eda.ipynb`

Analyzes zero inflation at the station level, runs K-means elbow to select cluster count (K=8), maps clusters, and compares zero inflation before and after aggregation.

**Outputs:** `data/processed/station_clusters.csv`, `results/eda_*.png`

### 3. `notebooks/feature_engineering.ipynb`

Merges pickups, clusters, and weather; builds 12 features including a 24-hour demand lag; performs chronological train/val/test split.

**Outputs:** `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`

### 4. `notebooks/baseline_model.ipynb`

Fits Linear Regression on the training set and evaluates on the test set.

**Outputs:** `results/baseline_metrics.json`, `results/baseline_pred_vs_actual.png`

### 5. `notebooks/tree_model_tuning.ipynb`

Grid-searches Random Forest and XGBoost hyperparameters using validation RMSE, then evaluates the best configurations on the test set.

**Outputs:** `results/tree_model_metrics.csv`, `results/tree_model_best_params.json`, `results/best_tuned_tree_pred_vs_actual.png`

### 6. `notebooks/pytorch_mlp_tuning.ipynb`

Trains a feedforward MLP ([128, 64, 32], dropout=0.2, Softplus output) with early stopping on validation RMSE.

**Outputs:** `results/pytorch_mlp_metrics.csv`, `results/pytorch_mlp_best_params.json`, `results/pytorch_mlp_loss_curve.png`, `results/pytorch_mlp_pred_vs_actual.png`

---

## Feature Schema

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `cluster_capacity` | continuous | total dock capacity in cluster |
| 2 | `hour_of_day` | int 0-23 | hour of the day |
| 3 | `day_of_week` | int 0-6 | 0 = Monday |
| 4 | `month` | int 1-12 | month of the year |
| 5 | `is_weekend` | binary | day_of_week >= 5 |
| 6 | `is_rush_hour` | binary | 7-9am or 4-6pm on a weekday |
| 7 | `is_holiday` | binary | Ontario statutory holidays 2025-2026 |
| 8 | `temperature_c` | continuous | hourly temperature |
| 9 | `precipitation_mm` | continuous | hourly precipitation |
| 10 | `wind_speed_kmh` | continuous | hourly wind speed |
| 11 | `is_rainy` | binary | precipitation_mm > 0 |
| 12 | `lag_24h` | continuous | actual pickups same cluster, 24 hours prior |

---

## Data Split

| Split | Period | Rows |
|---|---|---|
| Train | 2025-01-02 to 2025-10-31 | 58,176 |
| Validation | 2026-01-02 to 2026-01-31 | 5,760 |
| Test | 2026-02-01 to 2026-02-28 | 5,376 |

Nov/Dec 2025 are excluded from training (no source data available). Jan 1, 2026 is dropped from validation because its lag pulls from the empty Dec 31 entry. The val set is reserved for hyperparameter tuning; final metrics are reported on the test set only.

---

## File Structure

```
toronto-bikeshare-data/     <- monthly ridership CSVs (not tracked in git)
data/
  raw/                      <- rides_raw.csv, stations.csv (not tracked)
  processed/                <- hourly_pickups, clusters, train/val/test (not tracked)
notebooks/
  data_ingestion.ipynb
  eda.ipynb
  feature_engineering.ipynb
  baseline_model.ipynb
  tree_model_tuning.ipynb
  pytorch_mlp_tuning.ipynb
results/                    <- metrics, plots, best params
```

---

## Authors

Abdul Mohammed, Esam Uddin, Axel Tang: University of Toronto ECE
