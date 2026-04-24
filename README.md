# NIS Introduction Risk Prediction

Machine learning research project that predicts the risk of Non-Indigenous Species (NIS) introduction via ballast water discharge from ships. Given a maritime route defined by daily temperature and salinity conditions, the models predict whether the zooplankton carried in ballast water will survive the voyage and pose an introduction risk at the destination port.

---

## Background

Ships take on ballast water at origin ports (carrying local zooplankton) and discharge it at destination ports. Survival of those organisms depends on the environmental conditions (temperature, salinity) they experience along the route. This project trains classifiers on controlled tank experiments that simulate route conditions, then scores real planned shipping routes to estimate NIS introduction risk.

---

## Project Structure

```
GoogleShips/
├── prepare_environment_data/   # Download and convert NASA/CEDA environmental datasets
├── prepare_routes/             # Build route condition profiles from coordinates
├── analyze_expirement_results/ # Core ML pipeline (training, evaluation, scoring)
│   └── score_routes/           # Inference pipeline for scoring real shipping routes
├── tests/                      # Unit and integration tests
└── outputs_revision/           # Experiment outputs (generated, not for editing)
```

---

## Pipelines

### 1. `prepare_environment_data/`
One-time setup scripts to download sea surface temperature and salinity datasets from NASA and CEDA, and convert them to a common format for spatial lookup.

- `download_dbs.py` — downloads raw NetCDF datasets
- `convert_nc_file.py` / `convert_nasa_dbs_to_ceda_format.py` — normalise datasets to a common schema

### 2. `prepare_routes/`
Processes raw ship route coordinates into daily environmental condition profiles used as model input.

- `prepare_routes.py` — main entry point; queries environmental datasets via KDTree spatial lookup for each route waypoint and produces `routes_with_conditions.csv`
- `sample_routes.py` — randomly samples routes from the full set for planned-route scoring
- `add_new_sampled_routes.py` — extends an existing sample with new routes

### 3. `analyze_expirement_results/`
The core ML training and evaluation pipeline.

| Script | Purpose |
|--------|---------|
| `main.py` | Generates all combinatorial configurations and dispatches training jobs (locally or via SLURM) |
| `run_analysis_of_config.py` | Runs one configuration end-to-end: preprocess → visualise → train → aggregate |
| `preprocess.py` | Loads `Final_Data_Voyages.xlsx`, engineers temporal features (30-day windows, day-based intervals, acclimation days), splits train/test |
| `model.py` | Wraps sklearn classifiers with GridSearchCV, optional RFECV/Optuna feature selection, and produces feature importance and PDP+ICE plots |
| `model_lstm.py` + `train_lstm_cv.py` | PyTorch Lightning LSTM trained on (Temperature, Salinity, Time) sequences with 5-fold CV |
| `configuration.py` | Defines all experimental dimensions (interval lengths, metrics, balancing strategies, etc.) |
| `learning_curve_analysis.py` | Post-hoc analysis: trains the best RandomForest model on increasing subsets of the training data and plots test-set metrics vs. train set size |

**Models trained:** Decision Tree, Random Forest, KNN, Logistic Regression, Gradient Boosting, XGBoost, MLP, LSTM, and a soft VotingClassifier over the three best tree-based models.

**Cross-validation:** 5-fold stratified CV; class imbalance handled via `RandomUnderSampler`.

### 4. `analyze_expirement_results/score_routes/`
Inference pipeline that applies trained models to real planned shipping routes.

| Script | Purpose |
|--------|---------|
| `main.py` | Orchestrator — runs the three steps below in sequence |
| `prepare_full_routes.py` | Fills missing route days with planned-route conditions |
| `score_full_routes.py` | Runs the best trained model on each route and aggregates daily survival probabilities (min / max / mean / log-multiply) |
| `classify_routes_risk.py` | Fits a logistic regression on the aggregated scores to produce a binary and 3-class NIS introduction risk label per route |

---

## Key Data Files

| File | Description |
|------|-------------|
| `analyze_expirement_results/data/Final_Data_Voyages.xlsx` | Master experimental dataset |
| `analyze_expirement_results/data/preprocess/Final_Data_Voyages_Processed_30.csv` | Preprocessed cache (auto-generated) |
| `analyze_expirement_results/score_routes/planned_routes/all_sampled_routes.csv` | Planned shipping routes used for scoring |

---

## Running the Pipeline

### Train models (local)
```bash
cd analyze_expirement_results
python main.py \
  --outputs_dir_name outputs_test \
  --cpus 1 \
  --do_feature_selection False \
  --run_configurations_in_parallel False \
  --run_lstm_configurations_in_parallel False
```

### Train a single configuration
```bash
python analyze_expirement_results/run_analysis_of_config.py <path_to_config.csv>
```

### Score trained models on real routes
```bash
python analyze_expirement_results/score_routes/main.py --base_dir outputs_revision/configuration_0
```

### Learning curve analysis (best model)
```bash
python analyze_expirement_results/learning_curve_analysis.py --base_dir outputs_revision/configuration_0
```

---

## Configuration System

`configuration.py` defines all experimental dimensions as lists that are crossed into a full combinatorial grid:

| Parameter | Options |
|-----------|---------|
| `INTERVAL_LENGTH` | [1, 2, 3, 4] days of history per prediction |
| `METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS` | `mcc`, `f1` |
| `BALANCE_CLASSES_IN_TRAINING` | True / False |
| `STRATIFY_TRAIN_TEST_SPLIT` | True / False |
| `INCLUDE_CONTROL_ROUTES` | True / False |

**DEBUG_MODE** is auto-detected: Windows → `True` (small datasets, few epochs/trials); Linux → `False` (full scale).

---
