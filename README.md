# Churn Prediction: Community Engagement Data (tmw)

Independent, unguided study project focused on predicting user churn from an online community's engagement data, transactions, loyalty points, chat activity, and live-stream attendance, using an Analytical Base Table (ABT) and a tuned Random Forest classifier.

## Objective

Practice the end-to-end workflow of a churn model: starting from an already-built ABT, selecting relevant features, handling an out-of-time validation window, tuning a model with cross-validation, and tracking experiments with MLflow.

This is a study repo, kept intentionally small and focused on the modeling step (`train.py`) rather than the ABT construction itself.

## What's in this repo

| File | Content |
|---|---|
| `train.py` | Full modeling pipeline: sampling, feature selection, preprocessing, training, and evaluation |
| `data/abt_churn.csv` | Analytical Base Table (ABT) used as model input |

## The data

`abt_churn.csv` is a monthly snapshot table at the **user × reference month** grain (`dtRef`), covering **14 months** (March 2024 to April 2025), **2,251 unique users**, and **5,496 rows**, with an overall churn rate of **~47%**.

Each row describes a user's engagement in that period through 43 columns, grouped as:

- **Overall activity**: number of transactions, active days, points balance (positive/negative), distinct items purchased (SKUs), chat messages, live-stream presence, and community-specific actions (daily loot claims, RPG item sales, streak days, etc.).
- **Recency/frequency windows**: the same core metrics (transactions, active days, points balance) recomputed over the last **7, 14, and 28 days** (`*D7`, `*D14`, `*D28`).
- **Trend features** (`propAvg*`): each user's recent activity compared proportionally against their own historical average, to capture acceleration or slowdown in engagement.
- **Target**: `flagChurn`, a binary flag indicating whether the user churned.

## Modeling approach (`train.py`)

- **Out-of-time validation**: the most recent `dtRef` is held out as an out-of-time (OOT) set; the remaining months are split into train/test (80/20, stratified on the target).
- **Feature selection**: a baseline `DecisionTreeClassifier` ranks feature importance, and only the features covering the top 96% of cumulative importance are kept.
- **Preprocessing pipeline** (via [`feature-engine`](https://feature-engine.trainindata.com/)):
  - `DecisionTreeDiscretiser` to bin the selected numeric variables
  - One-Hot Encoding on the resulting bins
- **Model**: `RandomForestClassifier`, tuned with `GridSearchCV` (3-fold CV, scored on ROC-AUC) across `min_samples_leaf`, `n_estimators`, and `criterion`.
- **Experiment tracking**: MLflow autologging, with accuracy and ROC-AUC logged for train, test, and out-of-time sets, plus a combined ROC curve plot.

## Tech stack

Python · pandas · scikit-learn · feature-engine · MLflow · Matplotlib

## Running it

`train.py` expects an MLflow tracking server at `http://127.0.0.1:5000/` (adjust `mlflow.set_tracking_uri` / `set_experiment` as needed) and reads the ABT from `../data/abt_churn.csv`, so it should be run from a subfolder (e.g. `src/` or `notebooks/`) — or the path can be updated to `data/abt_churn.csv` when run from the repo root.

Required packages: `pandas`, `scikit-learn`, `mlflow`, `feature-engine`, `matplotlib`.