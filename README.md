# 🧑‍💼 HR Attrition Analytics Platform

An enterprise-grade HR analytics platform that predicts employee attrition risk,
explains *why* each employee is at risk, and turns that into actionable retention
recommendations — with a live dashboard and a Power BI-ready export.

Built on the IBM HR Analytics Employee Attrition & Performance dataset (1,470 employees,
35 attributes, ~16% attrition rate).

---

## 1. What this project demonstrates

| Area | What's in here |
|---|---|
| **Data Engineering** | Automatic missing-value detection, IQR + Isolation Forest outlier detection, rare-category grouping, reusable `sklearn` `Pipeline`/`ColumnTransformer` |
| **Feature Engineering** | 20+ HR-domain features (career stagnation, retention score, employee value score, engagement/compensation composites, etc.) |
| **Feature Selection** | Variance filtering + Mutual Information ranking, saved and reused at inference time |
| **Machine Learning** | 14+ classifiers trained and benchmarked (Logistic Regression → Gradient Boosting, XGBoost, LightGBM, CatBoost, Voting & Stacking ensembles) |
| **Imbalance Handling** | SMOTE / ADASYN / Borderline-SMOTE via `imbalanced-learn` |
| **Hyperparameter Tuning** | `RandomizedSearchCV` with `StratifiedKFold` on the top-performing candidates |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC AUC, PR AUC, Log Loss, MCC, Cohen's Kappa, Balanced Accuracy, Specificity/Sensitivity, calibration & learning curves |
| **Explainable AI** | SHAP global importance, beeswarm plots, permutation importance, per-model feature importance |
| **MLOps** | Versioned model artifacts (timestamped), full metadata JSON, structured logging, reproducible pipeline |
| **Dashboarding** | 8-tab Streamlit app: executive dashboard, risk heatmaps, employee search, single/batch prediction, department analytics, SHAP explainability, employee comparison |
| **Business Intelligence** | Power BI-ready CSV export with risk score, segment, department KPIs, and per-employee retention recommendations |

---

## 2. Project structure

```
attrition_project/
├── data/
│   └── employee_attrition.csv       # IBM HR Analytics raw dataset
├── models/                          # Trained model + preprocessing artifacts (generated)
│   ├── best_model.pkl
│   ├── best_model_<timestamp>.pkl   # versioned copy
│   ├── best_model_info.json         # metrics, feature list, version, timestamp
│   ├── encoders.pkl / scaler.pkl / feature_names.pkl / selected_features.json
│   └── shap_explainer_cache.pkl
├── outputs/                         # Generated reports (generated)
│   ├── model_comparison.csv / model_leaderboard.json
│   ├── feature_importance.csv / permutation_importance.csv / shap_feature_importance.csv
│   ├── feature_selection_mutual_info.csv
│   └── powerbi_attrition_dataset.csv
├── eda_plots/                       # 30+ EDA visualizations + EDA_REPORT.md (generated)
├── eval_plots/                      # Confusion matrices, ROC/PR curves, SHAP plots (generated)
├── logs/                            # pipeline.log (generated)
├── config.py                        # Paths, constants, logging setup
├── preprocessing.py                 # Cleaning, feature engineering, encoding, feature selection
├── eda.py                           # Exploratory data analysis (30+ charts)
├── train_models.py                  # Model bake-off, tuning, evaluation, SHAP, versioning
├── utils.py                         # Inference, risk scoring, HR recommendations, Power BI export
├── app.py                           # Streamlit dashboard (8 tabs)
├── requirements.txt
└── README.md
```

### Pipeline flow

```
Raw CSV
   │
   ▼
clean_data()          # duplicates, dtype fixes, imputation, rare-category grouping
   │
   ▼
engineer_features()   # 20+ HR domain features
   │
   ▼
encode_features()     # label encode binaries, one-hot encode multi-class
   │
   ▼
select_features()     # mutual information ranking → top 45 features
   │
   ▼
train/test split → StandardScaler
   │
   ▼
train_models.py       # 14+ models → SMOTE → tune top 3 → evaluate → SHAP → save best
   │
   ▼
models/best_model.pkl + metadata  ──▶  utils.predict_attrition_risk()  ──▶  app.py / Power BI CSV
```

---

## 3. How to run

```bash
pip install -r requirements.txt

# 1. Explore the data — saves 30+ plots + EDA_REPORT.md to eda_plots/
python eda.py

# 2. Train, tune, evaluate and explain all models — saves best model to models/
python train_models.py

# 3. Generate the Power BI-ready dataset (also runs automatically the first
#    time app.py loads the Executive Dashboard)
python utils.py

# 4. Launch the dashboard
streamlit run app.py
```

Each script is independently runnable and writes its outputs to disk, so the whole
pipeline can also be orchestrated by a scheduler (Airflow, cron, GitHub Actions) for
periodic retraining.

---

## 4. Preprocessing & feature engineering highlights

- **Missing values:** automatically detected; median/mode imputation by default,
  KNN imputation available (`impute_missing(df, strategy="knn")`).
- **Outliers:** univariate IQR flags + multivariate Isolation Forest flags, both
  visualized in EDA (not silently dropped — outliers here are legitimate employees).
- **Engineered features** include `RetentionScore`, `EmployeeValueScore`,
  `CareerStagnationScore`, `PromotionFrequency`, `CompanyLoyalty`, `WorkloadScore`,
  `ManagerRelationshipScore`, `TravelRisk`, and tenure/age/income bands — designed to
  give both the model *and* SHAP explanations meaningful, business-readable signal.
- **Feature selection:** Mutual Information ranks all 70+ candidate features (raw +
  engineered) and keeps the top 45; the selected list is persisted to
  `models/selected_features.json` so inference always uses the exact training schema.

---

## 5. Model performance

Full leaderboard is written to `outputs/model_comparison.csv` after each training run
(exact numbers vary run-to-run due to SMOTE resampling and CV splits). In a
representative run:

| Model | ROC AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Gradient Boosting | 0.838 | 0.43 | 0.50 | 0.46 |
| AdaBoost | 0.829 | 0.51 | 0.47 | 0.49 |
| Voting Ensemble | 0.828 | 0.62 | 0.52 | 0.56 |
| Logistic Regression | 0.821 | 0.79 | 0.37 | 0.50 |

The pipeline selects the **best model by ROC AUC** automatically, but note the
precision/recall trade-off: Logistic Regression catches far more true leavers
(recall 0.79) at the cost of more false alarms — worth reconsidering as the
"deployed" model if HR intervention capacity is limited and missing a flight risk is
more costly than an unnecessary check-in. This trade-off is intentionally left visible
in the leaderboard rather than hidden behind a single metric.

---

## 6. Explainability

`train_models.py` generates SHAP global importance and beeswarm plots for the winning
model (`eval_plots/shap_global_importance.png`, `shap_beeswarm.png`), plus a
permutation-importance CSV as a model-agnostic sanity check against SHAP. The
Streamlit app's **Explainability** tab surfaces these directly, and every scored
employee also gets a rule-based `HR_Recommendation` string derived from the same
known attrition drivers (overtime, satisfaction, promotion gap, travel, manager
tenure) — so the "why" is legible to HR without requiring them to read a SHAP plot.

---

## 7. Streamlit dashboard

8 tabs: **Executive Dashboard** (KPIs, risk distribution, top-15 at-risk employees),
**Risk Heatmap** (department × role, tenure trend), **Employee Search**, **Single
Prediction** (what-if form), **Batch Prediction** (CSV upload), **Department
Analytics**, **Explainability (SHAP)**, and **Employee Comparison**. Includes a
dark-mode toggle and sidebar filters that apply across the dashboard.

Run with: `streamlit run app.py`

---

## 8. Power BI

`utils.export_powerbi_dataset()` writes `outputs/powerbi_attrition_dataset.csv` with
the original HR attributes plus `Attrition_Probability`, `Risk_Score`, `Risk_Segment`,
`HR_Recommendation`, `Priority_Score`, `Department_Avg_Risk`, and banded columns
(`Tenure_Band`, `Income_Band`, `Age_Band`) for clean slicers. Suggested dashboard
pages once imported into Power BI Desktop:

1. **Executive Summary** — headcount, risk segment cards, predicted attrition rate
2. **Risk Segmentation** — donut/bar of `Risk_Segment`, trend by `Tenure_Band`
3. **Department & Job Role** — matrix of `Department_Avg_Risk`, drill into `JobRole`
4. **Salary & Promotion** — `MonthlyIncome` vs `Attrition_Probability`, promotion gap
5. **Retention Priority** — table sorted by `Priority_Score` with `HR_Recommendation`

(Building the `.pbix` file itself requires Power BI Desktop — this repo produces the
model-ready CSV that plugs directly into it.)

---

## 9. Business recommendations (from EDA + feature importance)

- **Overtime is the strongest single risk driver** — review workload distribution in
  overtime-heavy roles (Sales, Lab Technicians).
- **Promotion cadence matters** — employees several years past their last promotion
  show meaningfully higher attrition; the `HR_Recommendation` column flags these
  automatically.
- **Frequent travel + low satisfaction compound risk** — prioritize retention
  conversations where both are true.
- **Compensation alone doesn't explain attrition** — satisfaction and work-life
  balance are levers HR can act on faster than pay bands.

---

## 10. Engineering notes

- All paths/constants live in `config.py`; every module uses a shared logger that
  writes to both console and `logs/pipeline.log`.
- `preprocessing.py` exposes reusable, independently testable functions
  (`clean_data`, `engineer_features`, `encode_features`, `select_features`,
  `build_preprocessing_pipeline`) rather than one monolithic script.
- Inference (`utils.prepare_features_for_inference`) re-derives engineered features
  from raw input and re-aligns columns to the training schema, so a single
  hand-typed employee record and a full CSV batch go through the identical code path
  as training — no train/serve skew.
- Models are versioned on every training run (`models/best_model_<timestamp>.pkl`),
  with metrics, feature list, and resampling method captured in
  `models/best_model_info.json`.

---

## 11. Tech stack

Python · pandas · scikit-learn · XGBoost · LightGBM · CatBoost · imbalanced-learn ·
SHAP · Optuna-ready tuning hooks · matplotlib/seaborn · Streamlit · Power BI

---

## 12. Troubleshooting

**`ModuleNotFoundError` / pickle `find_class` errors when loading `best_model.pkl`**
Pickled scikit-learn models are not portable across scikit-learn versions. Either
`pip install -r requirements.txt` (now pinned to exact versions) so your environment
matches the one the shipped model was trained with, or simply retrain locally —
`python train_models.py` regenerates every `.pkl` against whatever versions you have
installed.

**Single-employee "what-if" prediction used to crash with `ValueError: Bin edges must
be unique`** — fixed. Originally `CompensationScore` used `pd.qcut()` inside
`engineer_features()`, which computes quantile bin edges from the *current batch*.
That's fine for the full 1,470-row workforce but breaks (or silently misbehaves) on a
batch of one, since one identical value can't be split into quartiles. The fix:
`compute_income_bins()` learns fixed bin edges once from the training population,
persists them to `models/feature_reference_stats.pkl`, and `engineer_features()` now
takes those edges as a parameter (via `pd.cut`, not `pd.qcut`) so training and
inference — single record or batch — always produce identical, reproducible feature
values. The same fix was applied to `RetentionScore`'s income normalization, which
had the same class of bug (recomputing `min()`/`max()` from the current batch instead
of a fixed reference range).

---

## 13. Future improvements

- Swap the mutual-information feature selector for a full RFE / SHAP-based selection
  loop and compare downstream model performance.
- Add Optuna Bayesian optimization as an alternative to `RandomizedSearchCV` for the
  final tuning pass (the dependency is already in `requirements.txt`).
- Containerize (`Dockerfile`) and add a CI workflow that retrains and re-benchmarks
  on every data update.
- Add drift monitoring (e.g. `evidently`) to detect when the workforce population
  shifts away from the training distribution.
