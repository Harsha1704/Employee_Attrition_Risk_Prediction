"""
preprocessing.py
-----------------
Data cleaning, feature engineering, encoding, scaling and feature
selection pipeline for the HR Attrition Analytics platform.

Design goals
------------
- Reusable, sklearn-native building blocks (Pipeline / ColumnTransformer)
  so the exact same transformation can be replayed at training time and
  at inference time (no train/serve skew).
- Automatic, generic handling of missing values, duplicates and outliers
  so the pipeline is not hard-coded to "this dataset happens to be clean".
- A rich set of HR-domain engineered features that give the models (and
  the explainability layer) genuinely meaningful signal to work with.

Run: python preprocessing.py   -> prints a summary of the processed data
"""

from __future__ import annotations

import os
import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from config import (
    DROP_COLS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
    MODELS_DIR,
    OUTPUTS_DIR,
    get_logger,
)

logger = get_logger(__name__)

# Scalers models tend to prefer. Distance/gradient-sensitive models get
# StandardScaler; tree ensembles don't need scaling at all but we still
# provide a RobustScaler variant (useful for SVM/KNN/MLP with outliers).
MODEL_SCALER_MAP = {
    "tree": None,                 # Random Forest, XGBoost, LightGBM, CatBoost, etc.
    "linear": StandardScaler,     # Logistic Regression
    "distance": RobustScaler,     # KNN, SVM
    "neural": MinMaxScaler,       # MLP
}


# ------------------------------------------------------------------
# 1. Loading
# ------------------------------------------------------------------
def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw HR CSV file."""
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# ------------------------------------------------------------------
# 2. Cleaning: duplicates, dtype fixes, missing values, outliers
# ------------------------------------------------------------------
def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce obviously-numeric columns stored as strings, and vice versa."""
    for col in df.columns:
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().mean() > 0.95 and df[col].nunique() > 2:
                df[col] = coerced
    return df


def _group_rare_categories(df: pd.DataFrame, min_freq: float = 0.02) -> pd.DataFrame:
    """Collapse rarely-seen categorical levels into an 'Other' bucket so
    one-hot encoding doesn't explode into sparse, noisy columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if col == TARGET_COL:
            continue
        freqs = df[col].value_counts(normalize=True)
        rare = freqs[freqs < min_freq].index
        if len(rare) > 0:
            df[col] = df[col].where(~df[col].isin(rare), other="Other")
    return df


def impute_missing(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """
    Automatic missing-value detection + imputation.
    strategy: 'auto' picks median for numeric, mode for categorical.
              'knn' uses KNNImputer for numeric columns.
    """
    missing_report = df.isnull().sum()
    missing_cols = missing_report[missing_report > 0]
    if len(missing_cols) == 0:
        logger.info("No missing values detected.")
        return df

    logger.info(f"Missing values detected in {len(missing_cols)} column(s):\n{missing_cols}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if strategy == "knn":
        num_missing = [c for c in numeric_cols if c in missing_cols.index]
        if num_missing:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    else:
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def detect_outliers_iqr(df: pd.DataFrame, cols: Optional[list] = None) -> pd.DataFrame:
    """Return a boolean mask column-wise flag DataFrame using the IQR rule."""
    cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
    flags = pd.DataFrame(False, index=df.index, columns=cols)
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flags[col] = (df[col] < lower) | (df[col] > upper)
    return flags


def flag_outliers_isolation_forest(df: pd.DataFrame, contamination: float = 0.03) -> pd.Series:
    """Multivariate outlier flag using Isolation Forest (-1 = outlier)."""
    numeric_df = df.select_dtypes(include=[np.number])
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    preds = iso.fit_predict(numeric_df)
    return pd.Series(preds == -1, index=df.index, name="is_outlier_iforest")


def clean_data(df: pd.DataFrame, impute_strategy: str = "auto") -> pd.DataFrame:
    """Full cleaning pass: duplicates -> dtype fixes -> drop uninformative
    columns -> missing value imputation -> rare-category grouping."""
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    if before != len(df):
        logger.info(f"Removed {before - len(df)} duplicate rows")

    df = _fix_dtypes(df)

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped uninformative columns: {cols_to_drop}")

    df = impute_missing(df, strategy=impute_strategy)
    df = _group_rare_categories(df)

    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# 3. Feature engineering — HR domain features
# ------------------------------------------------------------------
DEFAULT_INCOME_BINS = [0, 3000, 6000, 10000, np.inf]
# Fixed reference range for MonthlyIncome normalization (RetentionScore).
# Learned once from the full training population — see get_processed_splits().
INCOME_REFERENCE_RANGE = (1009, 19999)


def compute_income_bins(df: pd.DataFrame, n_bins: int = 4) -> list:
    """Compute quartile bin edges for MonthlyIncome from a full training
    dataset. Must be called once at TRAINING time (on many rows) and the
    resulting edges persisted — never recomputed with qcut() on a single
    inference record, since qcut needs a distribution of values to form
    quantiles and breaks (ValueError: Bin edges must be unique) when all
    values in the slice are identical, e.g. a batch of size 1.
    """
    _, edges = pd.qcut(df["MonthlyIncome"], n_bins, retbins=True, duplicates="drop")
    edges = list(edges)
    edges[0], edges[-1] = -np.inf, np.inf  # make edges open-ended so any income at inference falls inside
    return edges


def engineer_features(df: pd.DataFrame, income_bins: Optional[list] = None,
                       income_range: Optional[tuple] = None) -> pd.DataFrame:
    """Create advanced HR analytics features from the cleaned base columns.

    Every feature here is a pure, row-wise function of that row's own
    columns *plus* fixed reference constants (e.g. income bin edges) — never
    a statistic recomputed from the current batch. That's what makes it safe
    to call on a single hand-typed employee record at inference time as well
    as on the full training set: a batch-of-one and a batch-of-thousands
    must produce identical feature values for the same employee.

    income_bins: fixed MonthlyIncome bin edges for CompensationScore. Pass
    the edges learned at training time (see compute_income_bins) for
    inference; defaults to DEFAULT_INCOME_BINS if not provided.
    """
    df = df.copy()
    eps = 1e-6

    # --- Compensation & career progression -----------------------------
    df["IncomePerYear"] = df["MonthlyIncome"] * 12
    df["IncomePerYearOfExperience"] = df["IncomePerYear"] / (df["TotalWorkingYears"] + 1)
    df["PromotionDelay"] = df["YearsSinceLastPromotion"]
    df["YearsWithoutPromotion"] = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]
    df["ExperienceRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + eps)
    df["IncomeGrowthPotential"] = df["JobLevel"] * df["PercentSalaryHike"]
    df["CareerStagnationScore"] = (
        df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    ).clip(0, 1)
    df["PromotionFrequency"] = (df["YearsAtCompany"] + 1) / (df["YearsSinceLastPromotion"] + 1)
    df["CareerProgressionRate"] = df["JobLevel"] / (df["TotalWorkingYears"] + eps)
    df["CompanyLoyalty"] = df["YearsAtCompany"] / (df["NumCompaniesWorked"] + 1)
    df["JobSwitchFrequency"] = df["NumCompaniesWorked"] / (df["TotalWorkingYears"] + 1)

    # --- Composite satisfaction / engagement scores ---------------------
    df["SatisfactionScore"] = df[
        ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"]
    ].mean(axis=1)
    df["EngagementScore"] = df[["JobInvolvement", "JobSatisfaction"]].mean(axis=1)
    df["ManagerRelationshipScore"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)
    df["WorkloadScore"] = np.where(df["OverTime"] == "Yes", 1, 0) * (5 - df["WorkLifeBalance"])
    df["WorkLifeStress"] = (5 - df["WorkLifeBalance"]) + np.where(df["OverTime"] == "Yes", 1, 0)
    bins = income_bins if income_bins is not None else DEFAULT_INCOME_BINS
    df["CompensationScore"] = pd.cut(
        df["MonthlyIncome"], bins=bins, labels=[1, 2, 3, 4], include_lowest=True
    ).astype(int)
    df["PerformanceRisk"] = np.where(
        (df["PerformanceRating"] <= 3) & (df["JobSatisfaction"] <= 2), 1, 0
    )

    # --- Travel / distance risk -----------------------------------------
    travel_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    df["TravelRisk"] = df["BusinessTravel"].map(travel_map).fillna(1) + (df["DistanceFromHome"] > 15).astype(int)

    # --- Binned categorical features (useful for BI slicers + trees) ----
    df["TenureCategory"] = pd.cut(
        df["YearsAtCompany"], bins=[-1, 1, 3, 7, 100], labels=["New (0-1y)", "Early (1-3y)", "Established (3-7y)", "Veteran (7y+)"]
    ).astype(str)
    df["AgeCategory"] = pd.cut(
        df["Age"], bins=[17, 25, 35, 45, 60], labels=["18-25", "26-35", "36-45", "46-60"]
    ).astype(str)
    df["IncomeCategory"] = pd.cut(
        df["MonthlyIncome"], bins=[0, 3000, 6000, 10000, 100000], labels=["<3K", "3K-6K", "6K-10K", "10K+"]
    ).astype(str)

    # --- Composite retention / value scores (0-100, higher = better) ----
    # NOTE: income normalization uses a FIXED reference range (learned from
    # the training population, see INCOME_REFERENCE_RANGE below) rather than
    # this batch's own min/max. Using batch min/max would make a single
    # employee's score meaningless (min == max == their own salary, so
    # norm_income always evaluates to 0 regardless of actual pay) and would
    # make even multi-row batches inconsistent with the values computed
    # during training.
    ref_min, ref_max = income_range if income_range is not None else INCOME_REFERENCE_RANGE
    norm_sat = df["SatisfactionScore"] / 4 * 100
    norm_wlb = df["WorkLifeBalance"] / 4 * 100
    norm_income = ((df["MonthlyIncome"] - ref_min) / (ref_max - ref_min + eps) * 100).clip(0, 100)
    overtime_penalty = np.where(df["OverTime"] == "Yes", 15, 0)
    stagnation_penalty = df["CareerStagnationScore"] * 20

    df["RetentionScore"] = (
        0.35 * norm_sat + 0.25 * norm_wlb + 0.20 * norm_income - overtime_penalty - stagnation_penalty
    ).clip(0, 100)
    df["EmployeeValueScore"] = (
        0.4 * (df["PerformanceRating"] / 4 * 100)
        + 0.3 * (df["JobLevel"] / 5 * 100)
        + 0.3 * (df["TotalWorkingYears"].clip(upper=20) / 20 * 100)
    )

    return df


# ------------------------------------------------------------------
# 4. Encoding + scaling via sklearn ColumnTransformer / Pipeline
# ------------------------------------------------------------------
def build_preprocessing_pipeline(df: pd.DataFrame, model_family: str = "tree") -> ColumnTransformer:
    """
    Build a reusable ColumnTransformer that one-hot encodes categoricals
    and (optionally) scales numerics based on the target model family.
    model_family in {'tree', 'linear', 'distance', 'neural'} — see
    MODEL_SCALER_MAP for which scaler each maps to.
    """
    X = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    scaler_cls = MODEL_SCALER_MAP.get(model_family)
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    if scaler_cls is not None:
        num_pipeline.steps.append(("scaler", scaler_cls()))

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def encode_features(df: pd.DataFrame, fit: bool = True, encoders: dict = None):
    """
    Backwards-compatible simple encoder (label-encode binary categoricals,
    one-hot encode the rest) used by utils.py for single-record inference,
    where round-tripping a fitted ColumnTransformer is overkill.
    """
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    encoders = encoders or {}

    categorical_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    if TARGET_COL in categorical_cols:
        categorical_cols.remove(TARGET_COL)

    binary_cols = [c for c in categorical_cols if df[c].nunique() == 2]
    multi_cols = [c for c in categorical_cols if df[c].nunique() > 2]

    for col in binary_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    if TARGET_COL in df.columns:
        if fit:
            target_le = LabelEncoder()
            df[TARGET_COL] = target_le.fit_transform(df[TARGET_COL])
            encoders["_target"] = target_le
        else:
            target_le = encoders["_target"]
            df[TARGET_COL] = target_le.transform(df[TARGET_COL])

    return df, encoders


# ------------------------------------------------------------------
# 5. Feature selection
# ------------------------------------------------------------------
def select_features(X: pd.DataFrame, y: pd.Series, top_k: int = 40) -> list:
    """
    Rank features by a blend of Mutual Information + variance filtering,
    then keep the top_k. Cheap, model-agnostic, and fast enough to run
    on every training call. (RFE / permutation importance / SHAP-based
    selection are additionally reported per-model in train_models.py.)
    """
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X)
    kept_by_variance = X.columns[vt.get_support()].tolist()
    X_var = X[kept_by_variance]

    mi = mutual_info_classif(X_var, y, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi, index=X_var.columns).sort_values(ascending=False)

    selected = mi_series.head(top_k).index.tolist()

    mi_series.to_csv(os.path.join(OUTPUTS_DIR, "feature_selection_mutual_info.csv"))
    with open(os.path.join(MODELS_DIR, "selected_features.json"), "w") as f:
        json.dump(selected, f, indent=2)

    logger.info(f"Selected {len(selected)} / {X.shape[1]} features via mutual information.")
    return selected


# ------------------------------------------------------------------
# 6. End-to-end split used by train_models.py
# ------------------------------------------------------------------
def get_processed_splits(test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE,
                          save_artifacts: bool = True, use_feature_selection: bool = True):
    """
    Full pipeline: load -> clean -> engineer features -> encode -> split ->
    scale (StandardScaler, suitable for linear/distance models; tree models
    are scale-invariant so this has no effect on their performance).
    Returns X_train, X_test, y_train, y_test, feature_names, encoders, scaler
    """
    df = load_raw_data()
    df = clean_data(df)

    # Learn income reference stats ONCE from the full training population,
    # then reuse these exact fixed values at inference time (single record
    # or batch) so engineered features are consistent and never break on
    # a batch of size 1. See engineer_features() docstring for why this
    # matters.
    income_bins = compute_income_bins(df)
    income_range = (df["MonthlyIncome"].min(), df["MonthlyIncome"].max())

    df = engineer_features(df, income_bins=income_bins, income_range=income_range)

    df_encoded, encoders = encode_features(df, fit=True)

    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL]

    if use_feature_selection:
        selected = select_features(X, y, top_k=min(45, X.shape[1]))
        X = X[selected]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[feature_names] = scaler.fit_transform(X_train[feature_names])
    X_test_scaled[feature_names] = scaler.transform(X_test[feature_names])

    if save_artifacts:
        joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
        joblib.dump({"income_bins": income_bins, "income_range": income_range},
                    os.path.join(MODELS_DIR, "feature_reference_stats.pkl"))

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, encoders, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, encoders, scaler = get_processed_splits()
    print("Processed data shapes:")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("Attrition rate in train set: {:.2%}".format(y_train.mean()))
    print("Attrition rate in test set: {:.2%}".format(y_test.mean()))
    print("Number of selected features:", len(feature_names))
    print(feature_names)
