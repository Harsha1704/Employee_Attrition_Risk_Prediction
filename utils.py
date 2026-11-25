"""
utils.py
--------
Shared helpers used by the Streamlit app and reporting scripts:
    - Loading trained model + preprocessing artifacts
    - Transforming raw employee records into model-ready features
    - Risk segmentation + business rule-based HR recommendations
    - Power BI-ready dataset export with KPI / recommendation columns
"""

import os
import json
from typing import List

import joblib
import numpy as np
import pandas as pd

from config import MODELS_DIR, OUTPUTS_DIR, RISK_THRESHOLDS, get_logger
from preprocessing import (
    DROP_COLS,
    TARGET_COL,
    clean_data,
    encode_features,
    engineer_features,
    load_raw_data,
)

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Artifact loading
# ------------------------------------------------------------------
def load_artifacts():
    """Load the trained model and all preprocessing artifacts needed for inference."""
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

    info_path = os.path.join(MODELS_DIR, "best_model_info.json")
    model_info = {}
    if os.path.exists(info_path):
        with open(info_path) as f:
            model_info = json.load(f)

    return model, encoders, scaler, feature_names, model_info


def load_reference_stats():
    """Load the fixed income bin edges / range learned at training time,
    so single-record and batch inference reproduce training-time feature
    values exactly instead of recomputing batch-dependent statistics
    (which breaks or silently misbehaves on a batch of size 1)."""
    path = os.path.join(MODELS_DIR, "feature_reference_stats.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return {"income_bins": None, "income_range": None}  # falls back to engineer_features() defaults


def risk_bucket(probability: float) -> str:
    """Convert an attrition probability into a human-readable risk segment."""
    if probability >= RISK_THRESHOLDS["Medium Risk"]:
        return "High Risk"
    elif probability >= RISK_THRESHOLDS["Low Risk"]:
        return "Medium Risk"
    return "Low Risk"


# ------------------------------------------------------------------
# Inference feature prep
# ------------------------------------------------------------------
def prepare_features_for_inference(df: pd.DataFrame, encoders: dict, scaler, feature_names: List[str]) -> pd.DataFrame:
    """Take a raw employee dataframe and transform it into the exact
    engineered + encoded + scaled feature matrix the model expects."""
    df = df.copy()

    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    ref_stats = load_reference_stats()
    df = engineer_features(df, income_bins=ref_stats["income_bins"], income_range=ref_stats["income_range"])

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    binary_cols = [c for c in categorical_cols if c in encoders]
    for col in binary_cols:
        le = encoders[col]
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    multi_cols = [c for c in categorical_cols if c not in binary_cols]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    df[feature_names] = scaler.transform(df[feature_names])

    return df


def predict_attrition_risk(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Given a dataframe of employee records (raw/original schema), return the
    original data plus predicted attrition probability, risk segment, and
    actionable HR recommendations."""
    model, encoders, scaler, feature_names, _ = load_artifacts()
    X = prepare_features_for_inference(df_raw, encoders, scaler, feature_names)

    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    result = df_raw.copy().reset_index(drop=True)
    result["Attrition_Probability"] = np.round(proba, 4)
    result["Risk_Score"] = np.round(proba * 100, 1)
    result["Predicted_Attrition"] = np.where(pred == 1, "Yes", "No")
    result["Risk_Segment"] = [risk_bucket(p) for p in proba]
    result["HR_Recommendation"] = generate_recommendations(result)
    result["Priority_Score"] = compute_priority_score(result)

    return result


# ------------------------------------------------------------------
# Business intelligence: rule-based HR recommendations
# ------------------------------------------------------------------
def generate_recommendations(df: pd.DataFrame) -> List[str]:
    """
    Generate a short, human-readable retention recommendation per employee
    by combining the model's risk score with the strongest known HR
    attrition drivers (overtime, satisfaction, promotion gap, pay, travel,
    manager tenure). This is intentionally rule-based (not a black box) so
    HR can trust and act on it directly.
    """
    recs = []
    for _, row in df.iterrows():
        reasons = []
        risk = row.get("Risk_Segment", "Low Risk")

        if row.get("OverTime") == "Yes":
            reasons.append("reduce overtime / rebalance workload")
        if row.get("JobSatisfaction", 4) <= 2:
            reasons.append("address low job satisfaction")
        if row.get("EnvironmentSatisfaction", 4) <= 2:
            reasons.append("improve work environment fit")
        if row.get("WorkLifeBalance", 4) <= 2:
            reasons.append("support work-life balance")
        if row.get("YearsSinceLastPromotion", 0) >= 4 and row.get("YearsAtCompany", 0) >= 4:
            reasons.append("promotion overdue — schedule career conversation")
        if row.get("BusinessTravel") == "Travel_Frequently":
            reasons.append("evaluate travel burden")
        if row.get("YearsWithCurrManager", 5) <= 1 and row.get("YearsAtCompany", 0) >= 3:
            reasons.append("recent manager change — check relationship fit")
        if row.get("RelationshipSatisfaction", 4) <= 2:
            reasons.append("review manager/team relationship")
        if row.get("StockOptionLevel", 1) == 0 and row.get("JobLevel", 3) <= 2:
            reasons.append("consider equity/compensation review")

        if risk == "Low Risk" and not reasons:
            recs.append("No action needed — low predicted risk.")
        elif not reasons:
            recs.append("Model flags elevated risk without a clear single driver — recommend a general check-in.")
        else:
            prefix = "URGENT: " if risk == "High Risk" else ""
            recs.append(prefix + "; ".join(reasons[:3]).capitalize() + ".")
    return recs


def compute_priority_score(df: pd.DataFrame) -> pd.Series:
    """Blend attrition risk with employee value (performance/tenure) so HR
    can prioritize outreach to high-risk, high-value employees first."""
    perf = df.get("PerformanceRating", pd.Series(3, index=df.index)).fillna(3) / 4
    tenure_value = df.get("YearsAtCompany", pd.Series(0, index=df.index)).fillna(0).clip(upper=15) / 15
    risk = df["Attrition_Probability"]
    priority = (0.6 * risk + 0.25 * perf + 0.15 * tenure_value) * 100
    return priority.round(1)


# ------------------------------------------------------------------
# Power BI export
# ------------------------------------------------------------------
def export_powerbi_dataset(output_path: str = None) -> str:
    """Score the full workforce and export a Power BI-ready CSV containing
    original HR attributes + predictions + risk segmentation + BI-friendly
    banded columns + department KPIs + recommendations."""
    if output_path is None:
        output_path = os.path.join(OUTPUTS_DIR, "powerbi_attrition_dataset.csv")

    df_raw = load_raw_data()
    df_raw = clean_data(df_raw)

    scored = predict_attrition_risk(df_raw)

    scored["Tenure_Band"] = pd.cut(
        scored["YearsAtCompany"], bins=[-1, 1, 3, 5, 10, 100],
        labels=["<1 yr", "1-3 yrs", "3-5 yrs", "5-10 yrs", "10+ yrs"]
    )
    scored["Income_Band"] = pd.cut(
        scored["MonthlyIncome"], bins=[0, 3000, 6000, 10000, 100000],
        labels=["<3K", "3K-6K", "6K-10K", "10K+"]
    )
    scored["Age_Band"] = pd.cut(
        scored["Age"], bins=[17, 25, 35, 45, 60],
        labels=["18-25", "26-35", "36-45", "46-60"]
    )

    # Department-level KPI columns (broadcast back onto every row — handy for Power BI matrix visuals)
    dept_kpi = scored.groupby("Department")["Attrition_Probability"].mean().rename("Department_Avg_Risk")
    scored = scored.merge(dept_kpi, on="Department", how="left")

    scored.to_csv(output_path, index=False)
    logger.info(f"Power BI dataset exported to: {output_path} ({len(scored)} rows, {len(scored.columns)} cols)")
    return output_path


if __name__ == "__main__":
    path = export_powerbi_dataset()
    print(f"Power BI dataset exported to: {path}")
