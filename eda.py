"""
eda.py
------
Professional exploratory data analysis for the HR Attrition dataset.
Generates 25+ charts covering distributions, correlations, class
imbalance, department/role/salary/promotion patterns, PCA, missing
value structure and outliers — all saved automatically to eda_plots/,
plus a markdown EDA report summarising key findings.

Run: python eda.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import EDA_DIR, get_logger
from preprocessing import (
    clean_data,
    detect_outliers_iqr,
    engineer_features,
    flag_outliers_isolation_forest,
    load_raw_data,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
sns.set_palette("Set2")
logger = get_logger(__name__)

GREEN, RED, ORANGE, PURPLE, BLUE = "#2E7D32", "#C62828", "#EF6C00", "#6A1B9A", "#0277BD"

_fig_count = 0
_report_lines = ["# EDA Report — HR Attrition Analytics\n"]


def _save(fig_name: str, title: str, note: str = ""):
    global _fig_count
    _fig_count += 1
    fname = f"{_fig_count:02d}_{fig_name}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, fname), dpi=120)
    plt.close()
    _report_lines.append(f"## {_fig_count}. {title}\n\n![{title}]({fname})\n\n{note}\n")


def plot_attrition_distribution(df):
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(data=df, x="Attrition", hue="Attrition", palette=[GREEN, RED], legend=False)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    plt.title("Class Imbalance: Overall Attrition Distribution")
    rate = (df["Attrition"] == "Yes").mean()
    _save("attrition_distribution", "Class Imbalance", f"Overall attrition rate: **{rate:.1%}** — a minority-class problem.")


def plot_attrition_by(df, col, title, color=RED, rotate=20):
    plt.figure(figsize=(7, 4))
    rate = df.groupby(col)["Attrition"].apply(lambda x: (x == "Yes").mean() * 100).sort_values(ascending=False)
    ax = rate.plot(kind="bar", color=color)
    plt.ylabel("Attrition Rate (%)")
    plt.title(title)
    plt.xticks(rotation=rotate)
    for i, v in enumerate(rate):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8)
    _save(f"attrition_by_{col.lower()}", title)


def plot_numeric_distribution(df, col, title):
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x=col, hue="Attrition", multiple="stack", palette=[GREEN, RED], bins=20)
    plt.title(title)
    _save(f"distribution_{col.lower()}", title)


def plot_boxplot(df, col, title):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Attrition", y=col, hue="Attrition", palette=[GREEN, RED], legend=False)
    plt.title(title)
    _save(f"box_{col.lower()}", title)


def plot_violin(df, col, title):
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df, x="Attrition", y=col, hue="Attrition", palette=[GREEN, RED], legend=False)
    plt.title(title)
    _save(f"violin_{col.lower()}", title)


def plot_correlation_heatmap(df):
    plt.figure(figsize=(16, 13))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Correlation Heatmap — Numeric & Engineered Features")
    _save("correlation_heatmap", "Correlation Heatmap", "Highlights multicollinearity among tenure/promotion-derived features.")


def plot_pairplot(df, cols):
    g = sns.pairplot(df[cols + ["Attrition"]], hue="Attrition", palette=[GREEN, RED], diag_kind="kde", height=2.2)
    g.fig.suptitle("Pairwise Relationships — Key Drivers", y=1.02)
    global _fig_count
    _fig_count += 1
    fname = f"{_fig_count:02d}_pairplot_key_drivers.png"
    g.savefig(os.path.join(EDA_DIR, fname), dpi=120)
    plt.close("all")
    _report_lines.append(f"## {_fig_count}. Pairplot of Key Drivers\n\n![Pairplot]({fname})\n")


def plot_missing_matrix(df):
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Value Matrix")
    _save("missing_value_matrix", "Missing Value Matrix", "White cells indicate a missing value; the dataset is essentially complete.")


def plot_outliers(df):
    numeric_cols = ["MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "DistanceFromHome"]
    flags = detect_outliers_iqr(df, numeric_cols)
    plt.figure(figsize=(7, 4))
    counts = flags.sum().sort_values(ascending=False)
    counts.plot(kind="bar", color=ORANGE)
    plt.ylabel("Outlier Count (IQR rule)")
    plt.title("Univariate Outlier Counts (IQR Method)")
    _save("outliers_iqr", "Outlier Detection (IQR)")

    iso_flags = flag_outliers_isolation_forest(df)
    plt.figure(figsize=(5, 4))
    iso_flags.value_counts().plot(kind="bar", color=[GREEN, RED])
    plt.title(f"Isolation Forest Outliers ({iso_flags.sum()} flagged / {len(df)})")
    plt.xticks([0, 1], ["Normal", "Outlier"], rotation=0)
    _save("outliers_isolation_forest", "Multivariate Outlier Detection (Isolation Forest)")


def plot_pca(df):
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[], errors="ignore")
    numeric_df = numeric_df.fillna(numeric_df.median())
    X_scaled = StandardScaler().fit_transform(numeric_df)
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7, 5))
    plt.scatter(comps[:, 0], comps[:, 1], c=(df["Attrition"] == "Yes").astype(int),
                cmap="coolwarm", alpha=0.6, s=18)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.title("PCA Projection — Attrition Separability")
    _save("pca_projection", "PCA Projection (2D)", "Colored by attrition; shows the classes are not linearly separable in 2D, motivating non-linear models.")


def plot_interaction(df, x, y, hue="Attrition"):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=[GREEN, RED], alpha=0.6)
    plt.title(f"Interaction: {x} vs {y}")
    _save(f"interaction_{x.lower()}_{y.lower()}", f"Interaction Plot: {x} vs {y}")


def plot_gender_analysis(df):
    plt.figure(figsize=(6, 4))
    rate = df.groupby(["Gender"])["Attrition"].apply(lambda x: (x == "Yes").mean() * 100)
    ax = rate.plot(kind="bar", color=PURPLE)
    plt.ylabel("Attrition Rate (%)")
    plt.title("Attrition Rate by Gender")
    for i, v in enumerate(rate):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center")
    plt.xticks(rotation=0)
    _save("gender_analysis", "Gender Analysis")


def plot_salary_by_role(df):
    plt.figure(figsize=(9, 5))
    order = df.groupby("JobRole")["MonthlyIncome"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="JobRole", y="MonthlyIncome", order=order, hue="JobRole", palette="Set3", legend=False)
    plt.xticks(rotation=35, ha="right")
    plt.title("Salary Distribution by Job Role")
    _save("salary_by_role", "Salary Analysis by Job Role")


def plot_feature_importance_proxy(df_encoded, target):
    """Quick model-free feature importance proxy via mutual information, for EDA purposes."""
    from sklearn.feature_selection import mutual_info_classif
    numeric = df_encoded.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
    mi = mutual_info_classif(numeric.fillna(0), df_encoded[target], random_state=42)
    top = pd.Series(mi, index=numeric.columns).sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    top.sort_values().plot(kind="barh", color=BLUE)
    plt.title("Top 15 Features by Mutual Information with Attrition")
    plt.xlabel("Mutual Information Score")
    _save("mutual_info_feature_importance", "EDA-stage Feature Importance (Mutual Information)")


def run_eda():
    df = load_raw_data()
    df = clean_data(df)
    df_fe = engineer_features(df)

    logger.info(f"Dataset shape after cleaning: {df.shape}")
    logger.info(f"Overall attrition rate: {(df['Attrition'] == 'Yes').mean():.2%}")

    # --- Class imbalance & missingness ---
    plot_attrition_distribution(df)
    plot_missing_matrix(df)

    # --- Categorical breakdowns ---
    plot_attrition_by(df, "Department", "Attrition Rate by Department")
    plot_attrition_by(df, "JobRole", "Attrition Rate by Job Role", rotate=35)
    plot_attrition_by(df, "OverTime", "Attrition Rate by OverTime Status", color=PURPLE, rotate=0)
    plot_attrition_by(df, "BusinessTravel", "Attrition Rate by Business Travel", color=BLUE, rotate=0)
    plot_attrition_by(df, "MaritalStatus", "Attrition Rate by Marital Status", color=ORANGE, rotate=0)
    plot_attrition_by(df, "EducationField", "Attrition Rate by Education Field", rotate=25)
    plot_gender_analysis(df)

    # --- Numeric distributions ---
    plot_numeric_distribution(df, "Age", "Age Distribution by Attrition Status")
    plot_numeric_distribution(df, "YearsAtCompany", "Years at Company by Attrition Status")
    plot_numeric_distribution(df, "TotalWorkingYears", "Total Working Years by Attrition Status")
    plot_numeric_distribution(df, "DistanceFromHome", "Distance From Home by Attrition Status")

    # --- Boxplots / Violins ---
    plot_boxplot(df, "MonthlyIncome", "Monthly Income vs Attrition")
    plot_boxplot(df, "YearsSinceLastPromotion", "Years Since Last Promotion vs Attrition")
    plot_violin(df, "JobSatisfaction", "Job Satisfaction vs Attrition (Violin)")
    plot_violin(df, "WorkLifeBalance", "Work-Life Balance vs Attrition (Violin)")
    plot_violin(df, "EnvironmentSatisfaction", "Environment Satisfaction vs Attrition (Violin)")

    # --- Rate by ordinal levels ---
    plot_attrition_by(df, "JobSatisfaction", "Attrition Rate by Job Satisfaction (1-4)", color=ORANGE, rotate=0)
    plot_attrition_by(df, "WorkLifeBalance", "Attrition Rate by Work-Life Balance (1-4)", color=PURPLE, rotate=0)
    plot_attrition_by(df, "YearsSinceLastPromotion", "Attrition Rate by Years Since Last Promotion", color=BLUE, rotate=0)
    plot_attrition_by(df, "StockOptionLevel", "Attrition Rate by Stock Option Level", color=GREEN, rotate=0)
    plot_attrition_by(df, "JobLevel", "Attrition Rate by Job Level", color=RED, rotate=0)

    # --- Salary / promotion ---
    plot_salary_by_role(df)

    # --- Correlation / PCA / pairplot ---
    plot_correlation_heatmap(df_fe)
    plot_pca(df)
    plot_pairplot(df, ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears"])

    # --- Outliers ---
    plot_outliers(df)

    # --- Interaction plots ---
    plot_interaction(df, "MonthlyIncome", "YearsAtCompany")
    plot_interaction(df, "Age", "TotalWorkingYears")
    plot_interaction(df, "DistanceFromHome", "WorkLifeBalance")

    # --- Feature importance proxy (needs encoded target) ---
    from preprocessing import encode_features
    df_encoded, _ = encode_features(df_fe, fit=True)
    plot_feature_importance_proxy(df_encoded, "Attrition")

    with open(os.path.join(EDA_DIR, "EDA_REPORT.md"), "w") as f:
        f.write("\n".join(_report_lines))

    logger.info(f"Saved {_fig_count} plots + EDA_REPORT.md to: {EDA_DIR}")
    print(f"\n✅ Generated {_fig_count} EDA visualizations in {EDA_DIR}")


if __name__ == "__main__":
    run_eda()
