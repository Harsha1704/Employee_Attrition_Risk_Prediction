"""
app.py
------
Enterprise HR Analytics Streamlit application for attrition risk
management: executive dashboard, employee search, single/batch
prediction, department analytics, risk heatmaps, SHAP explanations,
top-risk employees, employee comparison, and CSV export.

Run: streamlit run app.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from preprocessing import load_raw_data, clean_data
from utils import load_artifacts, predict_attrition_risk, risk_bucket

st.set_page_config(page_title="HR Attrition Analytics Platform", layout="wide", page_icon="🧑‍💼")

RISK_COLORS = {"Low Risk": "#2E7D32", "Medium Risk": "#F9A825", "High Risk": "#C62828"}
RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"]

# ------------------------------------------------------------------
# Theme
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    dark_mode = st.toggle("Dark mode", value=False)

if dark_mode:
    plt.style.use("dark_background")
    sns.set_style("darkgrid")
    st.markdown("""<style>
        .stApp { background-color: #0E1117; color: #FAFAFA; }
        </style>""", unsafe_allow_html=True)
else:
    plt.style.use("default")
    sns.set_style("whitegrid")

st.title("🧑‍💼 HR Attrition Analytics Platform")
st.caption("Enterprise workforce risk intelligence · Explainable ML · Power BI-ready exports")

# ------------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------------
@st.cache_resource
def get_artifacts():
    return load_artifacts()

try:
    model, encoders, scaler, feature_names, model_info = get_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

if not model_loaded:
    st.error("No trained model found. Please run `python train_models.py` first to train and save a model.")
    st.stop()


@st.cache_data
def get_scored_workforce():
    df_raw = clean_data(load_raw_data())
    return predict_attrition_risk(df_raw)


scored = get_scored_workforce()

# ------------------------------------------------------------------
# Sidebar: model info + global filters
# ------------------------------------------------------------------
st.sidebar.header("📌 Model Info")
st.sidebar.write(f"**Model:** {model_info.get('model_name', 'Unknown')}")
metrics = model_info.get("metrics", {})
if metrics:
    st.sidebar.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
    st.sidebar.metric("Recall (catches leavers)", f"{metrics.get('recall', 0):.3f}")
    st.sidebar.metric("Precision", f"{metrics.get('precision', 0):.3f}")
st.sidebar.caption(f"Trained: {model_info.get('trained_at', 'n/a')} · v{model_info.get('version', '1.0')}")

st.sidebar.header("🔎 Filters")
dept_filter = st.sidebar.multiselect("Department", sorted(scored["Department"].unique()),
                                      default=sorted(scored["Department"].unique()))
risk_filter = st.sidebar.multiselect("Risk Segment", RISK_ORDER, default=RISK_ORDER)

filtered = scored[scored["Department"].isin(dept_filter) & scored["Risk_Segment"].isin(risk_filter)]

tabs = st.tabs([
    "📊 Executive Dashboard", "🌡️ Risk Heatmap", "🔍 Employee Search",
    "🧍 Single Prediction", "📁 Batch Prediction", "🏢 Department Analytics",
    "🧠 Explainability (SHAP)", "⚖️ Employee Comparison",
])

# ------------------------------------------------------------------
# TAB: Executive Dashboard
# ------------------------------------------------------------------
with tabs[0]:
    st.subheader("Executive Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Employees", len(filtered))
    c2.metric("🔴 High Risk", int((filtered["Risk_Segment"] == "High Risk").sum()))
    c3.metric("🟠 Medium Risk", int((filtered["Risk_Segment"] == "Medium Risk").sum()))
    c4.metric("🟢 Low Risk", int((filtered["Risk_Segment"] == "Low Risk").sum()))
    c5.metric("Predicted Attrition Rate", f"{filtered['Attrition_Probability'].mean():.1%}")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Risk Segment Distribution**")
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = filtered["Risk_Segment"].value_counts().reindex(RISK_ORDER).fillna(0)
        ax.bar(counts.index, counts.values, color=[RISK_COLORS[r] for r in RISK_ORDER])
        ax.set_ylabel("Employees")
        st.pyplot(fig)

    with colB:
        st.markdown("**Avg. Predicted Attrition Risk by Department**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        dept_risk = filtered.groupby("Department")["Attrition_Probability"].mean().sort_values(ascending=False) * 100
        dept_risk.plot(kind="bar", ax=ax2, color="#5E35B1")
        ax2.set_ylabel("Avg. Predicted Attrition Risk (%)")
        plt.xticks(rotation=15)
        st.pyplot(fig2)

    st.markdown("**Top 15 Highest-Risk Employees**")
    display_cols = ["EmployeeNumber", "Age", "Department", "JobRole", "MonthlyIncome", "YearsAtCompany",
                     "OverTime", "Attrition_Probability", "Risk_Segment", "HR_Recommendation"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    top_risk = filtered.sort_values("Attrition_Probability", ascending=False)[display_cols].head(15)
    st.dataframe(top_risk, use_container_width=True)

    st.download_button(
        "⬇️ Download Full Scored Dataset (CSV for Power BI)",
        scored.to_csv(index=False).encode("utf-8"),
        file_name="powerbi_attrition_dataset.csv", mime="text/csv"
    )

# ------------------------------------------------------------------
# TAB: Risk Heatmap
# ------------------------------------------------------------------
with tabs[1]:
    st.subheader("Risk Heatmap: Department × Job Role")
    pivot = filtered.pivot_table(index="JobRole", columns="Department",
                                  values="Attrition_Probability", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn_r", ax=ax, cbar_kws={"label": "Avg. Attrition Risk"})
    st.pyplot(fig)

    st.markdown("**Risk Trend by Tenure Band**")
    tenure_band = pd.cut(filtered["YearsAtCompany"], bins=[-1, 1, 3, 5, 10, 100],
                          labels=["<1yr", "1-3yrs", "3-5yrs", "5-10yrs", "10+yrs"])
    trend = filtered.groupby(tenure_band, observed=True)["Attrition_Probability"].mean() * 100
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    trend.plot(marker="o", ax=ax2, color="#C62828")
    ax2.set_ylabel("Avg. Predicted Attrition Risk (%)")
    st.pyplot(fig2)

# ------------------------------------------------------------------
# TAB: Employee Search
# ------------------------------------------------------------------
with tabs[2]:
    st.subheader("Employee Search")
    search_col1, search_col2 = st.columns([1, 3])
    with search_col1:
        emp_id = st.number_input("EmployeeNumber (search)", min_value=0, value=0, step=1)
    if emp_id and "EmployeeNumber" in scored.columns and emp_id in scored["EmployeeNumber"].values:
        emp = scored[scored["EmployeeNumber"] == emp_id].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Segment", emp["Risk_Segment"])
        c2.metric("Attrition Probability", f"{emp['Attrition_Probability']:.1%}")
        c3.metric("Priority Score", f"{emp['Priority_Score']:.1f}")
        st.info(f"**Recommendation:** {emp['HR_Recommendation']}")
        st.dataframe(emp.to_frame().T, use_container_width=True)
    else:
        st.write("Enter a valid EmployeeNumber to view details, or browse/filter the full table below.")
        st.dataframe(filtered, use_container_width=True, height=400)

# ------------------------------------------------------------------
# TAB: Single Prediction (what-if form)
# ------------------------------------------------------------------
with tabs[3]:
    st.subheader("Score a Single Employee")
    raw_df = load_raw_data()

    with st.form("single_employee_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 65, 30)
            department = st.selectbox("Department", sorted(raw_df["Department"].unique()))
            job_role = st.selectbox("Job Role", sorted(raw_df["JobRole"].unique()))
            education_field = st.selectbox("Education Field", sorted(raw_df["EducationField"].unique()))
            business_travel = st.selectbox("Business Travel", sorted(raw_df["BusinessTravel"].unique()))
            gender = st.selectbox("Gender", sorted(raw_df["Gender"].unique()))
            marital_status = st.selectbox("Marital Status", sorted(raw_df["MaritalStatus"].unique()))

        with c2:
            monthly_income = st.number_input("Monthly Income", 1000, 25000, 5000, step=100)
            job_level = st.slider("Job Level", 1, 5, 2)
            job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
            environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
            work_life_balance = st.slider("Work-Life Balance (1=Bad, 4=Best)", 1, 4, 3)
            job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
            relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)

        with c3:
            total_working_years = st.number_input("Total Working Years", 0, 40, 8)
            years_at_company = st.number_input("Years at Company", 0, 40, 5)
            years_in_role = st.number_input("Years in Current Role", 0, 20, 3)
            years_since_promotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
            years_with_manager = st.number_input("Years with Current Manager", 0, 20, 3)
            overtime = st.selectbox("OverTime", ["Yes", "No"])
            distance_from_home = st.number_input("Distance From Home (km)", 1, 30, 5)
            num_companies_worked = st.number_input("Num Companies Worked", 0, 10, 2)
            stock_option_level = st.slider("Stock Option Level (0-3)", 0, 3, 0)

        submitted = st.form_submit_button("Predict Attrition Risk")

    if submitted:
        record = {
            "Age": age, "BusinessTravel": business_travel, "DailyRate": 800,
            "Department": department, "DistanceFromHome": distance_from_home,
            "Education": 3, "EducationField": education_field,
            "EnvironmentSatisfaction": environment_satisfaction, "Gender": gender,
            "HourlyRate": 65, "JobInvolvement": job_involvement, "JobLevel": job_level,
            "JobRole": job_role, "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status, "MonthlyIncome": monthly_income,
            "MonthlyRate": 15000, "NumCompaniesWorked": num_companies_worked,
            "OverTime": overtime, "PercentSalaryHike": 13, "PerformanceRating": 3,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option_level, "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": 2, "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company, "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promotion, "YearsWithCurrManager": years_with_manager,
        }
        result = predict_attrition_risk(pd.DataFrame([record]))
        prob, segment, rec = result["Attrition_Probability"].iloc[0], result["Risk_Segment"].iloc[0], result["HR_Recommendation"].iloc[0]

        emoji = {"High Risk": "🔴", "Medium Risk": "🟠", "Low Risk": "🟢"}[segment]
        st.markdown(f"### {emoji} Predicted Risk Segment: **{segment}**")
        st.progress(min(float(prob), 1.0))
        st.write(f"**Attrition Probability:** {prob:.1%}")
        (st.warning if segment == "High Risk" else st.info if segment == "Medium Risk" else st.success)(rec)

# ------------------------------------------------------------------
# TAB: Batch Prediction
# ------------------------------------------------------------------
with tabs[4]:
    st.subheader("Batch Score Employees from CSV")
    st.write("Upload a CSV with the same schema as the training dataset.")
    uploaded_file = st.file_uploader("Upload employee CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        try:
            scored_batch = predict_attrition_risk(batch_df)
            st.success(f"Scored {len(scored_batch)} employees.")
            st.dataframe(scored_batch, use_container_width=True)
            st.download_button("⬇️ Download Scored Results", scored_batch.to_csv(index=False).encode("utf-8"),
                                file_name="scored_employees.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not score this file. Make sure column names match the training schema. Error: {e}")

# ------------------------------------------------------------------
# TAB: Department Analytics
# ------------------------------------------------------------------
with tabs[5]:
    st.subheader("Department Deep-Dive")
    selected_dept = st.selectbox("Select Department", sorted(scored["Department"].unique()))
    dept_df = scored[scored["Department"] == selected_dept]

    c1, c2, c3 = st.columns(3)
    c1.metric("Headcount", len(dept_df))
    c2.metric("Avg. Predicted Risk", f"{dept_df['Attrition_Probability'].mean():.1%}")
    c3.metric("High Risk Count", int((dept_df["Risk_Segment"] == "High Risk").sum()))

    colA, colB = st.columns(2)
    with colA:
        fig, ax = plt.subplots(figsize=(5, 4))
        role_risk = dept_df.groupby("JobRole")["Attrition_Probability"].mean().sort_values(ascending=False)
        role_risk.plot(kind="barh", ax=ax, color="#0277BD")
        ax.set_xlabel("Avg. Predicted Risk")
        st.pyplot(fig)
    with colB:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=dept_df, x="Risk_Segment", y="MonthlyIncome", order=RISK_ORDER,
                    palette=RISK_COLORS, hue="Risk_Segment", legend=False, ax=ax2)
        st.pyplot(fig2)

    st.dataframe(dept_df.sort_values("Attrition_Probability", ascending=False), use_container_width=True)

# ------------------------------------------------------------------
# TAB: Explainability (SHAP)
# ------------------------------------------------------------------
with tabs[6]:
    st.subheader("Model Explainability")
    st.write("Global feature importance and drivers behind the model's predictions "
             "(generated during training — see `eval_plots/`).")
    import os
    shap_global = "eval_plots/shap_global_importance.png"
    shap_bee = "eval_plots/shap_beeswarm.png"
    colA, colB = st.columns(2)
    if os.path.exists(shap_global):
        colA.image(shap_global, caption="SHAP Global Feature Importance")
    if os.path.exists(shap_bee):
        colB.image(shap_bee, caption="SHAP Beeswarm (impact direction)")

    fi_path = "outputs/feature_importance.csv"
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path).head(15)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#6A1B9A")
        ax.set_title("Top 15 Model Feature Importances")
        st.pyplot(fig)

# ------------------------------------------------------------------
# TAB: Employee Comparison
# ------------------------------------------------------------------
with tabs[7]:
    st.subheader("Compare Two Employees")
    ids = scored["EmployeeNumber"].tolist() if "EmployeeNumber" in scored.columns else scored.index.tolist()
    c1, c2 = st.columns(2)
    emp_a = c1.selectbox("Employee A", ids, index=0)
    emp_b = c2.selectbox("Employee B", ids, index=min(1, len(ids) - 1))

    key_col = "EmployeeNumber" if "EmployeeNumber" in scored.columns else scored.index
    row_a = scored[scored["EmployeeNumber"] == emp_a].iloc[0] if "EmployeeNumber" in scored.columns else scored.loc[emp_a]
    row_b = scored[scored["EmployeeNumber"] == emp_b].iloc[0] if "EmployeeNumber" in scored.columns else scored.loc[emp_b]

    compare_cols = ["Department", "JobRole", "MonthlyIncome", "YearsAtCompany", "OverTime",
                     "JobSatisfaction", "WorkLifeBalance", "Attrition_Probability", "Risk_Segment", "HR_Recommendation"]
    compare_cols = [c for c in compare_cols if c in scored.columns]
    comparison = pd.DataFrame({"Employee A": row_a[compare_cols], "Employee B": row_b[compare_cols]})
    st.dataframe(comparison, use_container_width=True)
