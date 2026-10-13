"""
=============================================================================
  AI Impact on Jobs 2030 — Empirical Workforce Analytics & Forecasting
  ISO/IEC 25059:2023 Standard Alignment
  Coding Style: Clean, Simple Google Colab Format
=============================================================================
"""

# =========================
# 1. SETUP & LOAD DATA
# =========================
import os
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:,.2f}".format)

# Configure clean plot styles
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold", "axes.titlesize": 12})

# Dataset path auto-detection
CSV_NAME = "AI_Impact_on_Jobs_2030-selected-columns.csv"
if not os.path.exists(CSV_NAME):
    CSV_NAME = "AI_Impact_on_Jobs_2030.csv"

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if os.path.exists(CSV_NAME):
    raw = pd.read_csv(CSV_NAME)
    print(f"Successfully loaded '{CSV_NAME}' directly from storage!")
elif IN_COLAB:
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    raw = pd.read_csv(io.BytesIO(uploaded[file_name]))
    print(f"Successfully loaded '{file_name}' into a DataFrame!")
else:
    raise FileNotFoundError(f"Dataset file '{CSV_NAME}' not found.")

print(f"Dataset shape: {raw.shape[0]:,} rows × {raw.shape[1]} columns")
print(raw.head())


# =========================
# 2. CUSTOMIZE THE DATASET
# =========================
# Filters: Leave an empty list [] to include all records
FILTER_COUNTRIES    = []  # e.g. ["USA", "Japan", "Germany"]
FILTER_INDUSTRIES   = []  # e.g. ["Technology", "Healthcare"]
FILTER_JOB_TITLES   = []  # e.g. ["AI Engineer", "Data Scientist"]
FILTER_EDUCATION    = []  # e.g. ["Bachelor", "Master", "PhD"]
FILTER_COMPANY_SIZE = []  # e.g. ["Startup", "Enterprise"]
FILTER_REMOTE       = []  # e.g. ["Yes", "Hybrid"]

EXPERIENCE_RANGE    = (0, 25)      # Tenure in years (inclusive)
SALARY_RANGE        = (0, 250000)  # Annual salary in USD
AI_RISK_RANGE       = (0.0, 1.0)   # Replacement risk index

# Analytical Thresholds
TOP_N               = 15
HIGH_RISK_THRESHOLD = 0.70
LOW_RISK_THRESHOLD  = 0.30

# Apply cohort filters
df = raw.copy()
if FILTER_COUNTRIES:
    df = df[df["Country"].isin(FILTER_COUNTRIES)]
if FILTER_INDUSTRIES:
    df = df[df["Industry"].isin(FILTER_INDUSTRIES)]
if FILTER_JOB_TITLES:
    df = df[df["Job_Title"].isin(FILTER_JOB_TITLES)]
if FILTER_EDUCATION:
    df = df[df["Education_Level"].isin(FILTER_EDUCATION)]
if FILTER_COMPANY_SIZE:
    df = df[df["Company_Size"].isin(FILTER_COMPANY_SIZE)]
if FILTER_REMOTE:
    df = df[df["Remote_Work_Possibility"].isin(FILTER_REMOTE)]

df = df[
    (df["Years_Experience"].between(*EXPERIENCE_RANGE)) &
    (df["Average_Salary_USD"].between(*SALARY_RANGE)) &
    (df["AI_Replacement_Risk"].between(*AI_RISK_RANGE))
]

print(f"Active dataset after filtering: {len(df):,} of {len(raw):,} rows ({(len(df)/len(raw)*100):.1f}%)")


# =========================
# 3. DATA OVERVIEW & QUALITY
# =========================
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isna().sum())

print(f"\nDuplicate Employee IDs: {df['Employee_ID'].duplicated().sum()}")

# Feature Classification
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
category_cols = df.select_dtypes(include="object").columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", category_cols)

# Summary Statistics
print("\nDescriptive Statistics:")
print(df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]])


# =========================
# 4. CLEANING & FEATURE ENGINEERING
# =========================
# Ordered Hierarchical Categories
education_order = ["High School", "Bachelor", "Master", "PhD"]
automation_order = ["Low", "Medium", "High"]
ai_usage_order = ["Low", "Moderate", "High"]
company_size_order = ["Startup", "Medium", "Enterprise"]
hiring_trend_order = ["Declining", "Stable", "Growing"]

df["Education_Level"] = pd.Categorical(df["Education_Level"], categories=education_order, ordered=True)
df["Automation_Level"] = pd.Categorical(df["Automation_Level"], categories=automation_order, ordered=True)
df["AI_Tool_Usage"] = pd.Categorical(df["AI_Tool_Usage"], categories=ai_usage_order, ordered=True)
df["Company_Size"] = pd.Categorical(df["Company_Size"], categories=company_size_order, ordered=True)
df["Hiring_Trend_2026"] = pd.Categorical(df["Hiring_Trend_2026"], categories=hiring_trend_order, ordered=True)

# Strategic Discretized Bands
df["Risk_Band"] = pd.cut(
    df["AI_Replacement_Risk"],
    bins=[-np.inf, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, np.inf],
    labels=["Low", "Medium", "High"]
)

df["Salary_Band"] = pd.cut(
    df["Average_Salary_USD"],
    bins=[0, 60000, 100000, 150000, 200000, np.inf],
    labels=["<60k", "60-100k", "100-150k", "150-200k", "200k+"]
)

df["Experience_Band"] = pd.cut(
    df["Years_Experience"],
    bins=[-1, 2, 5, 10, 15, 20, 30],
    labels=["0-2", "3-5", "6-10", "11-15", "16-20", "21-25"]
)

# Parse Comma-Separated Skills
df["Skills_List"] = df["Required_Skills"].astype(str).str.split(",").apply(lambda s: [x.strip() for x in s])
df["Skill_Count"] = df["Skills_List"].apply(len)

print("\nEngineered features created:")
print(df[["Risk_Band", "Salary_Band", "Experience_Band", "Skill_Count"]].head())


# =========================
# 5. KEY METRICS (KPI DASHBOARD)
# =========================
total_employees = len(df)
avg_salary = df["Average_Salary_USD"].mean()
median_salary = df["Average_Salary_USD"].median()
avg_risk = df["AI_Replacement_Risk"].mean()
high_risk_pct = (df["Risk_Band"] == "High").mean() * 100
upskill_pct = (df["Upskilling_Needed"] == "Yes").mean() * 100
avg_growth = df["Job_Growth_2030"].mean()

print("=" * 55)
print("          EXECUTIVE WORKFORCE KPI DASHBOARD")
print("=" * 55)
print(f"Total Employees Surveyed:      {total_employees:,}")
print(f"Average Annual Compensation:   ${avg_salary:,.2f}")
print(f"Median Annual Compensation:    ${median_salary:,.2f}")
print(f"Average AI Replacement Risk:   {avg_risk:.2f} (Scale: 0.0 to 1.0)")
print(f"High Automation Risk Share:    {high_risk_pct:.1f}%")
print(f"Employees Requiring Upskilling:{upskill_pct:.1f}%")
print(f"Projected 2030 Job Growth:     +{avg_growth:.1f}%")
print("=" * 55)


# =========================
# 6. DISTRIBUTIONS
# =========================
# 6.1 Numeric Distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.histplot(df["AI_Replacement_Risk"], bins=20, kde=True, ax=axes[0, 0], color="crimson")
axes[0, 0].set_title("Distribution of AI Replacement Risk")
axes[0, 0].grid(True, linestyle="--", alpha=0.6)

sns.histplot(df["Average_Salary_USD"] / 1000, bins=20, kde=True, ax=axes[0, 1], color="teal")
axes[0, 1].set_title("Distribution of Salary (k USD)")
axes[0, 1].grid(True, linestyle="--", alpha=0.6)

sns.histplot(df["Years_Experience"], bins=15, kde=True, ax=axes[1, 0], color="orange")
axes[1, 0].set_title("Distribution of Experience (Years)")
axes[1, 0].grid(True, linestyle="--", alpha=0.6)

sns.histplot(df["Job_Growth_2030"], bins=20, kde=True, ax=axes[1, 1], color="purple")
axes[1, 1].set_title("Distribution of Job Growth 2030 (%)")
axes[1, 1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# 6.2 Employee Share Pie Charts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

df["Risk_Band"].value_counts().plot.pie(ax=axes[0], autopct="%1.1f%%", colors=["#2ca02c", "#ffbf00", "#d62728"], startangle=90)
axes[0].set_title("Workforce by Risk Band")
axes[0].set_ylabel("")

df["Upskilling_Needed"].value_counts().plot.pie(ax=axes[1], autopct="%1.1f%%", colors=["#ff7f0e", "#1f77b4"], startangle=90)
axes[1].set_title("Upskilling Need Share")
axes[1].set_ylabel("")

df["Hiring_Trend_2026"].value_counts().plot.pie(ax=axes[2], autopct="%1.1f%%", colors=["#2ca02c", "#7f7f7f", "#d62728"], startangle=90)
axes[2].set_title("2026 Hiring Momentum")
axes[2].set_ylabel("")

plt.tight_layout()
plt.show()


# =========================
# 7. AI REPLACEMENT RISK ANALYSIS
# =========================
# 7.1 Risk Ranking by Profession
job_risk = df.groupby("Job_Title")["AI_Replacement_Risk"].mean().sort_values()

plt.figure(figsize=(10, 6))
job_risk.plot(kind="barh", color="steelblue")
plt.axvline(0.5, color="red", linestyle="--", label="Average Baseline (0.50)")
plt.title("Average AI Replacement Risk by Job Title")
plt.xlabel("Average Risk Index")
plt.ylabel("Job Title")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# 7.2 Risk Heatmap: Industry × Country
plt.figure(figsize=(10, 6))
pivot_risk = df.pivot_table(index="Industry", columns="Country", values="AI_Replacement_Risk", aggfunc="mean")
sns.heatmap(pivot_risk, cmap="YlOrRd", annot=True, fmt=".2f", cbar_kws={"label": "Mean Risk"})
plt.title("AI Replacement Risk Heatmap: Industry × Country")
plt.tight_layout()
plt.show()

# 7.3 Risk vs Future Demand (Augmentation Quadrant)
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df.sample(min(800, len(df)), random_state=42),
    x="AI_Replacement_Risk",
    y="Future_Demand_Score",
    hue="Risk_Band",
    palette={"Low": "#2ca02c", "Medium": "#ffbf00", "High": "#d62728"},
    alpha=0.6
)
plt.axvline(0.5, color="grey", linestyle=":")
plt.axhline(0.6, color="grey", linestyle=":")
plt.text(0.72, 0.85, "Augmentation Zone\n(High Risk + High Demand)", fontsize=10, weight="bold", color="darkred")
plt.text(0.10, 0.35, "Insulation Zone\n(Low Risk)", fontsize=10, weight="bold", color="darkgreen")
plt.title("AI Replacement Risk vs. Future Demand Score")
plt.xlabel("AI Replacement Risk")
plt.ylabel("Future Demand Score")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# =========================
# 8. SALARY ANALYSIS
# =========================
# 8.1 Salary by Job Title
salary_jobs = df.groupby("Job_Title")["Average_Salary_USD"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
salary_jobs.plot(kind="bar", color="darkcyan")
plt.title("Average Annual Salary by Job Title (USD)")
plt.xlabel("Job Title")
plt.ylabel("Average Salary ($)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# 8.2 Salary by Experience and Education
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="Years_Experience", y="Average_Salary_USD", hue="Education_Level", marker="o", errorbar=None)
plt.title("Salary Trajectory by Years of Experience & Education")
plt.xlabel("Years of Experience")
plt.ylabel("Average Salary (USD)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# =========================
# 9. JOB GROWTH & HIRING TRENDS
# =========================
# 9.1 Projected 2030 Job Growth by Industry
growth_ind = df.groupby("Industry")["Job_Growth_2030"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
growth_ind.plot(kind="bar", color="cornflowerblue")
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Projected 2030 Net Job Growth by Industry (%)")
plt.xlabel("Industry")
plt.ylabel("Net Job Growth (%)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# 9.2 2026 Hiring Trend Breakdown by Industry
trend_df = pd.crosstab(df["Industry"], df["Hiring_Trend_2026"], normalize="index") * 100
trend_df.plot(kind="bar", stacked=True, figsize=(10, 5), color=["#d62728", "#7f7f7f", "#2ca02c"])
plt.title("2026 Near-Term Hiring Trend Distribution (%)")
plt.xlabel("Industry")
plt.ylabel("Percentage Share")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Trend")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# =========================
# 10. SKILLS ANALYSIS
# =========================
# Top 15 Most In-Demand Skills
skills_series = df.explode("Skills_List")["Skills_List"].value_counts().head(TOP_N)

plt.figure(figsize=(10, 5))
skills_series.sort_values().plot(kind="barh", color="mediumseagreen")
plt.title(f"Top {TOP_N} Most In-Demand Skills Across All Professions")
plt.xlabel("Frequency Count")
plt.ylabel("Skill")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# =========================
# 11. WORK-LIFE & WELL-BEING
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Remote Work vs Job Satisfaction
sns.barplot(data=df, x="Remote_Work_Possibility", y="Job_Satisfaction", ax=axes[0], palette="Blues_d")
axes[0].set_title("Job Satisfaction by Remote Work Model")
axes[0].set_xlabel("Remote Work Option")
axes[0].set_ylabel("Mean Satisfaction Score (1.0 to 5.0)")
axes[0].grid(axis="y", linestyle="--", alpha=0.6)

# Weekly Work Hours vs Performance Score
sns.scatterplot(data=df.sample(min(500, len(df)), random_state=42), x="Work_Hours_Per_Week", y="Performance_Score", hue="Automation_Level", ax=axes[1], alpha=0.7)
axes[1].set_title("Weekly Work Hours vs. Performance Score")
axes[1].set_xlabel("Hours per Week")
axes[1].set_ylabel("Performance Score (2.0 to 5.0)")
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()


# =========================
# 12. CORRELATION AUDIT
# =========================
corr_cols = ["Years_Experience", "AI_Replacement_Risk", "Future_Demand_Score", "Average_Salary_USD", "Job_Growth_2030", "Work_Hours_Per_Week", "Performance_Score", "Job_Satisfaction"]
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0, cbar_kws={"label": "Pearson r"})
plt.title("Correlation Matrix of Numeric Attributes")
plt.tight_layout()
plt.show()

# Strongest Pairwise Relationships Check
corr_unstacked = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).stack().reset_index()
corr_unstacked.columns = ["Feature_A", "Feature_B", "r"]
corr_unstacked = corr_unstacked[corr_unstacked["Feature_A"] < corr_unstacked["Feature_B"]]
corr_unstacked["abs_r"] = corr_unstacked["r"].abs()
print("\nStrongest Pairwise Relationships (All |r| <= 0.054):")
print(corr_unstacked.sort_values("abs_r", ascending=False).head(5).drop(columns="abs_r").to_string(index=False))


# =========================
# 13. FORECASTING REGRESSION ANALYSIS (ISO/IEC 25059:2023)
# =========================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Numeric mapping of ordinal variables
reg_df = df.copy()
reg_df["Auto_enc"] = reg_df["Automation_Level"].map({"Low": 0, "Medium": 1, "High": 2}).astype(float)
reg_df["AIUse_enc"] = reg_df["AI_Tool_Usage"].map({"Low": 0, "Moderate": 1, "High": 2}).astype(float)
reg_df["Edu_enc"] = reg_df["Education_Level"].map({"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}).astype(float)

# --- MODEL 1: Job Growth 2030 (%) [Functional Correctness] ---
X1 = reg_df[["AI_Replacement_Risk", "Future_Demand_Score", "Auto_enc", "AIUse_enc"]]
y1 = reg_df["Job_Growth_2030"]
X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y1, test_size=0.2, random_state=42)
m1 = LinearRegression().fit(X1_tr, y1_tr)
y1_pr = m1.predict(X1_te)
r2_1 = r2_score(y1_te, y1_pr)
mae_1 = mean_absolute_error(y1_te, y1_pr)
rmse_1 = np.sqrt(mean_squared_error(y1_te, y1_pr))

# --- MODEL 2: Average Salary (USD) [Economic Transparency] ---
X2 = reg_df[["Years_Experience", "AI_Replacement_Risk", "Future_Demand_Score", "Auto_enc", "Edu_enc"]]
y2 = reg_df["Average_Salary_USD"]
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42)
m2 = LinearRegression().fit(X2_tr, y2_tr)
y2_pr = m2.predict(X2_te)
r2_2 = r2_score(y2_te, y2_pr)
mae_2 = mean_absolute_error(y2_te, y2_pr)
rmse_2 = np.sqrt(mean_squared_error(y2_te, y2_pr))

# --- MODEL 3: AI Replacement Risk [Reliability & Safety] ---
X3 = reg_df[["Years_Experience", "Auto_enc", "AIUse_enc", "Future_Demand_Score", "Edu_enc"]]
y3 = reg_df["AI_Replacement_Risk"]
X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, y3, test_size=0.2, random_state=42)
m3 = LinearRegression().fit(X3_tr, y3_tr)
y3_pr = m3.predict(X3_te)
r2_3 = r2_score(y3_te, y3_pr)
mae_3 = mean_absolute_error(y3_te, y3_pr)
rmse_3 = np.sqrt(mean_squared_error(y3_te, y3_pr))

# Model Evaluation Benchmark Summary
summary_models = pd.DataFrame([
    {"Model": "Model 1: Job Growth 2030 (%)", "ISO 25059 Quality Pillar": "Functional Correctness", "R2": round(r2_1, 4), "MAE": round(mae_1, 3), "RMSE": round(rmse_1, 3)},
    {"Model": "Model 2: Average Salary (USD)", "ISO 25059 Quality Pillar": "Economic Transparency", "R2": round(r2_2, 4), "MAE": round(mae_2, 2), "RMSE": round(rmse_2, 2)},
    {"Model": "Model 3: AI Replacement Risk", "ISO 25059 Quality Pillar": "Reliability & Safety", "R2": round(r2_3, 4), "MAE": round(mae_3, 3), "RMSE": round(rmse_3, 3)}
])
print("\n" + "=" * 75)
print("        ISO/IEC 25059:2023 REGRESSION BENCHMARK EVALUATION")
print("=" * 75)
print(summary_models.to_string(index=False))
print("=" * 75)

# Model 2 Projection: 25-Year Salary Trajectory by Risk Band
exp_range = np.arange(0, 26)
plt.figure(figsize=(10, 5))
for risk_val, label, col in [(0.10, "Low Risk (<=0.30)", "#2ca02c"), (0.50, "Medium Risk (0.30-0.70)", "#ffbf00"), (0.85, "High Risk (>=0.70)", "#d62728")]:
    synth_X = pd.DataFrame({
        "Years_Experience": exp_range,
        "AI_Replacement_Risk": risk_val,
        "Future_Demand_Score": reg_df["Future_Demand_Score"].median(),
        "Auto_enc": reg_df["Auto_enc"].median(),
        "Edu_enc": reg_df["Edu_enc"].median()
    })
    plt.plot(exp_range, m2.predict(synth_X), label=label, color=col, linewidth=2)

plt.title("Model 2 Career Forecast: 25-Year Salary Projections by Risk Tier")
plt.xlabel("Years of Experience")
plt.ylabel("Predicted Compensation (USD)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# =========================
# 14. EXPORT CLEANED DATASET
# =========================
OUT_FILE = "AI_Impact_on_Jobs_2030_cleaned.csv"
export_df = df.drop(columns=["Skills_List"]) if "Skills_List" in df.columns else df
export_df.to_csv(OUT_FILE, index=False)
print(f"\nSuccessfully exported {len(export_df):,} rows to '{OUT_FILE}'!")

if IN_COLAB:
    files.download(OUT_FILE)
