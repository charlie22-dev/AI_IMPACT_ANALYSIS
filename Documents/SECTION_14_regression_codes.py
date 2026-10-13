"""
=============================================================================
  SECTION 14 — Forecasting Regression Analysis (ISO/IEC 25059:2023)
  AI Impact on Jobs 2030 Notebook
=============================================================================

HOW TO ADD THIS TO THE NOTEBOOK
--------------------------------
  1. Open   AI_Impact_on_Jobs_2030_Analysis.ipynb
  2. Scroll to the very bottom (after Section 13 – Export)
  3. Add ONE NEW MARKDOWN CELL, then paste CELL-A content
  4. Add cells in order: CELL-B, CELL-C, CELL-D, CELL-E, CELL-F, CELL-G, CELL-H, CELL-I

  Each block below is clearly labelled with its cell type.
  In Jupyter: click "+ Code" or "+ Markdown" in the toolbar to add each cell.

=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL-A  |  TYPE: Markdown  |  PLACEMENT: After Section 13 (last cell)
# ─────────────────────────────────────────────────────────────────────────────
"""
## 14. Forecasting Regression Analysis (ISO/IEC 25059:2023)

This section applies **multiple linear regression** models grounded in the
case-study context and the quality characteristics defined by
**ISO/IEC 25059:2023** (AI systems — Quality model).

### Models fitted

| # | Target (Y) | Predictors (X) | ISO 25059 Quality Characteristic |
|---|-----------|----------------|----------------------------------|
| 1 | `Job_Growth_2030` (%) | AI Risk, Demand Score, Automation, AI Tool Usage | **Functional Correctness** — predictive accuracy of AI-driven displacement |
| 2 | `Average_Salary_USD` | Experience, AI Risk, Demand Score, Automation, Education | **Economic Impact** — how AI attributes drive compensation |
| 3 | `AI_Replacement_Risk` | Experience, Automation, AI Tool Usage, Demand, Education | **Reliability & Robustness** — decoding the AI risk model |

Each model produces:
- 📊 Coefficient bar chart (feature importance with sign)
- 🎯 Actual vs Predicted scatter + identity line
- 📉 Residual distribution histogram
- 📈 **Forecast projection** plot — sweeping the key predictor across its full range
"""


# ─────────────────────────────────────────────────────────────────────────────
# CELL-B  |  TYPE: Code  |  PLACEMENT: Immediately after CELL-A
#           PURPOSE: Shared setup — imports & ordinal encoding
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def encode_ordinal(series, order):
    """Map an ordered category list to integers 0, 1, 2, ..."""
    return series.map({v: i for i, v in enumerate(order)})


# Work on a copy of the filtered dataframe
reg_df = df.copy()
reg_df['Auto_enc']  = encode_ordinal(reg_df['Automation_Level'],
                                      ['Low', 'Medium', 'High'])
reg_df['AIUse_enc'] = encode_ordinal(reg_df['AI_Tool_Usage'],
                                      ['Low', 'Moderate', 'High'])
reg_df['Edu_enc']   = encode_ordinal(reg_df['Education_Level'],
                                      ['High School', 'Bachelor', 'Master', 'PhD'])

print(f'Regression dataset: {reg_df.shape[0]:,} rows')
print('Encoded ordinal columns → Auto_enc (0-2), AIUse_enc (0-2), Edu_enc (0-3)')


# ─────────────────────────────────────────────────────────────────────────────
# CELL-C  |  TYPE: Code  |  PLACEMENT: After CELL-B
#           PURPOSE: Model 1 — Forecast Job_Growth_2030 (%)
#                    ISO 25059:2023 → Functional Correctness
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_M1 = ['AI_Replacement_Risk', 'Future_Demand_Score', 'Auto_enc', 'AIUse_enc']
TARGET_M1   = 'Job_Growth_2030'

X1 = reg_df[FEATURES_M1].dropna()
y1 = reg_df.loc[X1.index, TARGET_M1]

X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y1, test_size=0.2, random_state=42)
m1    = LinearRegression().fit(X1_tr, y1_tr)
y1_pr = m1.predict(X1_te)

r2_m1   = r2_score(y1_te, y1_pr)
mae_m1  = mean_absolute_error(y1_te, y1_pr)
rmse_m1 = np.sqrt(mean_squared_error(y1_te, y1_pr))
coef_m1 = pd.Series(m1.coef_, index=FEATURES_M1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    'Model 1 — Forecast: Job Growth 2030 (%)\n'
    '(ISO 25059:2023 — Functional Correctness)',
    fontsize=13, fontweight='bold')

# A) Feature Coefficients
ax = axes[0]
cols_m1 = ['#e74c3c' if v < 0 else '#2ecc71' for v in coef_m1]
ax.barh(coef_m1.index, coef_m1.values, color=cols_m1)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient')
ax.set_title('Feature Coefficients')

# B) Actual vs Predicted
ax = axes[1]
ax.scatter(y1_te, y1_pr, alpha=0.4, edgecolors='none', color='steelblue')
lm1 = [min(y1_te.min(), y1_pr.min()), max(y1_te.max(), y1_pr.max())]
ax.plot(lm1, lm1, 'r--', linewidth=1.5, label='Perfect fit')
ax.set_xlabel('Actual Job Growth 2030 (%)')
ax.set_ylabel('Predicted Job Growth 2030 (%)')
ax.set_title(f'Actual vs Predicted\nR²={r2_m1:.3f}  MAE={mae_m1:.2f}  RMSE={rmse_m1:.2f}')
ax.legend()

# C) Residual Distribution
ax = axes[2]
ax.hist(y1_te - y1_pr, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linewidth=1.2, linestyle='--')
ax.set_xlabel('Residual')
ax.set_ylabel('Frequency')
ax.set_title('Residual Distribution')

plt.tight_layout()
plt.show()

print(f'\nModel 1  |  Intercept: {m1.intercept_:.3f}')
print(f'         |  R²={r2_m1:.3f}   MAE={mae_m1:.3f}   RMSE={rmse_m1:.3f}')
print('\nCoefficients:')
print(coef_m1.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# CELL-D  |  TYPE: Code  |  PLACEMENT: After CELL-C
#           PURPOSE: Model 1 Projection chart
#                    Sweep AI_Replacement_Risk → projected Job Growth
# ─────────────────────────────────────────────────────────────────────────────

_med_fds1   = reg_df['Future_Demand_Score'].median()
_med_auto1  = reg_df['Auto_enc'].median()
_med_aiuse1 = reg_df['AIUse_enc'].median()
risk_sweep  = np.linspace(0.05, 0.95, 200)

X_p1 = pd.DataFrame({
    'AI_Replacement_Risk': risk_sweep,
    'Future_Demand_Score': _med_fds1,
    'Auto_enc':            _med_auto1,
    'AIUse_enc':           _med_aiuse1,
})

fig, ax = plt.subplots(figsize=(10, 5))
samp1 = reg_df.sample(min(500, len(reg_df)), random_state=42)
ax.scatter(samp1['AI_Replacement_Risk'], samp1[TARGET_M1],
           alpha=0.25, color='steelblue', edgecolors='none', label='Observed data')
ax.plot(risk_sweep, m1.predict(X_p1), color='red', linewidth=2.5,
        label='Regression forecast (other vars at median)')
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.set_xlabel('AI Replacement Risk  (0 = safe  →  1 = fully replaceable)')
ax.set_ylabel('Projected Job Growth 2030 (%)')
ax.set_title(
    'Forecast Projection: AI Replacement Risk → Job Growth 2030\n'
    '(ISO 25059:2023 — Functional Correctness / Predictive Accuracy)')
ax.legend()
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL-E  |  TYPE: Code  |  PLACEMENT: After CELL-D
#           PURPOSE: Model 2 — Forecast Average_Salary_USD
#                    ISO 25059:2023 → Economic Impact / Transparency
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_M2 = ['Years_Experience', 'AI_Replacement_Risk',
                'Future_Demand_Score', 'Auto_enc', 'Edu_enc']
TARGET_M2   = 'Average_Salary_USD'

X2 = reg_df[FEATURES_M2].dropna()
y2 = reg_df.loc[X2.index, TARGET_M2]

X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42)
m2    = LinearRegression().fit(X2_tr, y2_tr)
y2_pr = m2.predict(X2_te)

r2_m2   = r2_score(y2_te, y2_pr)
mae_m2  = mean_absolute_error(y2_te, y2_pr)
rmse_m2 = np.sqrt(mean_squared_error(y2_te, y2_pr))
coef_m2 = pd.Series(m2.coef_, index=FEATURES_M2)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    'Model 2 — Forecast: Average Salary (USD)\n'
    '(ISO 25059:2023 — Economic Impact of AI Attributes)',
    fontsize=13, fontweight='bold')

# A) Feature Coefficients
ax = axes[0]
cols_m2 = ['#e74c3c' if v < 0 else '#2ecc71' for v in coef_m2]
ax.barh(coef_m2.index, coef_m2.values, color=cols_m2)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient (USD / unit)')
ax.set_title('Feature Coefficients')

# B) Actual vs Predicted
ax = axes[1]
ax.scatter(y2_te / 1000, y2_pr / 1000, alpha=0.4, edgecolors='none', color='darkorange')
lm2 = [min(y2_te.min(), y2_pr.min()) / 1000, max(y2_te.max(), y2_pr.max()) / 1000]
ax.plot(lm2, lm2, 'r--', linewidth=1.5, label='Perfect fit')
ax.set_xlabel('Actual Salary (k USD)')
ax.set_ylabel('Predicted Salary (k USD)')
ax.set_title(f'Actual vs Predicted\nR²={r2_m2:.3f}  MAE={mae_m2/1000:.1f}k  RMSE={rmse_m2/1000:.1f}k')
ax.legend()

# C) Residual Distribution
ax = axes[2]
ax.hist((y2_te - y2_pr) / 1000, bins=30, color='darkorange', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linewidth=1.2, linestyle='--')
ax.set_xlabel('Residual (k USD)')
ax.set_ylabel('Frequency')
ax.set_title('Residual Distribution')

plt.tight_layout()
plt.show()

print(f'\nModel 2  |  Intercept: ${m2.intercept_:,.0f}')
print(f'         |  R²={r2_m2:.3f}   MAE=${mae_m2:,.0f}   RMSE=${rmse_m2:,.0f}')
print('\nCoefficients (USD per unit):')
print(coef_m2.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# CELL-F  |  TYPE: Code  |  PLACEMENT: After CELL-E
#           PURPOSE: Model 2 Projection chart
#                    Salary vs Experience — stratified by AI Risk tier
# ─────────────────────────────────────────────────────────────────────────────

_exp_sw2  = np.arange(0, 26)
_med_au2  = reg_df['Auto_enc'].median()
_med_ed2  = reg_df['Edu_enc'].median()
_med_fd2  = reg_df['Future_Demand_Score'].median()

risk_tiers = [
    (0.10, 'Low AI Risk  (<=0.30)',       '#2ecc71'),
    (0.50, 'Medium AI Risk (0.30-0.70)',  '#f39c12'),
    (0.85, 'High AI Risk  (>=0.70)',      '#e74c3c'),
]

fig, ax = plt.subplots(figsize=(11, 5))
for rv, lbl, col in risk_tiers:
    Xp2 = pd.DataFrame({
        'Years_Experience':    _exp_sw2,
        'AI_Replacement_Risk': rv,
        'Future_Demand_Score': _med_fd2,
        'Auto_enc':            _med_au2,
        'Edu_enc':             _med_ed2,
    })
    ax.plot(_exp_sw2, m2.predict(Xp2) / 1000, label=lbl, color=col, linewidth=2.2)

samp2 = reg_df.sample(min(400, len(reg_df)), random_state=7)
ax.scatter(samp2['Years_Experience'], samp2['Average_Salary_USD'] / 1000,
           alpha=0.18, color='gray', edgecolors='none', s=20, zorder=0, label='Observed')

ax.set_xlabel('Years of Experience')
ax.set_ylabel('Projected Avg Salary (k USD)')
ax.set_title(
    'Salary Forecast by Experience & AI Replacement Risk Tier\n'
    '(ISO 25059:2023 — Economic Impact Transparency)')
ax.legend()
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL-G  |  TYPE: Code  |  PLACEMENT: After CELL-F
#           PURPOSE: Model 3 — Forecast AI_Replacement_Risk
#                    ISO 25059:2023 → Reliability & Robustness
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_M3 = ['Years_Experience', 'Auto_enc', 'AIUse_enc',
                'Future_Demand_Score', 'Edu_enc']
TARGET_M3   = 'AI_Replacement_Risk'

X3 = reg_df[FEATURES_M3].dropna()
y3 = reg_df.loc[X3.index, TARGET_M3]

X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, y3, test_size=0.2, random_state=42)
m3    = LinearRegression().fit(X3_tr, y3_tr)
y3_pr = m3.predict(X3_te)

r2_m3   = r2_score(y3_te, y3_pr)
mae_m3  = mean_absolute_error(y3_te, y3_pr)
rmse_m3 = np.sqrt(mean_squared_error(y3_te, y3_pr))
coef_m3 = pd.Series(m3.coef_, index=FEATURES_M3)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    'Model 3 — Forecast: AI Replacement Risk\n'
    '(ISO 25059:2023 — Reliability & Robustness)',
    fontsize=13, fontweight='bold')

# A) Feature Coefficients
ax = axes[0]
cols_m3 = ['#e74c3c' if v < 0 else '#2ecc71' for v in coef_m3]
ax.barh(coef_m3.index, coef_m3.values, color=cols_m3)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient')
ax.set_title('Feature Coefficients')

# B) Actual vs Predicted
ax = axes[1]
ax.scatter(y3_te, y3_pr, alpha=0.4, edgecolors='none', color='purple')
ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect fit')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Actual AI Replacement Risk')
ax.set_ylabel('Predicted AI Replacement Risk')
ax.set_title(f'Actual vs Predicted\nR²={r2_m3:.3f}  MAE={mae_m3:.3f}  RMSE={rmse_m3:.3f}')
ax.legend()

# C) Residual Distribution
ax = axes[2]
ax.hist(y3_te - y3_pr, bins=30, color='purple', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linewidth=1.2, linestyle='--')
ax.set_xlabel('Residual')
ax.set_ylabel('Frequency')
ax.set_title('Residual Distribution')

plt.tight_layout()
plt.show()

print(f'\nModel 3  |  Intercept: {m3.intercept_:.4f}')
print(f'         |  R²={r2_m3:.3f}   MAE={mae_m3:.4f}   RMSE={rmse_m3:.4f}')
print('\nCoefficients:')
print(coef_m3.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# CELL-H  |  TYPE: Code  |  PLACEMENT: After CELL-G
#           PURPOSE: Model 3 Projection chart
#                    AI Risk vs Experience — stratified by Automation Level
# ─────────────────────────────────────────────────────────────────────────────

_exp_sw3 = np.arange(0, 26)
_med_ai3 = reg_df['AIUse_enc'].median()
_med_fd3 = reg_df['Future_Demand_Score'].median()
_med_ed3 = reg_df['Edu_enc'].median()

auto_levels = [
    (0, 'Low Automation',    '#2ecc71'),
    (1, 'Medium Automation', '#f39c12'),
    (2, 'High Automation',   '#e74c3c'),
]

fig, ax = plt.subplots(figsize=(11, 5))
for av, lbl, col in auto_levels:
    Xp3 = pd.DataFrame({
        'Years_Experience':    _exp_sw3,
        'Auto_enc':            av,
        'AIUse_enc':           _med_ai3,
        'Future_Demand_Score': _med_fd3,
        'Edu_enc':             _med_ed3,
    })
    ax.plot(_exp_sw3, m3.predict(Xp3), label=lbl, color=col, linewidth=2.2)

# Risk threshold reference lines (uses HIGH_RISK_THRESHOLD & LOW_RISK_THRESHOLD
# that are already defined in Section 2 of the notebook)
ax.axhline(HIGH_RISK_THRESHOLD, color='red',   linestyle=':', linewidth=1,
           label=f'High Risk threshold ({HIGH_RISK_THRESHOLD})')
ax.axhline(LOW_RISK_THRESHOLD,  color='green', linestyle=':', linewidth=1,
           label=f'Low Risk threshold ({LOW_RISK_THRESHOLD})')

samp3 = reg_df.sample(min(400, len(reg_df)), random_state=3)
ax.scatter(samp3['Years_Experience'], samp3['AI_Replacement_Risk'],
           alpha=0.15, color='gray', edgecolors='none', s=20, zorder=0, label='Observed')

ax.set_ylim(0, 1)
ax.set_xlabel('Years of Experience')
ax.set_ylabel('Projected AI Replacement Risk')
ax.set_title(
    'AI Replacement Risk Forecast by Experience & Automation Level\n'
    '(ISO 25059:2023 — Reliability, Robustness & Safety)')
ax.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL-I  |  TYPE: Code  |  PLACEMENT: After CELL-H  (final cell of section)
#           PURPOSE: Summary comparison table for all 3 models
# ─────────────────────────────────────────────────────────────────────────────

summary_reg = pd.DataFrame([
    {
        'Model':                   'M1 — Job Growth 2030 (%)',
        'ISO 25059 Characteristic': 'Functional Correctness',
        'R2':   round(r2_m1, 4),
        'MAE':  round(mae_m1, 3),
        'RMSE': round(rmse_m1, 3),

    },
    {
        'Model':                   'M2 — Average Salary (USD)',
        'ISO 25059 Characteristic': 'Economic Impact',
        'R2':   round(r2_m2, 4),
        'MAE':  round(mae_m2, 1),
        'RMSE': round(rmse_m2, 1),
    },
    {
        'Model':                   'M3 — AI Replacement Risk',
        'ISO 25059 Characteristic': 'Reliability & Robustness',
        'R2':   round(r2_m3, 4),
        'MAE':  round(mae_m3, 4),
        'RMSE': round(rmse_m3, 4),
    },
]) 

print('=' * 80)
print('FORECASTING REGRESSION SUMMARY  —  ISO/IEC 25059:2023 Alignment')
print('=' * 80)
print(summary_reg.to_string(index=False))
print()
print('Note: Low R2 is expected for cross-sectional workforce data — many unobserved')
print('factors (industry cycles, policy, geography) also drive outcomes.')
print('Coefficients remain directionally interpretable and meaningful for the case study.')
