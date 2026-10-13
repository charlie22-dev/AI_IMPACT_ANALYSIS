# AI Impact on Jobs 2030: Presentation Guide

## Presentation plan

The presentation runs about 10–12 minutes, plus Q&A. It has two parts: a short spoken intro, then a live walkthrough of the Colab notebook, one section at a time.

| Part | Who | Sections | Time |
| --- | --- | --- | --- |
| Intro: title, abstract, dataset | Member A | none (no code) | 1.5 min |
| Data and charts | Member A | Title cell, Sections 1–4 | 4 min |
| Machine learning and results | Member B | Sections 5–10 | 6 min |
| Q&A | Both | | after |

**Before class:**

- [ ] Upload `AI_Impact_on_Jobs_2030_Final.ipynb` to Colab and run all cells once, so every chart is already showing if the internet or Colab is slow.
- [ ] Keep the CSV file ready in case you need to re-upload it.
- [ ] Check the cluster table in Section 7 and update the conclusion sentence if Colab numbered the groups differently.
- [ ] Practice the handoff from Member A to Member B once.

Explain the output and why each step was done. Don't read the code line by line.

## Part 1: Intro (Member A, about 1.5 min)

**Title:** AI Impact on Jobs 2030: Workforce Analysis and Job Growth Forecasting Using Machine Learning

**Abstract (read or paraphrase):**

> This study analyzes 3,000 employee records from a Kaggle dataset to understand how AI affects jobs across 10 industries and 10 countries. We cleaned the data and visualized patterns in AI replacement risk, salary and job growth. We then used supervised learning (regression) to forecast job growth up to 2030, and unsupervised learning (K-Means) to group employees with similar profiles. All models are evaluated against ISO/IEC 25059:2023, the quality model for AI systems.

**Dataset:** AI Impact on Jobs 2030, from Kaggle. It has 3,000 rows and 20 columns, including job title, industry, country, education, AI replacement risk, future demand score, salary, automation level and projected job growth for 2030.

## Member A: Title cell and Sections 1–4

### Title cell

**Show:** the notebook flow list.

> This is our roadmap. We start with the data, clean it, visualize it, then build models and forecast.

### Section 1: Import Libraries & Upload Dataset

**What it does:** loads the tools, then uploads the Kaggle CSV into Colab.

> We use pandas and numpy to handle the data, matplotlib and seaborn for charts, and scikit-learn for machine learning. Then we upload the CSV. The output confirms 3,000 rows and 20 columns.

### Section 2: Dataset Overview

**What it does:** lists the columns and their types, then shows summary statistics.

> We split the columns into numeric ones, like salary, risk and experience, and category ones, like industry and education. The summary shows an average AI replacement risk of about 0.50, an average salary of about $126,000, and average projected job growth of about +17%.

### Section 3: Data Cleaning

**What it does:** 7 steps.

1. Check for missing values and fill them if any.
2. Remove duplicate rows.
3. Trim extra spaces in text columns.
4. Drop `Employee_ID`.
5. Check for outliers with the IQR rule.
6. Turn ordered categories into numbers (Low=0, Medium=1, High=2).
7. One-hot encode `Industry` (one 0/1 column per industry).

> Bad data gives bad models, so cleaning comes first. Our dataset had 0 missing values, 0 duplicates and 0 outliers. It was already clean, but we ran the checks to prove it. We dropped Employee_ID because an ID number can't predict anything.
>
> Models only understand numbers, so we converted categories. Ordered ones like Low, Medium, High became 0, 1, 2. Industry has no natural order, so we gave each industry its own 0/1 column. This is called one-hot encoding.

### Section 4: Data Visualization

One sentence per chart:

| Chart | What to say |
| --- | --- |
| Bar chart | Average AI risk by job title. The jobs on the left face the highest risk of being replaced. |
| Pie chart | Share of employees in the Low, Medium and High risk bands. (Read the percentages from your output.) |
| Histogram | Salaries are spread fairly evenly from about $30k to $220k. |
| Scatter plot | AI risk vs. future demand. The dots are scattered everywhere, so high risk doesn't automatically mean low demand. |
| Boxplot | Salary by education level. The medians are similar, so education alone doesn't decide salary in this dataset. |
| Heatmap | The important one. Every correlation is close to 0, so the columns are barely related. Remember this: it explains our model scores later. |

**Handoff to Member B:**

> Now that we understand the data, let's build the models.

## Member B: Sections 5–10

### Section 5: Supervised Learning (Regression)

**What it does:** trains 2 models to predict `Job_Growth_2030` from AI risk, future demand, automation level, AI tool usage, experience, education, salary and industry.

> Supervised means we train on examples where the answer is already known. Our target is Job Growth 2030. We train on 80% of the data and keep 20% hidden to test on data the model has never seen.
>
> We compare Linear Regression, which is simple and explainable, with Random Forest, which can capture complex patterns. We score them with three metrics. R² is how much of the variation the model explains, where 1.0 is perfect. MAE is the average error in percentage points. RMSE is like MAE but punishes big errors more.

| Model | R² | MAE | RMSE |
| --- | --- | --- | --- |
| Linear Regression | −0.021 | 14.64 | 16.77 |
| Random Forest | −0.035 | 14.71 | 16.89 |

> Our R² is close to zero. That matches the heatmap: the columns are barely related, so there is very little pattern to learn. We report this openly instead of hiding it.

**Coefficients table:**

> Linear Regression shows why it predicts what it does. Future Demand Score has the biggest positive effect, at +4.5. AI Replacement Risk has a negative effect, so higher risk means less growth. Government and Finance push growth up, and Healthcare and Manufacturing pull it down.

**Actual vs. predicted plot:**

> If the predictions were perfect, every dot would sit on the red line. Ours form a flat band, which confirms the low R².

**Feature importance:**

> This shows which features the Random Forest relied on most.

### Section 6: Forecasting

**What it does:** uses the Linear Regression model to forecast job growth for each industry, year by year from 2026 to 2030.

> This is the forecasting our instructor asked for. The model predicts each industry's total growth by 2030. We convert that into a yearly rate and project it from a starting index of 100 jobs in 2026.
>
> By 2030, Government and Finance grow the most, to about 119. Healthcare and Manufacturing grow the least, to about 116. Every industry still grows, so AI changes jobs more than it removes them.

**New employee forecast (6.1):**

> We can also forecast for someone who isn't in the dataset. Take a Retail worker with high risk, high automation and low AI tool usage. Linear Regression predicts about +15.7% job growth by 2030.

Tip: if there's time, change `new_industry` to `"Finance"` and re-run the cell live.

### Section 7: Unsupervised Learning (K-Means)

**What it does:** groups employees without using a target column.

> Unsupervised means there's no answer column. The algorithm finds groups on its own. First we scale the features, so a big number like salary doesn't outweigh a small one like risk.
>
> The Elbow Method helps us choose the number of groups: we pick the point where the curve bends. We chose 3. One group has high automation, one has low job growth of about +4%, and one has high job growth of about +32%. PCA squeezes the 7 features into 2 dimensions so we can plot the groups.

Check the cluster table in your Colab output before presenting. The group numbers can shift between runs.

### Section 8: ISO/IEC 25059:2023 Summary

> ISO/IEC 25059:2023 is the international quality model for AI systems. We mapped each part of our notebook to it:

| Quality characteristic | Where it shows in the notebook |
| --- | --- |
| Functional correctness | Testing on unseen data with R², MAE and RMSE (Section 5) |
| Data quality | Missing value, duplicate and outlier checks (Section 3) |
| Transparency | Coefficients and feature importance explain the model (Section 5) |
| Adaptability | The model works on new inputs: the forecast and the new employee (Section 6) |
| Robustness | Two different algorithms give similar results (Section 5) |

### Section 9: Export CSV

> Finally, we export two files: the cleaned dataset with predictions and cluster labels, and the industry forecast table. Others can reuse them, for example in Excel or Power BI.

### Section 10: Conclusion

> 1. The data was clean, and we verified it. 2. AI risk varies by job, but every industry is forecast to grow by 2030. 3. Our models have low accuracy because the dataset has weak relationships, and we reported that honestly, as ISO 25059 requires. 4. K-Means found 3 clear groups of employees.

## Q&A prep

The question you're most likely to get is why R² is so low. Both members should be able to answer it.

| Question | Answer |
| --- | --- |
| Why is R² negative or so low? | The heatmap shows every correlation is under 0.06, so the features barely relate to job growth. The model can't learn a pattern that isn't in the data. A negative R² means it does about as well as guessing the average. |
| Then why use the model at all? | It still shows the direction of each effect: demand raises growth and risk lowers it. The same steps would work on a better dataset. Reporting low scores honestly is part of ISO 25059 transparency. |
| Why Linear Regression for the forecast, not Random Forest? | Both scored about the same, and Linear Regression is explainable: we can show exactly how each feature affects the result. |
| What's the difference between supervised and unsupervised? | Supervised learns from known answers (job growth). Unsupervised finds groups with no answers given (K-Means). |
| Why scale the data before K-Means? | K-Means measures distance. Salary is in the hundred-thousands and risk is between 0 and 1, so without scaling, salary would decide everything. |
| Why 3 clusters? | That's where the elbow curve bends. More clusters after that barely improve the fit. |
| Why one-hot encode Industry? | Industries have no order. Numbering them 0–9 would wrongly suggest one industry is "bigger" than another. |
| What is ISO/IEC 25059:2023? | The international standard that defines quality characteristics for AI systems, such as correctness, transparency and robustness. |
| How was the forecast calculated? | The model predicts each industry's total growth by 2030. We turn that into a yearly rate, (1 + total)^(1/4) − 1, and apply it each year from a base of 100 in 2026. |
| What would you improve? | Use a real-world dataset with stronger relationships, or one with a real date column, so we could do true time-series forecasting. |

## Key numbers and final tips

Memorize three numbers: **0.50** average AI risk, **+17%** average job growth by 2030, and **R² ≈ 0** for both models.

- Explain the output and why each step was done. Don't read the code line by line.
- If a cell fails live, scroll to the output from your earlier run and keep going.
- If you change the new-employee profile live, say what you changed before you re-run the cell.
- End with the conclusion, then invite questions.
