# Documentation Checklist: AI Impact on Jobs 2030

What to put in each section of the documentation, with the numbers to use. The full ready-written text for every section is in AI_Impact_on_Jobs_2030_Case_Study_Documentation.docx.

Check the Random Forest and cluster numbers against your own Colab output before pasting. They can differ slightly between library versions.

## Title page

- **Case study title:** AI Impact on Jobs 2030: Workforce Analysis and Job Growth Forecasting Using Data Science and Machine Learning
- **Dataset:** AI Impact on Jobs 2030 (Kaggle)
- **Tools:** Python, Google Colab, Pandas, Matplotlib, Seaborn, Scikit-learn
- **Standard:** ISO/IEC 25059:2023
- **Fill in yourself:** your names, course code and section, instructor, date submitted

## Abstract

One paragraph covering:

- **What:** 3,000 employees, 20 job titles, 10 industries, 10 countries
- **Why it matters:** shows which jobs face AI risk and which industries will grow
- **Data cleaning:** 0 missing values, 0 duplicates, 0 outliers; all 3,000 rows kept
- **Visualization:** 11 charts
- **Supervised:** Linear Regression and Random Forest predict Job Growth 2030. R² is −0.021 and −0.035, average error about 14.6 points, because every correlation is 0.054 or less
- **Forecasting:** Government and Finance about +19%; Manufacturing and Healthcare about +16%
- **Unsupervised:** K-Means finds 3 groups: high automation, low growth (about +4%) and high growth (about +32%)
- **ISO/IEC 25059:2023** evaluation, plus a note that the study shows patterns, not causes
- **Keywords:** artificial intelligence, job growth, forecasting, Linear Regression, Random Forest, K-Means clustering, ISO/IEC 25059:2023, Python, Pandas

## 1. Introduction

- AI is changing work: some tasks are automated, and new roles are appearing
- What the study does: clean, explore, visualize, then regression, forecasting and clustering
- Purpose: practice the full data science process and explain results honestly. It does not prove causes

## 2. Dataset Description

- **Table 1: all 20 columns.** For each, give the description, data type and values found
- Example: AI_Replacement_Risk is 0.05–0.95, average 0.50
- Example: Average_Salary_USD is $30,221–$219,998, average $126,642
- Example: Job_Growth_2030 is −10% to +45%, average +17.24%
- Example: Education_Level is High School 727, Bachelor 782, Master 712, PhD 779
- **Table 2: summary.** 3,000 rows, 20 columns; 12 categorical/text columns, 8 numeric; target Job_Growth_2030; 0 missing, 0 duplicates
- **Table 3: first 5 rows** (screenshot of df.head() from Colab)
- **Note:** the data looks generated, because categories have equal counts and the columns are unrelated

## 3. Data Cleaning

**Table 4: steps and results**

| Step | Result |
| --- | --- |
| Upload and load CSV | 3,000 rows, 20 columns |
| Check data types | 12 text, 4 integer, 4 decimal |
| Missing values | 0 |
| Duplicates | 0, nothing removed |
| Trim spaces | Text consistent |
| Drop Employee_ID | 19 columns left |
| Outliers (IQR rule) | 0 in every numeric column |
| Encode ordered categories | 6 new number columns (e.g. Low=0, Medium=1, High=2) |
| One-hot encode Industry | 10 new 0/1 columns |
| Create Risk Band | Low ≤0.30, Medium 0.30–0.70, High >0.70 |
| Final check | 3,000 rows, 36 columns, 0 missing |

Also explain **why encoding is needed**: models only understand numbers, and Industry has no natural order, so it gets one-hot encoding instead of numbers 0–9.

## 4. Exploratory Data Analysis

- **Table 5: summary statistics.** Average risk 0.50, average salary $126,642, average growth +17.24%, average experience 12.38 years
- **Table 6: risk bands.** Low 848 (28.3%), Medium 1,297 (43.2%), High 855 (28.5%). Also: 49.6% of employees need upskilling
- **Table 7: by industry.** Finance has the highest growth (18.57%) and salary ($132,017). Transportation is lowest (15.87%). The gap is only about 2.7 points
- **Table 8: by education.** Salary does not rise with education: High School $128,510 is highest and Master $124,825 is lowest
- **By job title:** highest risk is DevOps Engineer 0.569 and Cloud Architect 0.564; lowest is UX Designer 0.460. Top skills: Cloud Computing, Leadership, Communication, Cybersecurity

## 5. Data Visualization

Take a screenshot of each chart from Colab and write 2–3 sentences under it.

| Figure | Chart | Key point |
| --- | --- | --- |
| 1 | Bar: AI risk by job title | Every job averages 0.46–0.57, so no job is clearly safe |
| 2 | Pie: risk bands | Medium 43.2%; Low and High about 28% each |
| 3 | Histogram: salary | Almost flat from $30k to $220k, which suggests generated data |
| 4 | Scatter: risk vs demand | No trend (correlation 0.008) |
| 5 | Boxplot: salary by education | All four boxes are about the same |
| 6 | Heatmap | Every correlation is between −0.047 and 0.054, which explains the weak models |

## 6. Supervised Learning (Regression)

- What supervised learning and regression are
- **Features:** 17 in total (risk, demand, automation, AI tool use, experience, education, salary, plus 10 industry columns). **Target:** Job_Growth_2030
- **Split:** 80/20, which is 2,400 to train and 600 to test
- **The two models:** Linear Regression draws a straight line; Random Forest averages 200 trees
- **Metrics:** what R², MAE and RMSE mean
- **Table 9: results.** Linear Regression R² −0.021, MAE 14.64, RMSE 16.77. Random Forest R² −0.035, MAE 14.71, RMSE 16.89
- **Comparison:** guessing the average gives MAE 14.52, which is slightly better than both models
- **Figure 7:** actual vs predicted. The predictions sit in a flat band around 17%
- **Table 10: coefficients.** Future Demand +4.50; Government +1.79; Finance +1.78; AI Risk −0.80; Manufacturing −1.36; Healthcare −1.41
- **Figure 8:** feature importance. Salary 0.246, AI risk 0.196, demand 0.176
- **Important note:** the low R² is not a coding error. The data has no pattern to learn

## 7. Forecasting

- **Method:** predict each industry's average growth, turn it into a yearly rate with (1 + total)^(1/4) − 1, then project from 100 in 2026
- **Table 11: index for 2026–2030 by industry.** Government 119.0, Finance 118.8, Education 117.8, Technology 117.6, Media 117.3, Energy 116.9, Transportation 116.1, Retail 115.8, Healthcare 115.7, Manufacturing 115.7
- **Figure 9:** forecast line chart. Every industry grows
- **New employee example:** a Retail worker with high risk. Linear Regression gives +15.69% and Random Forest gives +19.27%. The two disagree, which shows the forecast is uncertain
- **Note:** this is not a true time series, because the dataset has no dates

## 8. Unsupervised Learning (K-Means)

- What unsupervised learning is, and how K-Means works
- **Features:** 7 columns, standardized with StandardScaler
- **Figure 10: elbow chart.** The curve has no sharp bend; K=3 was chosen for simple interpretation
- **Table 12, Cluster 0:** 950 employees, high automation (2.00), growth +17.18%
- **Table 12, Cluster 1:** 1,098 employees, low growth +4.28%
- **Table 12, Cluster 2:** 952 employees, high growth +32.25%, highest demand 0.64
- All three clusters have the same AI risk (0.50)
- **Figure 11: PCA plot.** The colors overlap; the 2 components keep only about 31% of the information
- **Silhouette score:** 0.102, so the groups are weakly separated
- **Note:** cluster numbers can change between runs. Read the averages before naming the groups

## 9. ISO/IEC 25059:2023 Quality Evaluation

**Table 13**

| Characteristic | How it's addressed | Result |
| --- | --- | --- |
| Functional correctness | Tested on 600 unseen employees | R² ≈ 0, reported openly |
| Data quality | Missing, duplicate and outlier checks | All 0 |
| Transparency | Coefficients and feature importance | Demand +4.50, risk −0.80 |
| Functional adaptability | Industry forecast and new employee | Works on new inputs |
| Robustness | Two algorithms compared | Both agree |

## 10. Results and Discussion

For each part, write **What was observed? / What does it mean? / Why is it important?** like your sample. Cover data cleaning, EDA, visualization, supervised learning, forecasting and unsupervised learning.

## 11. Limitations

- The data appears to be generated
- R² is near zero, so the forecasts shouldn't be used for real decisions
- The forecast is not a true time series, because there are no dates
- The clusters overlap (silhouette about 0.10)
- No causes are proven

## 12. Conclusion

- The data was clean
- Most jobs are expected to grow (+17.24% on average)
- The models are weak because the data has no strong patterns
- Every industry is forecast to grow, led by Government and Finance
- K-Means found 3 groups
- The study meets the ISO characteristics except correctness
- Lesson: good models need data with real patterns

## 13. References

- The Kaggle dataset page (paste the exact link you downloaded from)
- ISO/IEC 25059:2023
- Documentation for Pandas, Matplotlib, Seaborn and Scikit-learn
- Google Colab
