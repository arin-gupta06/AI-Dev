# EDA and Data Cleaning: 7-Step Practical Playbook

## Step 1: Define the Objective
Write one sentence goal and list key columns.  
Example: "Understand score patterns by gender and prepare clean data for analysis."  
Columns: gender, math_score, science_score, english_score


---

## Step 2: Build a Data Dictionary
| Column | Type | Meaning | Expected Range | Nullable |
|---|---|---|---|---|
| gender | category | student gender | Male, Female | No |
| math_score | numeric | math marks | 0–100 | No |

---

## Step 3: Run Initial Data Audit
Check: shape, dtypes, missing values, duplicates, invalid categories, out-of-range values.

```python
df.shape
df.info()
df.isna().sum()
df.duplicated().sum()
df.describe()
```

---

## Step 4: Clean Data in Small Passes
1. Standardize text (strip, title case)
2. Convert types (safe numeric coercion)
3. Handle missing values (median, mode, or drop)
4. Remove duplicates
5. Remove invalid ranges/impossible values
6. Document all decisions

```python
clean_df = df.copy()
clean_df["gender"] = clean_df["gender"].str.strip().str.title()
clean_df = clean_df.drop_duplicates()
clean_df["math_score"] = pd.to_numeric(clean_df["math_score"], errors="coerce")
clean_df["math_score"].fillna(clean_df["math_score"].median(), inplace=True)
clean_df = clean_df[(clean_df["math_score"] >= 0) & (clean_df["math_score"] <= 100)]
```

---

## Step 5: Validate After Cleaning
Re-run Step 3 checks and compare before vs after. Confirm:
- Missing values reduced/eliminated
- Duplicates removed
- Dtypes correct
- Ranges valid
- Categories standardized

---

## Step 6: Save Clean Data with Version
Never overwrite raw data. Export as: `dataset_clean_v1.csv`

```python
clean_df.to_csv("student_performance_clean_v1.csv", index=False)
```

---

## Step 7: Write 5-Line Insight Summary
1. Key quality issues found
2. Cleaning actions taken
3. Remaining risks or decisions
4. Top 2 data insights
5. Next analysis step

Example:  
*Found 5 missing math scores, 2 duplicates, standardized gender text. Filled missing with median. No remaining issues. Math scores lowest (mean 63.5), English most consistent (std 8.8). Next: Compare genders with t-test.*

---

## Key Facts to Remember

**Core Principles**
- Garbage in → garbage out. Poor data quality breaks insights.
- All cleaning decisions must be justified and documented.
- Small validated changes are safer than big uncontrolled edits.

**Missing Values**
- Numeric: use median (robust to outliers)
- Categorical: use mode or business rule
- Never drop many rows without justification

**Outliers**
- Outlier ≠ error; can be a real extreme case
- Use IQR method consistently
- Log rows removed

**Type Safety**
- Convert numeric safely: `pd.to_numeric(..., errors="coerce")`
- Check dtypes after conversion
- Don't mix numbers with text placeholders

**Category Cleaning**
- Standardize: strip spaces, title case
- Detect near-duplicates (male, Male, MALE)
- Keep known allowed-values list

**Never Interpret Before Audit**
Run quality checks before drawing conclusions.

**Document Everything**
Keep a cleaning log with date, version, and why each decision was made.

---

## Quick EDA Command Reference

```python
# Structure
df.shape
df.info()
df.columns.tolist()

# Quality
df.isna().sum()
df.duplicated().sum()

# Numeric summaries
df[numeric_cols].describe()
df[numeric_cols].corr()

# Category analysis
df["gender"].value_counts()
df.groupby("gender")[numeric_cols].mean()

# Outlier detection (IQR)
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
outliers = df[((df[cols] < Q1 - 1.5*(Q3-Q1)) | (df[cols] > Q3 + 1.5*(Q3-Q1))).any(axis=1)]
```

---

## One-Page Execution Routine
1. State objective.
2. Build data dictionary.
3. Run audit (shape, info, missing, duplicates).
4. Clean in small passes (text, types, missing, duplicates, invalid ranges).
5. Validate again (compare before/after).
6. Save with version (dataset_clean_v1.csv).
7. Write 5-line summary (issues, actions, risks, insights, next step).

Use this exact routine every time until automatic.

---

# EDA Analysis: 4 Deeper Steps (Post-Cleaning)

After cleaning is complete, run these 4 analyses to extract insights before visualization.

## 1) Univariate Analysis (Each Column Alone)
Understand distribution, spread, and shape of individual columns.

```python
score_cols = ["math_score", "science_score", "english_score"]

print("Descriptive statistics:\n", df[score_cols].describe())
print("\nSkewness (distribution tilt):\n", df[score_cols].skew())
print("\nKurtosis (tail weight):\n", df[score_cols].kurt())
```

**What to look for:**
- Mean vs median: if different, data is skewed
- Std dev: high = diverse scores, low = clustered scores
- Skewness > 0: right tail (high outliers); < 0: left tail (low outliers)

---

## 2) Bivariate Analysis (Gender Comparisons)
Compare numeric columns across gender groups.

```python
print("Mean by gender:\n", df.groupby("gender")[score_cols].mean())
print("\nMedian by gender:\n", df.groupby("gender")[score_cols].median())
print("\nStd dev by gender:\n", df.groupby("gender")[score_cols].std())
```

**What to look for:**
- Are means different between genders? How much?
- Is spread (std) different? Shows consistency differences
- Is median close to mean? If not, outliers present

---

## 3) Correlation Analysis
Measure linear relationships between score columns.

```python
print("Correlation matrix:\n", df[score_cols].corr())
```

**What to look for:**
- Values near 1.0: strong positive (both increase together)
- Values near -1.0: strong negative (one increases, other decreases)
- Values near 0: no linear relationship
- High correlation: subjects reinforce each other or measure similar ability

---

## 4) Outlier Detection (IQR Method)
Identify extreme values that may distort analysis.

```python
Q1 = df[score_cols].quantile(0.25)
Q3 = df[score_cols].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_mask = ((df[score_cols] < lower_bound) | (df[score_cols] > upper_bound)).any(axis=1)
print("Outlier rows count:", outlier_mask.sum())
print("\nOutlier records:\n", df[outlier_mask])
```

**What to look for:**
- How many outliers? (few = OK; many = check data quality)
- Are outliers realistic or data errors?
- Decision: keep (real extreme case) or flag for further investigation

---

## After These 4 Steps → Visualization
Once you have statistical insights, move to charts:
- Histograms (distribution shape)
- Boxplots (outliers and spread by gender)
- Scatter plots (correlations)
- Heatmaps (correlation matrix)
