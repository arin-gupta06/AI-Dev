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

---

# Visualization Guide: Matplotlib & Seaborn (Step-by-Step)

## Step 1: Setup & Imports
Import required libraries and configure display settings.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure visualization settings
plt.style.use('seaborn-v0_8-darkgrid')  # Clean, professional theme
sns.set_palette("husl")  # Color palette
plt.rcParams['figure.figsize'] = (12, 6)  # Default figure size
plt.rcParams['font.size'] = 10  # Font size
```

**Why these settings matter:**
- `style`: Makes charts look cleaner and more professional
- `palette`: Ensures colors are distinct and colorblind-friendly
- `figsize`: Bigger charts are easier to read
- `font.size`: Ensures text is readable

---

## Step 2: Univariate Visualization (Single Column Analysis)

### 2.1) Histogram – View Distribution Shape
Shows how data spreads across a numeric range. **Use when:** Understanding overall pattern of one continuous variable.

```python
import matplotlib.pyplot as plt

# Basic histogram
plt.figure(figsize=(10, 5))
plt.hist(df['math_score'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Math Scores', fontsize=14, fontweight='bold')
plt.xlabel('Math Score', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.show()
```

**What to look for:**
- **Symmetric (bell-shaped):** Normal distribution; most data centered with few extremes
- **Left-skewed:** Tail on left; most scores cluster high
- **Right-skewed:** Tail on right; most scores cluster low
- **Bimodal (two peaks):** Two distinct groups; might indicate different subpopulations
- **Flat:** Uniform spread; each range has similar frequency

**Interpretation example:**
- If histogram shows right-skew with most students at 60–80 and few at 90–100, students generally perform OK but excel is rare.

---

### 2.2) Kernel Density Estimate (KDE) – Smooth Distribution Curve
Shows probability distribution as a smooth curve instead of bars. **Use when:** You want to see the trend-line instead of raw binning.

```python
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='math_score', kde=True, bins=20, color='steelblue')
plt.title('Math Score Distribution with Trend Line', fontsize=14, fontweight='bold')
plt.xlabel('Math Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.show()
```

**When to use histogram vs KDE:**
- **Histogram:** Raw counts; see actual bin sizes
- **KDE:** Smooth trend; ignore bin artifacts

---

### 2.3) Box Plot (Univariate) – Show Spread & Outliers
Box shows data quartiles; whiskers extend to min/max; points beyond are outliers. **Use when:** Identifying outliers and understanding typical range.

```python
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, y='math_score', color='lightblue')
plt.title('Math Score Box Plot', fontsize=14, fontweight='bold')
plt.ylabel('Math Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.show()
```

**How to read:**
- **Line inside box:** Median (50th percentile; half above, half below)
- **Box edges (Q1, Q3):** 25th and 75th percentile (middle 50% of data)
- **Whiskers:** Extend to lower/upper bounds (typically 1.5 × IQR)
- **Dots:** Outliers beyond whiskers
- **Box height:** Interquartile range (IQR); larger = more spread

**Example interpretation:**
- If median is at 70, Q1 at 60, Q3 at 80: "Half students score 60–70, half score 70–80"

---

## Step 3: Bivariate Visualization (Compare Two Variables)

### 3.1) Grouped Box Plots – Compare Groups
Compare numeric column across categories. **Use when:** Comparing distributions between gender, region, or any category.

```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='gender', y='math_score', palette='Set2')
plt.title('Math Score Distribution by Gender', fontsize=14, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Math Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.show()
```

**What to look for:**
- **Different median positions:** One gender scores higher on average
- **Different box sizes:** One gender has more consistent scores
- **Outlier patterns:** Do outliers appear in specific groups?

**Example reading:** If Female box median is at 75 and Male is at 65, females score 10 points higher on average.

---

### 3.2) Violin Plot – Show Full Distribution by Group
Combines box plot with KDE to show full distribution shape per group. **Use when:** You want to see both the spread AND the distribution shape by group.

```python
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='gender', y='math_score', palette='muted')
plt.title('Math Score Distribution by Gender (Detailed)', fontsize=14, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Math Score', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.show()
```

**Why use violin instead of box:**
- **Wider sections:** Higher density of data at that score
- **Narrow sections:** Few students at that score
- Example: If female violin is wider at 75–85, many females score in that range

---

### 3.3) Scatter Plot – Show Relationship Between Two Numerics
Shows each data point as (X, Y) coordinate. **Use when:** Checking if two numeric columns correlate.

```python
plt.figure(figsize=(10, 6))
plt.scatter(df['math_score'], df['english_score'], alpha=0.5, s=50, color='darkblue')
plt.title('Math Score vs English Score', fontsize=14, fontweight='bold')
plt.xlabel('Math Score', fontsize=12)
plt.ylabel('English Score', fontsize=12)
plt.grid(alpha=0.3)

# Optional: Add trend line
z = np.polyfit(df['math_score'].dropna(), df['english_score'].dropna(), 1)
p = np.poly1d(z)
plt.plot(sorted(df['math_score'].dropna()), p(sorted(df['math_score'].dropna())), 
         "r--", linewidth=2, label='Trend')
plt.legend()
plt.show()
```

**What to look for:**
- **Points go upward right:** Positive correlation (both increase together)
- **Points go downward right:** Negative correlation (one increases, other decreases)
- **Random scatter:** No correlation (no relationship)
- **Tight cluster:** Strong correlation; tight linear bond
- **Loose cloud:** Weak correlation; relationship exists but scattered

---

### 3.4) Scatter Plot with Hue (Color by Category)
Add a third dimension by coloring points by category. **Use when:** Checking if relationship differs by group.

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='math_score', y='english_score', hue='gender', s=100, alpha=0.6)
plt.title('Math vs English Score (by Gender)', fontsize=14, fontweight='bold')
plt.xlabel('Math Score', fontsize=12)
plt.ylabel('English Score', fontsize=12)
plt.grid(alpha=0.3)
plt.legend(title='Gender', loc='best')
plt.show()
```

**Insight:** Do males and females show different trends? Do they cluster separately?

---

## Step 4: Multivariate Visualization (Many Columns at Once)

### 4.1) Correlation Heatmap – See All Pairwise Correlations
Displays correlation matrix as color grid. **Use when:** Understanding how all numeric columns relate to each other.

```python
# Select only numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**How to read:**
- **Dark red (close to 1.0):** Strong positive correlation
- **Dark blue (close to -1.0):** Strong negative correlation
- **White (close to 0):** No correlation
- **Numbers in cells:** Exact correlation value (−1 to +1)

**Business interpretation:**
- If math_score ↔ english_score is 0.85 (dark red): Both move together; students good at one tend to be good at the other
- If math_score ↔ return_days is 0.02 (white): No relationship

---

### 4.2) Pairplot – Scatterplots for All Numeric Pairs
Creates grid of scatter plots comparing every numeric column pair. **Use when:** Exploring all relationships at once.

```python
# Subset to avoid too many plots (limit to 4–5 columns)
plot_cols = ['math_score', 'english_score', 'science_score']
sns.pairplot(df[plot_cols], diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Pairwise Relationships (Math, English, Science)', 
             fontsize=14, fontweight='bold', y=1.001)
plt.tight_layout()
plt.show()
```

**Layout:**
- **Diagonal:** Histogram of each column (univariate)
- **Off-diagonal:** Scatter plots between pairs (bivariate)

**Use case:** Quick scan to spot strong correlations or clusters

---

### 4.3) Categorical Heatmap – Pivot Table as Heatmap
Show aggregated data (e.g., average score by gender and category). **Use when:** Comparing numeric values across two categorical dimensions.

```python
# Create pivot table (rows = gender, columns = category, values = average math_score)
pivot_table = df.pivot_table(values='math_score', index='gender', columns='category', aggfunc='mean')

plt.figure(figsize=(10, 5))
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={"label": "Avg Math Score"})
plt.title('Average Math Score: Gender × Category', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Gender', fontsize=12)
plt.tight_layout()
plt.show()
```

**Reading:** Which gender + category combo has highest average score? Lowest?

---

## Step 5: Categorical Data Visualization

### 5.1) Bar Chart – Count Categories
Shows frequency of each category. **Use when:** Comparing counts across categories.

```python
plt.figure(figsize=(10, 5))
category_counts = df['category'].value_counts()
plt.bar(category_counts.index, category_counts.values, color='coral', edgecolor='black')
plt.title('Number of Transactions by Category', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

**Or using seaborn:**
```python
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='category', palette='pastel', order=df['category'].value_counts().index)
plt.title('Number of Transactions by Category', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

---

### 5.2) Grouped Bar Chart – Compare Across Two Categories
Compare counts split by a second category. **Use when:** Comparing category A across levels of category B.

```python
plt.figure(figsize=(12, 5))
# Count by category and gender
grouped = df.groupby(['category', 'gender']).size().unstack()
grouped.plot(kind='bar', ax=plt.gca(), color=['#FF6B6B', '#4ECDC4'])
plt.title('Transaction Count by Category and Gender', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Gender', loc='best')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Step 6: Best Practices & Styling Tips

### 6.1) Readable Titles & Labels
```python
plt.title('Average Math Score by Gender', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Gender', fontsize=12, labelpad=10)
plt.ylabel('Average Score', fontsize=12, labelpad=10)
```
- **Why:** Large, bold titles stand out; labelpad adds breathing room

---

### 6.2) Add Legends When Using Hue/Color
```python
plt.legend(title='Gender', loc='upper right', frameon=True, shadow=True, fontsize=10)
```
- **Locations:** 'upper right', 'lower left', 'best' (auto-picks non-overlapping)
- **frameon=True:** Box around legend
- **shadow=True:** Drop shadow for depth

---

### 6.3) Use Grid Lines for Readability
```python
plt.grid(alpha=0.3, linestyle='--')  # Dashed grid
plt.grid(axis='y', alpha=0.3)  # Only horizontal grid
```
- **alpha=0.3:** Subtle (doesn't overpower data)
- **axis='y':** Only helps read values on Y-axis

---

### 6.4) Adjust Layout to Prevent Label Cutoff
```python
plt.tight_layout()  # Auto-adjusts spacing
# OR manual control:
plt.subplots_adjust(bottom=0.2, left=0.1)
```

---

### 6.5) Rotate X-axis Labels if Text Overlaps
```python
plt.xticks(rotation=45, ha='right')  # ha='right' aligns rotated text to the right
```

---

## Step 7: Multi-Plot Layouts (Dashboard View)

### 7.1) Create Multiple Plots in One Figure (Subplots)
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Histogram (top-left)
axes[0, 0].hist(df['math_score'], bins=20, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Math Score Distribution')
axes[0, 0].set_xlabel('Score')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Box by gender (top-right)
df.boxplot(column='math_score', by='gender', ax=axes[0, 1])
axes[0, 1].set_title('Math Score by Gender')

# Plot 3: Scatter (bottom-left)
axes[1, 0].scatter(df['math_score'], df['english_score'], alpha=0.5)
axes[1, 0].set_title('Math vs English')
axes[1, 0].set_xlabel('Math Score')
axes[1, 0].set_ylabel('English Score')

# Plot 4: Correlation heatmap (bottom-right)
sns.heatmap(df[['math_score', 'english_score', 'science_score']].corr(), 
            annot=True, ax=axes[1, 1], cmap='coolwarm', center=0)
axes[1, 1].set_title('Correlation Matrix')

plt.suptitle('Student Performance Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()
```

**Benefits:**
- Single unified view of multiple insights
- Professional look; suitable for reports/presentations

---

## Step 8: Real-World Example Walkthrough

**Scenario:** You've cleaned the E-Commerce Transactions dataset and want to visualize order patterns.

### Phase 1: Univariate (Understand Each Column)
```python
# What does payment_method distribution look like?
plt.figure(figsize=(10, 5))
sns.countplot(data=df_clean, x='payment_method', palette='husl')
plt.title('Payment Methods Used', fontsize=14, fontweight='bold')
plt.xlabel('Payment Method', fontsize=12)
plt.ylabel('Number of Transactions', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
# Insight: Credit Card dominates; PayPal barely used → focus on credit card optimization
```

### Phase 2: Bivariate (Compare Two Columns)
```python
# Do ratings vary by payment method?
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clean, x='payment_method', y='rating', palette='Set2')
plt.title('Rating Distribution by Payment Method', fontsize=14, fontweight='bold')
plt.xlabel('Payment Method', fontsize=12)
plt.ylabel('Customer Rating', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
# Insight: Credit card users rate ~4.2, Cash users rate ~3.8 → credit card = satisfied customers
```

### Phase 3: Multivariate (See Full Picture)
```python
# Relationship between price, quantity, and total_amount
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_clean['unit_price'], df_clean['quantity'], 
                     c=df_clean['total_amount'], s=100, alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Total Amount ($)')
plt.title('Price vs Quantity (colored by Total Amount)', fontsize=14, fontweight='bold')
plt.xlabel('Unit Price ($)', fontsize=12)
plt.ylabel('Quantity', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# Insight: More variance in high-price items; bulk orders rare for expensive products
```

---

## Quick Visualization Decision Tree

```
START: What question do I have?

├─ "What's the shape of ONE column?"
│  └─ Histogram or KDE
│
├─ "How does ONE column spread/vary?"
│  └─ Box plot
│
├─ "How does ONE column change ACROSS GROUPS?"
│  ├─ Few groups (2–3)? → Box plot by group
│  └─ Many groups (4+)? → Violin plot
│
├─ "Do TWO numeric columns relate?"
│  ├─ Linear trend? → Scatter + trend line
│  └─ By groups? → Scatter with hue
│
├─ "How do ALL numeric columns relate?"
│  ├─ See exact correlations → Heatmap
│  └─ Explore all pairs → Pairplot
│
├─ "What's the COUNT across CATEGORIES?"
│  └─ Bar chart
│
└─ "How do COUNTS differ by GROUP?"
   └─ Grouped bar chart
```

---

## Summary: Visualization Workflow

1. **Clean data first** (always!)
2. Start with **univariate** (histograms, box plots per column)
3. Move to **bivariate** (grouped box, scatter, correlation)
4. Finish with **multivariate** (heatmaps, pairplots, dashboards)
5. **Style consistently:** Same colors, fonts, grid style across all plots
6. **Label clearly:** Every chart needs title, axis labels, legend (if needed)
7. **Tell a story:** Order plots logically to build insight; not random charts

**Golden Rule:** Every visualization should answer a specific question your stakeholders care about. If a plot doesn't reveal insight, remove it.
