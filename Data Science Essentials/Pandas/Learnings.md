# Pandas - Key Learnings (Complete Guide for Data Science)

> A comprehensive reference covering all essential Pandas concepts — from Series and DataFrames to advanced statistical analysis, file I/O, and data transformations.

---

## Table of Contents

1. [Introduction to Pandas](#1-introduction-to-pandas)
2. [Series (1-D Labeled Array)](#2-series-1-d-labeled-array)
3. [Creating DataFrames](#3-creating-dataframes)
4. [Reading and Writing Data](#4-reading-and-writing-data)
5. [Basic DataFrame Operations](#5-basic-dataframe-operations)
6. [Indexing with loc and iloc](#6-indexing-with-loc-and-iloc)
7. [Filtering DataFrames](#7-filtering-dataframes)
8. [Column Name Matching in CSV Files](#8-column-name-matching-in-csv-files)
9. [Adding and Removing Data — drop() Method](#9-adding-and-removing-data--drop-method)
10. [sort_values() — Sorting Data](#10-sort_values--sorting-data)
11. [rank() — Ranking Data](#11-rank--ranking-data)
12. [unique() and nunique() — Unique Values](#12-unique-and-nunique--unique-values)
13. [isin() — Membership Testing](#13-isin--membership-testing)
14. [map() — Element-wise Mapping](#14-map--element-wise-mapping)
15. [apply() — Applying Functions](#15-apply--applying-functions)
16. [applymap() / map() on DataFrames — Element-wise DataFrame Operations](#16-applymap--map-on-dataframes--element-wise-dataframe-operations)
17. [GroupBy Operations](#17-groupby-operations)
18. [Merging and Joining DataFrames](#18-merging-and-joining-dataframes)
19. [Pivot Tables](#19-pivot-tables)
20. [Handling Missing Values](#20-handling-missing-values)
21. [Data Type Conversions](#21-data-type-conversions)
22. [String Operations](#22-string-operations)
23. [Working with Dates and Times](#23-working-with-dates-and-times)
24. [Skewness — Measuring Asymmetry](#24-skewness--measuring-asymmetry)
25. [Kurtosis — Measuring Tail Heaviness](#25-kurtosis--measuring-tail-heaviness)
26. [Reading Text Files](#26-reading-text-files)
27. [Reading JSON Files](#27-reading-json-files)
28. [Binning and Discretization](#28-binning-and-discretization)
29. [Cross-Tabulation](#29-cross-tabulation)
30. [Performance Tips](#30-performance-tips)
31. [Quick Reference Summary](#31-quick-reference-summary)

---

# 1. Introduction to Pandas

## What is Pandas?

Pandas is a Python library for **data manipulation and analysis**. It provides two primary data structures:

| Structure | Dimensions | Analogy |
|-----------|-----------|---------|
| **Series** | 1-D | A single column of a spreadsheet |
| **DataFrame** | 2-D | An entire spreadsheet / SQL table |

### Installation

```python
pip install pandas
```

### Import Convention

```python
import pandas as pd
import numpy as np   # Often used alongside pandas
```

---

# 2. Series (1-D Labeled Array)

## What is a Series?

A **Series** is a one-dimensional labeled array that can hold any data type (integers, strings, floats, Python objects, etc.). Think of it as a **single column** of a spreadsheet — it has values and an index (row labels).

```
Index  |  Value
--------------
  0    |   10
  1    |   20
  2    |   30
```

## Creating a Series

### From a List

```python
import pandas as pd

# Default integer index (0, 1, 2, ...)
s = pd.Series([10, 20, 30, 40])
print(s)
```
```
0    10
1    20
2    30
3    40
dtype: int64
```

### With Custom Index

```python
s = pd.Series([10, 30, 20], index=['a', 'b', 'c'])
print(s)
```
```
a    10
b    30
c    20
dtype: int64
```

### From a Dictionary

```python
data = {'Math': 90, 'Science': 85, 'English': 78}
s = pd.Series(data)
print(s)
```
```
Math       90
Science    85
English    78
dtype: int64
```

When you create a Series from a dictionary:
- **Keys** become the **index**
- **Values** become the **data**

### From a Scalar (Single Value)

```python
# Must provide index — the scalar is repeated for each index entry
s = pd.Series(5, index=['a', 'b', 'c'])
print(s)
```
```
a    5
b    5
c    5
dtype: int64
```

---

## Series Attributes

```python
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='scores')

s.values       # array([10, 20, 30])         — underlying NumPy array
s.index        # Index(['a', 'b', 'c'])      — index labels
s.dtype        # int64                        — data type
s.name         # 'scores'                     — name of the Series
s.shape        # (3,)                         — shape tuple
s.size         # 3                            — number of elements
s.nbytes       # 24                           — memory usage in bytes
s.is_unique    # True                         — are all values unique?
s.empty        # False                        — is the Series empty?
```

---

## Accessing Series Elements

### By Label (like `loc`)

```python
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

s['a']           # 10                — single element
s[['a', 'c']]    # a: 10, c: 30     — multiple elements
s['a':'c']        # a: 10, b: 20, c: 30 — range (inclusive!)
```

### By Position (like `iloc`)

```python
s.iloc[0]        # 10              — first element
s.iloc[-1]       # 30              — last element
s.iloc[0:2]      # a: 10, b: 20   — range (exclusive end)
```

---

## Series Operations

### Arithmetic (Vectorized — operates on every element)

```python
s = pd.Series([10, 20, 30])

s + 5       # [15, 25, 35]
s * 2       # [20, 40, 60]
s ** 2      # [100, 400, 900]
s / 10      # [1.0, 2.0, 3.0]
```

### Between Two Series (aligned by index)

```python
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

s1 + s2     # a: 11, b: 22, c: 33
s1 * s2     # a: 10, b: 40, c: 90
```

> **Important:** If indices don't match, unmatched positions become `NaN`.

```python
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20], index=['a', 'b'])

s1 + s2
# a    11.0
# b    22.0
# c     NaN   ← no 'c' in s2
```

---

## Series Statistical Methods

```python
s = pd.Series([10, 20, 30, 40, 50])

s.mean()      # 30.0     — average
s.median()    # 30.0     — middle value
s.mode()      # all values (equal frequency)
s.std()       # 15.81    — standard deviation
s.var()       # 250.0    — variance
s.sum()       # 150      — total
s.min()       # 10       — minimum
s.max()       # 50       — maximum
s.count()     # 5        — non-null count
s.describe()  # summary statistics
s.quantile(0.25)  # 20.0 — 25th percentile
```

---

## Series Boolean Filtering

```python
s = pd.Series([10, 20, 30, 40, 50])

# Get elements greater than 25
s[s > 25]          # 30, 40, 50

# Multiple conditions
s[(s > 15) & (s < 45)]    # 20, 30, 40
```

---

## Series vs DataFrame — Key Differences

| Feature | Series | DataFrame |
|---------|--------|-----------|
| Dimensions | 1-D | 2-D |
| Structure | Single column | Multiple columns |
| Creation | `pd.Series(data)` | `pd.DataFrame(data)` |
| Access column | Not applicable | `df['col']` returns a Series |
| Analogy | One column in Excel | Entire Excel sheet |

> **Key Insight:** A DataFrame is essentially a **collection of Series** that share the same index. Each column of a DataFrame is a Series.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

type(df['A'])   # <class 'pandas.core.series.Series'>
```

---

# 3. Creating DataFrames

## From Lists

### Basic Concept

When you have data in a list format where the first row contains column names (headers) and the rest contain actual data, you need to separate them properly.

```python
dataset = [
    ["Name", "Age", "City", "Salary"],  # Header row (index 0)
    ["Alice", 30, "New York", 70000],    # Data row (index 1)
    ["Bob", 25, "Los Angeles", 65000],   # Data row (index 2)
    ["Charlie", 35, "Chicago", 72000]    # Data row (index 3)
]
```

### The Correct Way

```python
import pandas as pd

df = pd.DataFrame(dataset[1:], columns=dataset[0])
```

---

## Understanding `dataset[1:]`

### What It Does

`dataset[1:]` is a **slice** that extracts all rows **except the first one** (index 0).

```python
dataset[1:]  # "From index 1 to the end"
```

**Returns:**
```python
[
    ["Alice", 30, "New York", 70000],
    ["Bob", 25, "Los Angeles", 65000],
    ["Charlie", 35, "Chicago", 72000]
]
```

### Why Skip Index 0?

Index 0 contains the **header** (column names), not actual data:
```python
dataset[0] = ["Name", "Age", "City", "Salary"]  # Headers, not data!
```

If you include it:
```python
# WRONG: Using dataset[0:]
df = pd.DataFrame(dataset[0:], columns=dataset[0])
```

**Result (Incorrect):**
```
      Name  Age         City  Salary
0     Name  Age         City  Salary  ← Header appears as data row (wrong!)
1    Alice   30     New York   70000
2      Bob   25  Los Angeles   65000
```

### Why `[1:]` Instead of `[1:100]`?

- `[1:]` means "from index 1 **to the end**" (flexible, works for any size)
- `[1:100]` only gets up to index 99 (might miss data)
- No need to count or hardcode the number of rows

---

## Understanding `columns=dataset[0]`

### What It Does

The `columns` parameter tells pandas **what to name each column** in the DataFrame.

```python
dataset[0]  # Returns: ["Name", "Age", "City", "Salary"]
```

This list becomes the **column labels** for the DataFrame.

### Without `columns=dataset[0]` (Bad)

```python
df = pd.DataFrame(dataset[1:])
print(df)
```

**Result:**
```
       0    1              2       3
0  Alice   30       New York   70000
1    Bob   25    Los Angeles   65000
2  Charlie 35        Chicago   72000
```

**Problems:**
- Columns are numbered: `0, 1, 2, 3` (meaningless!)
- Have to remember: "0 is Name, 1 is Age, 2 is City, 3 is Salary"
- Accessing data: `df[0]`, `df[1]` (not intuitive)

### With `columns=dataset[0]` (Good)

```python
df = pd.DataFrame(dataset[1:], columns=dataset[0])
print(df)
```

**Result:**
```
      Name  Age         City  Salary
0    Alice   30     New York   70000
1      Bob   25  Los Angeles   65000
2  Charlie   35      Chicago   72000
```

**Benefits:**
- Columns have meaningful names: `Name, Age, City, Salary`
- Self-documenting code
- Easy to access: `df["Name"]`, `df["Age"]`, `df["Salary"]`

---

## Visual Breakdown

```python
dataset = [
    ["Name", "Age", "City", "Salary"],  ← dataset[0] → columns parameter
    ─────────────────────────────────
    ["Alice", 30, "New York", 70000],   ┐
    ["Bob", 25, "Los Angeles", 65000],  │ dataset[1:] → DataFrame data
    ["Charlie", 35, "Chicago", 72000]   ┘
]
```

## From a Dictionary

```python
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [30, 25, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)
```

**Result:**
```
      Name  Age         City
0    Alice   30     New York
1      Bob   25  Los Angeles
2  Charlie   35      Chicago
```

---

# 4. Reading and Writing Data

## Reading CSV Files

### Basic CSV Reading

```python
import pandas as pd

# Basic read
df = pd.read_csv('file.csv')

# With specific encoding
df = pd.read_csv('file.csv', encoding='utf-8')

# With custom delimiter
df = pd.read_csv('file.csv', sep=';')  # For semicolon-separated files
df = pd.read_csv('file.tsv', sep='\t')  # For tab-separated files

# Skip rows
df = pd.read_csv('file.csv', skiprows=2)  # Skip first 2 rows

# Specify which row contains column names
df = pd.read_csv('file.csv', header=0)  # First row (default)
df = pd.read_csv('file.csv', header=None)  # No header row

# Select specific columns
df = pd.read_csv('file.csv', usecols=['Name', 'Age', 'City'])

# Handle missing values
df = pd.read_csv('file.csv', na_values=['NA', 'N/A', 'missing'])
```

### Common `read_csv` Parameters

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `filepath_or_buffer` | File path or URL | `'data.csv'` |
| `sep` | Delimiter | `','` (default), `';'`, `'\t'` |
| `header` | Row number(s) for column names | `0` (default), `None` |
| `names` | Custom column names | `['col1', 'col2']` |
| `index_col` | Column to use as index | `0` or `'ID'` |
| `usecols` | Columns to read | `['Name', 'Age']` |
| `skiprows` | Rows to skip | `2` or `[0, 1, 5]` |
| `nrows` | Number of rows to read | `1000` |
| `na_values` | Values to treat as NaN | `['NA', 'N/A']` |
| `encoding` | File encoding | `'utf-8'`, `'latin1'` |

---

## Writing to Files

### Writing CSV

```python
# Basic write
df.to_csv('output.csv')

# Without index column
df.to_csv('output.csv', index=False)

# With custom separator
df.to_csv('output.csv', sep=';')

# With specific columns
df.to_csv('output.csv', columns=['Name', 'Age'])

# With custom encoding
df.to_csv('output.csv', encoding='utf-8')
```

### Writing Excel

```python
# Write to Excel
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)

# Multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sales')
    df2.to_excel(writer, sheet_name='Expenses')
```

---

## Reading from Excel

```python
# Read Excel file
df = pd.read_excel('file.xlsx')

# Read specific sheet
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# Read multiple sheets
dfs = pd.read_excel('file.xlsx', sheet_name=['Sheet1', 'Sheet2'])
# Returns dictionary: {'Sheet1': df1, 'Sheet2': df2}

# Read all sheets
dfs = pd.read_excel('file.xlsx', sheet_name=None)
```

---

# 5. Basic DataFrame Operations

## Viewing Data

### Quick Look at Data

```python
# First 5 rows (default)
df.head()

# First 10 rows
df.head(10)

# Last 5 rows (default)
df.tail()

# Last 3 rows
df.tail(3)

# Random sample of rows
df.sample(5)       # 5 random rows
df.sample(frac=0.1)  # 10% of data
```

---

## Getting DataFrame Information

### Basic Info

```python
# Shape (rows, columns)
print(df.shape)
# Output: (100, 5) means 100 rows, 5 columns

# Number of rows
len(df)

# Number of columns
len(df.columns)

# Column names
print(df.columns)
# Output: Index(['Name', 'Age', 'City', 'Salary'], dtype='object')

# Convert to list
print(df.columns.tolist())
# Output: ['Name', 'Age', 'City', 'Salary']

# Index (row labels)
print(df.index)

# Data types of each column
print(df.dtypes)
```

### Detailed Info

```python
# Comprehensive information
df.info()
"""
Output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100 entries, 0 to 99
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   Name    100 non-null    object
 1   Age     98 non-null     int64 
 2   City    100 non-null    object
 3   Salary  95 non-null     float64
dtypes: float64(1), int64(1), object(2)
memory usage: 3.2+ KB
"""

# Statistical summary for numerical columns
df.describe()
"""
Output:
              Age        Salary
count   98.000000     95.000000
mean    32.500000  68500.000000
std      8.234567   5234.123456
min     22.000000  55000.000000
25%     27.000000  65000.000000
50%     31.000000  68000.000000
75%     37.000000  72000.000000
max     45.000000  80000.000000
"""

# Include all columns (including object types)
df.describe(include='all')
```

---

## Selecting Data

### Selecting Columns

```python
# Single column (returns Series)
df['Name']
df.Name  # Alternative (not recommended if column name has spaces)

# Multiple columns (returns DataFrame)
df[['Name', 'Age']]

# Select first 3 columns
df.iloc[:, :3]

# Select columns by position
df.iloc[:, [0, 2, 4]]  # Columns 0, 2, and 4
```

### Selecting Rows

```python
# First 5 rows
df[:5]
df[0:5]

# Rows 10 to 20
df[10:20]

# Every 2nd row
df[::2]

# Rows where condition is True
df[df['Age'] > 30]
```

---

## Adding Columns

```python
# Add new column with constant value
df['Country'] = 'USA'

# Add column based on calculation
df['Salary_Monthly'] = df['Salary'] / 12

# Add column based on condition
df['Age_Group'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Senior')

# Insert column at specific position
df.insert(2, 'Department', 'IT')  # Insert at position 2
```

---

## Adding Rows

```python
# Append single row (create dictionary)
new_row = {'Name': 'Eve', 'Age': 28, 'City': 'Boston', 'Salary': 69000}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# Append another DataFrame
df2 = pd.DataFrame([
    {'Name': 'Frank', 'Age': 31, 'City': 'Seattle', 'Salary': 71000},
    {'Name': 'Grace', 'Age': 29, 'City': 'Austin', 'Salary': 67000}
])
df = pd.concat([df, df2], ignore_index=True)
```

---

## Resetting and Setting Index

```python
# Reset index (make it a column)
df = df.reset_index()

# Reset and drop old index
df = df.reset_index(drop=True)

# Set column as index
df = df.set_index('ID')

# Set multiple columns as index
df = df.set_index(['City', 'Department'])

# Reset multi-index
df = df.reset_index()
```

---

# 6. Indexing with loc and iloc

## Understanding loc vs iloc

| Feature | `loc` | `iloc` |
|---------|-------|--------|
| **Type** | Label-based | Position-based |
| **Uses** | Column names, row labels | Integer positions |
| **Slicing** | Inclusive end | Exclusive end |
| **Example** | `df.loc[0:5, 'Name']` | `df.iloc[0:5, 0]` |

---

## loc — Label-Based Indexing

### Basic Usage

```python
# Single row by label
df.loc[0]  # Row with index 0

# Multiple rows
df.loc[[0, 5, 10]]

# Row range (INCLUSIVE end)
df.loc[0:5]  # Rows 0, 1, 2, 3, 4, 5 (includes 5!)

# Single cell
df.loc[0, 'Name']  # Row 0, column 'Name'

# Multiple cells
df.loc[[0, 1], ['Name', 'Age']]

# All rows, specific columns
df.loc[:, ['Name', 'Age']]

# Specific rows, all columns
df.loc[0:5, :]
```

### With Boolean Conditions

```python
# Rows where Age > 30
df.loc[df['Age'] > 30]

# Select specific columns for filtered rows
df.loc[df['Age'] > 30, ['Name', 'Salary']]

# Multiple conditions
df.loc[(df['Age'] > 30) & (df['City'] == 'New York'), ['Name', 'Salary']]
```

### Modifying Data

```python
# Update single cell
df.loc[0, 'Age'] = 31

# Update multiple cells
df.loc[0:2, 'Salary'] = 70000

# Update based on condition
df.loc[df['Age'] > 40, 'Status'] = 'Senior'

# Update multiple columns
df.loc[df['Age'] > 40, ['Status', 'Bonus']] = ['Senior', 5000]
```

---

## iloc — Position-Based Indexing

### Basic Usage

```python
# Single row by position
df.iloc[0]  # First row

# Multiple rows
df.iloc[[0, 5, 10]]

# Row range (EXCLUSIVE end)
df.iloc[0:5]  # Rows 0, 1, 2, 3, 4 (excludes 5!)

# Single cell
df.iloc[0, 0]  # First row, first column

# Multiple cells
df.iloc[[0, 1], [0, 1]]  # First 2 rows, first 2 columns

# All rows, specific columns
df.iloc[:, [0, 2]]  # All rows, columns 0 and 2

# Last row
df.iloc[-1]

# Last 5 rows
df.iloc[-5:]

# Every 2nd row
df.iloc[::2]
```

### Slicing

```python
# First 3 rows, first 2 columns
df.iloc[:3, :2]

# Rows 5-10, columns 1-3
df.iloc[5:10, 1:3]

# All rows, last column
df.iloc[:, -1]

# Reverse row order
df.iloc[::-1]
```

---

## at and iat — Fast Scalar Access

```python
# at - label-based (faster for single values)
value = df.at[0, 'Name']
df.at[0, 'Name'] = 'Alice'

# iat - position-based (faster for single values)
value = df.iat[0, 0]
df.iat[0, 0] = 'Alice'
```

**When to use**: Only for getting/setting **single values**. Much faster than loc/iloc for single-cell access.

---

# 7. Filtering DataFrames

## Boolean Operators

When filtering with multiple conditions, use:
- `&` for AND
- `|` for OR
- `~` for NOT
- **Always wrap each condition in parentheses!**

---

## Examples

### Single Condition

```python
# Keep rows where sepal length > 5
filtered = df[df["sepal length (cm)"] > 5]
```

### Multiple Conditions (AND)

```python
# Keep rows where sepal length > 5 AND species is setosa
filtered = df[(df["sepal length (cm)"] > 5) & (df["species"] == "setosa")]
#            ↑                               ↑                            ↑
#         parentheses                       &                     parentheses
```

### Multiple Conditions (OR)

```python
# Keep rows where sepal length > 7 OR petal width > 2
filtered = df[(df["sepal length (cm)"] > 7) | (df["petal width (cm)"] > 2)]
```

### Using NOT

```python
# Keep rows where species is NOT setosa
filtered = df[~(df["species"] == "setosa")]

# Or equivalently
filtered = df[df["species"] != "setosa"]
```

### Common Mistakes

```python
# ❌ Wrong - Missing Parentheses
filtered = df[df["sepal length (cm)"] > 5 & df["species"] == "setosa"]
# Error! Operator precedence issues

# ❌ Wrong - Using 'and' instead of '&'
filtered = df[(df["sepal length (cm)"] > 5) and (df["species"] == "setosa")]
# Error! Use & not 'and'

# ✅ Correct
filtered = df[(df["sepal length (cm)"] > 5) & (df["species"] == "setosa")]
```

---

# 8. Column Name Matching in CSV Files

When working with CSV files, **column names must match exactly** (including spaces and special characters).

### Example: Iris Dataset

CSV Header:
```
sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),species
```

```python
# ❌ Wrong - Missing units
df[df["sepal length"] > 5]      # KeyError!

# ✅ Correct - Exact match
df[df["sepal length (cm)"] > 5]  # Works!
```

### How to Check Column Names

```python
# Method 1: Print column names
print(df.columns)
# Output: Index(['sepal length (cm)', 'sepal width (cm)', ...])

# Method 2: Use df.info()
df.info()

# Method 3: List columns as list
print(df.columns.tolist())
```

---

# 9. Adding and Removing Data — drop() Method

## What is drop()?

The `drop()` method **removes rows or columns** from a DataFrame. It returns a new DataFrame by default (does not modify the original unless `inplace=True`).

### Syntax

```python
df.drop(labels, axis=0, inplace=False, errors='raise')
```

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `labels` | Row index or column name(s) to drop | Required |
| `axis` | `0` = drop rows, `1` = drop columns | `0` |
| `inplace` | `True` = modify original, `False` = return new | `False` |
| `errors` | `'raise'` = error if label missing, `'ignore'` = skip | `'raise'` |

---

## Dropping Columns

```python
# Drop single column
df_new = df.drop('Country', axis=1)

# Drop multiple columns
df_new = df.drop(['Country', 'Department'], axis=1)

# Using `columns` parameter (clearer)
df_new = df.drop(columns=['Country', 'Department'])

# Drop in-place (modifies original DataFrame)
df.drop('Country', axis=1, inplace=True)

# Drop by column position
df_new = df.drop(df.columns[2], axis=1)  # Drop 3rd column

# Alternative: del (in-place, no return)
del df['Country']
```

---

## Dropping Rows

```python
# Drop by index label
df_new = df.drop(0)          # Drop row with index 0
df_new = df.drop([0, 5, 10])  # Drop multiple rows

# Drop in-place
df.drop(0, inplace=True)

# Drop based on condition (inverse filter approach)
df_new = df[df['Age'] >= 25]  # Keep only Age >= 25 (drops others)

# Drop rows with missing values
df_new = df.dropna()               # Drop rows with ANY NaN
df_new = df.dropna(how='all')       # Drop rows where ALL values are NaN
df_new = df.dropna(subset=['Name']) # Drop rows where 'Name' is NaN
```

---

## Dropping Duplicates

```python
# Drop all duplicate rows
df_clean = df.drop_duplicates()

# Drop duplicates based on specific columns
df_clean = df.drop_duplicates(subset=['Name', 'City'])

# Keep last occurrence instead of first
df_clean = df.drop_duplicates(keep='last')

# Mark all duplicates (don't keep any)
df_clean = df.drop_duplicates(keep=False)

# Reset index after dropping
df_clean = df_clean.reset_index(drop=True)
```

---

## drop() vs del vs pop()

| Method | Returns New? | In-Place | Can Drop Rows? |
|--------|-------------|----------|----------------|
| `df.drop()` | Yes (default) | Optional (`inplace=True`) | Yes |
| `del df['col']` | No | Always | No (columns only) |
| `df.pop('col')` | Returns the dropped column | Always | No (columns only) |

```python
# pop() example — removes column AND returns it
removed_col = df.pop('Salary')
print(removed_col)  # The Salary Series
# df no longer has 'Salary'
```

---

## Practical Example

```python
import pandas as pd

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice'],
    'Age': [30, 25, 35, 30],
    'City': ['NY', 'LA', 'CHI', 'NY'],
    'Temp': [None, None, None, None]
})

# Workflow: clean this DataFrame
df = df.drop_duplicates()                      # Remove duplicate rows
df = df.drop(columns=['Temp'])                 # Remove useless column
df = df.dropna()                               # Remove rows with NaN
df = df.reset_index(drop=True)                 # Clean up index
print(df)
```

---

# 10. sort_values() — Sorting Data

## What is sort_values()?

`sort_values()` sorts a DataFrame or Series by the values in one or more columns. It returns a **new sorted DataFrame** by default.

### Syntax

```python
df.sort_values(
    by,                    # Column name(s) to sort by
    ascending=True,        # True = A→Z / 0→9, False = Z→A / 9→0
    inplace=False,         # Modify in place?
    na_position='last',    # Where to put NaN: 'first' or 'last'
    ignore_index=False,    # Reset index after sorting?
    key=None               # Function to apply before sorting
)
```

---

## Sort by Single Column

```python
# Sort ascending (default)
df_sorted = df.sort_values('Age')

# Sort descending
df_sorted = df.sort_values('Age', ascending=False)

# Sort in-place
df.sort_values('Age', inplace=True)

# Reset index after sorting
df_sorted = df.sort_values('Age', ignore_index=True)
```

---

## Sort by Multiple Columns

```python
# Sort by City ascending, then by Age descending within each city
df_sorted = df.sort_values(['City', 'Age'], ascending=[True, False])
```

**How it works:**
1. First sorts by `City` (A→Z)
2. Within each city group, sorts by `Age` (highest first)

```
     City  Age
0  Chicago  35
1  Chicago  28
2       LA  30
3       LA  22
4       NY  40
5       NY  25
```

---

## Handling Missing Values in sort_values()

```python
# Put NaN values first
df_sorted = df.sort_values('Age', na_position='first')

# Put NaN values last (default)
df_sorted = df.sort_values('Age', na_position='last')
```

---

## Sorting with a Key Function

```python
# Sort strings by length instead of alphabetical order
df_sorted = df.sort_values('Name', key=lambda col: col.str.len())

# Sort strings case-insensitively
df_sorted = df.sort_values('Name', key=lambda col: col.str.lower())
```

---

## Sorting a Series

```python
s = pd.Series([30, 10, 20, 50, 40])

s_sorted = s.sort_values()               # [10, 20, 30, 40, 50]
s_sorted = s.sort_values(ascending=False) # [50, 40, 30, 20, 10]
```

---

## Sort by Index

```python
# Sort by row index
df_sorted = df.sort_index()

# Sort descending
df_sorted = df.sort_index(ascending=False)

# Sort columns by name (axis=1)
df_sorted = df.sort_index(axis=1)
```

---

# 11. rank() — Ranking Data

## What is rank()?

`rank()` assigns a **rank** (position) to each value in a Series or DataFrame column. The highest/lowest value gets rank 1, the next gets rank 2, and so on.

### Syntax

```python
df['col'].rank(
    ascending=True,   # True = smallest gets rank 1
    method='average',  # How to handle ties
    na_option='keep'   # How to handle NaN
)
```

---

## Basic Ranking

```python
import pandas as pd

df = pd.DataFrame({
    'Student': ['Alice', 'Bob', 'Charlie', 'David'],
    'Score': [85, 92, 78, 92]
})

# Rank ascending (lowest score = rank 1)
df['Rank_Asc'] = df['Score'].rank()
print(df)
```
```
   Student  Score  Rank_Asc
0    Alice     85       2.0
1      Bob     92       3.5    ← tied with David, average of 3 and 4
2  Charlie     78       1.0
3    David     92       3.5    ← tied with Bob
```

```python
# Rank descending (highest score = rank 1)
df['Rank_Desc'] = df['Score'].rank(ascending=False)
```
```
   Student  Score  Rank_Desc
0    Alice     85       3.0
1      Bob     92       1.5
2  Charlie     78       4.0
3    David     92       1.5
```

---

## Tie-Breaking Methods

The `method` parameter controls how ties (identical values) are ranked:

| Method | Behavior | Example (tied at positions 3, 4) |
|--------|----------|----------------------------------|
| `'average'` | Average of tied positions | Both get 3.5 |
| `'min'` | Lowest tied position | Both get 3 |
| `'max'` | Highest tied position | Both get 4 |
| `'first'` | Order of appearance | First gets 3, second gets 4 |
| `'dense'` | Like `min`, but no gaps | Both get 3, next gets 4 |

```python
scores = pd.Series([88, 92, 92, 78, 85])

scores.rank(method='average')  # [3.0, 4.5, 4.5, 1.0, 2.0]
scores.rank(method='min')      # [3.0, 4.0, 4.0, 1.0, 2.0]
scores.rank(method='max')      # [3.0, 5.0, 5.0, 1.0, 2.0]
scores.rank(method='first')    # [3.0, 4.0, 5.0, 1.0, 2.0]
scores.rank(method='dense')    # [3.0, 4.0, 4.0, 1.0, 2.0]
```

### Understanding Dense Rank

Dense rank never skips a rank number. Think of it like a competition where tied competitors share a medal:

```
Score  | average | min | dense
-------|---------|-----|------
  92   |   4.5   |  4  |   4
  92   |   4.5   |  4  |   4
  88   |   3.0   |  3  |   3
  85   |   2.0   |  2  |   2
  78   |   1.0   |  1  |   1
```

With `min`, the next rank after two 4s is 6 (skips 5). With `dense`, it's 5 (no gap).

---

## Handling NaN in Ranking

```python
s = pd.Series([10, np.nan, 30, 20])

s.rank(na_option='keep')     # [1.0, NaN, 3.0, 2.0]  ← NaN stays NaN
s.rank(na_option='top')      # [2.0, 1.0, 4.0, 3.0]  ← NaN gets rank 1
s.rank(na_option='bottom')   # [1.0, 4.0, 3.0, 2.0]  ← NaN gets last rank
```

---

## Rank by Group

```python
# Rank within each group
df['Rank_by_City'] = df.groupby('City')['Salary'].rank(ascending=False, method='dense')
```

This gives each person a rank **within their city**, not globally.

---

## Practical Example: Student Leaderboard

```python
df = pd.DataFrame({
    'student_id': [1, 2, 3, 4, 5],
    'final_grade': [88, 92, 78, 95, 82]
})

df['rank'] = df['final_grade'].rank(ascending=False, method='min')
print(df.sort_values('rank'))
```
```
   student_id  final_grade  rank
3           4           95   1.0
1           2           92   2.0
0           1           88   3.0
4           5           82   4.0
2           3           78   5.0
```

---

# 12. unique() and nunique() — Unique Values

## What are unique() and nunique()?

- `unique()` — Returns an **array of all distinct values** in a Series
- `nunique()` — Returns the **count** of distinct values
- `value_counts()` — Returns a Series with the **frequency** of each unique value

---

## unique() — Get Distinct Values

```python
s = pd.Series(['apple', 'banana', 'apple', 'cherry', 'banana', 'apple'])

s.unique()
# array(['apple', 'banana', 'cherry'], dtype=object)
```

**Key facts:**
- Returns a **NumPy array**, not a Series
- Preserves the **order of first appearance**
- Includes `NaN` if present

```python
s = pd.Series([1, 2, np.nan, 2, 3, np.nan])
s.unique()
# array([ 1.,  2., nan,  3.])   ← NaN included
```

---

## nunique() — Count Distinct Values

```python
s = pd.Series(['apple', 'banana', 'apple', 'cherry', 'banana'])

s.nunique()        # 3 (apple, banana, cherry)

# Include NaN in count
s2 = pd.Series(['apple', 'banana', np.nan])
s2.nunique()                 # 2 (excludes NaN by default)
s2.nunique(dropna=False)     # 3 (includes NaN)
```

### On a DataFrame

```python
# Count unique values per column
df.nunique()
"""
Name       5
Age        4
City       3
dtype: int64
"""

# Count unique values per row
df.nunique(axis=1)
```

---

## value_counts() — Frequency of Each Value

```python
s = pd.Series(['apple', 'banana', 'apple', 'cherry', 'banana', 'apple'])

s.value_counts()
```
```
apple     3
banana    2
cherry    1
dtype: int64
```

### Useful Parameters

```python
# As percentages (proportions)
s.value_counts(normalize=True)
# apple     0.50
# banana    0.333
# cherry    0.167

# Include NaN
s.value_counts(dropna=False)

# Sort by value instead of frequency
s.value_counts().sort_index()

# Bin continuous data
df['Age'].value_counts(bins=5)
```

---

## Practical Data Science Use Cases

```python
# How many unique cities?
print(f"Unique cities: {df['City'].nunique()}")

# What are they?
print(f"Cities: {df['City'].unique()}")

# Distribution of categories
print(df['Gender'].value_counts())

# Check cardinality before encoding
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].nunique()} unique values")
```

---

# 13. isin() — Membership Testing

## What is isin()?

`isin()` checks whether each element in a Series is contained in a given list (or set/array) of values. It returns a **boolean Series** — `True` if the value is in the list, `False` otherwise.

**Analogy**: Like asking "Is this student's city one of New York, Chicago, or Boston?"

### Syntax

```python
df['col'].isin(values)    # values = list, set, Series, or array
```

---

## Basic Usage

```python
import pandas as pd

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'City': ['NY', 'LA', 'Chicago', 'NY', 'Boston']
})

# Check which cities are in the list
df['City'].isin(['NY', 'Chicago'])
```
```
0     True
1    False
2     True
3     True
4    False
dtype: bool
```

---

## Filtering with isin()

```python
# Keep only rows where City is NY or Chicago
target_cities = ['NY', 'Chicago']
filtered = df[df['City'].isin(target_cities)]
print(filtered)
```
```
      Name     City
0    Alice       NY
2  Charlie  Chicago
3    David       NY
```

### Exclude with ~isin() (NOT in)

```python
# Keep rows where City is NOT in the list
filtered = df[~df['City'].isin(['NY', 'LA'])]
print(filtered)
```
```
      Name     City
2  Charlie  Chicago
4      Eve   Boston
```

---

## isin() with Multiple Columns

```python
# Filter using isin on multiple columns
df[(df['City'].isin(['NY', 'LA'])) & (df['Age'].isin([25, 30]))]
```

---

## isin() with a Dictionary (DataFrame-level)

```python
# Check membership across multiple columns at once
df.isin({'City': ['NY', 'LA'], 'Age': [25, 30]})
```
```
    Name   City    Age
0  False   True  False
1  False   True   True
2  False  False  False
3  False   True  False
4  False  False  False
```

---

## isin() vs == (When to Use Which)

| Scenario | Use | Example |
|----------|-----|---------|
| Check one value | `==` | `df[df['City'] == 'NY']` |
| Check multiple values | `isin()` | `df[df['City'].isin(['NY', 'LA'])]` |
| Exclude one value | `!=` | `df[df['City'] != 'NY']` |
| Exclude multiple values | `~isin()` | `df[~df['City'].isin(['NY', 'LA'])]` |

---

## Practical Example: Filtering Student Data

```python
# List of student IDs you want to analyze
target_ids = [101, 105, 110, 115, 120]

# Get only those students
selected = df[df['student_id'].isin(target_ids)]

# Get grade categories of interest
passing_grades = ['A', 'B', 'C']
passed = df[df['grade'].isin(passing_grades)]
```

---

# 14. map() — Element-wise Mapping

## What is map()?

`map()` is a **Series method** that transforms each element using:
- A **dictionary** (lookup table)
- A **function** (applied to each element)
- Another **Series** (index-based lookup)

**Analogy**: Like a translator — give it a word, it returns the translation.

### Syntax

```python
series.map(arg, na_action=None)
```

| Parameter | Purpose |
|-----------|---------|
| `arg` | Dictionary, function, or Series for mapping |
| `na_action` | If `'ignore'`, NaN values are skipped (not passed to the function) |

---

## map() with a Dictionary

```python
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Grade': ['A', 'B', 'C']
})

# Create a mapping dictionary
grade_map = {'A': 'Excellent', 'B': 'Good', 'C': 'Average'}

# Apply the mapping
df['Grade_Label'] = df['Grade'].map(grade_map)
print(df)
```
```
      Name Grade Grade_Label
0    Alice     A   Excellent
1      Bob     B        Good
2  Charlie     C     Average
```

> **Important:** If a value is not in the dictionary, it becomes `NaN`.

```python
grade_map = {'A': 'Excellent', 'B': 'Good'}  # 'C' missing!
df['Grade'].map(grade_map)
# 0    Excellent
# 1         Good
# 2          NaN   ← 'C' not found → NaN
```

---

## map() with a Function

```python
# Apply a function to each element
df['Name_Upper'] = df['Name'].map(str.upper)
# ['ALICE', 'BOB', 'CHARLIE']

# With lambda
df['Name_Length'] = df['Name'].map(lambda x: len(x))
# [5, 3, 7]

# Custom function
def categorize_score(score):
    if score >= 90:
        return 'High'
    elif score >= 70:
        return 'Medium'
    else:
        return 'Low'

df['Category'] = df['Score'].map(categorize_score)
```

---

## map() with na_action='ignore'

```python
s = pd.Series([1, 2, np.nan, 4])

# Without na_action — NaN is passed to the function (may cause errors)
s.map(lambda x: x * 2)
# 0    2.0
# 1    4.0
# 2    NaN   ← NaN * 2 = NaN (works here, but risky)
# 3    8.0

# With na_action='ignore' — NaN is left as-is
s.map(lambda x: x * 2, na_action='ignore')
# Same result, but safer for functions that can't handle NaN
```

---

## map() vs replace()

| Feature | `map()` | `replace()` |
|---------|---------|-------------|
| Unmapped values | Become `NaN` | Stay unchanged |
| Works on | Series only | Series and DataFrame |
| Use case | Full transformation | Selective substitution |

```python
grade_map = {'A': 'Excellent', 'B': 'Good'}  # No mapping for 'C'

df['Grade'].map(grade_map)      # C → NaN
df['Grade'].replace(grade_map)  # C stays as 'C'
```

---

# 15. apply() — Applying Functions

## What is apply()?

`apply()` lets you run a **function on each element, row, or column** of a Series/DataFrame. It is the most versatile transformation tool in Pandas.

### Key Difference from map()

| Feature | `map()` | `apply()` |
|---------|---------|-----------|
| Works on | Series only | Series AND DataFrame |
| Input | Single element | Element (Series), row/column (DataFrame) |
| Flexibility | Simple mapping | Complex logic, multiple columns |

---

## apply() on a Series

Applies a function to **each individual element** of the Series.

```python
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Score': [85, 72, 91]
})

# Lambda function
df['Grade'] = df['Score'].apply(lambda x: 'Pass' if x >= 75 else 'Fail')
print(df)
```
```
      Name  Score Grade
0    Alice     85  Pass
1      Bob     72  Fail
2  Charlie     91  Pass
```

### Named Function

```python
def classify_score(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    else:
        return 'F'

df['Letter_Grade'] = df['Score'].apply(classify_score)
```

---

## apply() on a DataFrame — Row-wise (axis=1)

When `axis=1`, the function receives an **entire row** as a Series. You can access multiple columns.

```python
df = pd.DataFrame({
    'First': ['Alice', 'Bob'],
    'Last': ['Smith', 'Jones'],
    'Salary': [70000, 65000]
})

# Combine first and last name
df['Full_Name'] = df.apply(lambda row: f"{row['First']} {row['Last']}", axis=1)
```
```
  First   Last  Salary     Full_Name
0  Alice  Smith   70000   Alice Smith
1    Bob  Jones   65000     Bob Jones
```

### Complex Row-wise Logic

```python
df = pd.DataFrame({
    'math': [85, 92, 78],
    'science': [90, 88, 76],
    'english': [78, 95, 82]
})

# Calculate whether student passed overall (>= 80 average)
def check_pass(row):
    avg = (row['math'] + row['science'] + row['english']) / 3
    return 'Pass' if avg >= 80 else 'Fail'

df['Result'] = df.apply(check_pass, axis=1)
```

---

## apply() on a DataFrame — Column-wise (axis=0)

When `axis=0` (default), the function receives an **entire column** as a Series.

```python
# Normalize each numeric column (min-max scaling)
df_normalized = df[['math', 'science', 'english']].apply(
    lambda col: (col - col.min()) / (col.max() - col.min())
)
```

---

## apply() with Arguments

```python
# Function with extra arguments
def add_bonus(salary, bonus_pct):
    return salary * (1 + bonus_pct / 100)

# Pass extra arguments via args
df['New_Salary'] = df['Salary'].apply(add_bonus, args=(10,))  # 10% bonus

# Or with keyword arguments
df['New_Salary'] = df['Salary'].apply(add_bonus, bonus_pct=10)
```

---

## apply() for Conditional Filling (Advanced)

```python
# Fill missing Sales_Category based on Sales amount
df['Sales_Category'] = df.apply(
    lambda row: row['Sales_Category'] if pd.notna(row['Sales_Category'])
                else ('High' if row['Sales'] >= 1000 else 'Low'),
    axis=1
)
```

---

## apply() Performance Tip

`apply()` is flexible but **slower than vectorized operations**. Prefer vectorized methods when possible:

```python
# ❌ Slow — apply
df['double'] = df['Score'].apply(lambda x: x * 2)

# ✅ Fast — vectorized
df['double'] = df['Score'] * 2

# ❌ Slow — apply with condition
df['category'] = df['Score'].apply(lambda x: 'High' if x > 80 else 'Low')

# ✅ Faster — np.where (vectorized)
import numpy as np
df['category'] = np.where(df['Score'] > 80, 'High', 'Low')
```

> **Rule of thumb:** Use `apply()` only when the logic is too complex for vectorized operations.

---

# 16. applymap() / map() on DataFrames — Element-wise DataFrame Operations

## What is applymap()?

`applymap()` applies a function to **every single element** in a DataFrame. Unlike `apply()` which works on rows/columns, `applymap()` works on **each individual cell**.

> **Note:** In **pandas ≥ 2.1.0**, `applymap()` has been **deprecated** in favor of `DataFrame.map()`. Both do the same thing — apply a function element-wise to every cell.

### Syntax

```python
# pandas < 2.1.0
df.applymap(func)

# pandas >= 2.1.0 (preferred)
df.map(func)
```

---

## How It Works

```
DataFrame:
      A     B     C
0    10    20    30
1    40    50    60

applymap(lambda x: x * 2):

      A     B     C
0    20    40    60       ← every cell is doubled
1    80   100   120
```

---

## Basic Examples

### Format All Values

```python
df = pd.DataFrame({
    'A': [1.23456, 2.34567],
    'B': [3.45678, 4.56789]
})

# Round every value to 2 decimal places
df_rounded = df.applymap(lambda x: round(x, 2))
# Or in pandas >= 2.1.0:
df_rounded = df.map(lambda x: round(x, 2))
print(df_rounded)
```
```
      A     B
0  1.23  3.46
1  2.35  4.57
```

### Convert All to String

```python
df_str = df.applymap(str)
# Every cell is now a string: '1.23456', '3.45678', etc.
```

### Conditional Formatting

```python
# Mark values > 50 with a star
df_marked = df.applymap(lambda x: f"★{x}" if x > 50 else str(x))
```

---

## applymap() vs apply() vs map()

| Method | Works On | Input to Function | Use Case |
|--------|----------|-------------------|----------|
| `Series.map()` | Series only | Each element | Map/transform values |
| `Series.apply()` | Series | Each element | Complex element logic |
| `DataFrame.apply()` | DataFrame | Each row or column | Row/column operations |
| `DataFrame.applymap()` | DataFrame | Each individual cell | Format every cell |
| `DataFrame.map()` (≥2.1) | DataFrame | Each individual cell | Same as applymap |

### Visual Comparison

```python
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

# map() on Series — each element
df['A'].map(lambda x: x * 10)         # Series: [10, 20]

# apply() on Series — each element
df['A'].apply(lambda x: x * 10)       # Series: [10, 20]

# apply(axis=0) on DataFrame — each column
df.apply(lambda col: col.sum())        # Series: A=3, B=7

# apply(axis=1) on DataFrame — each row
df.apply(lambda row: row.sum(), axis=1) # Series: 0=4, 1=6

# applymap() on DataFrame — each cell
df.applymap(lambda x: x * 10)          # DataFrame: [[10,30],[20,40]]
```

---

## Practical Example: Data Cleaning

```python
# Clean a messy DataFrame — strip whitespace and lowercase everything
df_text = pd.DataFrame({
    'Name': ['  Alice ', ' BOB  ', 'Charlie '],
    'City': [' New York', 'LA  ', '  Chicago']
})

df_clean = df_text.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
print(df_clean)
```
```
      Name       City
0    alice   new york
1      bob         la
2  charlie    chicago
```

---

# 17. GroupBy Operations

## Basic Grouping

### Concept

**Analogy**: Like organizing students by class, then calculating average scores for each class.

```python
# Group by single column
grouped = df.groupby('City')

# Group by multiple columns
grouped = df.groupby(['City', 'Department'])
```

---

## Aggregation Functions

### Single Aggregation

```python
# Mean of all numeric columns per group
df.groupby('City').mean()

# Sum
df.groupby('City').sum()

# Count (number of rows per group)
df.groupby('City').count()

# Size (number of rows per group, including NaN)
df.groupby('City').size()

# Min/Max
df.groupby('City').min()
df.groupby('City').max()

# First/Last row per group
df.groupby('City').first()
df.groupby('City').last()
```

### Specific Column Aggregation

```python
# Mean salary per city
df.groupby('City')['Salary'].mean()

# Multiple columns
df.groupby('City')[['Salary', 'Age']].mean()
```

---

## Multiple Aggregations

### Using agg()

```python
# Multiple functions on one column
df.groupby('City')['Salary'].agg(['mean', 'sum', 'count'])

# Different functions for different columns
df.groupby('City').agg({
    'Salary': ['mean', 'sum'],
    'Age': ['min', 'max'],
    'Name': 'count'
})

# Custom aggregation names
df.groupby('City')['Salary'].agg(
    Average='mean',
    Total='sum',
    Count='count'
)

# Custom function
df.groupby('City')['Salary'].agg(lambda x: x.max() - x.min())
```

---

## Filtering Groups

```python
# Filter groups where count > 2
df.groupby('City').filter(lambda x: len(x) > 2)

# Filter groups where mean salary > 70000
df.groupby('City').filter(lambda x: x['Salary'].mean() > 70000)
```

---

## Transform

**Purpose**: Apply function to each group and return same-shaped DataFrame.

```python
# Subtract group mean from each value
df['Salary_Centered'] = df.groupby('City')['Salary'].transform(lambda x: x - x.mean())

# Fill missing with group mean
df['Salary'] = df.groupby('City')['Salary'].transform(lambda x: x.fillna(x.mean()))
```

---

## Apply Custom Functions to Groups

```python
# Custom function per group
def analyze_group(group):
    return pd.Series({
        'total': group['Salary'].sum(),
        'average': group['Salary'].mean(),
        'count': len(group)
    })

result = df.groupby('City').apply(analyze_group)
```

---

# 18. Merging and Joining DataFrames

## Types of Merges

| Type | SQL Equivalent | Description |
|------|----------------|-------------|
| `inner` | INNER JOIN | Only matching rows |
| `left` | LEFT JOIN | All from left, matching from right |
| `right` | RIGHT JOIN | All from right, matching from left |
| `outer` | FULL OUTER JOIN | All rows from both |

---

## Merge — Combining on Common Columns

### Basic Merge

```python
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David']
})

df2 = pd.DataFrame({
    'ID': [2, 3, 4, 5],
    'Salary': [65000, 72000, 68000, 75000]
})

# Inner merge (default) — only matching IDs
merged = pd.merge(df1, df2, on='ID')

# Left merge — all from df1
merged = pd.merge(df1, df2, on='ID', how='left')

# Right merge — all from df2
merged = pd.merge(df1, df2, on='ID', how='right')

# Outer merge — all from both
merged = pd.merge(df1, df2, on='ID', how='outer')
```

---

## Merge on Different Column Names

```python
merged = pd.merge(df1, df2, left_on='Employee_ID', right_on='ID')
```

## Merge on Multiple Columns

```python
merged = pd.merge(df1, df2, on=['ID', 'Department'], how='inner')
```

## Handling Duplicate Column Names

```python
merged = pd.merge(df1, df2, on='ID', suffixes=('_left', '_right'))
```

---

## Concat — Stacking DataFrames

### Vertical Stacking (Default)

```python
combined = pd.concat([df1, df2], ignore_index=True)
```

### Horizontal Stacking

```python
combined = pd.concat([df1, df2], axis=1)
```

---

## Join — Merge on Index

```python
df1 = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
df2 = pd.DataFrame({'B': [4, 5, 6]}, index=['b', 'c', 'd'])

result = df1.join(df2, how='left')
```

---

# 19. Pivot Tables

## What is a Pivot Table?

A pivot table reshapes data to summarize and analyze it from different perspectives. It's like reorganizing a spreadsheet so you can see patterns more clearly.

### Simple Analogy
- **Before pivot**: Long list (Date, Student, Present/Absent)
- **After pivot**: Grid showing students as rows and dates as columns, with attendance status in cells

---

## Basic Example

```python
data = {
    'Date': ['Jan', 'Jan', 'Feb', 'Feb', 'Mar', 'Mar'],
    'Store': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [100, 150, 120, 160, 110, 170]
}
df = pd.DataFrame(data)

pivot = df.pivot_table(
    values='Sales',      # What numbers to put in cells
    index='Date',        # What to use as rows
    columns='Store',     # What to use as columns
    aggfunc='sum'        # How to combine multiple values
)
print(pivot)
```
```
Store    A    B
Date            
Feb    120  160
Jan    100  150
Mar    110  170
```

---

## All Pivot Table Parameters

```python
df.pivot_table(
    values=None,           # Column(s) to aggregate
    index=None,            # Row grouping column(s)
    columns=None,          # Column grouping column(s)
    aggfunc='mean',        # Aggregation function(s)
    fill_value=None,       # Value to replace NaN
    margins=False,         # Add row/column totals
    dropna=True,           # Exclude columns with all NaN
    margins_name='All',    # Name for margin row/column
    observed=False         # For categorical data (advanced)
)
```

---

## Parameter Details

### 1. `values` — What to Aggregate

```python
# Single value column
pivot = df.pivot_table(values='Sales', index='Date', columns='Store')

# Multiple value columns
pivot = df.pivot_table(values=['Sales', 'Profit'], index='Date', columns='Store')
```

### 2. `index` — Rows of Pivot Table

```python
# Single index
pivot = df.pivot_table(values='Sales', index='Date', columns='Store')

# Multiple indices (hierarchical)
pivot = df.pivot_table(values='Sales', index=['Year', 'Date'], columns='Store')
```

### 3. `columns` — Columns of Pivot Table

```python
# Single column category
pivot = df.pivot_table(values='Sales', index='Date', columns='Store')

# Multiple column categories
pivot = df.pivot_table(values='Sales', index='Date', columns=['Region', 'Store'])
```

### 4. `aggfunc` — Aggregation Function

```python
# Single function
pivot = df.pivot_table(values='Sales', index='Date', columns='Store', aggfunc='sum')

# Multiple functions
pivot = df.pivot_table(values='Sales', index='Date', columns='Store', 
                       aggfunc=['sum', 'mean', 'count'])

# Different functions for different columns
pivot = df.pivot_table(values=['Sales', 'Profit'], index='Date', columns='Store',
                       aggfunc={'Sales': 'sum', 'Profit': 'mean'})

# Custom function
pivot = df.pivot_table(values='Sales', index='Date', columns='Store',
                       aggfunc=lambda x: x.max() - x.min())
```

### 5. `fill_value` — Replace Missing Values

```python
pivot = df.pivot_table(values='Sales', index='Date', columns='Store', fill_value=0)
```

### 6. `margins` — Add Totals Row/Column

```python
pivot = df.pivot_table(values='Sales', index='Date', columns='Store', 
                       aggfunc='sum', margins=True)
```

### 7. `margins_name` — Label for Totals

```python
pivot = df.pivot_table(values='Sales', index='Date', columns='Store',
                       aggfunc='sum', margins=True, margins_name='Grand Total')
```

---

## Parameter Quick Reference

| Parameter | Purpose | Common Values |
|-----------|---------|---------------|
| `values` | What to aggregate | Column name(s) |
| `index` | Row categories | Column name(s) |
| `columns` | Column categories | Column name(s) |
| `aggfunc` | How to combine | `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'` |
| `fill_value` | Replace NaN | `0`, `''`, `-1` |
| `margins` | Add totals | `True` / `False` |
| `margins_name` | Label for totals | `'All'`, `'Total'` |

---

# 20. Handling Missing Values

## Types of Missing Values

- **Numerical Missing Values** (NaN) — When numeric data is missing
- **Textual Missing Values** (NaN or empty strings) — When text data is missing

---

## Method 1: `fillna()` — Fill with Specific Value

```python
# Fill with 0
df['Sales'] = df['Sales'].fillna(0)

# Fill with specific text
df['Product_Code'] = df['Product_Code'].fillna('UNKNOWN')

# Fill different columns with different values
df = df.fillna({'Sales': 0, 'Profit': 0, 'Product_Code': 'N/A'})
```

---

## Method 2: Fill with Mean/Median/Mode

```python
# Fill with mean (average)
df['Sales'] = df['Sales'].fillna(df['Sales'].mean())

# Fill with median (middle value) — better when data has outliers
df['Sales'] = df['Sales'].fillna(df['Sales'].median())

# Fill with mode (most common value) — best for categorical data
df['Category'] = df['Category'].fillna(df['Category'].mode()[0])
```

---

## Method 3: `interpolate()` — Estimate from Neighbors

```python
data = {'Day': [1, 2, 3, 4, 5], 'Sales': [100, np.nan, 120, np.nan, 140]}
df = pd.DataFrame(data)

df['Sales'] = df['Sales'].interpolate()
# Result: [100, 110, 120, 130, 140]
```

---

## Method 4: Forward Fill (`ffill`) — Copy Previous Value

```python
df['Status'] = df['Status'].ffill()
# ['Active', NaN, NaN, 'Inactive'] → ['Active', 'Active', 'Active', 'Inactive']
```

## Method 5: Backward Fill (`bfill`) — Copy Next Value

```python
df['Status'] = df['Status'].bfill()
# ['Active', NaN, NaN, 'Inactive'] → ['Active', 'Inactive', 'Inactive', 'Inactive']
```

---

## Conditional Filling with apply()

```python
df['Sales_Category'] = df.apply(
    lambda row: row['Sales_Category'] if pd.notna(row['Sales_Category'])
                else ('High' if row['Sales'] >= 1000 else 'Low'),
    axis=1
)
```

---

## Code Pattern Analysis — Auto-generating Missing Codes

### Sequential Numbering

```python
counter = 1

def generate_code(row):
    global counter
    if pd.isna(row['Product_Code']):
        new_code = f"PROD-{counter:03d}"
        counter += 1
        return new_code
    return row['Product_Code']

df['Product_Code'] = df.apply(generate_code, axis=1)
```

### Department-Based Codes

```python
def analyze_code(row):
    if pd.isna(row['Code']):
        same_dept_codes = df[
            (df['Department'] == row['Department']) & 
            (df['SubDept'] == row['SubDept']) & 
            (df['Code'].notna())
        ]['Code']
        
        if len(same_dept_codes) > 0:
            last_code = same_dept_codes.iloc[-1]
            parts = last_code.split('-')
            last_number = int(parts[-1])
            new_number = last_number + 1
            return f"{row['Department']}-{row['SubDept']}-{new_number:03d}"
        else:
            return f"{row['Department']}-{row['SubDept']}-001"
    return row['Code']

df['Code'] = df.apply(analyze_code, axis=1)
```

---

## Filling Methods Comparison

| Method | Best For | Example Use Case |
|--------|----------|------------------|
| `fillna(value)` | Constant replacement | Fill missing categories with "Unknown" |
| `fillna(mean/median)` | Numerical gaps | Fill missing prices with average |
| `interpolate()` | Sequential data | Fill missing temperature readings |
| `ffill()` | Carry forward | Fill status changes (keep last status) |
| `bfill()` | Carry backward | Fill from future known values |
| `apply() + lambda` | Conditional logic | Fill based on other column values |

---

# 21. Data Type Conversions

## Checking Data Types

```python
print(df.dtypes)          # All column types
print(df['Age'].dtype)    # Specific column
```

---

## Converting Types with astype()

```python
df['Age'] = df['Age'].astype(int)
df['Salary'] = df['Salary'].astype(float)
df['ID'] = df['ID'].astype(str)
df['City'] = df['City'].astype('category')  # Saves memory

# Convert multiple columns
df[['Age', 'Salary']] = df[['Age', 'Salary']].astype(float)
```

## Handling Errors in Conversion

```python
# Ignore errors (invalid → NaN)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Raise error on invalid conversion
df['Age'] = pd.to_numeric(df['Age'], errors='raise')

# Keep original value if conversion fails
df['Age'] = pd.to_numeric(df['Age'], errors='ignore')
```

## Converting Strings to Dates

```python
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Extract components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.day_name()
```

---

# 22. String Operations

## Basic String Methods

```python
df['Name'] = df['Name'].str.lower()       # lowercase
df['Name'] = df['Name'].str.upper()       # UPPERCASE
df['Name'] = df['Name'].str.capitalize()  # First letter
df['Name'] = df['Name'].str.title()       # Title Case
df['Name'] = df['Name'].str.strip()       # Remove whitespace
df['Name'] = df['Name'].str.lstrip()      # Left trim
df['Name'] = df['Name'].str.rstrip()      # Right trim
```

---

## String Checks

```python
df[df['Name'].str.contains('Alice')]           # Contains substring
df[df['Name'].str.contains('alice', case=False)] # Case-insensitive
df[df['Name'].str.startswith('A')]             # Starts with
df[df['Name'].str.endswith('son')]             # Ends with
df['ID'].str.isnumeric()                       # Is numeric?
df['Name'].str.isalpha()                       # Is alphabetic?
```

---

## String Manipulation

```python
# Replace
df['Name'] = df['Name'].str.replace('Bob', 'Robert')

# Split
df['First_Name'] = df['Name'].str.split(' ').str[0]
df[['First', 'Last']] = df['Name'].str.split(' ', expand=True)

# Join
df['Full_Name'] = df['First_Name'] + ' ' + df['Last_Name']

# Extract with regex
df['Code'] = df['ID'].str.extract(r'([A-Z]+)-(\d+)')

# String length
df['Name_Length'] = df['Name'].str.len()

# Pad with characters
df['ID'] = df['ID'].str.pad(width=5, side='left', fillchar='0')
```

---

# 23. Working with Dates and Times

## Creating Date Columns

```python
df['Today'] = pd.Timestamp.now()
df['Date'] = pd.Timestamp('2024-01-15')
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# Date range
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
```

---

## Extracting Date Components

```python
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Hour'] = df['Date'].dt.hour
df['DayOfWeek'] = df['Date'].dt.dayofweek   # 0=Monday, 6=Sunday
df['DayName'] = df['Date'].dt.day_name()     # 'Monday', etc.
df['Quarter'] = df['Date'].dt.quarter
df['IsWeekend'] = df['Date'].dt.dayofweek >= 5
```

---

## Date Arithmetic

```python
df['Tomorrow'] = df['Date'] + pd.Timedelta(days=1)
df['Yesterday'] = df['Date'] - pd.Timedelta(days=1)
df['NextWeek'] = df['Date'] + pd.Timedelta(weeks=1)
df['NextMonth'] = df['Date'] + pd.DateOffset(months=1)
df['NextYear'] = df['Date'] + pd.DateOffset(years=1)
df['Days_Since'] = (pd.Timestamp.now() - df['Date']).dt.days
df['Duration'] = (df['End_Date'] - df['Start_Date']).dt.days
```

---

## Filtering by Date

```python
df[df['Date'] == '2024-01-15']
df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2024-12-31')]
df[df['Date'].dt.year == 2024]
df[df['Date'].dt.month == 1]
df[df['Date'].dt.dayofweek >= 5]   # Weekends
```

---

# 24. Skewness — Measuring Asymmetry

## What is Skewness?

Skewness measures the **asymmetry** of the distribution of values around the mean. In simple terms, it tells you whether the data is **lopsided** to the left or right.

### Visual Analogy

Imagine a see-saw (balance beam):
- **Perfectly balanced** → Symmetric (bell curve) → Skewness = 0
- **Heavier on the right** → Long tail stretches right → **Positive skew**
- **Heavier on the left** → Long tail stretches left → **Negative skew**

---

## Types of Skewness

| Type | Skewness Value | Tail Direction | Mean vs Median |
|------|---------------|----------------|----------------|
| **Positive (Right)** | > 0 | Long right tail | Mean > Median |
| **Symmetric** | ≈ 0 | No tail | Mean ≈ Median |
| **Negative (Left)** | < 0 | Long left tail | Mean < Median |

### Real-World Examples

| Distribution | Skewness | Why |
|-------------|----------|-----|
| Income | Positive | Most people earn average, few earn millions (right tail) |
| Age at retirement | Negative | Most retire at 60-65, few retire very early (left tail) |
| Test scores (well-designed) | ≈ 0 | Roughly bell-shaped |

---

## How to Calculate in Pandas

```python
import pandas as pd

df = pd.read_csv('student_performance.csv')

# Skewness for a single column
skew_math = df['math_score'].skew()
print(f"Math Score Skewness: {skew_math:.3f}")

# Skewness for all numeric columns
print(df.select_dtypes(include='number').skew())
```

---

## Interpreting Skewness Values

| Value Range | Interpretation |
|-------------|----------------|
| -0.5 to +0.5 | Approximately symmetric |
| -1.0 to -0.5 or +0.5 to +1.0 | Moderately skewed |
| < -1.0 or > +1.0 | Highly skewed |

```python
skewness = df['final_grade'].skew()

if abs(skewness) < 0.5:
    print("Distribution is approximately symmetric")
elif abs(skewness) < 1:
    print("Distribution is moderately skewed")
else:
    print("Distribution is highly skewed")

if skewness > 0:
    print("Right-skewed (positive): tail extends to the right")
elif skewness < 0:
    print("Left-skewed (negative): tail extends to the left")
else:
    print("Perfectly symmetric")
```

---

## Why Skewness Matters for Data Science

1. **Choosing the right statistical measure:**
   - Symmetric data → **Mean** is a good measure of center
   - Skewed data → **Median** is better (not influenced by extreme values)

2. **Machine Learning preprocessing:**
   - Many algorithms assume normally distributed features
   - Highly skewed features may need **log transformation** or **Box-Cox transformation**

3. **Outlier detection:**
   - High skewness often indicates the presence of outliers

```python
# Log transformation to reduce positive skewness
import numpy as np
df['log_income'] = np.log1p(df['income'])  # log1p handles 0 values safely
print(f"Before: {df['income'].skew():.3f}")
print(f"After:  {df['log_income'].skew():.3f}")
```

---

# 25. Kurtosis — Measuring Tail Heaviness

## What is Kurtosis?

Kurtosis measures the **"tailedness"** of a distribution — how heavy or light the tails are compared to a normal distribution. It tells you about the likelihood of **extreme values (outliers)**.

### Visual Analogy

Imagine comparing mountains:
- **Normal mountain** (mesokurtic) → Standard bell shape → Kurtosis ≈ 0
- **Tall, sharp peak** (leptokurtic) → Data concentrated in center with fat tails → Kurtosis > 0
- **Short, flat peak** (platykurtic) → Data spread out, thin tails → Kurtosis < 0

---

## Types of Kurtosis

> **Note:** Pandas uses **excess kurtosis** (kurtosis relative to normal distribution). A normal distribution has excess kurtosis = 0.

| Type | Excess Kurtosis | Shape | Outlier Risk |
|------|----------------|-------|-------------|
| **Leptokurtic** | > 0 | Sharp peak, fat tails | Higher chance of outliers |
| **Mesokurtic** | ≈ 0 | Normal bell curve | Normal outlier probability |
| **Platykurtic** | < 0 | Flat peak, thin tails | Lower chance of outliers |

### Real-World Examples

| Distribution | Kurtosis | Why |
|-------------|----------|-----|
| Stock returns | High positive | Most days normal, occasional crashes/spikes |
| Human heights | Near 0 | Follows normal distribution |
| Uniform (dice roll) | Negative | All outcomes equally likely, no peak |

---

## How to Calculate in Pandas

```python
import pandas as pd

df = pd.read_csv('student_performance.csv')

# Kurtosis for a single column
kurt_math = df['math_score'].kurtosis()
print(f"Math Score Kurtosis: {kurt_math:.3f}")

# Kurtosis for all numeric columns
print(df.select_dtypes(include='number').kurtosis())
```

---

## Interpreting Kurtosis Values

| Value Range | Interpretation |
|-------------|----------------|
| ≈ 0 | Normal-like distribution |
| > 0 (positive) | Heavy tails — more outliers than normal |
| < 0 (negative) | Light tails — fewer outliers than normal |
| > 3 | Very heavy tails — extreme outlier risk |

```python
kurt = df['final_grade'].kurtosis()

if abs(kurt) < 0.5:
    print("Approximately normal (mesokurtic)")
elif kurt > 0:
    print(f"Leptokurtic (kurtosis={kurt:.3f}): heavy tails, more outliers")
else:
    print(f"Platykurtic (kurtosis={kurt:.3f}): light tails, fewer outliers")
```

---

## Skewness + Kurtosis Together

Using both metrics gives you a fuller picture of the distribution:

```python
def distribution_summary(series, name=""):
    skew = series.skew()
    kurt = series.kurtosis()
    
    print(f"--- {name} ---")
    print(f"Mean: {series.mean():.2f}, Median: {series.median():.2f}")
    print(f"Skewness: {skew:.3f} → {'Right' if skew > 0 else 'Left' if skew < 0 else 'Symmetric'}")
    print(f"Kurtosis: {kurt:.3f} → {'Heavy tails' if kurt > 0 else 'Light tails' if kurt < 0 else 'Normal'}")
    print()

distribution_summary(df['math_score'], "Math Score")
distribution_summary(df['final_grade'], "Final Grade")
```

---

## Why Kurtosis Matters for Data Science

1. **Risk assessment:** High kurtosis in financial data means more extreme events (crashes/booms)
2. **Outlier awareness:** Leptokurtic data has more outliers — plan your cleaning strategy accordingly
3. **Model assumptions:** Many statistical tests assume normality (kurtosis ≈ 0)
4. **Feature engineering:** May need transformation if kurtosis is too high

---

# 26. Reading Text Files

## What are Text Files?

Text files are plain files that store data as human-readable characters. Unlike CSV or Excel, text files can have **various formats and separators**. Common extensions: `.txt`, `.log`, `.dat`, `.tsv`.

---

## Using `read_csv()` for Text Files

Despite its name, `pd.read_csv()` can read **any delimited text file**, not just CSVs. The key is to specify the correct **separator/delimiter**.

### Tab-Separated Files (.tsv or .txt)

```python
# Tab-separated values
df = pd.read_csv('data.tsv', sep='\t')
df = pd.read_csv('data.txt', sep='\t')
```

### Space-Separated Files

```python
# Space-separated
df = pd.read_csv('data.txt', sep=' ')

# Multiple spaces (variable whitespace)
df = pd.read_csv('data.txt', sep='\s+', engine='python')
# \s+ matches one or more whitespace characters
```

### Pipe-Separated Files

```python
# Pipe-separated values
df = pd.read_csv('data.txt', sep='|')
```

### Custom Delimiter

```python
# Semicolon-separated
df = pd.read_csv('data.txt', sep=';')

# Tilde-separated
df = pd.read_csv('data.txt', sep='~')
```

---

## Using `read_table()` — Dedicated for Tab-Separated

```python
# read_table() defaults to tab separator (unlike read_csv which defaults to comma)
df = pd.read_table('data.txt')              # Tab-separated (default)
df = pd.read_table('data.txt', sep='|')     # Override with pipe
```

---

## Using `read_fwf()` — Fixed-Width Files

Some text files use **fixed column widths** instead of delimiters:

```
Name       Age  City          Salary
Alice       30  New York       70000
Bob         25  Los Angeles    65000
Charlie     35  Chicago        72000
```

```python
# Auto-detect column widths
df = pd.read_fwf('data.txt')

# Specify column widths manually
df = pd.read_fwf('data.txt', widths=[10, 5, 15, 10])

# Specify column positions (start, end tuples)
df = pd.read_fwf('data.txt', colspecs=[(0, 10), (10, 15), (15, 30), (30, 40)])
```

---

## Common Parameters for Text File Reading

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `sep` / `delimiter` | Column separator | `'\t'`, `' '`, `'\|'` |
| `header` | Row containing column names | `0` (first row), `None` (no header) |
| `names` | Custom column names | `['col1', 'col2']` |
| `skiprows` | Skip initial rows | `3` or `[0, 2, 5]` |
| `skipfooter` | Skip rows at end | `2` |
| `comment` | Character marking comment lines | `'#'` |
| `encoding` | File encoding | `'utf-8'`, `'latin1'`, `'cp1252'` |
| `engine` | Parser engine | `'python'` (for regex separators) |
| `error_bad_lines` | Skip malformed lines | `False` |

---

## Handling Comment Lines

```python
# Skip lines starting with '#'
df = pd.read_csv('data.txt', comment='#', sep='\t')
```

**File content:**
```
# This is a comment
# Another comment
Name	Age	City
Alice	30	New York
Bob	25	LA
```

---

## Reading Raw Text (Line by Line)

When the file is unstructured and doesn't fit a table format:

```python
# Read entire file into a single-column DataFrame
df = pd.read_csv('log.txt', header=None, names=['line'])
# Each line becomes one row

# Then parse as needed
df['timestamp'] = df['line'].str[:19]  # Extract first 19 chars as timestamp
df['message'] = df['line'].str[20:]    # Rest is the message
```

### Using Python's Built-in File Reading

```python
# For files that need custom parsing
with open('data.txt', 'r') as f:
    lines = f.readlines()

# Strip whitespace and split
data = [line.strip().split('|') for line in lines]

# Convert to DataFrame
df = pd.DataFrame(data[1:], columns=data[0])
```

---

## Practical Example: Reading a Log File

```python
# Common log format: timestamp | level | message
log_data = pd.read_csv(
    'app.log',
    sep='|',
    header=None,
    names=['timestamp', 'level', 'message'],
    encoding='utf-8'
)

# Clean up whitespace
log_data = log_data.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

# Filter errors only
errors = log_data[log_data['level'] == 'ERROR']
print(errors)
```

---

# 27. Reading JSON Files

## What is JSON?

JSON (JavaScript Object Notation) is a lightweight data format that stores data as **key-value pairs** and **arrays**. It's the most common format for APIs and web data.

### JSON Structure

```json
{
    "name": "Alice",
    "age": 30,
    "city": "New York",
    "scores": [85, 90, 78]
}
```

**Key concepts:**
- `{}` — Object (like a Python dictionary)
- `[]` — Array (like a Python list)
- `"key": value` — Key-value pair
- Values can be: strings, numbers, booleans, null, arrays, or nested objects

---

## Basic JSON Reading

### Simple (Flat) JSON — Records Style

**File: `data.json`**
```json
[
    {"name": "Alice", "age": 30, "city": "New York"},
    {"name": "Bob", "age": 25, "city": "LA"},
    {"name": "Charlie", "age": 35, "city": "Chicago"}
]
```

```python
import pandas as pd

df = pd.read_json('data.json')
print(df)
```
```
      name  age      city
0    Alice   30  New York
1      Bob   25        LA
2  Charlie   35   Chicago
```

---

## JSON Orientations (orient parameter)

JSON data can be structured differently. The `orient` parameter tells pandas how to interpret the structure.

### 1. Records Orientation (Default for arrays)

Each item is a dictionary (row). **Most common for API responses.**

```json
[
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25}
]
```

```python
df = pd.read_json('data.json', orient='records')
```

### 2. Columnar Orientation

Data grouped by columns (like a dictionary of lists).

```json
{
    "name": ["Alice", "Bob", "Charlie"],
    "age": [30, 25, 35],
    "city": ["New York", "LA", "Chicago"]
}
```

```python
df = pd.read_json('data.json', orient='columns')
```

### 3. Index Orientation

Outer keys are row indices.

```json
{
    "0": {"name": "Alice", "age": 30},
    "1": {"name": "Bob", "age": 25}
}
```

```python
df = pd.read_json('data.json', orient='index')
```

### 4. Split Orientation

Explicitly splits columns, index, and data.

```json
{
    "columns": ["name", "age"],
    "index": [0, 1],
    "data": [["Alice", 30], ["Bob", 25]]
}
```

```python
df = pd.read_json('data.json', orient='split')
```

### 5. Values Orientation

Just the data as arrays (no column names or index).

```json
[
    ["Alice", 30],
    ["Bob", 25]
]
```

```python
df = pd.read_json('data.json', orient='values')
df.columns = ['name', 'age']  # Assign column names manually
```

---

## Handling Nested JSON

Real-world JSON is often **nested** (objects within objects). Pandas provides `json_normalize()` to flatten it.

### Example: Nested JSON

```json
[
    {
        "name": "Alice",
        "age": 30,
        "address": {
            "city": "New York",
            "state": "NY",
            "zip": "10001"
        },
        "scores": [85, 90, 78]
    },
    {
        "name": "Bob",
        "age": 25,
        "address": {
            "city": "LA",
            "state": "CA",
            "zip": "90001"
        },
        "scores": [72, 88, 91]
    }
]
```

### Flatten with json_normalize()

```python
import json

# Load JSON data
with open('data.json', 'r') as f:
    data = json.load(f)

# Normalize (flatten) the nested structure
df = pd.json_normalize(data)
print(df)
```
```
    name  age       scores  address.city address.state address.zip
0  Alice   30  [85, 90, 78]     New York            NY       10001
1    Bob   25  [72, 88, 91]           LA            CA       90001
```

**What happened:**
- Nested `address` object was flattened into `address.city`, `address.state`, `address.zip`
- `scores` array stayed as a list (not flattened — arrays of values need extra handling)

### Deeper Nesting

```python
# Specify the record path for deeply nested data
df = pd.json_normalize(
    data,
    record_path='scores',     # Path to the array to expand
    meta=['name', 'age'],     # Fields to include from parent
    meta_prefix='student_'    # Prefix for parent fields
)
```

---

## Reading JSON from a String

```python
import json

json_string = '{"name": "Alice", "age": 30, "city": "NY"}'

# Method 1: read_json with StringIO
from io import StringIO
df = pd.read_json(StringIO(json_string), orient='index')

# Method 2: Parse dict then create DataFrame
data = json.loads(json_string)
df = pd.DataFrame([data])
```

---

## Reading JSON from an API URL

```python
# Read directly from a URL
df = pd.read_json('https://api.example.com/data')

# With authentication (use requests library)
import requests

response = requests.get('https://api.example.com/data', headers={'Authorization': 'Bearer token'})
data = response.json()
df = pd.json_normalize(data)
```

---

## Writing JSON

```python
# Write DataFrame to JSON file
df.to_json('output.json')

# Pretty-printed (human readable)
df.to_json('output.json', indent=4)

# Different orientations
df.to_json('output.json', orient='records', indent=2)
df.to_json('output.json', orient='columns')
df.to_json('output.json', orient='split')
```

---

## Common read_json Parameters

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `path_or_buf` | File path, URL, or JSON string | `'data.json'` |
| `orient` | JSON structure format | `'records'`, `'columns'`, `'index'`, `'split'` |
| `typ` | Return type | `'frame'` (DataFrame), `'series'` (Series) |
| `encoding` | File encoding | `'utf-8'` |
| `lines` | Read JSON Lines format (one object per line) | `True` / `False` |
| `convert_dates` | Try to parse date columns | `True` / `False` |

---

## JSON Lines Format (.jsonl)

Some files have **one JSON object per line** (common in log files and streaming data):

```
{"name": "Alice", "age": 30}
{"name": "Bob", "age": 25}
{"name": "Charlie", "age": 35}
```

```python
df = pd.read_json('data.jsonl', lines=True)
```

---

## Practical Example: API Data Processing

```python
import pandas as pd
import json

# Simulating API response (nested JSON)
api_response = [
    {
        "id": 1,
        "user": {"name": "Alice", "email": "alice@example.com"},
        "orders": [
            {"product": "Laptop", "price": 999},
            {"product": "Mouse", "price": 29}
        ]
    },
    {
        "id": 2,
        "user": {"name": "Bob", "email": "bob@example.com"},
        "orders": [
            {"product": "Keyboard", "price": 79}
        ]
    }
]

# Step 1: Flatten user info
df_users = pd.json_normalize(api_response, meta=['id'])
print(df_users[['id', 'user.name', 'user.email']])

# Step 2: Expand orders into separate rows
df_orders = pd.json_normalize(
    api_response,
    record_path='orders',
    meta=['id', ['user', 'name']]
)
print(df_orders)
```

---

# 28. Binning and Discretization

## pd.cut() — Equal-Width Bins

```python
# Create age groups
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 30, 50, 100], 
                          labels=['Child', 'Young Adult', 'Adult', 'Senior'])

# Equal-width bins (auto)
df['Salary_Bin'] = pd.cut(df['Salary'], bins=5)
```

## pd.qcut() — Equal-Frequency Bins (Quantiles)

```python
df['Salary_Quartile'] = pd.qcut(df['Salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

---

# 29. Cross-Tabulation

## What is Cross-Tabulation?

Cross-tabulation (`pd.crosstab()`) creates a frequency table showing how often combinations of values occur across two or more categorical columns.

```python
# Basic crosstab
ct = pd.crosstab(df['Gender'], df['Passed'])
print(ct)
```
```
Passed   No  Yes
Gender          
Female   12   38
Male     15   35
```

### With Percentages

```python
# Row percentages (each row sums to 100%)
ct_pct = pd.crosstab(df['Gender'], df['Passed'], normalize='index') * 100
print(ct_pct)
```
```
Passed      No    Yes
Gender                
Female   24.0   76.0
Male     30.0   70.0
```

### With Margins (Totals)

```python
ct = pd.crosstab(df['Gender'], df['Passed'], margins=True)
```

---

# 30. Performance Tips

## Vectorization vs Loops

```python
# ❌ Slow — Using loop
for i in range(len(df)):
    df.loc[i, 'New_Col'] = df.loc[i, 'Col1'] * 2

# ✅ Fast — Vectorized operation
df['New_Col'] = df['Col1'] * 2
```

## Memory Optimization

```python
# Check memory usage
df.memory_usage(deep=True)

# Convert to categorical to save memory
df['City'] = df['City'].astype('category')

# Downcast numeric types
df['Age'] = pd.to_numeric(df['Age'], downcast='integer')

# Read only needed columns
df = pd.read_csv('file.csv', usecols=['Name', 'Age', 'Salary'])

# Use chunks for large files
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)
```

---

## Best Practices

1. **Use vectorized operations** instead of loops
2. **Use `loc`/`iloc`** for explicit indexing
3. **Chain operations carefully** — sometimes breaking them up is clearer
4. **Use `inplace=True` sparingly** — makes debugging harder
5. **Check dtypes early** — correct types save memory and improve speed
6. **Use `category` dtype** for low-cardinality string columns
7. **Reset index after filtering** to avoid gaps
8. **Use `copy()`** when you want to avoid modifying original DataFrame

---

## Replace Values

```python
# Replace specific value
df['Status'] = df['Status'].replace('N/A', 'Unknown')

# Replace multiple values
df['Status'] = df['Status'].replace(['N/A', 'NA', 'Missing'], 'Unknown')

# Replace with dictionary
df['Status'] = df['Status'].replace({'Y': 'Yes', 'N': 'No', 'M': 'Maybe'})

# Replace across entire DataFrame
df = df.replace('N/A', np.nan)

# Replace with regex
df['Code'] = df['Code'].str.replace(r'[^A-Za-z0-9]', '', regex=True)
```

---

## Renaming

```python
# Rename columns
df = df.rename(columns={'Name': 'Employee_Name', 'Age': 'Employee_Age'})

# Rename with function
df = df.rename(columns=str.lower)
df = df.rename(columns=str.upper)

# Set column names directly
df.columns = ['col1', 'col2', 'col3']

# Clean column names
df.columns = df.columns.str.strip().str.replace(' ', '_')
```

---

## Handling Duplicates

```python
# Check for duplicates
df.duplicated()                         # Boolean Series
df.duplicated(subset=['Name', 'City'])  # Check specific columns
df.duplicated(keep=False)               # Mark ALL duplicates as True

# Drop duplicates
df_clean = df.drop_duplicates()
df_clean = df.drop_duplicates(subset=['Name', 'City'])
df_clean = df.drop_duplicates(keep='last')
```

---

# 31. Quick Reference Summary

## Essential Operations Cheat Sheet

| Task | Command |
|------|---------|
| **Create Series** | `pd.Series([1, 2, 3])` |
| **Create DataFrame** | `pd.DataFrame(data)` |
| **Reading Data** | `pd.read_csv()`, `pd.read_json()`, `pd.read_excel()` |
| **Writing Data** | `df.to_csv()`, `df.to_json()`, `df.to_excel()` |
| **View Data** | `df.head()`, `df.tail()`, `df.sample()` |
| **Info** | `df.info()`, `df.describe()`, `df.shape` |
| **Select Column** | `df['col']`, `df[['col1', 'col2']]` |
| **Select Rows** | `df.loc[label]`, `df.iloc[position]` |
| **Filter** | `df[df['col'] > value]` |
| **Filter (multiple)** | `df[df['col'].isin(['a', 'b'])]` |
| **Sort** | `df.sort_values('col')` |
| **Rank** | `df['col'].rank()` |
| **Unique** | `df['col'].unique()`, `df['col'].nunique()` |
| **Value Counts** | `df['col'].value_counts()` |
| **Group** | `df.groupby('col').agg()` |
| **Merge** | `pd.merge(df1, df2, on='key')` |
| **Concat** | `pd.concat([df1, df2])` |
| **Pivot Table** | `df.pivot_table(values, index, columns)` |
| **Cross-Tab** | `pd.crosstab(df['col1'], df['col2'])` |
| **Missing Values** | `df.fillna()`, `df.dropna()` |
| **Drop Column** | `df.drop('col', axis=1)` |
| **Drop Row** | `df.drop(index)` |
| **Rename** | `df.rename(columns={'old': 'new'})` |
| **Map Values** | `df['col'].map(dict_or_func)` |
| **Apply Function** | `df.apply(func)`, `df['col'].apply(func)` |
| **Element-wise** | `df.applymap(func)` or `df.map(func)` (≥2.1) |
| **Skewness** | `df['col'].skew()` |
| **Kurtosis** | `df['col'].kurtosis()` |
| **Read Text** | `pd.read_csv('file.txt', sep='\t')` |
| **Read JSON** | `pd.read_json('file.json')` |
| **Flatten JSON** | `pd.json_normalize(data)` |

---

## Key Takeaways

1. **Series is the building block**: Every DataFrame column is a Series
2. **Separate headers from data**: Use `dataset[1:]` for data, `dataset[0]` for column names
3. **Named columns are essential**: Makes DataFrames readable and easy to work with
4. **Column names must match exactly**: Including spaces, units, and special characters
5. **Multiple conditions need parentheses**: Always wrap each condition when using `&` or `|`
6. **Use `&` and `|`, not `and` and `or`**: For boolean operations on DataFrames
7. **`drop()` returns a new DataFrame**: Use `inplace=True` to modify the original
8. **`sort_values()` handles multiple columns**: Pass a list with corresponding `ascending` list
9. **`rank()` has 5 tie-breaking methods**: `average`, `min`, `max`, `first`, `dense`
10. **`isin()` > multiple `==`**: Use `isin()` when checking against a list of values
11. **`map()` ≠ `apply()`**: `map()` is for simple 1:1 transforms; `apply()` handles complex logic
12. **`applymap()` → `DataFrame.map()`**: Renamed in pandas ≥ 2.1.0
13. **Skewness tells you direction**: Positive = right tail, Negative = left tail
14. **Kurtosis tells you tail weight**: High = more outliers, Low = fewer outliers
15. **`read_csv()` reads ANY delimited text**: Just change the `sep` parameter
16. **`json_normalize()` flattens nested JSON**: Essential for API data
17. **Vectorize whenever possible**: `apply()` is flexible but slow — prefer direct operations
