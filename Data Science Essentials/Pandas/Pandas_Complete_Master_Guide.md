# 📘 Pandas Complete Master Guide

## (Syllabus Aligned + Data Science Ready)

------------------------------------------------------------------------

# 1. Introduction to Pandas

Pandas is a powerful Python library used for: - Data manipulation - Data
cleaning - Data transformation - Statistical analysis - Handling
structured tabular data

Core Data Structures: - **Series** (1D labeled array) - **DataFrame**
(2D labeled table)

------------------------------------------------------------------------

# 2. Pandas Series

## Creating Series

``` python
import pandas as pd

# From list
s = pd.Series([10, 20, 30])

# With custom index
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# From dictionary
s = pd.Series({'A': 100, 'B': 200})
```

## Accessing Series

``` python
s[0]
s['A']
s.iloc[0]
s.loc['A']
```

## Series Operations

``` python
s.mean()
s.median()
s.std()
s.sum()
s.max()
s.min()
```

------------------------------------------------------------------------

# 3. Creating DataFrames

## From Dictionary

``` python
data = {'Name': ['A', 'B'], 'Age': [20, 25]}
df = pd.DataFrame(data)
```

## From List with Header

``` python
df = pd.DataFrame(dataset[1:], columns=dataset[0])
```

------------------------------------------------------------------------

# 4. Data Inspection

``` python
df.head()
df.tail()
df.info()
df.describe()
df.shape
df.columns
df.dtypes
```

------------------------------------------------------------------------

# 5. Indexing & Selection

## Column Selection

``` python
df['Age']
df[['Name', 'Age']]
```

## Row Selection

``` python
df.loc[0]
df.iloc[0]
```

------------------------------------------------------------------------

# 6. Filtering Data

``` python
df[df['Age'] > 20]
df[(df['Age'] > 20) & (df['Salary'] > 50000)]
df[df['City'].isin(['Delhi', 'Mumbai'])]
```

Operators: - & (AND) - \| (OR) - \~ (NOT)

------------------------------------------------------------------------

# 7. Dropping Entries

``` python
df.drop('Age', axis=1)
df.drop(0)
df.drop('Age', axis=1, inplace=True)
```

------------------------------------------------------------------------

# 8. Handling Missing Values

``` python
df.isnull().sum()
df.fillna(0)
df['col'].fillna(df['col'].mean())
df.ffill()
df.bfill()
df.interpolate()
```

------------------------------------------------------------------------

# 9. Sorting and Ranking

## Sorting

``` python
df.sort_values('Age')
df.sort_values('Age', ascending=False)
```

## Ranking

``` python
df['rank'] = df['marks'].rank()
df['rank_desc'] = df['marks'].rank(ascending=False)
```

------------------------------------------------------------------------

# 10. Unique Values and Membership

``` python
df['col'].unique()
df['col'].value_counts()
df['col'].isin(['A', 'B'])
```

------------------------------------------------------------------------

# 11. Function Application

## map()

``` python
df['grade'] = df['marks'].map(lambda x: 'High' if x > 80 else 'Low')
```

## apply()

``` python
df['new_col'] = df['col'].apply(lambda x: x * 2)
df.apply(lambda row: row['A'] + row['B'], axis=1)
```

## applymap()

``` python
df.applymap(lambda x: x * 2)
```

------------------------------------------------------------------------

# 12. GroupBy & Aggregation

``` python
df.groupby('gender')['marks'].mean()
df.groupby(['gender', 'age']).agg(['mean', 'sum'])
```

------------------------------------------------------------------------

# 13. Pivot Tables

``` python
df.pivot_table(
    values='Sales',
    index='Date',
    columns='Store',
    aggfunc='sum',
    fill_value=0,
    margins=True
)
```

------------------------------------------------------------------------

# 14. Descriptive Statistics

``` python
df['marks'].mean()
df['marks'].std()
df['marks'].skew()
df['marks'].kurtosis()
```

------------------------------------------------------------------------

# 15. Percentiles & IQR

``` python
df['marks'].quantile(0.25)
df['marks'].quantile(0.75)
```

IQR = Q3 - Q1

------------------------------------------------------------------------

# 16. Correlation & Covariance

``` python
df.corr()
df['A'].corr(df['B'])
df['A'].cov(df['B'])
```

------------------------------------------------------------------------

# 17. Ranking & Binning

``` python
df['rank'] = df['marks'].rank()
df['category'] = pd.cut(df['marks'], bins=[0, 60, 80, 100], labels=['Low', 'Medium', 'High'])
df['quartile'] = pd.qcut(df['marks'], q=4)
```

------------------------------------------------------------------------

# 18. Crosstab

``` python
pd.crosstab(df['gender'], df['passed'], margins=True)
```

------------------------------------------------------------------------

# 19. Reading & Writing Data

## CSV

``` python
df = pd.read_csv('file.csv')
df.to_csv('output.csv', index=False)
```

## JSON

``` python
df = pd.read_json('file.json')
df.to_json('output.json')
```

------------------------------------------------------------------------

# Final Checklist (Syllabus Coverage)

-   Series
-   DataFrame
-   Indexing & Selection
-   Dropping Entries
-   Sorting & Ranking
-   Unique & Membership
-   Function Application (map, apply, applymap)
-   Mean, Std, Skewness, Kurtosis
-   Correlation & Covariance
-   Percentiles & IQR
-   Pivot Tables
-   Crosstab
-   Reading & Writing CSV & JSON
-   Missing Value Handling

------------------------------------------------------------------------

You can now confidently use this guide for: - University exams -
Practical exams - Viva - Any Data Science course
