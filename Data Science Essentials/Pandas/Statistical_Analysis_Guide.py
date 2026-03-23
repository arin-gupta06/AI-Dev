"""
Statistical Analysis Guide - Student Performance Dataset
This file demonstrates various statistical analysis techniques using pandas
"""

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('student_performance.csv')

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDataset Shape:", df.shape)

print("\n" + "=" * 80)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 80)

# Basic statistics for all numerical columns
print("\nBasic Statistics:")
print(df.describe())

# Statistics for specific columns
print("\nMath Score Statistics:")
print(f"Mean: {df['math_score'].mean():.2f}")
print(f"Median: {df['math_score'].median():.2f}")
print(f"Mode: {df['math_score'].mode().values[0]}")
print(f"Standard Deviation: {df['math_score'].std():.2f}")
print(f"Variance: {df['math_score'].var():.2f}")
print(f"Min: {df['math_score'].min()}")
print(f"Max: {df['math_score'].max()}")
print(f"Range: {df['math_score'].max() - df['math_score'].min()}")

# Quartiles and IQR
print("\nQuartile Analysis:")
Q1 = df['math_score'].quantile(0.25)
Q2 = df['math_score'].quantile(0.50)
Q3 = df['math_score'].quantile(0.75)
IQR = Q3 - Q1
print(f"Q1 (25th percentile): {Q1}")
print(f"Q2 (50th percentile/Median): {Q2}")
print(f"Q3 (75th percentile): {Q3}")
print(f"IQR (Interquartile Range): {IQR}")

print("\n" + "=" * 80)
print("2. CORRELATION ANALYSIS")
print("=" * 80)

# Correlation matrix for numerical columns
print("\nCorrelation Matrix:")
numerical_cols = ['study_hours_per_week', 'attendance_percentage', 'previous_grade', 
                  'sleep_hours', 'math_score', 'science_score', 'english_score', 'final_grade']
correlation_matrix = df[numerical_cols].corr()
print(correlation_matrix)

# Specific correlations
print("\nKey Correlations with Final Grade:")
print(correlation_matrix['final_grade'].sort_values(ascending=False))

print("\n" + "=" * 80)
print("3. GROUPBY ANALYSIS")
print("=" * 80)

# Group by gender
print("\nPerformance by Gender:")
gender_stats = df.groupby('gender')[['math_score', 'science_score', 'english_score', 'final_grade']].mean()
print(gender_stats)

# Group by extracurricular activities
print("\nPerformance by Extracurricular Activities:")
extra_stats = df.groupby('extracurricular')[['study_hours_per_week', 'final_grade']].mean()
print(extra_stats)

# Group by pass/fail
print("\nCharacteristics of Passed vs Failed Students:")
pass_stats = df.groupby('passed')[['study_hours_per_week', 'attendance_percentage', 'sleep_hours']].mean()
print(pass_stats)

print("\n" + "=" * 80)
print("4. DISTRIBUTION ANALYSIS")
print("=" * 80)

# Value counts
print("\nGender Distribution:")
print(df['gender'].value_counts())
print("\nPass/Fail Distribution:")
print(df['passed'].value_counts())

# Percentage distribution
print("\nPass Rate:")
print(df['passed'].value_counts(normalize=True) * 100)

print("\n" + "=" * 80)
print("5. CROSS-TABULATION ANALYSIS")
print("=" * 80)

# Cross-tabulation: Gender vs Passed
print("\nGender vs Pass/Fail:")
crosstab = pd.crosstab(df['gender'], df['passed'], margins=True)
print(crosstab)

# Cross-tabulation with percentages
print("\nGender vs Pass/Fail (Percentages):")
crosstab_pct = pd.crosstab(df['gender'], df['passed'], normalize='index') * 100
print(crosstab_pct)

print("\n" + "=" * 80)
print("6. CONDITIONAL ANALYSIS")
print("=" * 80)

# Students with high study hours (>20 hours/week)
high_study = df[df['study_hours_per_week'] > 20]
print(f"\nStudents studying >20 hours/week: {len(high_study)}")
print(f"Average final grade: {high_study['final_grade'].mean():.2f}")

# Students with low attendance (<70%)
low_attendance = df[df['attendance_percentage'] < 70]
print(f"\nStudents with <70% attendance: {len(low_attendance)}")
print(f"Average final grade: {low_attendance['final_grade'].mean():.2f}")

# High performers (final_grade >= 90)
high_performers = df[df['final_grade'] >= 90]
print(f"\nHigh performers (grade >= 90): {len(high_performers)}")
print("\nCharacteristics of high performers:")
print(high_performers[['study_hours_per_week', 'attendance_percentage', 'sleep_hours']].mean())

print("\n" + "=" * 80)
print("7. ADVANCED STATISTICAL MEASURES")
print("=" * 80)

# Skewness and Kurtosis
print("\nSkewness (measure of asymmetry):")
print(f"Math Score Skewness: {df['math_score'].skew():.3f}")
print(f"Final Grade Skewness: {df['final_grade'].skew():.3f}")

print("\nKurtosis (measure of tail heaviness):")
print(f"Math Score Kurtosis: {df['math_score'].kurtosis():.3f}")
print(f"Final Grade Kurtosis: {df['final_grade'].kurtosis():.3f}")

# Coefficient of Variation (CV)
print("\nCoefficient of Variation (CV = std/mean):")
cv_math = (df['math_score'].std() / df['math_score'].mean()) * 100
print(f"Math Score CV: {cv_math:.2f}%")

print("\n" + "=" * 80)
print("8. PERCENTILE ANALYSIS")
print("=" * 80)

# Custom percentiles
percentiles = [10, 25, 50, 75, 90, 95, 99]
print("\nFinal Grade Percentiles:")
for p in percentiles:
    value = df['final_grade'].quantile(p/100)
    print(f"{p}th percentile: {value:.2f}")

print("\n" + "=" * 80)
print("9. COVARIANCE ANALYSIS")
print("=" * 80)

# Covariance between study hours and final grade
cov_study_grade = df['study_hours_per_week'].cov(df['final_grade'])
print(f"\nCovariance (Study Hours & Final Grade): {cov_study_grade:.2f}")

# Covariance matrix
print("\nCovariance Matrix (selected features):")
cov_matrix = df[['study_hours_per_week', 'attendance_percentage', 'final_grade']].cov()
print(cov_matrix)

print("\n" + "=" * 80)
print("10. RANKING AND SORTING")
print("=" * 80)

# Rank students by final grade
df_ranked = df.copy()
df_ranked['rank'] = df_ranked['final_grade'].rank(ascending=False, method='min')
print("\nTop 10 Students:")
print(df_ranked[['student_id', 'final_grade', 'rank']].sort_values('rank').head(10))

print("\n" + "=" * 80)
print("EXERCISES TO PRACTICE:")
print("=" * 80)
print("""
1. Calculate the correlation between sleep_hours and final_grade
2. Find the average study hours for students who passed vs failed
3. Identify outliers in math_score using IQR method
4. Calculate the pass rate for each age group
5. Find students who study less than 10 hours but still passed
6. Calculate the median attendance percentage for each gender
7. Create age groups (18, 19, 20) and compare their average grades
8. Find the relationship between previous_grade and final_grade
9. Calculate z-scores for final_grade to identify exceptional students
10. Compare the variance in scores between genders
""")

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS COMPLETE!")
print("=" * 80)
