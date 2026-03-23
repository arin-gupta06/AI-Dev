"""
STATISTICAL ANALYSIS PRACTICE QUESTIONS - SOLUTIONS
Student Performance Dataset

Compare your answers with these solutions!
"""

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('student_performance.csv')

print("="*80)
print("SOLUTIONS - Compare your answers!")
print("="*80)

# ============================================================================
# SECTION 1: BASIC DESCRIPTIVE STATISTICS
# ============================================================================

print("\n📊 SECTION 1: BASIC DESCRIPTIVE STATISTICS\n")

# Question 1.1
print("Q1.1: Average final grade")
avg_grade = df['final_grade'].mean()
print(f"Answer: {avg_grade:.2f}")

# Question 1.2
print("\nQ1.2: Median study hours per week")
median_study = df['study_hours_per_week'].median()
print(f"Answer: {median_study}")

# Question 1.3
print("\nQ1.3: Standard deviation of math scores")
std_math = df['math_score'].std()
print(f"Answer: {std_math:.2f}")

# Question 1.4
print("\nQ1.4: Min and max attendance")
min_att = df['attendance_percentage'].min()
max_att = df['attendance_percentage'].max()
print(f"Answer: Min = {min_att}%, Max = {max_att}%")

# Question 1.5
print("\nQ1.5: Pass/Fail count")
pass_count = df['passed'].value_counts()
print(pass_count)

# ============================================================================
# SECTION 2: FILTERING AND CONDITIONAL ANALYSIS
# ============================================================================

print("\n\n🔍 SECTION 2: FILTERING AND CONDITIONAL ANALYSIS\n")

# Question 2.1
print("Q2.1: Students studying >15 hours/week")
high_study = df[df['study_hours_per_week'] > 15]
print(f"Answer: {len(high_study)} students")

# Question 2.2
print("\nQ2.2: Average grade for >90% attendance")
high_att = df[df['attendance_percentage'] > 90]
avg_grade_high_att = high_att['final_grade'].mean()
print(f"Answer: {avg_grade_high_att:.2f}")

# Question 2.3
print("\nQ2.3: Students with >90 in all subjects")
high_all = df[(df['math_score'] > 90) & (df['science_score'] > 90) & (df['english_score'] > 90)]
print(f"Answer: {len(high_all)} students")
print(high_all[['student_id', 'math_score', 'science_score', 'english_score']])

# Question 2.4
print("\nQ2.4: % of failed students with <70% attendance")
failed_students = df[df['passed'] == 'No']
failed_low_att = failed_students[failed_students['attendance_percentage'] < 70]
percentage = (len(failed_low_att) / len(failed_students)) * 100
print(f"Answer: {percentage:.1f}%")

# Question 2.5
print("\nQ2.5: Extracurricular participation by gender")
extra_by_gender = df[df['extracurricular'] == 'Yes'].groupby('gender').size()
print(extra_by_gender)

# ============================================================================
# SECTION 3: GROUPBY AND AGGREGATION
# ============================================================================

print("\n\n👥 SECTION 3: GROUPBY AND AGGREGATION\n")

# Question 3.1
print("Q3.1: Average final grade by gender")
gender_grades = df.groupby('gender')['final_grade'].mean()
print(gender_grades)

# Question 3.2
print("\nQ3.2: Average study hours (passed vs failed)")
study_by_pass = df.groupby('passed')['study_hours_per_week'].mean()
print(study_by_pass)

# Question 3.3
print("\nQ3.3: Average scores by age")
age_scores = df.groupby('age')[['math_score', 'science_score', 'english_score']].mean()
print(age_scores)

# Question 3.4
print("\nQ3.4: Median sleep hours by extracurricular")
sleep_by_extra = df.groupby('extracurricular')['sleep_hours'].median()
print(sleep_by_extra)

# Question 3.5
print("\nQ3.5: Pass rate by age")
age_pass_rate = df.groupby('age')['passed'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
print(age_pass_rate)

# ============================================================================
# SECTION 4: CORRELATION AND RELATIONSHIPS
# ============================================================================

print("\n\n🔗 SECTION 4: CORRELATION AND RELATIONSHIPS\n")

# Question 4.1
print("Q4.1: Correlation between study hours and final grade")
corr_study_grade = df['study_hours_per_week'].corr(df['final_grade'])
print(f"Answer: {corr_study_grade:.3f}")

# Question 4.2
print("\nQ4.2: Strongest correlation with final_grade")
correlations = df[['study_hours_per_week', 'attendance_percentage', 'previous_grade', 
                    'sleep_hours', 'math_score', 'science_score', 'english_score']].corrwith(df['final_grade'])
print(correlations.sort_values(ascending=False))
print(f"\nStrongest: {correlations.abs().idxmax()} ({correlations.abs().max():.3f})")

# Question 4.3
print("\nQ4.3: Correlation between sleep and final grade")
corr_sleep_grade = df['sleep_hours'].corr(df['final_grade'])
print(f"Answer: {corr_sleep_grade:.3f}")

# Question 4.4
print("\nQ4.4: Correlation matrix")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()
print(corr_matrix.round(2))

# Question 4.5
print("\nQ4.5: Covariance between attendance and final grade")
cov_att_grade = df['attendance_percentage'].cov(df['final_grade'])
print(f"Answer: {cov_att_grade:.2f}")

# ============================================================================
# SECTION 5: ADVANCED STATISTICS
# ============================================================================

print("\n\n📈 SECTION 5: ADVANCED STATISTICS\n")

# Question 5.1
print("Q5.1: IQR for final_grade")
Q1 = df['final_grade'].quantile(0.25)
Q3 = df['final_grade'].quantile(0.75)
IQR = Q3 - Q1
print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")

# Question 5.2
print("\nQ5.2: Outliers in math_score using IQR")
Q1_math = df['math_score'].quantile(0.25)
Q3_math = df['math_score'].quantile(0.75)
IQR_math = Q3_math - Q1_math
lower_bound = Q1_math - 1.5 * IQR_math
upper_bound = Q3_math + 1.5 * IQR_math
outliers = df[(df['math_score'] < lower_bound) | (df['math_score'] > upper_bound)]
print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
print(f"Number of outliers: {len(outliers)}")

# Question 5.3
print("\nQ5.3: Coefficient of variation for study hours")
mean_study = df['study_hours_per_week'].mean()
std_study = df['study_hours_per_week'].std()
cv = (std_study / mean_study) * 100
print(f"Answer: {cv:.2f}%")

# Question 5.4
print("\nQ5.4: Skewness of final_grade")
skewness = df['final_grade'].skew()
print(f"Answer: {skewness:.3f}")
print(f"Interpretation: {'Right-skewed' if skewness > 0 else 'Left-skewed' if skewness < 0 else 'Symmetric'}")

# Question 5.5
print("\nQ5.5: Students with |z-score| > 2")
mean_grade = df['final_grade'].mean()
std_grade = df['final_grade'].std()
df['z_score'] = (df['final_grade'] - mean_grade) / std_grade
extreme_students = df[abs(df['z_score']) > 2]
print(f"Number of students: {len(extreme_students)}")
print(extreme_students[['student_id', 'final_grade', 'z_score']].sort_values('z_score'))

# ============================================================================
# SECTION 6: PERCENTILES AND RANKING
# ============================================================================

print("\n\n🏆 SECTION 6: PERCENTILES AND RANKING\n")

# Question 6.1
print("Q6.1: 90th percentile of final_grade")
p90 = df['final_grade'].quantile(0.90)
print(f"Answer: {p90:.2f}")

# Question 6.2
print("\nQ6.2: Top 5 students by ranking")
df['rank'] = df['final_grade'].rank(ascending=False, method='min')
top5 = df.nsmallest(5, 'rank')[['student_id', 'final_grade', 'rank']]
print(top5)

# Question 6.3
print("\nQ6.3: Percentile for score of 82")
percentile = (df['final_grade'] < 82).sum() / len(df) * 100
print(f"Answer: {percentile:.1f}th percentile")

# Question 6.4
print("\nQ6.4: Study hours quartiles")
df['study_quartile'] = pd.qcut(df['study_hours_per_week'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
print(df['study_quartile'].value_counts().sort_index())

# Question 6.5
print("\nQ6.5: Top 10% by attendance")
top_10_percentile = df['attendance_percentage'].quantile(0.90)
top_attendance = df[df['attendance_percentage'] >= top_10_percentile]
print(f"Threshold: {top_10_percentile}%")
print(f"Number of students: {len(top_attendance)}")

# ============================================================================
# SECTION 7: CROSS-TABULATION AND PIVOT TABLES
# ============================================================================

print("\n\n📊 SECTION 7: CROSS-TABULATION AND PIVOT TABLES\n")

# Question 7.1
print("Q7.1: Crosstab gender vs extracurricular")
crosstab1 = pd.crosstab(df['gender'], df['extracurricular'])
print(crosstab1)

# Question 7.2
print("\nQ7.2: Pivot table gender vs passed")
pivot1 = df.pivot_table(values='final_grade', index='gender', columns='passed', aggfunc='mean')
print(pivot1)

# Question 7.3
print("\nQ7.3: Pass rate by gender and extracurricular")
pass_rate = df.groupby(['gender', 'extracurricular'])['passed'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
print(pass_rate)

# Question 7.4
print("\nQ7.4: Pivot study hours by age and gender")
pivot2 = df.pivot_table(values='study_hours_per_week', index='age', columns='gender', aggfunc='mean')
print(pivot2)

# Question 7.5
print("\nQ7.5: Crosstab age vs passed with percentages")
crosstab2 = pd.crosstab(df['age'], df['passed'], normalize='index') * 100
print(crosstab2.round(1))

# ============================================================================
# SECTION 8: COMPLEX ANALYSIS
# ============================================================================

print("\n\n🎯 SECTION 8: COMPLEX ANALYSIS\n")

# Question 8.1
print("Q8.1: Top factors differentiating passed vs failed")
comparison = df.groupby('passed')[['study_hours_per_week', 'attendance_percentage', 
                                    'previous_grade', 'sleep_hours']].mean()
print(comparison)
difference = comparison.loc['Yes'] - comparison.loc['No']
print("\nDifference (Passed - Failed):")
print(difference.sort_values(ascending=False))

# Question 8.2
print("\nQ8.2: Performance categories")
df['performance_category'] = pd.cut(df['final_grade'], 
                                     bins=[0, 60, 80, 100], 
                                     labels=['Low', 'Medium', 'High'])
print(df['performance_category'].value_counts())

# Question 8.3
print("\nQ8.3: Study hours sweet spot")
df['study_range'] = pd.cut(df['study_hours_per_week'], 
                            bins=[0, 10, 15, 20, 30], 
                            labels=['0-10', '10-15', '15-20', '20+'])
study_analysis = df.groupby('study_range')['final_grade'].agg(['mean', 'count'])
print(study_analysis)

# Question 8.4
print("\nQ8.4: Grade difference (extracurricular vs none)")
extra_yes = df[df['extracurricular'] == 'Yes']['final_grade'].mean()
extra_no = df[df['extracurricular'] == 'No']['final_grade'].mean()
pct_diff = ((extra_yes - extra_no) / extra_no) * 100
print(f"With extracurricular: {extra_yes:.2f}")
print(f"Without extracurricular: {extra_no:.2f}")
print(f"Percentage difference: {pct_diff:.1f}%")

# Question 8.5
print("\nQ8.5: Underperforming students")
df['grade_change'] = df['previous_grade'] - df['final_grade']
underperformers = df[df['grade_change'] > 10]
print(f"Number of underperformers: {len(underperformers)}")
print("\nCommon characteristics:")
print(underperformers[['study_hours_per_week', 'attendance_percentage', 'sleep_hours']].mean())

# ============================================================================
# SECTION 9: DATA QUALITY AND VALIDATION
# ============================================================================

print("\n\n🔍 SECTION 9: DATA QUALITY AND VALIDATION\n")

# Question 9.1
print("Q9.1: Missing values")
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing: {missing.sum()}")

# Question 9.2
print("\nQ9.2: Verify final_grade calculation")
df['calculated_avg'] = df[['math_score', 'science_score', 'english_score']].mean(axis=1)
df['difference'] = abs(df['final_grade'] - df['calculated_avg'])
print(f"Max difference: {df['difference'].max():.2f}")
print(f"Average difference: {df['difference'].mean():.2f}")

# Question 9.3
print("\nQ9.3: Validate attendance range")
invalid_att = df[(df['attendance_percentage'] < 0) | (df['attendance_percentage'] > 100)]
print(f"Invalid attendance records: {len(invalid_att)}")

# Question 9.4
print("\nQ9.4: Duplicate student IDs")
duplicates = df['student_id'].duplicated().sum()
print(f"Duplicate IDs: {duplicates}")

# Question 9.5
print("\nQ9.5: Inconsistent pass/fail data")
inconsistent = df[(df['passed'] == 'No') & (df['final_grade'] >= 60)]
print(f"Inconsistent records: {len(inconsistent)}")
if len(inconsistent) > 0:
    print(inconsistent[['student_id', 'final_grade', 'passed']])

# ============================================================================
# SECTION 10: BONUS CHALLENGES
# ============================================================================

print("\n\n🚀 SECTION 10: BONUS CHALLENGES\n")

# Question 10.1
print("Q10.1: Prediction score")
df['prediction_score'] = (df['study_hours_per_week'] * 2 + 
                          df['attendance_percentage'] + 
                          df['previous_grade'] * 0.5)
df['predicted_pass'] = df['prediction_score'].apply(lambda x: 'Yes' if x > 150 else 'No')
accuracy = (df['predicted_pass'] == df['passed']).sum() / len(df) * 100
print(f"Prediction accuracy: {accuracy:.1f}%")

# Question 10.2
print("\nQ10.2: Learning efficiency")
df['efficiency'] = df['final_grade'] / df['study_hours_per_week']
top_efficient = df.nlargest(5, 'efficiency')[['student_id', 'final_grade', 'study_hours_per_week', 'efficiency']]
print("Most efficient learners:")
print(top_efficient)

# Question 10.3
print("\nQ10.3: Multi-level groupby")
multi_group = df.groupby(['gender', 'extracurricular', 'age'])['final_grade'].mean()
print(multi_group)

# Question 10.4
print("\nQ10.4: Rolling average")
df_sorted = df.sort_values('student_id')
df_sorted['rolling_avg'] = df_sorted['final_grade'].rolling(window=5, min_periods=1).mean()
print(df_sorted[['student_id', 'final_grade', 'rolling_avg']].head(10))

# Question 10.5
print("\nQ10.5: Comprehensive comparison report")
report = df.groupby('passed').agg({
    'study_hours_per_week': ['mean', 'median', 'std'],
    'attendance_percentage': ['mean', 'median', 'std'],
    'sleep_hours': ['mean', 'median'],
    'math_score': ['mean', 'std'],
    'science_score': ['mean', 'std'],
    'english_score': ['mean', 'std'],
    'final_grade': ['mean', 'median', 'std', 'min', 'max']
})
print(report.round(2))

print("\n\n" + "="*80)
print("✅ ALL SOLUTIONS COMPLETE!")
print("="*80)
print("\nHow did you do? Keep practicing!")
