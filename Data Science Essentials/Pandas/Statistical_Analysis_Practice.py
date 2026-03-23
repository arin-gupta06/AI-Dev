"""
STATISTICAL ANALYSIS PRACTICE QUESTIONS
Student Performance Dataset

Instructions:
1. Use the 'student_performance.csv' dataset
2. Try to solve each question before checking the solutions file
3. Practice writing clean, readable pandas code
4. Focus on understanding the statistical concepts, not just getting answers
"""

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('student_performance.csv')

print("Dataset loaded successfully!")
print(f"Total records: {len(df)}")
print("\n" + "="*80)

# ============================================================================
# SECTION 1: BASIC DESCRIPTIVE STATISTICS (Easy)
# ============================================================================

print("\n📊 SECTION 1: BASIC DESCRIPTIVE STATISTICS")
print("="*80)

# Question 1.1: What is the average final grade of all students?
print("\nQ1.1: Calculate the average final grade of all students")
# YOUR CODE HERE


# Question 1.2: What is the median study hours per week?
print("\nQ1.2: Find the median study hours per week")
# YOUR CODE HERE


# Question 1.3: What is the standard deviation of math scores?
print("\nQ1.3: Calculate the standard deviation of math scores")
# YOUR CODE HERE


# Question 1.4: What are the minimum and maximum attendance percentages?
print("\nQ1.4: Find the minimum and maximum attendance percentages")
# YOUR CODE HERE


# Question 1.5: How many students passed and how many failed?
print("\nQ1.5: Count the number of students who passed and failed")
# YOUR CODE HERE


# ============================================================================
# SECTION 2: FILTERING AND CONDITIONAL ANALYSIS (Easy-Medium)
# ============================================================================

print("\n\n🔍 SECTION 2: FILTERING AND CONDITIONAL ANALYSIS")
print("="*80)

# Question 2.1: How many students study more than 15 hours per week?
print("\nQ2.1: Count students who study more than 15 hours per week")
# YOUR CODE HERE


# Question 2.2: What is the average final grade of students with attendance > 90%?
print("\nQ2.2: Calculate average final grade for students with >90% attendance")
# YOUR CODE HERE


# Question 2.3: Find all students who scored above 90 in all three subjects
print("\nQ2.3: Find students who scored >90 in math, science, AND english")
# YOUR CODE HERE


# Question 2.4: What percentage of students who failed had attendance below 70%?
print("\nQ2.4: Calculate percentage of failed students with <70% attendance")
# YOUR CODE HERE


# Question 2.5: How many male vs female students participate in extracurricular activities?
print("\nQ2.5: Count male and female students in extracurricular activities")
# YOUR CODE HERE


# ============================================================================
# SECTION 3: GROUPBY AND AGGREGATION (Medium)
# ============================================================================

print("\n\n👥 SECTION 3: GROUPBY AND AGGREGATION")
print("="*80)

# Question 3.1: Compare average final grades between male and female students
print("\nQ3.1: Calculate average final grade by gender")
# YOUR CODE HERE


# Question 3.2: What is the average study hours for students who passed vs failed?
print("\nQ3.2: Compare average study hours between passed and failed students")
# YOUR CODE HERE


# Question 3.3: Group by age and find the average of all score columns
print("\nQ3.3: Calculate average scores (math, science, english) for each age group")
# YOUR CODE HERE


# Question 3.4: Find the median sleep hours for students with/without extracurricular activities
print("\nQ3.4: Compare median sleep hours by extracurricular participation")
# YOUR CODE HERE


# Question 3.5: Which age group has the highest pass rate?
print("\nQ3.5: Calculate pass rate for each age group")
# YOUR CODE HERE


# ============================================================================
# SECTION 4: CORRELATION AND RELATIONSHIPS (Medium)
# ============================================================================

print("\n\n🔗 SECTION 4: CORRELATION AND RELATIONSHIPS")
print("="*80)

# Question 4.1: What is the correlation between study_hours_per_week and final_grade?
print("\nQ4.1: Calculate correlation between study hours and final grade")
# YOUR CODE HERE


# Question 4.2: Which variable has the strongest correlation with final_grade?
print("\nQ4.2: Find which variable correlates most strongly with final_grade")
# YOUR CODE HERE


# Question 4.3: Is there a correlation between sleep_hours and academic performance?
print("\nQ4.3: Calculate correlation between sleep hours and final grade")
# YOUR CODE HERE


# Question 4.4: Create a correlation matrix for all numeric columns
print("\nQ4.4: Display correlation matrix for all numeric variables")
# YOUR CODE HERE


# Question 4.5: What is the covariance between attendance_percentage and final_grade?
print("\nQ4.5: Calculate covariance between attendance and final grade")
# YOUR CODE HERE


# ============================================================================
# SECTION 5: ADVANCED STATISTICS (Medium-Hard)
# ============================================================================

print("\n\n📈 SECTION 5: ADVANCED STATISTICS")
print("="*80)

# Question 5.1: Calculate the interquartile range (IQR) for final_grade
print("\nQ5.1: Calculate Q1, Q3, and IQR for final_grade")
# YOUR CODE HERE


# Question 5.2: Identify outliers in math_score using IQR method (Q1-1.5*IQR, Q3+1.5*IQR)
print("\nQ5.2: Find outliers in math_score using IQR method")
# YOUR CODE HERE


# Question 5.3: Calculate the coefficient of variation (CV) for study_hours_per_week
print("\nQ5.3: Calculate coefficient of variation (std/mean * 100) for study hours")
# YOUR CODE HERE


# Question 5.4: What is the skewness of the final_grade distribution?
print("\nQ5.4: Calculate skewness of final_grade")
# YOUR CODE HERE


# Question 5.5: Calculate z-scores for final_grade and identify students with |z| > 2
print("\nQ5.5: Find students with final grades >2 standard deviations from mean")
# YOUR CODE HERE


# ============================================================================
# SECTION 6: PERCENTILES AND RANKING (Medium)
# ============================================================================

print("\n\n🏆 SECTION 6: PERCENTILES AND RANKING")
print("="*80)

# Question 6.1: What is the 90th percentile of final_grade?
print("\nQ6.1: Calculate the 90th percentile of final_grade")
# YOUR CODE HERE


# Question 6.2: Rank all students by their final_grade (highest = rank 1)
print("\nQ6.2: Create a ranking column and show top 5 students")
# YOUR CODE HERE


# Question 6.3: What percentile is a student with final_grade = 82?
print("\nQ6.3: Find what percentile a score of 82 represents")
# YOUR CODE HERE


# Question 6.4: Divide students into quartiles based on study_hours_per_week
print("\nQ6.4: Create quartile groups for study hours and show counts")
# YOUR CODE HERE


# Question 6.5: Find students in the top 10% by attendance_percentage
print("\nQ6.5: Identify students in top 10% of attendance")
# YOUR CODE HERE


# ============================================================================
# SECTION 7: CROSS-TABULATION AND PIVOT TABLES (Medium)
# ============================================================================

print("\n\n📊 SECTION 7: CROSS-TABULATION AND PIVOT TABLES")
print("="*80)

# Question 7.1: Create a cross-tabulation of gender vs extracurricular activities
print("\nQ7.1: Create crosstab of gender vs extracurricular participation")
# YOUR CODE HERE


# Question 7.2: Create a pivot table showing average final_grade by gender and passed status
print("\nQ7.2: Pivot table with gender as rows, passed as columns, final_grade as values")
# YOUR CODE HERE


# Question 7.3: What is the pass rate for each combination of gender and extracurricular?
print("\nQ7.3: Calculate pass rate for gender + extracurricular combinations")
# YOUR CODE HERE


# Question 7.4: Create a pivot table showing average study hours by age and gender
print("\nQ7.4: Pivot table of average study hours (age vs gender)")
# YOUR CODE HERE


# Question 7.5: Cross-tabulate age vs passed with percentages
print("\nQ7.5: Crosstab of age vs passed showing percentages")
# YOUR CODE HERE


# ============================================================================
# SECTION 8: COMPLEX ANALYSIS (Hard)
# ============================================================================

print("\n\n🎯 SECTION 8: COMPLEX ANALYSIS")
print("="*80)

# Question 8.1: Find the top 3 factors that differentiate passed vs failed students
print("\nQ8.1: Compare mean values of all numeric columns for passed vs failed")
# YOUR CODE HERE


# Question 8.2: Create a 'performance_category' (Low: <60, Medium: 60-80, High: >80)
print("\nQ8.2: Categorize students by final_grade and show distribution")
# YOUR CODE HERE


# Question 8.3: Is there a study hours "sweet spot"? Compare grades by study hour ranges
print("\nQ8.3: Group study hours into bins (0-10, 10-15, 15-20, 20+) and compare grades")
# YOUR CODE HERE


# Question 8.4: Calculate the percentage difference in final grades between students 
# with/without extracurricular activities
print("\nQ8.4: Calculate percentage difference in grades (extracurricular vs none)")
# YOUR CODE HERE


# Question 8.5: Find students who underperformed (previous_grade - final_grade > 10)
print("\nQ8.5: Identify underperforming students and find common characteristics")
# YOUR CODE HERE


# ============================================================================
# SECTION 9: DATA QUALITY AND VALIDATION (Medium)
# ============================================================================

print("\n\n🔍 SECTION 9: DATA QUALITY AND VALIDATION")
print("="*80)

# Question 9.1: Check for any missing values in the dataset
print("\nQ9.1: Count missing values in each column")
# YOUR CODE HERE


# Question 9.2: Verify that final_grade is approximately the average of the three subject scores
print("\nQ9.2: Calculate average of math, science, english and compare with final_grade")
# YOUR CODE HERE


# Question 9.3: Check if all attendance percentages are between 0 and 100
print("\nQ9.3: Validate attendance_percentage values are in valid range")
# YOUR CODE HERE


# Question 9.4: Identify any duplicate student_ids
print("\nQ9.4: Check for duplicate student IDs")
# YOUR CODE HERE


# Question 9.5: Find students with inconsistent data (e.g., failed but grade ≥ 60)
print("\nQ9.5: Find students marked as 'No' in passed but have final_grade >= 60")
# YOUR CODE HERE


# ============================================================================
# SECTION 10: BONUS CHALLENGES (Very Hard)
# ============================================================================

print("\n\n🚀 SECTION 10: BONUS CHALLENGES")
print("="*80)

# Question 10.1: Build a simple scoring system to predict pass/fail
# Use: study_hours * 2 + attendance + previous_grade * 0.5
print("\nQ10.1: Create prediction score and compare with actual pass/fail")
# YOUR CODE HERE


# Question 10.2: Find the "efficiency score" (final_grade / study_hours_per_week)
# Who are the most efficient learners?
print("\nQ10.2: Calculate and rank students by learning efficiency")
# YOUR CODE HERE


# Question 10.3: Create a multi-level groupby: gender -> extracurricular -> age
# Show average final grades
print("\nQ10.3: Multi-level groupby analysis")
# YOUR CODE HERE


# Question 10.4: Calculate the rolling average final_grade for students sorted by student_id
# (window size = 5)
print("\nQ10.4: Calculate 5-student rolling average of final grades")
# YOUR CODE HERE


# Question 10.5: Perform a comprehensive comparison: create a report showing
# all key statistics for passed vs failed students side by side
print("\nQ10.5: Comprehensive comparison report (passed vs failed)")
# YOUR CODE HERE


print("\n\n" + "="*80)
print("🎉 PRACTICE QUESTIONS COMPLETE!")
print("="*80)
print("\nTips:")
print("- Try solving without looking at solutions first")
print("- Use pandas documentation when stuck")
print("- Focus on understanding WHY, not just HOW")
print("- Check your answers with Statistical_Analysis_Solutions.py")
