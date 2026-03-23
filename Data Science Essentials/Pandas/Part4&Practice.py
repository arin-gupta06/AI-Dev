import pandas as pd

# Loading the dataset:
# Note: pd.read_csv() already returns a DataFrame, no need for pd.DataFrame()
dataset = pd.read_csv("Pandas/student_performance.csv")  # Use forward slash for cross-platform compatibility

#Section1:
print("="*80)
#Question 1.1: What is the average final grade of all students?
print("Average final grade of all students: \n",dataset["final_grade"].mean())

# Question 1.2: What is the median study hours per week?
print("Median of study hours per week: \n ", dataset["study_hours_per_week"].median())


# Question 1.3: What is the standard deviation of math scores?
print("The Standard Deviation of the math score is: \n", dataset["math_score"].std())

# Question 1.4: What are the minimum and maximum attendance percentages?
print(f"The Minimum attendance percentage is: {dataset['attendance_percentage'].min()}%")
print(f"The Maximum attendance percentage is: {dataset['attendance_percentage'].max()}%")

# Question 1.5: How many students passed and how many failed?
print("Total numbers of passed and fails are: \n")
print(dataset["passed"].value_counts())
print("="*80)


#Section2:
# Question 2.1: How many students study more than 15 hours per week?
more_than_15_hours = dataset[dataset["study_hours_per_week"] >= 15]
print("The total number of students who studied more than 15 hours are: \n", len(more_than_15_hours))

# Question 2.2: What is the average final grade of students with attendance > 90%?
filter_90 = dataset[dataset["attendance_percentage"] > 90]
print("The average final grade for the students whose attendance is more than 90 is: \n", filter_90["final_grade"].mean())

# Question 2.3: Find all students who scored above 90 in all three subjects
most_scored = dataset[(dataset["math_score"] > 90) & (dataset["science_score"] > 90) & (dataset["english_score"] > 90)]
print("\nStudents who scored >90 in all subjects:")
print(most_scored[["student_id", "math_score", "science_score", "english_score", "final_grade"]])


# Question 2.4: What percentage of students who failed had attendance below 70%?
failed = dataset[dataset["passed"] =="No"]
attendance_70 = failed[failed["attendance_percentage"] < 70]
percentage = (len(attendance_70)/len(dataset["student_id"]))* 100
print("The percentage of failed students with attendance of less than 70 %: \n ",percentage)

# Question 2.5: How many male vs female students participate in extracurricular activities?
participated_extracurricular = dataset[dataset["extracurricular"] == "Yes"]
gender_counts = participated_extracurricular["gender"].value_counts()
print("The number of males and females participated in extracurricular activities are: \n", gender_counts)

