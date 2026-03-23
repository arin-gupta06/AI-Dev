import pandas as pd
import numpy as np

dataset = {
    "Name": ["Alice", "Bob", np.nan, "David", "Eva"],
    "Age": [24, 30, 22, np.nan, 28],
    "City": [np.nan, "Los Angeles", "Chicago", "Houston", "Phoenix"],
    "Salary": [70000, 65000, 72000, np.nan, 64000]
}

data = pd.DataFrame(dataset)
print("Original DataFrame:\n", data)

# Handling missing values

data["Age"] = data["Age"].fillna(data["Age"].mean())
data["City"] = data["City"].fillna("Unknown")
data["Salary"] = data["Salary"].interpolate()
# print("DataFrame after handling missing values:\n", data)

data = data.rename(columns = {"Name": "Employee_name", "Salary": "Annual_salary"})

print("DataFrame after renaming columns:\n", data)