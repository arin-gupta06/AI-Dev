import pandas as pd
import numpy as np

dataset_01 = pd.DataFrame({
    "ID": [1, 2, 3, 4, 5],
    "Score": [85, 90, np.nan, 88, 92],
})

dataset_02 = pd.DataFrame({
    "ID": [3, 4, 5, 6, 7],
    "Grade": ['B', 'A', 'A', 'C', 'B'],
})

print("Dataset 01:\n", dataset_01)
print("Dataset 02:\n", dataset_02)

#Merging 
merged = pd.merge(dataset_01, dataset_02, on = "ID", how = "outer")
print("Merged DataFrame:\n", merged)

# Handling missing values
merged["Grade"] = merged.apply(
    lambda row: "A" if row["Score"] >= 90 else "B" if row["Score"] >= 80 else "C",
    axis=1
)
merged["Score"] = merged["Score"].fillna(merged["Score"].mean())  # Fill missing scores with mean
print("DataFrame after handling missing values:\n", merged)


# Modifying Score values for specific IDs FIRST
merged["Score"] = merged.apply(
    lambda row: 60 if row["ID"] == 3 else (70 if row["ID"] == 6 else row["Score"]), axis = 1
)
print("After modifying Score values:\n", merged)

# Adding Result column AFTER modifying scores
merged["Result"] = merged.apply(
    lambda row: "Passed" if row["Score"] >= 85 else "Fail", axis = 1
)   
print("After adding Result column:\n", merged)