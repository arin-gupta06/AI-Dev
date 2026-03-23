import pandas as pd

data = {
    "Name": ["A", "B", "C", "D", "E", "A", "C", "B", "E", "B", "D"],
    "Age": [24, 30, 22, 35, 28, 24, 22, 30, 28, 30, 35],
    "Score" : [45, 70, 78, 18, 92, 75, 62, 90, 42, 90, 68]
}
dataset = pd.DataFrame(data)
print("Original Dataset: \n",dataset)

#grouping data by Name
grouped = dataset.groupby("Name").mean()
# print(grouped)
print()
# Calculating the summary statistics of group data
statistics = dataset.groupby("Name").agg(
    {"Score": ["mean","min","max"],
     "Age": ["mean","min","max"]
     })

new_stats = pd.DataFrame(statistics)
print("The statistics of the given dataset are as follows: \n\n",new_stats)