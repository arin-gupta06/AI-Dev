#https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
import pandas as pd

#load the dataset

df = pd.read_csv("Practice\Pandas\iris.csv")
# print("First few rows of datasets: \n", df.tail(10))
# print(df.info())
# print(df.describe())
filtered_rows = df[(df["sepal width (cm)"] > 3) & (df["species"] == "setosa")]
print("New Rows: \n", filtered_rows)

