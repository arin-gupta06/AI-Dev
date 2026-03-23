import pandas as pd
import numpy as np
import calendar
#loading dataset
data = pd.read_csv("Pandas/index_1.csv")
dataset = pd.DataFrame(data)
# print(dataset)
 
 
# Handling the missing values:
def filling_up_missing_values(row):
    if(pd.isna(row["card_number/cash"]) and pd.notnull(row["money"]) and row["payment_mode"] == "cash"):
        row["card_number/cash"] = f"{row["money"]} is paid with cash."
        return row["card_number/cash"]
    else:
        return row["card_number/cash"]
dataset["card_number/cash"] = dataset.apply(filling_up_missing_values, axis=1)

# Seeing the changes
cash_transactions = dataset[dataset["payment_mode"] =="cash"]
# print(cash_transactions) 

       
    
#Statistical Analysis for the given dataset:

#Total sale of the year:
total_sale_in_yr = dataset["money"].sum()
# print("Total sale of the year: ",total_sale_in_yr)

#Total sale of the particular month
month = pd.to_datetime(dataset["date"], format= "%d-%m-%Y").dt.month
month_name = month.map(lambda x: calendar.month_name[x])
# print(month_name)
grouped = dataset.groupby(month_name)["money"].agg(
    Total_Revenue = "sum",
    Total_Sale = "count"
)
# print("Total revenue of the month: \n",grouped)

# Pivot_Table

pivot = dataset.pivot_table(
    values= "money",
    index="coffee_name",
    columns="payment_mode",
    aggfunc= "sum",
    fill_value= 0
)
print("Statistical analysis of the dataset (by mode of payment): \n",pivot)

# by month name:
dataset["month"] = month_name
monthly_stats = dataset.pivot_table(
    values= "money",
    index = "coffee_name",
    columns = "month",
    aggfunc= "sum",
    fill_value= 0,
    margins=True,
    margins_name= "Total"
)
print("Statistical analysis of the dataset (by monthly basis): \n",monthly_stats)
monthly_stats.to_csv("Pandas/Coffee_Stats.csv")