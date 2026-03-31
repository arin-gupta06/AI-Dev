# Let's do another Exploratory Data Analysis
import numpy as np
import pandas as pd
# let's load the data set
dataset = pd.read_csv("F://transfer3//AI DEV//Data Science Essentials//Pandas//Coffee_Stats.csv")

if dataset is not None:
   print("Dataset loaded succesfully.")
else:
   print("Dataset not loaded yet !!.")

# Let's build the Dataframe for the dataset

data = pd.DataFrame(dataset)

def divider(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# Let's see the general insights of the given dataset

def general_insights(df):
    divider("1) DATA OVERVIEW")
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nDescriptive statistics:\n", df.describe())
    print("\nData types:\n", df.dtypes)

# Let's examine the Different types of Variants of the dataset

def Univariant_comp(df):
    divider("2) UNIVARIANTS INSIGHTS")
    for cols in df.select_dtypes(include = [np.number]).columns:
        print(f"\n{cols.upper()}")
        print("-" * 40)
        print(f"Mean of {cols}:\n", df[cols].mean())
        print(f"Median of {cols}:\n", df[cols].median())
        print(f"Standard Deviation of {cols}:\n", df[cols].std())
        print(f"Skewness of {cols}:\n", df[cols].skew())
        print(f"Kurtosis of {cols}:\n", df[cols].kurt())

def Bivariants_comp(df):
    divider("3) BIVARIANTS INSIGHTS")
    months_lists = df.select_dtypes(include = [np.number]).columns
    for mon in months_lists:
        print(f"Mean of the month {mon} : \n", df[mon].mean())
        print(f"\n Standard Deviation of the month {mon} : \n", df[mon].std())
        print(f"\n Median of the month {mon} : \n", df[mon].median())

    


# Let's see the correlation analysis of the values in the dataset

def correlation_analysis(df):
    divider("4) CORRELATION OF COFFEE BRANDS IN DIFFERENT MONTHS.")
    months_col = df.select_dtypes(include = [np.number]).columns
    corr = df[months_col].corr()
    print(corr)
    return corr
    


def final_summary(df):
    divider("5) FINAL SUMMARY OF THE DATASET")
    months_col = df.select_dtypes(include = [np.number]).columns
    mean_sales = df[months_col].mean().sort_values(ascending = False)
    std_sales = df[months_col].std().sort_values()
    month_sales_grouped_mean = df.groupby("coffee_name")[months_col].mean()

    print("Top average sale of the month: \n", mean_sales.index[0], "=", round(mean_sales.iloc[0], 2))
    print("lowest average sale of the month: \n", mean_sales.index[-1], "=", round(mean_sales.iloc[-1], 2))
    print("Most consistent month sale: \n", std_sales.index[0], "=", round(mean_sales.iloc[0], 2))




# general_insights(data)
# correlation_analysis(data)

# print(data.columns[1:])

# correlation_analysis(data)
for cof1, cof2 in data["coffee_name"]:
    print(cof1, cof2)
