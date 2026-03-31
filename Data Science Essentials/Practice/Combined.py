# Combined practice of numpy and pandas
import numpy as np
import pandas as pd


# loading dataset
data = pd.read_csv("F://transfer3//AI DEV//Data Science Essentials//Pandas//student_performance.csv")

# Let's perform exploratory Data Analysis for this dataset
dataset = pd.DataFrame(data)


def divider(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# Let's get to the insights of the dataset
def print_details(df):
    divider("1) DATA OVERVIEW")
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nDescriptive statistics:\n", df.describe())
    print("\nData types:\n", df.dtypes)




# Let's examine the Different of Variants of the scores columns

scores_list = [c for c in dataset.columns if "score" in c]



def Univariant_comp(df):
    divider("2) UNIVARIATE ANALYSIS (SUBJECT-WISE)")
    for sub in scores_list:
        print(f"\n{sub.upper()}")
        print("-" * 40)
        print(f"Mean of {sub}:\n", df[sub].mean())
        print(f"Median of {sub}:\n", df[sub].median())
        print(f"Standard Deviation of {sub}:\n", df[sub].std())
        print(f"Skewness of {sub}:\n", df[sub].skew())
        print(f"Kurtosis of {sub}:\n", df[sub].kurt())


def Bivariant_comp(df):
    divider("3) BIVARIATE ANALYSIS (GENDER-WISE)")
    # Group ke basis pe comparison
    print("Mean of groups:\n", df.groupby("gender")[scores_list].mean())
    print("\nMedian of the group:\n", df.groupby("gender")[scores_list].median())
    print("\nStandard Deviation:\n", df.groupby("gender")[scores_list].std())

# Let's do Correlation analysis of the scores obtained
def correlation_analysis(df):
    divider("4) CORRELATION ANALYSIS")
    corr = df[scores_list].corr()
    print("Correlation analysis:\n", corr)
    return corr


def quick_insights(df, corr):
    divider("5) QUICK INSIGHTS")
    mean_scores = df[scores_list].mean().sort_values(ascending=False)
    std_scores = df[scores_list].std().sort_values()
    gender_means = df.groupby("gender")[scores_list].mean()

    print("Top average subject:", mean_scores.index[0], "=", round(mean_scores.iloc[0], 2))
    print("Lowest average subject:", mean_scores.index[-1], "=", round(mean_scores.iloc[-1], 2))
    print("Most consistent subject (lowest std):", std_scores.index[0], "=", round(std_scores.iloc[0], 2))

    if "Male" in gender_means.index and "Female" in gender_means.index:
        gap = gender_means.loc["Male"] - gender_means.loc["Female"]
        print("\nMale - Female mean gap by subject:\n", gap.round(2))

    corr_pairs = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack().sort_values(ascending=False)
    if len(corr_pairs) > 0:
        pair, value = corr_pairs.index[0], corr_pairs.iloc[0]
        print("\nStrongest positive correlation:", pair, "=", round(value, 3))


divider("EDA REPORT START")
print_details(dataset)
Univariant_comp(dataset)
Bivariant_comp(dataset)
corr_df = correlation_analysis(dataset)
quick_insights(dataset, corr_df)
divider("EDA REPORT END")

     