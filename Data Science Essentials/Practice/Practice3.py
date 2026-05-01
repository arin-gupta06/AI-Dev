"""
================================================================================
                    DATA SCIENCE - CUSTOMER ANALYSIS PIPELINE
================================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as m


# ============================================================================
# 1. DATA LOADING
# ============================================================================

try:
    df = pd.read_csv("D:\\AI DEV\\Data Science Essentials\\Dataset\\Synthetic_Dataset_Dirty.csv")
    print("\n✓ Data loaded successfully\n")
except Exception as e:
    print(f"\n✗ Error loading data: {e}\n")

copy_df = df.copy()
print(copy_df.head(10))


# ============================================================================
# 2. DATA EXPLORATION & INSIGHTS
# ============================================================================

def general_insights(df):
    """Display comprehensive dataset overview and statistics."""
    
    print("\n" + "="*80)
    print(" GENERAL INSIGHTS ".center(80))
    print("="*80 + "\n")
    
    print("-" * 80)
    print("Statistical Summary:")
    print("-" * 80)
    print(df.describe())
    
    print("\n" + "-" * 80)
    print("Data Types:")
    print("-" * 80)
    print(df.dtypes)
    
    print("\n" + "-" * 80)
    print("DataFrame Information:")
    print("-" * 80)
    df.info()
    
    print("\n" + "-" * 80)
    print("Missing Values:")
    print("-" * 80)
    print(df.isnull().sum())
    
    print("\n" + "-" * 80)
    print("Duplicate Values:")
    print("-" * 80)
    print(f"Total duplicates: {df.duplicated().sum()}")
    print("="*80 + "\n")

general_insights(copy_df)


# ============================================================================
# 3. DATA CLEANING & PREPROCESSING
# ============================================================================

# Remove duplicates
copy_df = copy_df.drop_duplicates(keep="first")
print("\nAfter Removing Duplicates: ", copy_df.duplicated().sum())



def handles_missing_values(df):
    """Handle missing values through mean imputation and mode filling."""
    
    print("\n" + "-" * 80)
    print("Processing Missing Values...")
    print("-" * 80 + "\n")

    string_col = df.select_dtypes(include=["object", "string"]).columns
    df[string_col] = df[string_col].apply(
        lambda col: col.astype("string").str.strip().str.title()
    )
    # string missing values
    for col in string_col:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])

    # numeric missing values
    nums_cols = df.select_dtypes(include="number").columns
    for nc in nums_cols:
        if nc == "User_ID": continue
        df[nc] = df[nc].fillna(df[nc].mean())



    return df
copy_df = handles_missing_values(copy_df)
print("\n✅ Cleaning done successfully!!")

def handle_outliers(df):
    nums_cols = df.select_dtypes(include="number").columns
    nums_cols = nums_cols.drop("User_ID", errors="ignore")
    for col in nums_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR


        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    print("\n✔️ Outliers are being capped successfully !!")

    return df

copy_df = handle_outliers(copy_df)



# ============================================================================
# 4. DATA VISUALIZATION
# ============================================================================


def vis_with_hist(df):
    filtered_nums_col = df.select_dtypes(include="number").columns
    filtered_nums_col = filtered_nums_col.drop("User_ID", errors="ignore")

    total_cols = len(filtered_nums_col)
    col_in_a_row = 3
    row = m.ceil(total_cols / col_in_a_row)

    plt.figure(figsize=[12,10])

    for i, col in enumerate(filtered_nums_col):
        plt.subplot(row, col_in_a_row, i + 1)
        plt.hist(df[col].dropna(), bins=20, edgecolor="black", alpha=0.7, color="green")
        plt.title(col, fontsize=11, fontweight="bold")
        plt.xticks(rotation=45)
    
    plt.tight_layout()

    # plt.show()


def vis_with_bar(df):

    filtered_str_col = df.select_dtypes(include=["object", "string"]).columns
    vis_mapping = {
        "Product_Category" : ["Purchase_Amount"],
        "Customer_Status" : ["Months_Active"],   
    }

    total_cols = sum(len(v) for v in vis_mapping.values())
    col_in_a_row = 3
    row = m.ceil(total_cols / col_in_a_row)

    plot_index = 1

    plt.figure(figsize=[12,10])

    for tc in vis_mapping:
        if tc not in filtered_str_col: continue
        for nc in vis_mapping[tc]:
            plt.subplot(row, col_in_a_row, plot_index)

            grouped = df.groupby(tc)[nc].mean()
            grouped.sort_values().plot(kind="bar", color="coral", edgecolor="black", alpha=0.8)

            plt.title(f"{nc} by {tc}", fontsize=11, fontweight="bold")
            plt.xlabel(tc)
            plt.ylabel(f"Avg {nc}")
            plt.xticks(rotation=45)

            plot_index += 1
    plt.tight_layout()
    # plt.show()

def vis_with_crosstab(df):
    Months = pd.to_datetime(df["Transaction_Date"]).dt.month_name()
    ct = pd.crosstab(
        Months, df["Region"], values=df["Satisfaction_Score"], aggfunc="var", normalize="index"
    )

    ct.plot(kind="bar",fontsize=11)
    plt.title("Product Category vs Review", fontsize=12, fontweight="bold")
    plt.xlabel("Product Category")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)

def visualize(df):
  vis_with_hist(copy_df)

  vis_with_bar(copy_df)
  vis_with_crosstab(df)

  plt.show()

visualize(copy_df)
    