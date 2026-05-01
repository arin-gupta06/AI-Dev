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
    df = pd.read_csv("D:\\AI DEV\\Data Science Essentials\\Dataset\\Customer_Behavioral_Data.csv")
    print("\n✓ Data loaded successfully\n")
except Exception as e:
    print(f"\n✗ Error loading data: {e}\n")

copy_df = df.copy()


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
    
    # Normalize string columns
    string_cols = df.select_dtypes(include=["object","string"]).columns
    df[string_cols] = df[string_cols].apply(
        lambda col: col.astype("string").str.strip().str.title()
    )
    
    # Fill numeric columns with mean (excluding CustomerID)
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if col == "CustomerID":
            continue
        df[col] = df[col].fillna(df[col].mean())

    # Fill string columns with mode
    for col in string_cols:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])

    print("✓ Missing values handled successfully\n")
    return df

copy_df = handles_missing_values(copy_df)

# print(copy_df.isnull().sum())
def handle_outlier(df):
    filter_nums_col = df.select_dtypes(include="number").columns
    filter_nums_col = filter_nums_col.drop("CustomerID", errors="ignore")
    for c in filter_nums_col:
        Q1 = df[c].quantile(0.25)
        Q3 = df[c].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR

        df[c] = np.where(df[c] < lower_bound, lower_bound, df[c])
        df[c] = np.where(df[c] > upper_bound, upper_bound, df[c])
    
    return df
copy_df = handle_outlier(copy_df)


# ============================================================================
# 4. DATA VISUALIZATION
# ============================================================================

import matplotlib.pyplot as plt
import math

def vis_with_hist(df):
    """Generate histogram plots for all numeric columns."""
    
    filtered_cols_nums = df.select_dtypes(include="number").columns
    filtered_cols_nums = filtered_cols_nums.drop("CustomerID", errors="ignore")
    total_cols = len(filtered_cols_nums)
    cols_in_a_row = 3
    row = m.ceil(total_cols / cols_in_a_row)
    
    plt.figure(figsize=(12, 10))
    
    for i, col in enumerate(filtered_cols_nums):
        plt.subplot(row, cols_in_a_row, i + 1)
        plt.hist(df[col].dropna(), bins=20, edgecolor="black", color="steelblue", alpha=0.7)
        plt.title(col, fontsize=11, fontweight="bold")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
    
    plt.tight_layout()


def vis_with_bar(df):
    """Generate bar plots comparing categorical and numeric columns."""
    
    # Select text columns
    filtered_col_text = df.select_dtypes(include=["object", "string"]).columns
    filtered_col_text = filtered_col_text.drop(
        ["Loyalty_Member", "Email_Campaign_Response", "Discount_Availed"], errors="ignore"
    )

    # Define mapping (selective plotting)
    mapping = {
        "Product_Category": ["Purchase_Amount"],
        "Payment_Method": ["Monthly_Spending"],
        "Product_Review": ["Satisfaction_Score"]
    }

    total_plots = sum(len(v) for v in mapping.values())
    cols_in_a_row = 3
    rows = math.ceil(total_plots / cols_in_a_row)

    plt.figure(figsize=(12, 10))

    plot_index = 1

    for text_col in mapping:
        if text_col not in filtered_col_text:
            continue

        for num_col in mapping[text_col]:
            plt.subplot(rows, cols_in_a_row, plot_index)

            grouped = df.groupby(text_col)[num_col].mean()
            grouped.sort_values().plot(kind="bar", color="coral", edgecolor="black", alpha=0.8)

            plt.title(f"{num_col} by {text_col}", fontsize=11, fontweight="bold")
            plt.xlabel(text_col)
            plt.ylabel(f"Avg {num_col}")
            plt.xticks(rotation=45)

            plot_index += 1

    plt.tight_layout()

def vis_with_crosstab(df):
    color_map = {
    "Positive": "#2ecc71",
    "Neutral": "#f1c40f",
    "Negative": "#e74c3c"
    }
    ct = pd.crosstab(
        df["Product_Category"],
        df["Product_Review"],
        normalize="index",
    )
    colors = [color_map[col] for col in ct.columns]
    ct.plot(kind="bar", stacked=True, color=colors) 
    plt.title("Product Category v/s Product Review")

def visualize(df):
    """Execute all visualization plots."""
    
    print("\n" + "="*80)
    print(" GENERATING VISUALIZATIONS ".center(80))
    print("="*80 + "\n")
    
    # Histogram plot for numerical values
    print("→ Generating histogram plots...")
    vis_with_hist(df)

    # Bar chart for categorical and numeric data
    print("→ Generating bar plots...")
    vis_with_bar(df)

    print("→ Generating crosstab plots...")
    vis_with_crosstab(df)
    
    plt.show()
    print("\n✓ Visualizations complete\n")

visualize(copy_df) 