# Let's do another Exploratory Data Analysis
import numpy as np
import pandas as pd
# let's load the data set
dataset = pd.read_csv("F://transfer3//AI DEV//Data Science Essentials//Dataset//Coffee_Stats.csv")
insights = {}
if dataset is not None:
   print("Dataset loaded succesfully.")
else:
   print("Dataset not loaded yet !!.")

# Let's build the Dataframe for the dataset

data = pd.DataFrame(dataset)

def divider(title):
    print("\n" + "█" * 80)
    print(f"  📊 {title}")
    print("█" * 80)
    
def small_divider(title):
    print("\n" + "▬" * 80)
    print(f"  ► {title}")
    print("▬" * 80)


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
    divider("2️⃣  UNIVARIANTS INSIGHTS - INDIVIDUAL COLUMN ANALYSIS")
    for cols in df.select_dtypes(include = [np.number]).columns:
        small_divider(f"📊 {cols.upper()} STATISTICS")
        print(f"\n  📈 DISTRIBUTION METRICS:")
        print(f"     • Mean (Average):        {df[cols].mean():>12.2f}")
        print(f"     • Median (Middle):       {df[cols].median():>12.2f}")
        print(f"     • Std Deviation:         {df[cols].std():>12.2f}")
        
        print(f"\n  🔄 SHAPE METRICS:")
        print(f"     • Skewness:              {df[cols].skew():>12.2f}  (Symmetry)")
        print(f"     • Kurtosis:              {df[cols].kurt():>12.2f}  (Tail Heaviness)")
        
        print(f"\n  💡 INTERPRETATION:")
        if abs(df[cols].skew()) < 0.5:
            skew_msg = "Fairly Symmetric"
        elif df[cols].skew() > 0:
            skew_msg = "Right-skewed (Long tail right)"
        else:
            skew_msg = "Left-skewed (Long tail left)"
        print(f"     • Distribution Shape: {skew_msg}")

def Bivariants_comp(df):
    divider("3️⃣  BIVARIANTS INSIGHTS - MONTHLY TRENDS")
    months_lists = df.select_dtypes(include = [np.number]).columns
    for mon in months_lists:
        small_divider(f"📅 {mon.upper()} MONTHLY OVERVIEW")
        mean_val = df[mon].mean()
        std_val = df[mon].std()
        median_val = df[mon].median()
        
        print(f"\n  📊 DISTRIBUTION METRICS:")
        print(f"     • Average Sales:        {mean_val:>12.2f} units")
        print(f"     • Median Sales:          {median_val:>12.2f} units")
        print(f"     • Variability (Std):     {std_val:>12.2f} units")
        
        print(f"\n  🎯 PERFORMANCE ANALYSIS:")
        cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
        print(f"     • Coefficient of Variation: {cv:>6.2f}%")
        if cv < 20:
            stability = "✅ Very Stable"
        elif cv < 50:
            stability = "⚠️  Moderately Stable"
        else:
            stability = "❌ Highly Variable"
        print(f"     • Stability Level: {stability}")

    


# Let's see the correlation analysis of the values in the dataset

def correlation_analysis(df):
    divider("4️⃣  CORRELATION ANALYSIS - BRAND RELATIONSHIPS")
    months_col = df.select_dtypes(include = [np.number]).columns
    corr = df[months_col].corr()
    
    small_divider("🔗 CORRELATION MATRIX")
    print("\n  Value Range: -1 (Negative) to +1 (Positive)")
    print("  • Near +1: Brands move together (positive relationship)")
    print("  • Near  0: No relationship")
    print("  • Near -1: Brands move oppositely\n")
    print(corr.to_string())
    
    small_divider("📈 KEY INSIGHTS")
    # Find highest and lowest correlations (excluding diagonal)
    corr_unstacked = corr.unstack()
    corr_unstacked = corr_unstacked[corr_unstacked != 1.0]  # Remove self-correlation
    
    print(f"\n  💪 Strongest Positive Relationships:")
    top_corr = corr_unstacked.nlargest(3)
    for idx, (pair, value) in enumerate(top_corr.items(), 1):
        print(f"     {idx}. {pair[0]} ↔ {pair[1]}: {value:.3f}")
    
    return corr
    


def final_summary(df):
    divider("5) FINAL SUMMARY OF THE DATASET")
    months_col = df.select_dtypes(include = [np.number]).columns
    coffee_brands_group = df.groupby("coffee_name")
    coffee_brands_list = df["coffee_name"].to_list()
    new_df = df.set_index("coffee_name")
    deviate = new_df[months_col].std(axis=1)
   
    for mon in months_col:
        brand_sales = coffee_brands_group[mon].mean()

        insights[mon] = brand_sales.to_dict()
        max_brand_value = brand_sales.max()
        max_brand = brand_sales.idxmax()
        min_brand_value = brand_sales.min()
        min_brand = brand_sales.idxmin()

        divider(f"{mon} MONTHLY SALES BREAKDOWN")
        
        total_sale = 0
        max_share = 0.0
        min_share = 0.0
        
        print("\n  ☕ BRAND SALES DETAILS:")
        print("  " + "─" * 50)
        for brand, sale in brand_sales.items():
            print(f"    • {brand:<25} → {sale:>10.2f} units")
            total_sale += sale
        print("  " + "─" * 50)
        
        max_share = (max_brand_value / total_sale) * 100
        min_share = (min_brand_value / total_sale) * 100
        
        small_divider("📈 TOP PERFORMER")
        print(f"    🏆 {max_brand.upper()}")
        print(f"    Sales: {max_brand_value:.2f} units")
        print(f"    Market Share: {max_share:.2f}%")
        
        small_divider("📉 LOWEST PERFORMER")
        print(f"    ⚠️  {min_brand.upper()}")
        print(f"    Sales: {min_brand_value:.2f} units")
        print(f"    Market Share: {min_share:.2f}%")
        
        small_divider("📊 MONTHLY SUMMARY")
        print(f"    Total {mon} Sales: {total_sale:.2f} units")
        print(f"    Average per Brand: {total_sale/len(brand_sales):.2f} units")
    small_divider("⭐ YEARLY CONSISTENCY ANALYSIS")
    print(f"\n  🎯 MOST CONSISTENT BRAND (Most Stable Sales):")
    print(f"     Brand: {deviate.idxmin().upper()}")
    print(f"     Variability Score: {deviate.min():.2f}")
    print(f"     → Best for predictable planning & inventory")
    
    print(f"\n  ⚡ MOST VOLATILE BRAND (Most Unpredictable Sales):")
    print(f"     Brand: {deviate.idxmax().upper()}")
    print(f"     Variability Score: {deviate.max():.2f}")
    print(f"     → Higher risk, seasonal fluctuations detected")
    
    small_divider("📌 YEAR-END INSIGHTS")
    print(f"\n  ✓ Total Brands Analyzed: {len(deviate)}")
    print(f"  ✓ Analysis Period: Full Year")
    print(f"  ✓ Consistency Range: {deviate.min():.2f} to {deviate.max():.2f}")
    print(f"  ✓ Average Variability: {deviate.mean():.2f}")




Univariant_comp(data)
Bivariants_comp(data)
correlation_analysis(data)

final_summary(data)

