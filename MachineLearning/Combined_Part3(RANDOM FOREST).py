import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ═══════════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                       DATA LOADING & INITIALIZATION                           ║
# ║                E-COMMERCE TRANSACTIONS RAW DATA PROCESSING                    ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATASET FROM CSV FILE
# ─────────────────────────────────────────────────────────────────────────────────
dataset = pd.read_csv("D://AI DEV//Data Science Essentials//Dataset//E_Commerce_Transactions_Raw.csv")
data = pd.DataFrame(dataset)
print(f"✓ Dataset loaded: {data.shape[0]} rows × {data.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 2: DEFINE UTILITY FUNCTION FOR DATA OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────────
def general(df):
    """Display comprehensive data overview including shape, info, stats, and duplicates"""
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.isna().sum())
    print(df.duplicated().sum())

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 3: CREATE WORKING COPY OF DATA
# ─────────────────────────────────────────────────────────────────────────────────
copy_data = data.copy()
print("✓ Data copy created for processing\n")

# ═══════════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                    MISSING VALUES HANDLING - STRING COLUMNS                   ║
# ║                  Fill gaps in categorical/text data                           ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 4: HANDLE MISSING CUSTOMER_ID VALUES
# ─────────────────────────────────────────────────────────────────────────────────
missing_count = copy_data["customer_id"].isna().sum()
copy_data["customer_id"] = copy_data["customer_id"].fillna("Unknown")
print(f"✓ Customer ID: Filled {missing_count} missing values with 'Unknown'")

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 5: HANDLE MISSING CITY VALUES
# ─────────────────────────────────────────────────────────────────────────────────
missing_count = copy_data["city"].isna().sum()
most_common_city = copy_data["city"].mode().values[0]
copy_data["city"] = copy_data["city"].fillna(most_common_city)
print(f"✓ City: Filled {missing_count} missing values with mode '{most_common_city}'")

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 6: HANDLE PRODUCT_NAME - CLEANUP & MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────────
copy_data["product_name"] = copy_data["product_name"].str.lower().str.strip()
missing_count = copy_data["product_name"].isna().sum()
most_common_product = copy_data["product_name"].mode().values[0]
copy_data["product_name"] = copy_data["product_name"].fillna(most_common_product)
print(f"✓ Product Name: Lowercase & trimmed | Filled {missing_count} missing values")

# ═══════════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                    MISSING VALUES HANDLING - NUMERIC COLUMNS                  ║
# ║                  Fill gaps in rating data & create categories                 ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 7: DEFINE RATING CATEGORIZATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────────
def category_rats_stat(rating):
    """
    Convert numeric rating to categorical description
    5.0+ → Excellent | 4.0+ → Very Good | 3.0+ → Good | 
    2.0+ → Average | 1.0+ → Bad | NaN → No Rating
    """
    if pd.isna(rating):
        return "No Rating"
    elif rating >= 5.0:
        return "Excellent"
    elif rating >= 4.0:
        return "Very Good"
    elif rating >= 3.0:
        return "Good"
    elif rating >= 2.0:
        return "Average"
    elif rating >= 1.0:
        return "Bad"
    else:
        return "Invalid"

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 8: HANDLE MISSING RATING VALUES
# ─────────────────────────────────────────────────────────────────────────────────
missing_count = copy_data["rating"].isna().sum()
mean_rating = copy_data["rating"].mean()
copy_data["rating"] = copy_data["rating"].apply(lambda x: mean_rating if pd.isna(x) else x)
copy_data["rating_stat"] = copy_data["rating"].apply(category_rats_stat)
print(f"✓ Rating: Filled {missing_count} missing values with mean={mean_rating:.2f}")
print(f"✓ Rating_Stat: Created categorical column from ratings\n")

# ═══════════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                    INTELLIGENT MISSING VALUE HANDLING - DELIVERY STATUS       ║
# ║                  Fill using cross-column logic (payment_method dependency)    ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 9: DEFINE DELIVERY STATUS HANDLER (PAYMENT-BASED)
# ─────────────────────────────────────────────────────────────────────────────────
def handle_delivery_stat(df):
    """
    Fill missing delivery_status intelligently based on payment_method:
    ├─ CASH payment → "Shipped" (immediate)
    ├─ Card payment → "Pending" (processing)
    └─ Other → "Processing" (default)
    
    Uses row-wise logic (axis=1) to access both columns
    """
    def fill_delivery(row):
        if pd.notna(row["delivery_status"]):
            return row["delivery_status"]
        else:
            payment = str(row["payment_method"]).lower().strip()
            if payment == "cash":
                return "Shipped"
            elif "card" in payment:
                return "Pending"
            else:
                return "Processing"
    
    df["delivery_status"] = df.apply(fill_delivery, axis=1)
    return df

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 10: APPLY DELIVERY STATUS HANDLER
# ─────────────────────────────────────────────────────────────────────────────────
copy_data = handle_delivery_stat(copy_data)
print(f"✓ Delivery Status: Filled missing values based on payment_method")
print(f"  Distribution: {dict(copy_data['delivery_status'].value_counts())}\n")



# ═══════════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║               NUMERIC COLUMNS TYPE CONVERSION & FINAL CLEANUP                 ║
# ║              Convert data types & prepare for model training                  ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 11: UNIT PRICE COLUMN - Type Conversion & Handling Missing Values
# ─────────────────────────────────────────────────────────────────────────────────
copy_data["unit_price"] = pd.to_numeric(copy_data["unit_price"], errors="coerce")
copy_data["unit_price"] = copy_data["unit_price"].fillna(copy_data["unit_price"].median())
copy_data["unit_price"] = copy_data["unit_price"].astype("int64")

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 12: TOTAL AMOUNT COLUMN - Type Conversion & Handling Missing Values
# ─────────────────────────────────────────────────────────────────────────────────
copy_data["total_amount"] = pd.to_numeric(copy_data["total_amount"], errors="coerce")
copy_data["total_amount"] = copy_data["total_amount"].fillna(copy_data["total_amount"].median())
copy_data["total_amount"] = copy_data["total_amount"].astype("int64")

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 13: RETURN DAYS COLUMN - Type Conversion & Handling Missing Values
# ─────────────────────────────────────────────────────────────────────────────────
copy_data["return_days"] = pd.to_numeric(copy_data["return_days"], errors="coerce")
copy_data["return_days"] = copy_data["return_days"].fillna(copy_data["return_days"].mean())
copy_data["return_days"] = copy_data["return_days"].astype("int64")

# ═══════════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                      DUPLICATE DETECTION & REMOVAL                            ║
# ║                Identify and remove exact duplicate transactions               ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 14: DEFINE INTELLIGENT DUPLICATE REMOVAL FUNCTION
# ─────────────────────────────────────────────────────────────────────────────────
def remove_duplicates(df, strategy="first"):
    """
    Remove high-confidence duplicate transactions intelligently
    
    Duplicates identified by: customer_id + product_name + date + quantity + total_amount + unit_price
    (Same customer, same product, same date, same quantities = definitely duplicate)
    
    Parameters:
    -----------
    df : pd.DataFrame - Input dataframe
    strategy : str - 'first' (keep original), 'last' (keep newest), 'max_qty' (keep highest qty)
    
    Returns:
    --------
    pd.DataFrame - Cleaned dataframe with duplicates removed
    """
    rows_before = len(df)
    
    if strategy == "first":
        df = df.drop_duplicates(
            subset=['customer_id', 'product_name', 'date', 'quantity', 'total_amount', 'unit_price'],
            keep='first'
        )
        reason = "Kept original transaction (most reliable)"
    elif strategy == "last":
        df = df.drop_duplicates(
            subset=['customer_id', 'product_name', 'date', 'quantity', 'total_amount', 'unit_price'],
            keep='last'
        )
        reason = "Kept most recent transaction"
    elif strategy == "max_qty":
        df = df.sort_values("quantity", ascending=False)
        df = df.drop_duplicates(
            subset=['customer_id', 'product_name', 'date'],
            keep='first'
        )
        reason = "Kept highest quantity transaction"
    
    rows_after = len(df)
    rows_removed = rows_before - rows_after
    
    return df, rows_before, rows_after, rows_removed, reason

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 15: APPLY DUPLICATE REMOVAL
# ─────────────────────────────────────────────────────────────────────────────────
copy_data, rows_before, rows_after, rows_removed, reason = remove_duplicates(copy_data, strategy="first")

print(f"\n✓ Duplicate Removal Complete:")
print(f"  Strategy: first ({reason})")
print(f"  Rows before: {rows_before} | Rows after: {rows_after}")
print(f"  Rows removed: {rows_removed}")
print(f"  Remaining duplicates: {copy_data.duplicated().sum()}")

# ═══════════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║            DATA VALUE STANDARDIZATION & CONSISTENCY NORMALIZATION            ║
# ║         Ensure consistent formatting across categorical columns              ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 16: DEFINE CITY STANDARDIZATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────────
def standardize_column_values(df, column_name, patterns_dict):
    """
    Standardize categorical values in a column using regex pattern matching
    
    This function:
    ├─ Removes special characters and extra whitespace
    ├─ Converts variations (e.g., NYC, New York → New York)
    ├─ Normalizes case (e.g., new york → New York)
    └─ Ensures consistency across the column
    
    Parameters:
    -----------
    df : pd.DataFrame - Input dataframe
    column_name : str - Column to standardize
    patterns_dict : dict - Mapping of regex patterns to standard values
                          key: pattern (regex), value: standard name
    
    Returns:
    --------
    tuple : (dataframe, unique_before, unique_after, changes_made)
    """
    
    # Count unique values before
    unique_before = df[column_name].nunique()
    original_values = set(df[column_name].dropna().unique())
    
    # Step 1: Remove special characters and extra spaces
    df[column_name] = (df[column_name]
                       .str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                       .str.strip()
                       .str.lower())
    
    # Step 2: Apply pattern replacements
    for pattern, standard_name in patterns_dict.items():
        df[column_name] = df[column_name].str.replace(
            pattern, 
            standard_name.lower(), 
            case=False, 
            regex=True
        )
    
    # Step 3: Format to proper case
    df[column_name] = df[column_name].str.title()
    
    # Count unique values after
    unique_after = df[column_name].nunique()
    new_values = set(df[column_name].dropna().unique())
    
    changes_made = unique_before - unique_after
    consolidated_values = original_values - new_values
    
    return df, unique_before, unique_after, changes_made, consolidated_values

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 17: DEFINE CITY STANDARDIZATION PATTERNS
# ─────────────────────────────────────────────────────────────────────────────────

CITY_PATTERNS = {
        # NEW YORK variations
        r'[\s\-_\.]*nyc[\s\-_\.]*|[\s\-_\.]*new[\s\-_\.]*york[\s\-_\.]*': 'New York',
        
        # LOS ANGELES variations
        r'[\s\-_\.]*la[\s\-_\.]*|[\s\-_\.]*los[\s\-_\.]*angeles[\s\-_\.]*': 'Los Angeles',
        
        # HOUSTON variations
        r'[\s\-_\.]*houston[\s\-_\.]*|[\s\-_\.]*htx[\s\-_\.]*': 'Houston',
        
        # CHICAGO variations
        r'[\s\-_\.]*chicago[\s\-_\.]*|[\s\-_\.]*chi[\s\-_\.]*': 'Chicago',
        
        # PHOENIX variations
        r'[\s\-_\.]*phoenix[\s\-_\.]*|[\s\-_\.]*phx[\s\-_\.]*': 'Phoenix',
    }

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 18: DISPLAY CITY STANDARDIZATION - BEFORE STATE
# ─────────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*80)
print("CITY COLUMN - DATA STANDARDIZATION".center(80))
print("─"*80)

print(f"\n📊 BEFORE STANDARDIZATION:")
print(f"   Unique city values: {copy_data['city'].nunique()}")
print(f"\n   Top 10 city values (with inconsistencies):")
city_before = copy_data['city'].value_counts().head(10)
for idx, (city, count) in enumerate(city_before.items(), 1):
    print(f"   {idx:2d}. {city:30s} → {count:4d} occurrences")

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 19: APPLY CITY STANDARDIZATION
# ─────────────────────────────────────────────────────────────────────────────────

copy_data, before_count, after_count, changes, consolidated = standardize_column_values(
    copy_data, 
    'city', 
    CITY_PATTERNS
)

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 20: DISPLAY CITY STANDARDIZATION - AFTER STATE
# ─────────────────────────────────────────────────────────────────────────────────

print(f"\n✅ AFTER STANDARDIZATION:")
print(f"   Unique city values: {copy_data['city'].nunique()}")
print(f"   Variations consolidated: {changes}")
print(f"\n   Standardized city values:")
city_after = copy_data['city'].value_counts()
for idx, (city, count) in enumerate(city_after.items(), 1):
    print(f"   {idx:2d}. {city:30s} → {count:4d} occurrences")

print(f"\n   Cities merged into standard format:")
if consolidated:
    for old_city in sorted(consolidated):
        print(f"   • '{old_city}' → merged into standard format")

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 21: STANDARDIZE PAYMENT METHOD COLUMN
# ─────────────────────────────────────────────────────────────────────────────────

payment_patterns = {
    # CARD variations
    r'(credit\s*card|debit\s*card|card|visa|mastercard)': 'Card',
    
    # UPI variations
    r'(upi|u\.p\.i)': 'UPI',
    
    # CASH variations
    r'(cash|cash\s*payment)': 'Cash',
    
    # WALLET variations
    r'(wallet|digital\s*wallet|e.wallet)': 'Wallet',
}

print(f"\n" + "─"*80)
print("PAYMENT METHOD - DATA STANDARDIZATION".center(80))
print("─"*80)

print(f"\n📊 BEFORE STANDARDIZATION:")
print(f"   Unique payment methods: {copy_data['payment_method'].nunique()}")
print(f"\n   Payment methods (with inconsistencies):")
payment_before = copy_data['payment_method'].value_counts()
for idx, (method, count) in enumerate(payment_before.items(), 1):
    print(f"   {idx:2d}. {method:30s} → {count:4d} occurrences")

copy_data, before_count_pay, after_count_pay, changes_pay, consolidated_pay = standardize_column_values(
    copy_data,
    'payment_method',
    payment_patterns
)

print(f"\n✅ AFTER STANDARDIZATION:")
print(f"   Unique payment methods: {copy_data['payment_method'].nunique()}")
print(f"   Variations consolidated: {changes_pay}")
print(f"\n   Standardized payment methods:")
payment_after = copy_data['payment_method'].value_counts()
for idx, (method, count) in enumerate(payment_after.items(), 1):
    print(f"   {idx:2d}. {method:30s} → {count:4d} occurrences")

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 22: STANDARDIZE DELIVERY STATUS COLUMN
# ─────────────────────────────────────────────────────────────────────────────────

delivery_patterns = {
    # SHIPPED variations
    r'(shipped|in.delivery|out.for.delivery|delivered)': 'Shipped',
    
    # PENDING variations
    r'(pending|processing|in.process)': 'Pending',
    
    # CANCELLED variations
    r'(cancelled|cancel|canceled)': 'Cancelled',
}

print(f"\n" + "─"*80)
print("DELIVERY STATUS - DATA STANDARDIZATION".center(80))
print("─"*80)

print(f"\n📊 BEFORE STANDARDIZATION:")
print(f"   Unique delivery statuses: {copy_data['delivery_status'].nunique()}")
print(f"\n   Delivery statuses (with inconsistencies):")
delivery_before = copy_data['delivery_status'].value_counts()
for idx, (status, count) in enumerate(delivery_before.items(), 1):
    print(f"   {idx:2d}. {status:30s} → {count:4d} occurrences")

copy_data, before_count_del, after_count_del, changes_del, consolidated_del = standardize_column_values(
    copy_data,
    'delivery_status',
    delivery_patterns
)

print(f"\n✅ AFTER STANDARDIZATION:")
print(f"   Unique delivery statuses: {copy_data['delivery_status'].nunique()}")
print(f"   Variations consolidated: {changes_del}")
print(f"\n   Standardized delivery statuses:")
delivery_after = copy_data['delivery_status'].value_counts()
for idx, (status, count) in enumerate(delivery_after.items(), 1):
    print(f"   {idx:2d}. {status:30s} → {count:4d} occurrences")

# ═══════════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                      DATA CLEANING PIPELINE SUMMARY                           ║
# ║          Complete overview of all transformations & improvements             ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────────
# STEP 23: FINAL COMPREHENSIVE DATA REPORT
# ─────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("DATA CLEANING PIPELINE - FINAL SUMMARY".center(80))
print("="*80)

print(f"\n📊 DATASET DIMENSIONS:")
print(f"   Total rows: {copy_data.shape[0]}")
print(f"   Total columns: {copy_data.shape[1]}")

print(f"\n📋 DATA QUALITY METRICS:")
print(f"   Missing values (total): {copy_data.isna().sum().sum()}")
print(f"   Duplicate rows: {copy_data.duplicated().sum()}")

print(f"\n🔢 DATA TYPES SUMMARY:")
dtype_counts = copy_data.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"   {dtype}: {count} columns")

print(f"\n✅ CLEANING STEPS COMPLETED:")
print(f"   1. ✓ Loaded raw e-commerce data ({copy_data.shape[0]} rows)")
print(f"   2. ✓ Handled string column missing values (customer_id, city, product_name)")
print(f"   3. ✓ Handled numeric column missing values (rating, unit_price, total_amount)")
print(f"   4. ✓ Created categorical transformations (rating_stat)")
print(f"   5. ✓ Applied intelligent missing value logic (delivery_status)")
print(f"   6. ✓ Converted numeric data types (int64 for prices & quantities)")
print(f"   7. ✓ Removed {rows_removed} high-confidence duplicate transactions")
print(f"   8. ✓ Standardized city names ({changes} variations consolidated)")
print(f"   9. ✓ Standardized payment methods ({changes_pay} variations consolidated)")
print(f"   10. ✓ Standardized delivery status ({changes_del} variations consolidated)")

print(f"\n🎯 FINAL DATA STATISTICS:")
print(f"   Unique cities: {copy_data['city'].nunique()}")
print(f"   Unique payment methods: {copy_data['payment_method'].nunique()}")
print(f"   Unique delivery statuses: {copy_data['delivery_status'].nunique()}")
print(f"   Unique product categories: {copy_data['rating_stat'].nunique()}")

print("\n" + "="*80)
print("✨ DATA IS CLEAN, STANDARDIZED & READY FOR ANALYSIS ✨".center(80))
print("="*80 + "\n")

print("\n🎯 Ready for visualization! Use: from CombinedPart3_Visualisation import ECommerceVisualizer\n")

# ═══════════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║              RANDOM FOREST MODEL - E-COMMERCE RATING PREDICTION              ║
# ║                                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("RANDOM FOREST MODEL TRAINING".center(80))
print("="*80)

# Data preparation for modeling
copy_data["date"] = pd.to_datetime(copy_data["date"], format='mixed', dayfirst=True)
copy_data["day"] = copy_data["date"].dt.day
copy_data["month"] = copy_data["date"].dt.month
copy_data["year"] = copy_data["date"].dt.year
copy_data["day_of_week"] = copy_data["date"].dt.dayofweek

copy_data["category"] = copy_data["category"].str.strip().str.lower()
copy_data["quantity"] = copy_data["quantity"].str.replace("x", "").astype(int)
copy_data["payment_method"] = copy_data["payment_method"].str.strip().str.title()
copy_data["payment_method"] = copy_data["payment_method"].replace({
    "Pay Pal": "PayPal",
    "Cc": "Card"
})
copy_data.columns = [
    col if col == "rating" else col.replace("_", " ").replace("&", "and").title()
    for col in copy_data.columns
]

# Dropping unuseful columns
copy_data = copy_data.drop(columns=[
    "Customer Id",
    "Total Amount",
    "Rating Stat",
    "Product Name",
    "Transaction Id",
    "Date"
])

# One-hot encoding for categorical variables
copy_data = pd.get_dummies(copy_data, drop_first=True)

print(f"\n📊 DATA SHAPE AFTER ENCODING: {copy_data.shape}")
print(f"   Features: {copy_data.shape[1] - 1}")
print(f"   Samples: {copy_data.shape[0]}")

# Setting up the input and output variable
X = copy_data.drop("rating", axis=1)
Y = copy_data["rating"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"\n📋 TRAIN-TEST SPLIT:")
print(f"   Training set: {X_train.shape[0]} samples ({(1-0.2)*100:.0f}%)")
print(f"   Testing set:  {X_test.shape[0]} samples ({0.2*100:.0f}%)")

# Train Random Forest Regressor model
model = RandomForestRegressor(
    n_estimators=100,           # Number of trees in the forest
    max_depth=15,               # Maximum depth of trees
    min_samples_split=5,        # Minimum samples required to split a node
    min_samples_leaf=2,         # Minimum samples required at leaf node
    random_state=42,
    n_jobs=-1                   # Use all available processors
)

model.fit(X_train, y_train)
print("\n✓ Random Forest Model trained successfully!")
print(f"  Number of trees: {model.n_estimators}")
print(f"  Number of features: {len(model.feature_importances_)}")

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n📊 MODEL PERFORMANCE METRICS:")
print(f"   R² Score:              {r2:.4f}")
print(f"   Mean Absolute Error:   {mae:.4f}")
print(f"   Mean Squared Error:    {mse:.4f}")
print(f"   Root Mean Sq. Error:   {rmse:.4f}")

# Feature importance
print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES (by importance):")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['Feature']:30s} → {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")

# Visualization
print(f"\n📈 GENERATING VISUALIZATION...")
sample_index = range(len(y_pred))
plt.figure(figsize=(14, 6))

# Plot 1: Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(sample_index, y_pred, label="Predicted", alpha=0.6, color='green')
plt.scatter(sample_index, y_test.values, label="Actual", alpha=0.6, color='red')
plt.xlabel("Sample Index")
plt.ylabel("Rating")
plt.legend()
plt.title("Random Forest: Actual vs Predicted Ratings")
plt.grid(True, alpha=0.3)

# Plot 2: Residuals
plt.subplot(1, 2, 2)
residuals = y_test.values - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, color='purple')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Rating")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("✨ RANDOM FOREST MODEL TRAINING COMPLETE ✨".center(80))
print("="*80 + "\n")
