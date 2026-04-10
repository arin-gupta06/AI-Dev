import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading dataset

dataset = pd.read_csv("D://AI DEV//Data Science Essentials//Dataset//E_Commerce_Transactions_Raw.csv")

# converting it into Dataframe
data = pd.DataFrame(dataset)

# Checking the data
# print(data.tail())


# General insights of the dataset

def general(df):
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.isna().sum())
    print(df.duplicated().sum())


copy_data = data.copy()

# Handling missing customer id
copy_data["customer_id"] = copy_data["customer_id"].fillna("Unknown")
# print(copy_data.tail())

# Handling the missing customer city
# print(copy_data["city"].mode())
copy_data["city"] = copy_data["city"].fillna(copy_data["city"].mode().values[0])

# Handles product names missing values
copy_data["product_name"] = copy_data["product_name"].str.lower().str.strip()

copy_data["product_name"] = copy_data["product_name"].fillna(copy_data["product_name"].mode().values[0])
# print(copy_data["rating"])
def category_rats_stat(rating):
  if pd.isna(rating):
    return "No Rating"
  elif rating >=5.0:
     return "Excellent"
  elif rating >=4.0:
     return "Very Good"
  elif rating >=3.0:
     return "Good"
  elif rating >=2.0:
     return "Average"
  elif rating >=1.0:
     return "Bad"
  else:
     return "Invalid"
  
   
mean_rating = copy_data["rating"].mean()
copy_data["rating"] = copy_data["rating"].apply(lambda x: mean_rating if pd.isna(x) else x)
copy_data["rating_stat"] = copy_data["rating"].apply(category_rats_stat)

# Handles delivery stats with payment method column

# print(copy_data["delivery_status"])
# print(copy_data["payment_method"])

def handle_delivery_stat(df):
    """
    Fill missing delivery_status based on payment_method:
    - CASH payment → "Shipped"
    - Card payment → "Pending"
    - Other → "Processing"
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
    
    # ✅ CORRECT: Use df.apply() with axis=1 (not df["col"].apply())
    df["delivery_status"] = df.apply(fill_delivery, axis=1)
    return df


# Call function and assign result
copy_data = handle_delivery_stat(copy_data)


print(copy_data.tail(20))

general(copy_data)

# ═══════════════════════════════════════════════════════════════════════════════════
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                         VISUALIZATION SECTION                                  ║
# ║                     ALL STANDARD PLOTS & CHARTS                               ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════════

# Configure visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("\n" + "="*80)
print("STARTING COMPREHENSIVE VISUALIZATION".center(80))
print("="*80 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 1: HISTOGRAM - Distribution of Ratings
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 5))
plt.hist(copy_data['rating'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('PLOT 1: Distribution of Customer Ratings (Histogram)', fontsize=14, fontweight='bold')
plt.xlabel('Rating Value', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 2: HISTOGRAM WITH KDE - Distribution with Smooth Curve
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 5))
sns.histplot(data=copy_data, x='rating', kde=True, bins=20, color='steelblue')
plt.title('PLOT 2: Rating Distribution with KDE Curve (Smooth Trend)', fontsize=14, fontweight='bold')
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 3: BOX PLOT - Univariate (Show Outliers and Spread)
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(8, 5))
sns.boxplot(data=copy_data, y='rating', color='lightblue')
plt.title('PLOT 3: Rating Box Plot (Median, Quartiles, Outliers)', fontsize=14, fontweight='bold')
plt.ylabel('Rating Value', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 4: GROUPED BOX PLOT - Compare by Delivery Status
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(12, 6))
sns.boxplot(data=copy_data, x='delivery_status', y='rating', palette='Set2')
plt.title('PLOT 4: Rating Distribution by Delivery Status (Grouped Box Plot)', fontsize=14, fontweight='bold')
plt.xlabel('Delivery Status', fontsize=12)
plt.ylabel('Rating', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 5: VIOLIN PLOT - Detailed Distribution by Group
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(12, 6))
sns.violinplot(data=copy_data, x='delivery_status', y='rating', palette='muted')
plt.title('PLOT 5: Rating Distribution by Delivery Status (Violin Plot - Shows Density)', fontsize=14, fontweight='bold')
plt.xlabel('Delivery Status', fontsize=12)
plt.ylabel('Rating', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 6: COUNT PLOT - Frequency of Each Category
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(12, 5))
sns.countplot(data=copy_data, x='delivery_status', palette='Set2', order=copy_data['delivery_status'].value_counts().index)
plt.title('PLOT 6: Number of Transactions by Delivery Status (Count Plot)', fontsize=14, fontweight='bold')
plt.xlabel('Delivery Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 7: BAR CHART - Average Rating by Delivery Status
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 5))
avg_rating_by_status = copy_data.groupby('delivery_status')['rating'].mean()
avg_rating_by_status.plot(kind='bar', color='coral', edgecolor='black', alpha=0.8)
plt.title('PLOT 7: Average Rating by Delivery Status (Bar Chart)', fontsize=14, fontweight='bold')
plt.xlabel('Delivery Status', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 8: GROUPED BAR CHART - Payment Method vs Delivery Status
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(12, 6))
grouped_data = copy_data.groupby(['payment_method', 'delivery_status']).size().unstack(fill_value=0)
grouped_data.plot(kind='bar', ax=plt.gca(), color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8)
plt.title('PLOT 8: Transaction Count by Payment Method & Delivery Status (Grouped Bar)', fontsize=14, fontweight='bold')
plt.xlabel('Payment Method', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Delivery Status', loc='best')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 9: SCATTER PLOT - Quantity vs Rating
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 6))
plt.scatter(copy_data['quantity'], copy_data['rating'], alpha=0.5, s=50, color='darkblue')
plt.title('PLOT 9: Quantity vs Rating (Scatter Plot - Check Correlation)', fontsize=14, fontweight='bold')
plt.xlabel('Quantity Ordered', fontsize=12)
plt.ylabel('Customer Rating', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 10: SCATTER PLOT WITH HUE - Colored by Delivery Status
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(12, 6))
sns.scatterplot(data=copy_data, x='quantity', y='rating', hue='delivery_status', s=100, alpha=0.6)
plt.title('PLOT 10: Quantity vs Rating (Colored by Delivery Status)', fontsize=14, fontweight='bold')
plt.xlabel('Quantity Ordered', fontsize=12)
plt.ylabel('Customer Rating', fontsize=12)
plt.legend(title='Delivery Status', loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 11: SCATTER PLOT - Unit Price vs Total Amount
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 6))
scatter = plt.scatter(copy_data['unit_price'], copy_data['total_amount'], 
                     c=copy_data['quantity'], s=100, alpha=0.6, cmap='viridis')
cbar = plt.colorbar(scatter)
cbar.set_label('Quantity', fontsize=10)
plt.title('PLOT 11: Unit Price vs Total Amount (Colored by Quantity)', fontsize=14, fontweight='bold')
plt.xlabel('Unit Price ($)', fontsize=12)
plt.ylabel('Total Amount ($)', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 12: LINE PLOT - Trends (Average Rating over Categories)
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 5))
category_rating = copy_data.groupby('category')['rating'].mean().sort_values()
category_rating.plot(kind='line', marker='o', linewidth=2, markersize=8, color='green')
plt.title('PLOT 12: Average Rating by Product Category (Line Plot)', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 13: PIE CHART - Distribution of Payment Methods
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 6))
payment_counts = copy_data['payment_method'].value_counts()
colors = sns.color_palette("husl", len(payment_counts))
plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 11})
plt.title('PLOT 13: Distribution of Payment Methods (Pie Chart)', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 14: DONUT CHART - Distribution of Delivery Status
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 6))
delivery_counts = copy_data['delivery_status'].value_counts()
colors = sns.color_palette("Set2", len(delivery_counts))
plt.pie(delivery_counts, labels=delivery_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, textprops={'fontsize': 11})
# Create donut hole
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
plt.gca().add_artist(centre_circle)
plt.title('PLOT 14: Distribution of Delivery Status (Donut Chart)', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 15: CORRELATION HEATMAP - See Relationships Between Numeric Columns
# ═══════════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 8))
numeric_cols = copy_data.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = copy_data[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('PLOT 15: Correlation Heatmap (Numeric Columns)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 16: PAIRPLOT - All Numeric Columns Relationships
# ═══════════════════════════════════════════════════════════════════════════════════
plot_cols = ['quantity', 'unit_price', 'rating', 'return_days']
plot_cols_valid = [col for col in plot_cols if col in copy_data.columns]
print(f"\nCreating PairPlot with columns: {plot_cols_valid}")
pairplot = sns.pairplot(copy_data[plot_cols_valid].dropna(), diag_kind='hist', plot_kws={'alpha': 0.6})
pairplot.fig.suptitle('PLOT 16: Pairplot (All Numeric Column Relationships)', 
                      fontsize=14, fontweight='bold', y=1.001)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 17: DISTRIBUTION COMPARISON - With vs Without Missing Values
# ═══════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot with missing values (most data)
sns.histplot(data=copy_data, x='rating', kde=True, bins=20, color='steelblue', ax=axes[0])
axes[0].set_title(f'Rating Distribution (Includes {copy_data["rating"].isna().sum()} NaN)', 
                  fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency')

# Plot without missing values
copy_data_clean = copy_data.dropna(subset=['rating'])
sns.histplot(data=copy_data_clean, x='rating', kde=True, bins=20, color='lightgreen', ax=axes[1])
axes[1].set_title(f'Rating Distribution (NaN Removed - {len(copy_data_clean)} clean rows)', 
                  fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency')

plt.suptitle('PLOT 17: Impact of Missing Values on Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# PLOT 18: MULTI-PLOT DASHBOARD - Comprehensive Overview
# ═══════════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Rating histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(copy_data['rating'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_title('Rating Distribution', fontweight='bold')
ax1.set_xlabel('Rating')
ax1.set_ylabel('Count')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Quantity histogram
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(copy_data['quantity'], bins=15, color='coral', edgecolor='black', alpha=0.7)
ax2.set_title('Quantity Distribution', fontweight='bold')
ax2.set_xlabel('Quantity')
ax2.set_ylabel('Count')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Unit Price histogram
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(copy_data['unit_price'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
ax3.set_title('Unit Price Distribution', fontweight='bold')
ax3.set_xlabel('Price ($)')
ax3.set_ylabel('Count')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Rating box plot
ax4 = fig.add_subplot(gs[1, 0])
sns.boxplot(data=copy_data, y='rating', color='lightblue', ax=ax4)
ax4.set_title('Rating Box Plot', fontweight='bold')
ax4.set_ylabel('Rating')
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Delivery Status count
ax5 = fig.add_subplot(gs[1, 1])
delivery_counts = copy_data['delivery_status'].value_counts()
ax5.bar(delivery_counts.index, delivery_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8)
ax5.set_title('Orders by Delivery Status', fontweight='bold')
ax5.set_xlabel('Status')
ax5.set_ylabel('Count')
ax5.tick_params(axis='x', rotation=45)
ax5.grid(axis='y', alpha=0.3)

# Plot 6: Payment Method count
ax6 = fig.add_subplot(gs[1, 2])
payment_counts = copy_data['payment_method'].value_counts()
ax6.bar(payment_counts.index, payment_counts.values, color='skyblue', edgecolor='black', alpha=0.8)
ax6.set_title('Orders by Payment Method', fontweight='bold')
ax6.set_xlabel('Method')
ax6.set_ylabel('Count')
ax6.tick_params(axis='x', rotation=45)
ax6.grid(axis='y', alpha=0.3)

# Plot 7: Quantity vs Rating scatter
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(copy_data['quantity'], copy_data['rating'], alpha=0.5, s=30, color='darkblue')
ax7.set_title('Quantity vs Rating', fontweight='bold')
ax7.set_xlabel('Quantity')
ax7.set_ylabel('Rating')
ax7.grid(alpha=0.3)

# Plot 8: Rating by Payment Method box plot
ax8 = fig.add_subplot(gs[2, 1])
sns.boxplot(data=copy_data, x='payment_method', y='rating', palette='Set2', ax=ax8)
ax8.set_title('Rating by Payment Method', fontweight='bold')
ax8.set_xlabel('Payment Method')
ax8.set_ylabel('Rating')
ax8.tick_params(axis='x', rotation=45)
ax8.grid(axis='y', alpha=0.3)

# Plot 9: Average rating by category
ax9 = fig.add_subplot(gs[2, 2])
category_avg = copy_data.groupby('category')['rating'].mean().sort_values()
ax9.barh(category_avg.index, category_avg.values, color='orange', edgecolor='black', alpha=0.8)
ax9.set_title('Avg Rating by Category', fontweight='bold')
ax9.set_xlabel('Average Rating')
ax9.grid(axis='x', alpha=0.3)

plt.suptitle('PLOT 18: E-Commerce Comprehensive Dashboard (9 Views)', fontsize=16, fontweight='bold', y=0.995)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS FOR VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("VISUALIZATION SUMMARY STATISTICS".center(80))
print("="*80)

print(f"\n{'RATINGS':30} | {'Value':15}")
print("-" * 50)
print(f"{'Average Rating':30} | {copy_data['rating'].mean():.2f}")
print(f"{'Median Rating':30} | {copy_data['rating'].median():.2f}")
print(f"{'Std Deviation':30} | {copy_data['rating'].std():.2f}")
print(f"{'Min/Max Rating':30} | {copy_data['rating'].min():.1f} / {copy_data['rating'].max():.1f}")

print(f"\n{'DELIVERY STATUS':30} | {'Count':15} | {'%':10}")
print("-" * 60)
delivery_dist = copy_data['delivery_status'].value_counts()
for status, count in delivery_dist.items():
    pct = (count / len(copy_data)) * 100
    print(f"{status:30} | {count:15} | {pct:6.1f}%")

print(f"\n{'PAYMENT METHOD':30} | {'Count':15} | {'%':10}")
print("-" * 60)
payment_dist = copy_data['payment_method'].value_counts()
for method, count in payment_dist.items():
    pct = (count / len(copy_data)) * 100
    print(f"{method:30} | {count:15} | {pct:6.1f}%")

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE! ✓".center(80))
print("="*80 + "\n")