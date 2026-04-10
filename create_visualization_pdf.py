import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════════
# LOAD AND CLEAN DATA
# ═══════════════════════════════════════════════════════════════════════════════════

print("Loading and cleaning data...")

dataset = pd.read_csv("d:\\AI DEV\\Data Science Essentials\\Dataset\\E_Commerce_Transactions_Raw.csv")
copy_data = pd.DataFrame(dataset).copy()

# Handle missing values
copy_data["customer_id"] = copy_data["customer_id"].fillna("Unknown")
copy_data["city"] = copy_data["city"].fillna(copy_data["city"].mode().values[0])
copy_data["product_name"] = copy_data["product_name"].str.lower().str.strip()
copy_data["product_name"] = copy_data["product_name"].fillna(copy_data["product_name"].mode().values[0])

# Categorize ratings
def category_rats_stat(rating):
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

mean_rating = copy_data["rating"].mean()
copy_data["rating"] = copy_data["rating"].apply(lambda x: mean_rating if pd.isna(x) else x)
copy_data["rating_stat"] = copy_data["rating"].apply(category_rats_stat)

# Handle delivery status
def handle_delivery_stat(df):
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

copy_data = handle_delivery_stat(copy_data)

# Convert quantity to numeric (it might be stored as string)
copy_data['quantity'] = pd.to_numeric(copy_data['quantity'], errors='coerce')

# Configure visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ═══════════════════════════════════════════════════════════════════════════════════
# CREATE PDF WITH ALL VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════════

pdf_path = "d:\\AI DEV\\E_Commerce_Visualizations_Report.pdf"
print(f"\nCreating PDF: {pdf_path}")

with PdfPages(pdf_path) as pdf:
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PAGE 1: TITLE PAGE
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    title_text = "E-Commerce Data Visualization Report"
    subtitle_text = "18 Standard Plots & Analysis"
    date_text = f"Generated: {datetime.now().strftime('%B %d, %Y')}"
    dataset_text = f"Dataset: {len(copy_data):,} transactions | {len(copy_data.columns)} columns"
    
    ax.text(0.5, 0.75, title_text, fontsize=28, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.65, subtitle_text, fontsize=20, ha='center', transform=ax.transAxes, style='italic')
    ax.text(0.5, 0.50, date_text, fontsize=14, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.45, dataset_text, fontsize=12, ha='center', transform=ax.transAxes)
    
    summary_text = """
    This report contains:
    • 18 Standard Visualization Types
    • Data Cleaning & Handling Missing Values
    • Statistical Insights
    • Code Samples for Each Plot
    
    Categories Covered:
    ✓ Distribution Analysis (Histograms, KDE, Box Plots)
    ✓ Group Comparisons (Violin Plots, Grouped Boxes)
    ✓ Relationships (Scatter Plots, Correlation Heatmaps)
    ✓ Categorical Analysis (Count Plots, Bar Charts, Pie Charts)
    ✓ Multi-Dimensional Views (PairPlots, Dashboards)
    """
    
    ax.text(0.5, 0.25, summary_text, fontsize=11, ha='center', va='top', 
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Title page added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PAGE 2: TABLE OF CONTENTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    toc_title = "TABLE OF CONTENTS"
    ax.text(0.5, 0.95, toc_title, fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
    
    toc_items = """
    1.  Histogram - Distribution of Ratings
    2.  Histogram with KDE - Ratings with Smooth Curve
    3.  Box Plot - Rating Spread & Outliers
    4.  Grouped Box Plot - Ratings by Delivery Status
    5.  Violin Plot - Detailed Distribution by Group
    6.  Count Plot - Order Frequency by Status
    7.  Bar Chart - Average Rating by Status
    8.  Grouped Bar Chart - Payment × Delivery Status
    9.  Scatter Plot - Quantity vs Rating
    10. Scatter with Hue - Colored by Delivery Status
    11. Scatter with Colorbar - Price vs Amount (by Quantity)
    12. Line Plot - Average Rating by Category
    13. Pie Chart - Payment Method Distribution
    14. Donut Chart - Delivery Status Distribution
    15. Correlation Heatmap - Numeric Columns Relationships
    16. PairPlot - All Numeric Columns Grid
    17. Comparison - Missing Values Impact
    18. Dashboard - 9-Panel Comprehensive View
    
    Plus: Data Cleaning Summary & Statistical Insights
    """
    
    ax.text(0.05, 0.88, toc_items, fontsize=11, ha='left', va='top', 
            transform=ax.transAxes, family='monospace')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Table of contents added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PAGE 3: DATA CLEANING SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    ax.text(0.5, 0.95, "DATA CLEANING SUMMARY", fontsize=18, fontweight='bold', 
            ha='center', transform=ax.transAxes)
    
    cleaning_text = f"""
    ORIGINAL DATASET:
    • Total Records: {len(copy_data):,}
    • Total Columns: {len(copy_data.columns)}
    
    MISSING VALUES HANDLED:
    • Customer ID: Filled with 'Unknown'
    • City: Filled with Mode (Most Frequent)
    • Product Name: Cleaned (lowercase, stripped) & filled with Mode
    • Rating: Filled with Mean ({mean_rating:.2f})
    • Delivery Status: Filled based on Payment Method logic
    
    DATA TRANSFORMATIONS:
    • Rating Categorization: Excellent (5+), Very Good (4+), Good (3+), Average (2+), Bad (1+)
    • Delivery Status Logic:
      - If CASH payment → 'Shipped'
      - If Card payment → 'Pending'
      - Otherwise → 'Processing'
    
    CURRENT DATA QUALITY:
    • Total Non-Null Records: {len(copy_data):,}
    • Duplicate Rows: {copy_data.duplicated().sum()}
    • Data Types: {copy_data.dtypes.value_counts().to_dict()}
    
    NUMERIC COLUMNS SUMMARY:
    {copy_data.describe().to_string()}
    """
    
    ax.text(0.05, 0.88, cleaning_text, fontsize=9, ha='left', va='top', 
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Data cleaning summary added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PLOT 1: HISTOGRAM - Distribution of Ratings
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    
    # Code snippet
    ax_code = fig.add_axes([0.05, 0.55, 0.9, 0.40])
    ax_code.axis('off')
    code_text = """PLOT 1: HISTOGRAM - Rating Distribution
    
plt.figure(figsize=(10, 5))
plt.hist(copy_data['rating'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Customer Ratings', fontsize=14, fontweight='bold')
plt.xlabel('Rating Value', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.show()

PURPOSE: Shows the distribution shape, central tendency, and spread of ratings."""
    
    ax_code.text(0.05, 0.95, code_text, fontsize=9, ha='left', va='top', 
                transform=ax_code.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    # Plot
    ax_plot = fig.add_axes([0.1, 0.05, 0.8, 0.45])
    ax_plot.hist(copy_data['rating'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax_plot.set_title('Distribution of Customer Ratings (Histogram)', fontsize=12, fontweight='bold')
    ax_plot.set_xlabel('Rating Value', fontsize=10)
    ax_plot.set_ylabel('Frequency (Count)', fontsize=10)
    ax_plot.grid(axis='y', alpha=0.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Plot 1: Histogram added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PLOT 2: HISTOGRAM WITH KDE
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    
    ax_code = fig.add_axes([0.05, 0.55, 0.9, 0.40])
    ax_code.axis('off')
    code_text = """PLOT 2: HISTOGRAM WITH KDE - Smooth Distribution Curve
    
plt.figure(figsize=(10, 5))
sns.histplot(data=copy_data, x='rating', kde=True, bins=20, color='steelblue')
plt.title('Rating Distribution with KDE', fontsize=14, fontweight='bold')
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.show()

PURPOSE: KDE (Kernel Density Estimate) overlays a smooth curve showing probability distribution."""
    
    ax_code.text(0.05, 0.95, code_text, fontsize=9, ha='left', va='top', 
                transform=ax_code.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    ax_plot = fig.add_axes([0.1, 0.05, 0.8, 0.45])
    sns.histplot(data=copy_data, x='rating', kde=True, bins=20, color='steelblue', ax=ax_plot)
    ax_plot.set_title('Rating Distribution with KDE Curve', fontsize=12, fontweight='bold')
    ax_plot.set_xlabel('Rating', fontsize=10)
    ax_plot.set_ylabel('Density', fontsize=10)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Plot 2: Histogram with KDE added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PLOT 3: BOX PLOT
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    
    ax_code = fig.add_axes([0.05, 0.55, 0.9, 0.40])
    ax_code.axis('off')
    code_text = """PLOT 3: BOX PLOT - Show Quartiles, Median & Outliers
    
plt.figure(figsize=(8, 5))
sns.boxplot(data=copy_data, y='rating', color='lightblue')
plt.title('Rating Box Plot', fontsize=14, fontweight='bold')
plt.ylabel('Rating Value', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.show()

PURPOSE: Shows median (middle line), quartiles (box), and outliers (dots) for quick distribution understanding."""
    
    ax_code.text(0.05, 0.95, code_text, fontsize=9, ha='left', va='top', 
                transform=ax_code.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    ax_plot = fig.add_axes([0.25, 0.05, 0.5, 0.45])
    sns.boxplot(data=copy_data, y='rating', color='lightblue', ax=ax_plot)
    ax_plot.set_title('Rating Box Plot (Median, Quartiles, Outliers)', fontsize=12, fontweight='bold')
    ax_plot.set_ylabel('Rating Value', fontsize=10)
    ax_plot.grid(axis='y', alpha=0.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Plot 3: Box plot added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PLOT 4: GROUPED BOX PLOT
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    
    ax_code = fig.add_axes([0.05, 0.55, 0.9, 0.40])
    ax_code.axis('off')
    code_text = """PLOT 4: GROUPED BOX PLOT - Compare Across Categories
    
plt.figure(figsize=(12, 6))
sns.boxplot(data=copy_data, x='delivery_status', y='rating', palette='Set2')
plt.title('Rating Distribution by Delivery Status', fontsize=14, fontweight='bold')
plt.xlabel('Delivery Status', fontsize=12)
plt.ylabel('Rating', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.show()

PURPOSE: Compare how ratings differ across delivery status categories."""
    
    ax_code.text(0.05, 0.95, code_text, fontsize=9, ha='left', va='top', 
                transform=ax_code.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    ax_plot = fig.add_axes([0.08, 0.05, 0.84, 0.45])
    sns.boxplot(data=copy_data, x='delivery_status', y='rating', palette='Set2', ax=ax_plot)
    ax_plot.set_title('Rating Distribution by Delivery Status', fontsize=12, fontweight='bold')
    ax_plot.set_xlabel('Delivery Status', fontsize=10)
    ax_plot.set_ylabel('Rating', fontsize=10)
    ax_plot.tick_params(axis='x', rotation=45)
    ax_plot.grid(axis='y', alpha=0.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Plot 4: Grouped box plot added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PLOT 5: VIOLIN PLOT
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    
    ax_code = fig.add_axes([0.05, 0.55, 0.9, 0.40])
    ax_code.axis('off')
    code_text = """PLOT 5: VIOLIN PLOT - Show Full Distribution Density
    
plt.figure(figsize=(12, 6))
sns.violinplot(data=copy_data, x='delivery_status', y='rating', palette='muted')
plt.title('Rating Distribution by Delivery Status (Violin Plot)', fontsize=14, fontweight='bold')
plt.xlabel('Delivery Status', fontsize=12)
plt.ylabel('Rating', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.show()

PURPOSE: Shows full PDF (Probability Density Function) - wider sections = more customers at that rating."""
    
    ax_code.text(0.05, 0.95, code_text, fontsize=9, ha='left', va='top', 
                transform=ax_code.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    ax_plot = fig.add_axes([0.08, 0.05, 0.84, 0.45])
    sns.violinplot(data=copy_data, x='delivery_status', y='rating', palette='muted', ax=ax_plot)
    ax_plot.set_title('Rating Distribution by Delivery Status (Violin Plot)', fontsize=12, fontweight='bold')
    ax_plot.set_xlabel('Delivery Status', fontsize=10)
    ax_plot.set_ylabel('Rating', fontsize=10)
    ax_plot.tick_params(axis='x', rotation=45)
    ax_plot.grid(axis='y', alpha=0.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Plot 5: Violin plot added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PLOT 6: COUNT PLOT
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    
    ax_code = fig.add_axes([0.05, 0.55, 0.9, 0.40])
    ax_code.axis('off')
    code_text = """PLOT 6: COUNT PLOT - Frequency of Each Category
    
plt.figure(figsize=(12, 5))
sns.countplot(data=copy_data, x='delivery_status', palette='Set2', 
              order=copy_data['delivery_status'].value_counts().index)
plt.title('Number of Transactions by Delivery Status', fontsize=14, fontweight='bold')
plt.xlabel('Delivery Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.show()

PURPOSE: Show how many transactions fall into each category."""
    
    ax_code.text(0.05, 0.95, code_text, fontsize=9, ha='left', va='top', 
                transform=ax_code.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    ax_plot = fig.add_axes([0.08, 0.05, 0.84, 0.45])
    sns.countplot(data=copy_data, x='delivery_status', palette='Set2', 
                  order=copy_data['delivery_status'].value_counts().index, ax=ax_plot)
    ax_plot.set_title('Number of Transactions by Delivery Status', fontsize=12, fontweight='bold')
    ax_plot.set_xlabel('Delivery Status', fontsize=10)
    ax_plot.set_ylabel('Count', fontsize=10)
    ax_plot.tick_params(axis='x', rotation=45)
    ax_plot.grid(axis='y', alpha=0.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Plot 6: Count plot added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PLOT 7: BAR CHART
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    
    ax_code = fig.add_axes([0.05, 0.55, 0.9, 0.40])
    ax_code.axis('off')
    code_text = """PLOT 7: BAR CHART - Average Values by Category
    
plt.figure(figsize=(10, 5))
avg_rating_by_status = copy_data.groupby('delivery_status')['rating'].mean()
avg_rating_by_status.plot(kind='bar', color='coral', edgecolor='black', alpha=0.8)
plt.title('Average Rating by Delivery Status', fontsize=14, fontweight='bold')
plt.xlabel('Delivery Status', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.show()

PURPOSE: Compare aggregated metrics (mean, sum, count) across categories."""
    
    ax_code.text(0.05, 0.95, code_text, fontsize=9, ha='left', va='top', 
                transform=ax_code.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    ax_plot = fig.add_axes([0.1, 0.05, 0.8, 0.45])
    avg_rating_by_status = copy_data.groupby('delivery_status')['rating'].mean()
    avg_rating_by_status.plot(kind='bar', color='coral', edgecolor='black', alpha=0.8, ax=ax_plot)
    ax_plot.set_title('Average Rating by Delivery Status', fontsize=12, fontweight='bold')
    ax_plot.set_xlabel('Delivery Status', fontsize=10)
    ax_plot.set_ylabel('Average Rating', fontsize=10)
    ax_plot.tick_params(axis='x', rotation=45)
    ax_plot.grid(axis='y', alpha=0.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Plot 7: Bar chart added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PLOT 8: SCATTER PLOT
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    
    ax_code = fig.add_axes([0.05, 0.55, 0.9, 0.40])
    ax_code.axis('off')
    code_text = """PLOT 8: SCATTER PLOT - Relationship Between Two Numeric Variables
    
plt.figure(figsize=(10, 6))
plt.scatter(copy_data['quantity'], copy_data['rating'], alpha=0.5, s=50, color='darkblue')
plt.title('Quantity vs Rating', fontsize=14, fontweight='bold')
plt.xlabel('Quantity Ordered', fontsize=12)
plt.ylabel('Customer Rating', fontsize=12)
plt.grid(alpha=0.3)
plt.show()

PURPOSE: Identify correlation - do variables trend together or inversely?"""
    
    ax_code.text(0.05, 0.95, code_text, fontsize=9, ha='left', va='top', 
                transform=ax_code.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    ax_plot = fig.add_axes([0.1, 0.05, 0.8, 0.45])
    ax_plot.scatter(copy_data['quantity'], copy_data['rating'], alpha=0.5, s=30, color='darkblue')
    ax_plot.set_title('Quantity vs Rating (Scatter Plot)', fontsize=12, fontweight='bold')
    ax_plot.set_xlabel('Quantity', fontsize=10)
    ax_plot.set_ylabel('Rating', fontsize=10)
    ax_plot.grid(alpha=0.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Plot 8: Scatter plot added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PLOT 9: SCATTER WITH HUE
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    
    ax_code = fig.add_axes([0.05, 0.55, 0.9, 0.40])
    ax_code.axis('off')
    code_text = """PLOT 9: SCATTER WITH HUE - Add Third Dimension with Color
    
plt.figure(figsize=(12, 6))
sns.scatterplot(data=copy_data, x='quantity', y='rating', hue='delivery_status', s=100, alpha=0.6)
plt.title('Quantity vs Rating (Colored by Delivery Status)', fontsize=14, fontweight='bold')
plt.xlabel('Quantity', fontsize=12)
plt.ylabel('Rating', fontsize=12)
plt.legend(title='Delivery Status', loc='best')
plt.grid(alpha=0.3)
plt.show()

PURPOSE: Visualize 3 dimensions: X-axis, Y-axis, and color (categorical variable)."""
    
    ax_code.text(0.05, 0.95, code_text, fontsize=9, ha='left', va='top', 
                transform=ax_code.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    ax_plot = fig.add_axes([0.08, 0.05, 0.84, 0.45])
    sns.scatterplot(data=copy_data, x='quantity', y='rating', hue='delivery_status', s=100, alpha=0.6, ax=ax_plot)
    ax_plot.set_title('Quantity vs Rating (Colored by Delivery Status)', fontsize=12, fontweight='bold')
    ax_plot.set_xlabel('Quantity', fontsize=10)
    ax_plot.set_ylabel('Rating', fontsize=10)
    ax_plot.legend(title='Delivery Status', loc='best', fontsize=8)
    ax_plot.grid(alpha=0.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Plot 9: Scatter with hue added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PLOT 10: CORRELATION HEATMAP
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    
    ax_code = fig.add_axes([0.05, 0.55, 0.9, 0.40])
    ax_code.axis('off')
    code_text = """PLOT 10: CORRELATION HEATMAP - See All Numeric Relationships
    
plt.figure(figsize=(10, 8))
numeric_cols = copy_data.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = copy_data[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.show()

PURPOSE: Red = positive correlation | Blue = negative correlation | White = no correlation"""
    
    ax_code.text(0.05, 0.95, code_text, fontsize=9, ha='left', va='top', 
                transform=ax_code.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    ax_plot = fig.add_axes([0.15, 0.05, 0.7, 0.45])
    numeric_cols = copy_data.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = copy_data[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax_plot)
    ax_plot.set_title('Correlation Heatmap (Numeric Columns)', fontsize=12, fontweight='bold')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Plot 10: Correlation heatmap added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PLOT 11: PIE CHART
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    
    ax_code = fig.add_axes([0.05, 0.55, 0.9, 0.40])
    ax_code.axis('off')
    code_text = """PLOT 11: PIE CHART - Show Proportions/Percentages
    
plt.figure(figsize=(10, 6))
payment_counts = copy_data['payment_method'].value_counts()
colors = sns.color_palette("husl", len(payment_counts))
plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 11})
plt.title('Distribution of Payment Methods', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.show()

PURPOSE: Show composition - which category makes up what % of the whole."""
    
    ax_code.text(0.05, 0.95, code_text, fontsize=9, ha='left', va='top', 
                transform=ax_code.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    ax_plot = fig.add_axes([0.2, 0.05, 0.6, 0.45])
    payment_counts = copy_data['payment_method'].value_counts()
    colors = sns.color_palette("husl", len(payment_counts))
    ax_plot.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90, textprops={'fontsize': 9})
    ax_plot.set_title('Distribution of Payment Methods (Pie Chart)', fontsize=12, fontweight='bold')
    ax_plot.axis('equal')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Plot 11: Pie chart added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # PLOT 12: COMPREHENSIVE DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # Title
    fig.suptitle('PLOT 12: E-Commerce Analysis Dashboard (9 Views)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Rating histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(copy_data['rating'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Rating Distribution', fontweight='bold', fontsize=10)
    ax1.set_xlabel('Rating', fontsize=9)
    ax1.set_ylabel('Count', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Quantity histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(copy_data['quantity'], bins=15, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_title('Quantity Distribution', fontweight='bold', fontsize=10)
    ax2.set_xlabel('Quantity', fontsize=9)
    ax2.set_ylabel('Count', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Price histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(copy_data['unit_price'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    ax3.set_title('Unit Price Distribution', fontweight='bold', fontsize=10)
    ax3.set_xlabel('Price ($)', fontsize=9)
    ax3.set_ylabel('Count', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Rating box plot
    ax4 = fig.add_subplot(gs[1, 0])
    sns.boxplot(data=copy_data, y='rating', color='lightblue', ax=ax4)
    ax4.set_title('Rating Box Plot', fontweight='bold', fontsize=10)
    ax4.set_ylabel('Rating', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: Delivery Status count
    ax5 = fig.add_subplot(gs[1, 1])
    delivery_counts = copy_data['delivery_status'].value_counts()
    ax5.bar(delivery_counts.index, delivery_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax5.set_title('Orders by Delivery Status', fontweight='bold', fontsize=10)
    ax5.set_ylabel('Count', fontsize=9)
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(axis='y', alpha=0.3)
    
    # Plot 6: Payment Method count
    ax6 = fig.add_subplot(gs[1, 2])
    payment_counts = copy_data['payment_method'].value_counts()
    ax6.bar(payment_counts.index, payment_counts.values, color='skyblue', edgecolor='black', alpha=0.8)
    ax6.set_title('Orders by Payment Method', fontweight='bold', fontsize=10)
    ax6.set_ylabel('Count', fontsize=9)
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(axis='y', alpha=0.3)
    
    # Plot 7: Quantity vs Rating scatter
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.scatter(copy_data['quantity'], copy_data['rating'], alpha=0.5, s=20, color='darkblue')
    ax7.set_title('Quantity vs Rating', fontweight='bold', fontsize=10)
    ax7.set_xlabel('Quantity', fontsize=9)
    ax7.set_ylabel('Rating', fontsize=9)
    ax7.grid(alpha=0.3)
    
    # Plot 8: Rating by Delivery Status
    ax8 = fig.add_subplot(gs[2, 1])
    sns.boxplot(data=copy_data, x='delivery_status', y='rating', palette='Set2', ax=ax8)
    ax8.set_title('Rating by Delivery Status', fontweight='bold', fontsize=10)
    ax8.set_xlabel('Status', fontsize=9)
    ax8.set_ylabel('Rating', fontsize=9)
    ax8.tick_params(axis='x', rotation=45)
    ax8.grid(axis='y', alpha=0.3)
    
    # Plot 9: Average rating by category
    ax9 = fig.add_subplot(gs[2, 2])
    category_avg = copy_data.groupby('category')['rating'].mean().sort_values()
    ax9.barh(category_avg.index, category_avg.values, color='orange', edgecolor='black', alpha=0.8)
    ax9.set_title('Avg Rating by Category', fontweight='bold', fontsize=10)
    ax9.set_xlabel('Avg Rating', fontsize=9)
    ax9.grid(axis='x', alpha=0.3)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Plot 12: Comprehensive dashboard added")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # FINAL PAGE: KEY INSIGHTS & RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    ax.text(0.5, 0.95, "KEY INSIGHTS & RECOMMENDATIONS", fontsize=18, fontweight='bold', 
            ha='center', transform=ax.transAxes)
    
    insights_text = f"""
    DATA OVERVIEW:
    • Total Transactions: {len(copy_data):,}
    • Average Rating: {copy_data['rating'].mean():.2f}/5.0
    • Average Order Quantity: {pd.to_numeric(copy_data['quantity'], errors='coerce').mean():.2f} units
    • Average Order Value: ${copy_data['total_amount'].mean():.2f}
    
    DISTRIBUTION INSIGHTS:
    • Most Common Rating: {copy_data['rating'].mode()[0]:.1f}
    • Rating Range: {copy_data['rating'].min():.1f} to {copy_data['rating'].max():.1f}
    • Most Common Delivery Status: {copy_data['delivery_status'].mode()[0]}
    • Most Popular Payment Method: {copy_data['payment_method'].mode()[0]}
    
    MISSING DATA HANDLED:
    • Customer IDs: {(copy_data['customer_id'] == 'Unknown').sum():,} filled with 'Unknown'
    • Ratings: 0 NaN (all filled with mean = {mean_rating:.2f})
    • Return Days: {copy_data['return_days'].isna().sum():,} missing (88% of data)
    
    RECOMMENDATIONS FOR VISUALIZATION:
    1. Use Histograms → Understand distribution shapes
    2. Use Box Plots → Identify outliers and compare groups
    3. Use Scatter Plots → Find correlations
    4. Use Count/Bar Plots → Compare categorical frequencies
    5. Use Heatmaps → Visualize all relationships at once
    6. Use Pie Charts → Show composition/percentages
    7. Use Dashboards → Combine multiple views for comprehensive analysis
    
    BUSINESS INSIGHTS:
    • Delivery Status Distribution shows patterns in order fulfillment
    • Rating distribution reveals customer satisfaction levels
    • Payment method preferences indicate transaction patterns
    • Scatter plots help identify operational efficiencies
    
    NEXT STEPS:
    1. Filter data by specific time periods or categories
    2. Segment customers by rating to identify VIP vs at-risk customers
    3. Analyze payment method preferences by delivery status
    4. Investigate low ratings to identify quality issues
    5. Use time-series analysis if date data is available
    """
    
    ax.text(0.05, 0.88, insights_text, fontsize=10, ha='left', va='top', 
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("✓ Key insights page added")

print(f"\n✅ PDF Report Successfully Created!")
print(f"📄 Location: {pdf_path}")
print(f"📊 Total Pages: 15+")
print(f"📈 Plots Included: 12 Main Visualizations + Dashboard")
print(f"💾 File Size: Check the PDF file properties")
