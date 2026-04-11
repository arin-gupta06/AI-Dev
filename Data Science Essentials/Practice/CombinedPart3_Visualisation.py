"""
═══════════════════════════════════════════════════════════════════════════════════
                    VISUALIZATION MODULE - OOP STRUCTURE
                    E-COMMERCE DATA VISUALIZATION CLASS
═══════════════════════════════════════════════════════════════════════════════════

This module provides a comprehensive visualization system using Object-Oriented 
Programming (OOP) for modular, reusable, and maintainable code.

Usage:
    viz = ECommerceVisualizer(df)
    viz.plot_rating_histogram()
    viz.plot_correlation_heatmap()
    viz.plot_comprehensive_dashboard()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


class ECommerceVisualizer:
    """
    Main visualization class for E-Commerce transaction data analysis.
    
    Attributes:
        df (pd.DataFrame): Input dataframe containing e-commerce data
        fig_size (tuple): Default figure size for plots
        style (str): Matplotlib style theme
        palette (str): Seaborn color palette
    """
    
    def __init__(self, dataframe: pd.DataFrame, fig_size: tuple = (12, 6), 
                 style: str = 'seaborn-v0_8-darkgrid', palette: str = 'husl'):
        """
        Initialize the ECommerceVisualizer.
        
        Args:
            dataframe (pd.DataFrame): The data to visualize
            fig_size (tuple): Default figure size (width, height)
            style (str): Matplotlib style theme
            palette (str): Seaborn color palette
        """
        self.df = dataframe.copy()
        self.fig_size = fig_size
        self.style = style
        self.palette = palette
        self._setup_style()
    
    def _setup_style(self):
        """Configure visualization style settings."""
        plt.style.use(self.style)
        sns.set_palette(self.palette)
        plt.rcParams['figure.figsize'] = self.fig_size
        plt.rcParams['font.size'] = 10
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION 1: UNIVARIATE DISTRIBUTION PLOTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def plot_rating_histogram(self, bins: int = 20, title_suffix: str = "") -> None:
        """
        Plot histogram of customer ratings.
        
        Args:
            bins (int): Number of bins for histogram
            title_suffix (str): Additional text for title
        """
        plt.figure(figsize=(10, 5))
        plt.hist(self.df['rating'], bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of Customer Ratings (Histogram){title_suffix}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Rating Value', fontsize=12)
        plt.ylabel('Frequency (Count)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_rating_kde(self, bins: int = 20) -> None:
        """
        Plot histogram with KDE (Kernel Density Estimation) overlay.
        
        Args:
            bins (int): Number of bins for histogram
        """
        plt.figure(figsize=(10, 5))
        sns.histplot(data=self.df, x='rating', kde=True, bins=bins, color='steelblue')
        plt.title('Rating Distribution with KDE Curve (Smooth Trend)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_quantity_histogram(self, bins: int = 15) -> None:
        """
        Plot histogram of order quantities.
        
        Args:
            bins (int): Number of bins for histogram
        """
        plt.figure(figsize=(10, 5))
        plt.hist(self.df['quantity'], bins=bins, color='coral', edgecolor='black', alpha=0.7)
        plt.title('Distribution of Order Quantities (Histogram)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Quantity', fontsize=12)
        plt.ylabel('Frequency (Count)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_price_histogram(self, bins: int = 20) -> None:
        """
        Plot histogram of unit prices.
        
        Args:
            bins (int): Number of bins for histogram
        """
        plt.figure(figsize=(10, 5))
        plt.hist(self.df['unit_price'], bins=bins, color='lightgreen', 
                edgecolor='black', alpha=0.7)
        plt.title('Distribution of Unit Prices (Histogram)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Price ($)', fontsize=12)
        plt.ylabel('Frequency (Count)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION 2: BOX PLOTS & OUTLIER DETECTION
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def plot_rating_boxplot(self) -> None:
        """
        Plot box plot showing rating distribution, median, quartiles, and outliers.
        """
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.df, y='rating', color='lightblue')
        plt.title('Rating Box Plot (Median, Quartiles, Outliers)', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Rating Value', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_rating_by_delivery_boxplot(self) -> None:
        """
        Plot grouped box plot comparing ratings by delivery status.
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='delivery_status', y='rating', palette='Set2')
        plt.title('Rating Distribution by Delivery Status (Grouped Box Plot)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Delivery Status', fontsize=12)
        plt.ylabel('Rating', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_rating_by_payment_boxplot(self) -> None:
        """
        Plot grouped box plot comparing ratings by payment method.
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='payment_method', y='rating', palette='Set2')
        plt.title('Rating Distribution by Payment Method (Grouped Box Plot)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Payment Method', fontsize=12)
        plt.ylabel('Rating', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION 3: VIOLIN & DENSITY PLOTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def plot_rating_violinplot(self) -> None:
        """
        Plot violin plot showing detailed distribution by delivery status.
        """
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=self.df, x='delivery_status', y='rating', palette='muted')
        plt.title('Rating Distribution by Delivery Status (Violin Plot - Shows Density)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Delivery Status', fontsize=12)
        plt.ylabel('Rating', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION 4: COUNT & CATEGORY PLOTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def plot_delivery_countplot(self) -> None:
        """
        Plot count plot showing frequency of each delivery status.
        """
        plt.figure(figsize=(12, 5))
        sns.countplot(data=self.df, x='delivery_status', palette='Set2', 
                     order=self.df['delivery_status'].value_counts().index)
        plt.title('Number of Transactions by Delivery Status (Count Plot)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Delivery Status', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_payment_countplot(self) -> None:
        """
        Plot count plot showing frequency of each payment method.
        """
        plt.figure(figsize=(12, 5))
        sns.countplot(data=self.df, x='payment_method', palette='Set3', 
                     order=self.df['payment_method'].value_counts().index)
        plt.title('Number of Transactions by Payment Method (Count Plot)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Payment Method', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION 5: BAR CHARTS & AGGREGATIONS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def plot_avg_rating_by_delivery(self) -> None:
        """
        Plot bar chart showing average rating by delivery status.
        """
        plt.figure(figsize=(10, 5))
        avg_rating = self.df.groupby('delivery_status')['rating'].mean()
        avg_rating.plot(kind='bar', color='coral', edgecolor='black', alpha=0.8)
        plt.title('Average Rating by Delivery Status (Bar Chart)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Delivery Status', fontsize=12)
        plt.ylabel('Average Rating', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_avg_rating_by_category(self) -> None:
        """
        Plot bar chart showing average rating by product category.
        """
        plt.figure(figsize=(10, 5))
        category_avg = self.df.groupby('category')['rating'].mean().sort_values()
        category_avg.plot(kind='barh', color='orange', edgecolor='black', alpha=0.8)
        plt.title('Average Rating by Product Category (Horizontal Bar Chart)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Average Rating', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_avg_rating_by_payment(self) -> None:
        """
        Plot bar chart showing average rating by payment method.
        """
        plt.figure(figsize=(10, 5))
        avg_rating = self.df.groupby('payment_method')['rating'].mean()
        avg_rating.plot(kind='bar', color='steelblue', edgecolor='black', alpha=0.8)
        plt.title('Average Rating by Payment Method (Bar Chart)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Payment Method', fontsize=12)
        plt.ylabel('Average Rating', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_grouped_payment_delivery(self) -> None:
        """
        Plot grouped bar chart showing transaction counts by payment method and delivery status.
        """
        plt.figure(figsize=(12, 6))
        grouped_data = self.df.groupby(['payment_method', 'delivery_status']).size().unstack(fill_value=0)
        grouped_data.plot(kind='bar', ax=plt.gca(), 
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8)
        plt.title('Transaction Count by Payment Method & Delivery Status (Grouped Bar)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Payment Method', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Delivery Status', loc='best')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION 6: SCATTER PLOTS & RELATIONSHIPS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def plot_quantity_vs_rating(self) -> None:
        """
        Plot scatter plot showing relationship between quantity and rating.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['quantity'], self.df['rating'], alpha=0.5, s=50, color='darkblue')
        plt.title('Quantity vs Rating (Scatter Plot - Check Correlation)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Quantity Ordered', fontsize=12)
        plt.ylabel('Customer Rating', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_quantity_vs_rating_colored(self) -> None:
        """
        Plot scatter plot with points colored by delivery status.
        """
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=self.df, x='quantity', y='rating', 
                       hue='delivery_status', s=100, alpha=0.6)
        plt.title('Quantity vs Rating (Colored by Delivery Status)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Quantity Ordered', fontsize=12)
        plt.ylabel('Customer Rating', fontsize=12)
        plt.legend(title='Delivery Status', loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_price_vs_total_amount(self) -> None:
        """
        Plot scatter plot showing unit price vs total amount, colored by quantity.
        """
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.df['unit_price'], self.df['total_amount'], 
                             c=self.df['quantity'], s=100, alpha=0.6, cmap='viridis')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Quantity', fontsize=10)
        plt.title('Unit Price vs Total Amount (Colored by Quantity)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Unit Price ($)', fontsize=12)
        plt.ylabel('Total Amount ($)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION 7: LINE PLOTS & TRENDS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def plot_category_rating_trend(self) -> None:
        """
        Plot line plot showing average rating trend across categories.
        """
        plt.figure(figsize=(10, 5))
        category_rating = self.df.groupby('category')['rating'].mean().sort_values()
        category_rating.plot(kind='line', marker='o', linewidth=2, markersize=8, color='green')
        plt.title('Average Rating by Product Category (Line Plot)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Average Rating', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION 8: PIE & DONUT CHARTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def plot_payment_method_pie(self) -> None:
        """
        Plot pie chart showing distribution of payment methods.
        """
        plt.figure(figsize=(10, 6))
        payment_counts = self.df['payment_method'].value_counts()
        colors = sns.color_palette("husl", len(payment_counts))
        plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90, textprops={'fontsize': 11})
        plt.title('Distribution of Payment Methods (Pie Chart)', 
                 fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def plot_delivery_status_donut(self) -> None:
        """
        Plot donut chart showing distribution of delivery status.
        """
        plt.figure(figsize=(10, 6))
        delivery_counts = self.df['delivery_status'].value_counts()
        colors = sns.color_palette("Set2", len(delivery_counts))
        plt.pie(delivery_counts, labels=delivery_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 11})
        # Create donut hole
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        plt.gca().add_artist(centre_circle)
        plt.title('Distribution of Delivery Status (Donut Chart)', 
                 fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION 9: CORRELATION & HEATMAPS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def plot_correlation_heatmap(self) -> None:
        """
        Plot correlation heatmap showing relationships between numeric columns.
        """
        plt.figure(figsize=(10, 8))
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap (Numeric Columns)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION 10: PAIRPLOT & MULTI-RELATIONSHIPS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def plot_pairplot(self, columns: Optional[List[str]] = None) -> None:
        """
        Plot pairplot showing relationships between all numeric columns.
        
        Args:
            columns (List[str]): Specific columns to include. If None, uses default columns.
        """
        if columns is None:
            columns = ['quantity', 'unit_price', 'rating', 'return_days']
        
        valid_cols = [col for col in columns if col in self.df.columns]
        print(f"Creating PairPlot with columns: {valid_cols}")
        
        pairplot = sns.pairplot(self.df[valid_cols].dropna(), diag_kind='hist', 
                               plot_kws={'alpha': 0.6})
        pairplot.fig.suptitle('Pairplot (All Numeric Column Relationships)', 
                             fontsize=14, fontweight='bold', y=1.001)
        plt.tight_layout()
        plt.show()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION 11: COMPARISON & IMPACT ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def plot_missing_values_impact(self) -> None:
        """
        Plot comparison of rating distribution with and without missing values.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot with missing values
        sns.histplot(data=self.df, x='rating', kde=True, bins=20, 
                    color='steelblue', ax=axes[0])
        axes[0].set_title(f'Rating Distribution (Includes {self.df["rating"].isna().sum()} NaN)', 
                         fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency')
        
        # Plot without missing values
        df_clean = self.df.dropna(subset=['rating'])
        sns.histplot(data=df_clean, x='rating', kde=True, bins=20, 
                    color='lightgreen', ax=axes[1])
        axes[1].set_title(f'Rating Distribution (NaN Removed - {len(df_clean)} clean rows)', 
                         fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency')
        
        plt.suptitle('Impact of Missing Values on Distribution', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION 12: COMPREHENSIVE DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def plot_comprehensive_dashboard(self) -> None:
        """
        Plot comprehensive 9-plot dashboard with key metrics and visualizations.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Rating histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.df['rating'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_title('Rating Distribution', fontweight='bold')
        ax1.set_xlabel('Rating')
        ax1.set_ylabel('Count')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Quantity histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.df['quantity'], bins=15, color='coral', edgecolor='black', alpha=0.7)
        ax2.set_title('Quantity Distribution', fontweight='bold')
        ax2.set_xlabel('Quantity')
        ax2.set_ylabel('Count')
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Unit Price histogram
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(self.df['unit_price'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax3.set_title('Unit Price Distribution', fontweight='bold')
        ax3.set_xlabel('Price ($)')
        ax3.set_ylabel('Count')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Rating box plot
        ax4 = fig.add_subplot(gs[1, 0])
        sns.boxplot(data=self.df, y='rating', color='lightblue', ax=ax4)
        ax4.set_title('Rating Box Plot', fontweight='bold')
        ax4.set_ylabel('Rating')
        ax4.grid(axis='y', alpha=0.3)
        
        # Plot 5: Delivery Status count
        ax5 = fig.add_subplot(gs[1, 1])
        delivery_counts = self.df['delivery_status'].value_counts()
        ax5.bar(delivery_counts.index, delivery_counts.values, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.8)
        ax5.set_title('Orders by Delivery Status', fontweight='bold')
        ax5.set_xlabel('Status')
        ax5.set_ylabel('Count')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(axis='y', alpha=0.3)
        
        # Plot 6: Payment Method count
        ax6 = fig.add_subplot(gs[1, 2])
        payment_counts = self.df['payment_method'].value_counts()
        ax6.bar(payment_counts.index, payment_counts.values, color='skyblue', 
               edgecolor='black', alpha=0.8)
        ax6.set_title('Orders by Payment Method', fontweight='bold')
        ax6.set_xlabel('Method')
        ax6.set_ylabel('Count')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(axis='y', alpha=0.3)
        
        # Plot 7: Quantity vs Rating scatter
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter(self.df['quantity'], self.df['rating'], alpha=0.5, s=30, color='darkblue')
        ax7.set_title('Quantity vs Rating', fontweight='bold')
        ax7.set_xlabel('Quantity')
        ax7.set_ylabel('Rating')
        ax7.grid(alpha=0.3)
        
        # Plot 8: Rating by Payment Method box plot
        ax8 = fig.add_subplot(gs[2, 1])
        sns.boxplot(data=self.df, x='payment_method', y='rating', palette='Set2', ax=ax8)
        ax8.set_title('Rating by Payment Method', fontweight='bold')
        ax8.set_xlabel('Payment Method')
        ax8.set_ylabel('Rating')
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(axis='y', alpha=0.3)
        
        # Plot 9: Average rating by category
        ax9 = fig.add_subplot(gs[2, 2])
        category_avg = self.df.groupby('category')['rating'].mean().sort_values()
        ax9.barh(category_avg.index, category_avg.values, color='orange', 
                edgecolor='black', alpha=0.8)
        ax9.set_title('Avg Rating by Category', fontweight='bold')
        ax9.set_xlabel('Average Rating')
        ax9.grid(axis='x', alpha=0.3)
        
        plt.suptitle('E-Commerce Comprehensive Dashboard (9 Views)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.show()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION 13: SUMMARY STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    def print_visualization_summary(self) -> None:
        """
        Print comprehensive summary statistics for the dataset.
        """
        print("\n" + "="*80)
        print("VISUALIZATION SUMMARY STATISTICS".center(80))
        print("="*80)
        
        # Rating Statistics
        print(f"\n{'RATINGS':30} | {'Value':15}")
        print("-" * 50)
        print(f"{'Average Rating':30} | {self.df['rating'].mean():.2f}")
        print(f"{'Median Rating':30} | {self.df['rating'].median():.2f}")
        print(f"{'Std Deviation':30} | {self.df['rating'].std():.2f}")
        print(f"{'Min/Max Rating':30} | {self.df['rating'].min():.1f} / {self.df['rating'].max():.1f}")
        
        # Delivery Status Distribution
        print(f"\n{'DELIVERY STATUS':30} | {'Count':15} | {'%':10}")
        print("-" * 60)
        delivery_dist = self.df['delivery_status'].value_counts()
        for status, count in delivery_dist.items():
            pct = (count / len(self.df)) * 100
            print(f"{status:30} | {count:15} | {pct:6.1f}%")
        
        # Payment Method Distribution
        print(f"\n{'PAYMENT METHOD':30} | {'Count':15} | {'%':10}")
        print("-" * 60)
        payment_dist = self.df['payment_method'].value_counts()
        for method, count in payment_dist.items():
            pct = (count / len(self.df)) * 100
            print(f"{method:30} | {count:15} | {pct:6.1f}%")
        
        print("\n" + "="*80 + "\n")
    
    def get_available_methods(self) -> List[str]:
        """
        Return list of all available visualization methods.
        
        Returns:
            List[str]: List of method names
        """
        methods = [method for method in dir(self) 
                  if method.startswith('plot_') or method.startswith('print_')]
        return methods


# ═══════════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Example usage of the ECommerceVisualizer class.
    """
    
    # Load dataset
    dataset = pd.read_csv("D://AI DEV//Data Science Essentials//Dataset//E_Commerce_Transactions_Raw.csv")
    data = pd.DataFrame(dataset)
    
    # Data preprocessing (same as original)
    copy_data = data.copy()
    copy_data["customer_id"] = copy_data["customer_id"].fillna("Unknown")
    copy_data["city"] = copy_data["city"].fillna(copy_data["city"].mode().values[0])
    copy_data["product_name"] = copy_data["product_name"].str.lower().str.strip()
    copy_data["product_name"] = copy_data["product_name"].fillna(copy_data["product_name"].mode().values[0])
    
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
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # CREATE VISUALIZER INSTANCE
    # ═══════════════════════════════════════════════════════════════════════════════════
    viz = ECommerceVisualizer(copy_data)
    
    # Print available methods
    print("\n" + "="*80)
    print("AVAILABLE VISUALIZATION METHODS".center(80))
    print("="*80)
    for i, method in enumerate(viz.get_available_methods(), 1):
        print(f"{i:2d}. {method}")
    print("="*80 + "\n")
    
    # Print summary statistics
    viz.print_visualization_summary()
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # CALL SPECIFIC VISUALIZATIONS AS NEEDED
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    # Example: Call individual plot methods
    # viz.plot_rating_histogram()
    # viz.plot_correlation_heatmap()
    # viz.plot_comprehensive_dashboard()
    
    print("VISUALIZER READY! Use: viz.plot_<method_name>() to display any visualization\n")
