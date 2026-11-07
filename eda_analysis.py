import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# 1. LOAD DATASET
# =====================================

df = pd.read_csv("dataset/SuperMarket Analysis.csv")

print("DATASET LOADED SUCCESSFULLY!")

# =====================================
# 2. BASIC DATA EXPLORATION
# =====================================

print("\n========== FIRST FIVE ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# =====================================
# 3. DATA CLEANING
# =====================================

print("\n========== DATA CLEANING ==========")

print("Duplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Convert Date column
date_col = next(
    (col for col in df.columns if col.lower() == "date"),
    None
)

if date_col:
    df[date_col] = pd.to_datetime(
        df[date_col], errors="coerce"
    )

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv(
    "dataset/cleaned_supermarket_sales.csv",
    index=False
)

print("\nDATA CLEANING COMPLETED!")
print("Final Dataset Shape:", df.shape)

# =====================================
# 4. STATISTICAL ANALYSIS
# =====================================

print("\n========== STATISTICAL ANALYSIS ==========")

# Identify sales column automatically
sales_col = next(
    (col for col in df.columns
     if col.strip().lower() in ["sales", "total"]),
    None
)

if sales_col:

    df[sales_col] = pd.to_numeric(
        df[sales_col], errors="coerce"
    )

    print("\nAverage Sales:")
    print(df[sales_col].mean())

    print("\nMaximum Sales:")
    print(df[sales_col].max())

    print("\nMinimum Sales:")
    print(df[sales_col].min())

    print("\nTotal Revenue:")
    print(df[sales_col].sum())

    print("\nSales Standard Deviation:")
    print(df[sales_col].std())

else:
    print("Sales column not found!")

# =====================================
# 5. PRODUCT LINE ANALYSIS
# =====================================

product_col = next(
    (col for col in df.columns
     if col.strip().lower() == "product line"),
    None
)

if product_col and sales_col:

    print("\n========== PRODUCT LINE SALES ==========")

    product_sales = df.groupby(
        product_col
    )[sales_col].sum()

    print(product_sales)

# =====================================
# 6. PAYMENT METHOD ANALYSIS
# =====================================

payment_col = next(
    (col for col in df.columns
     if col.strip().lower() == "payment"),
    None
)

if payment_col:

    print("\n========== PAYMENT METHOD ANALYSIS ==========")

    print(df[payment_col].value_counts())

# =====================================
# 7. CUSTOMER ANALYSIS
# =====================================

customer_col = next(
    (col for col in df.columns
     if col.strip().lower() == "customer type"),
    None
)

if customer_col:

    print("\n========== CUSTOMER ANALYSIS ==========")

    print(df[customer_col].value_counts())

# =====================================
# 8. FINAL RESULT
# =====================================

print("\n====================================")
print("EDA BASIC ANALYSIS COMPLETED!")
print("====================================")

print("Total Records:", df.shape[0])
print("Total Columns:", df.shape[1])
print("Cleaned Dataset Saved Successfully!")

# =====================================
# 9. DATA VISUALIZATION
# =====================================

import os

os.makedirs("visualizations", exist_ok=True)

# BAR CHART - PRODUCT LINE SALES

product_sales = df.groupby(
    'Product line'
)[sales_col].sum()

plt.figure(figsize=(10, 6))

product_sales.plot(
    kind='bar',
    edgecolor='black'
)

plt.title("Sales by Product Line")
plt.xlabel("Product Line")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "visualizations/product_sales.png"
)

plt.show()

print("Bar Chart Saved Successfully!")

# PIE CHART - PAYMENT METHOD ANALYSIS

payment_counts = df['Payment'].value_counts()

plt.figure(figsize=(8, 8))

plt.pie(
    payment_counts,
    labels=payment_counts.index,
    autopct='%1.1f%%',
    startangle=90
)

plt.title("Payment Method Distribution")

plt.tight_layout()

plt.savefig(
    "visualizations/payment_distribution.png"
)

plt.show()
plt.close()

print("Pie Chart Saved Successfully!")

# HISTOGRAM - SALES DISTRIBUTION

plt.figure(figsize=(10, 6))

plt.hist(
    df[sales_col].dropna(),
    bins=20,
    edgecolor='black'
)

plt.title("Sales Distribution")
plt.xlabel("Sales Amount")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "visualizations/sales_histogram.png"
)

plt.show()
plt.close()

print("Histogram Saved Successfully!")

# CUSTOMER TYPE ANALYSIS - COUNT PLOT

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x='Customer type'
)

plt.title("Customer Type Distribution")
plt.xlabel("Customer Type")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "visualizations/customer_type_analysis.png"
)

plt.show()
plt.close()

print("Customer Type Chart Saved Successfully!")

# CORRELATION HEATMAP

numeric_df = df.select_dtypes(include=np.number)

correlation = numeric_df.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "visualizations/correlation_heatmap.png"
)

plt.show()
plt.close()

print("Correlation Heatmap Saved Successfully!")

# =====================================
# 10. KEY INSIGHTS
# =====================================

print("\n========== KEY INSIGHTS ==========")

# Best Selling Product
if product_col and sales_col:

    best_product = df.groupby(
        product_col
    )[sales_col].sum().idxmax()

    print("\nBest Selling Product:")
    print(best_product)

# Most Used Payment Method
if payment_col:

    popular_payment = df[payment_col].value_counts().idxmax()

    print("\nMost Popular Payment Method:")
    print(popular_payment)

# Most Common Customer Type
if customer_col:

    popular_customer = df[customer_col].value_counts().idxmax()

    print("\nMost Common Customer Type:")
    print(popular_customer)

# Best Performing Branch
branch_col = next(
    (col for col in df.columns
     if col.lower() == "branch"),
    None
)

if branch_col and sales_col:

    best_branch = df.groupby(
        branch_col
    )[sales_col].sum().idxmax()

    print("\nBest Performing Branch:")
    print(best_branch)

print("\nKEY INSIGHTS GENERATED SUCCESSFULLY!")