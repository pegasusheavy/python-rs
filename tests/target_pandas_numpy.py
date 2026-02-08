"""
Target test script: comprehensive Pandas + NumPy usage.

This script is the acceptance test for python-rs full compatibility.
It exercises: imports, classes, dunder methods, keyword arguments,
list/dict comprehensions, lambdas, closures, exceptions, slicing,
method chaining, string formatting, and the full Pandas/NumPy stack.

Must produce identical output when run under CPython 3.11+ and python-rs.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# =============================================================================
# 1. NumPy fundamentals
# =============================================================================

# Array creation and basic ops
a = np.array([1, 2, 3, 4, 5], dtype=np.float64)
b = np.arange(1, 6, dtype=np.float64)
assert np.array_equal(a, b), "arange mismatch"

# Vectorized arithmetic
c = a * 2 + b ** 2 - 1
print("vectorized:", c)  # [2. 7. 14. 23. 34.]

# Broadcasting
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row = np.array([10, 20, 30])
broadcast_result = matrix + row
print("broadcast:\n", broadcast_result)

# Reshape, transpose, slicing
reshaped = np.arange(12).reshape(3, 4)
print("reshaped:\n", reshaped)
print("transposed:\n", reshaped.T)
print("slice [1:, ::2]:", reshaped[1:, ::2])

# Linear algebra
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])
print("matmul:\n", A @ B)
print("det(A):", np.linalg.det(A))
print("inv(A):\n", np.linalg.inv(A))

eigenvalues, eigenvectors = np.linalg.eig(A)
print("eigenvalues:", eigenvalues)

# Aggregations with axis
data = np.random.seed(42)
rand_matrix = np.random.randn(4, 3)
print("col means:", rand_matrix.mean(axis=0))
print("row sums:", rand_matrix.sum(axis=1))
print("global std:", rand_matrix.std())

# Boolean indexing
arr = np.arange(20)
mask = (arr % 3 == 0) | (arr % 5 == 0)
print("fizzbuzz indices:", arr[mask])

# Fancy indexing + np.where
scores = np.array([85, 42, 91, 67, 73, 55, 98, 30])
grades = np.where(scores >= 70, "pass", "fail")
print("grades:", grades)

# Stacking and splitting
top = np.ones((2, 3))
bottom = np.zeros((2, 3))
stacked = np.vstack([top, bottom])
left, right = np.hsplit(stacked, [1])
print("vstack shape:", stacked.shape)
print("hsplit left shape:", left.shape)

# Structured array (record-like)
dt = np.dtype([("name", "U10"), ("age", "i4"), ("score", "f8")])
people = np.array([("Alice", 30, 9.5), ("Bob", 25, 8.1), ("Charlie", 35, 7.3)], dtype=dt)
print("structured names:", people["name"])
print("mean score:", people["score"].mean())


# =============================================================================
# 2. Pandas fundamentals
# =============================================================================

# DataFrame construction from dict
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve",
             "Frank", "Grace", "Hank", "Ivy", "Jack"],
    "department": ["Engineering", "Marketing", "Engineering", "Sales",
                   "Marketing", "Engineering", "Sales", "Marketing",
                   "Engineering", "Sales"],
    "salary": [95000, 62000, 87000, 71000, 58000,
               105000, 68000, 54000, 92000, 76000],
    "years_exp": [8, 3, 6, 4, 2, 12, 5, 1, 7, 3],
    "performance": [4.2, 3.8, 4.5, 3.1, 4.0, 4.8, 3.5, 2.9, 4.1, 3.7],
    "remote": [True, False, True, False, True,
               True, False, False, True, False],
})

print("\n=== DataFrame ===")
print(df.head())
print("\nshape:", df.shape)
print("\ndtypes:\n", df.dtypes)
print("\ndescribe:\n", df.describe())

# Column operations and new column creation
df["salary_k"] = df["salary"] / 1000
df["bonus"] = df.apply(
    lambda row: row["salary"] * 0.15 if row["performance"] >= 4.0 else row["salary"] * 0.08,
    axis=1,
)
df["total_comp"] = df["salary"] + df["bonus"]
print("\nwith bonus:\n", df[["name", "salary", "bonus", "total_comp"]].head())


# =============================================================================
# 3. Filtering, sorting, and selection
# =============================================================================

# Boolean filtering with multiple conditions
senior_high_perf = df[(df["years_exp"] >= 5) & (df["performance"] >= 4.0)]
print("\nsenior high performers:\n", senior_high_perf[["name", "years_exp", "performance"]])

# .query() method
marketing_or_sales = df.query("department in ['Marketing', 'Sales'] and salary > 60000")
print("\nmarketing/sales >60k:\n", marketing_or_sales[["name", "department", "salary"]])

# Sorting
sorted_df = df.sort_values(["department", "salary"], ascending=[True, False])
print("\nsorted by dept then salary desc:\n", sorted_df[["name", "department", "salary"]])

# .loc and .iloc
print("\nloc[2:4, 'name':'salary']:\n", df.loc[2:4, "name":"salary"])
print("\niloc[::3, [0, 2, 4]]:\n", df.iloc[::3, [0, 2, 4]])

# .at and .iat for scalar access
print("\nat[0, 'name']:", df.at[0, "name"])
print("iat[1, 2]:", df.iat[1, 2])


# =============================================================================
# 4. GroupBy — the core of Pandas analytics
# =============================================================================

print("\n=== GroupBy ===")

# Basic aggregation
dept_stats = df.groupby("department").agg(
    avg_salary=("salary", "mean"),
    max_salary=("salary", "max"),
    headcount=("name", "count"),
    avg_perf=("performance", "mean"),
)
print("\ndept stats:\n", dept_stats)

# Multiple aggregations on same column
salary_agg = df.groupby("department")["salary"].agg(["mean", "median", "std", "min", "max"])
print("\nsalary aggregations:\n", salary_agg)

# GroupBy with transform (broadcast back to original shape)
df["dept_avg_salary"] = df.groupby("department")["salary"].transform("mean")
df["salary_vs_dept"] = df["salary"] - df["dept_avg_salary"]
print("\nsalary vs dept avg:\n", df[["name", "department", "salary", "salary_vs_dept"]])

# GroupBy with filter
big_departments = df.groupby("department").filter(lambda g: len(g) >= 3)
print("\nbig departments only:\n", big_departments[["name", "department"]])

# GroupBy + apply with custom function
def dept_summary(group):
    return pd.Series({
        "total_salary": group["salary"].sum(),
        "salary_range": group["salary"].max() - group["salary"].min(),
        "pct_remote": group["remote"].mean() * 100,
    })

dept_custom = df.groupby("department").apply(dept_summary)
print("\ncustom dept summary:\n", dept_custom)


# =============================================================================
# 5. Pivot tables, crosstabs, reshaping
# =============================================================================

print("\n=== Reshaping ===")

# Pivot table
pivot = df.pivot_table(
    values="salary",
    index="department",
    columns="remote",
    aggfunc=["mean", "count"],
)
print("\npivot table:\n", pivot)

# Crosstab
ct = pd.crosstab(df["department"], df["remote"], margins=True)
print("\ncrosstab:\n", ct)

# Melt (wide → long)
wide_df = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "math": [90, 85],
    "science": [88, 92],
    "english": [76, 81],
})
long_df = wide_df.melt(id_vars="name", var_name="subject", value_name="score")
print("\nmelted:\n", long_df)

# Stack and unstack
stacked_series = df.set_index(["department", "name"])["salary"]
print("\nstacked:\n", stacked_series.head(6))


# =============================================================================
# 6. String operations
# =============================================================================

print("\n=== String Ops ===")

df["name_upper"] = df["name"].str.upper()
df["name_len"] = df["name"].str.len()
df["dept_short"] = df["department"].str[:3]
contains_a = df[df["name"].str.contains("a", case=False)]
print("names containing 'a':\n", contains_a["name"].tolist())
print("name lengths:", df["name_len"].tolist())


# =============================================================================
# 7. Time series
# =============================================================================

print("\n=== Time Series ===")

dates = pd.date_range("2024-01-01", periods=365, freq="D")
ts = pd.Series(np.cumsum(np.random.randn(365)) + 100, index=dates, name="price")
print("time series head:\n", ts.head())
print("time series tail:\n", ts.tail())

# Resampling
monthly = ts.resample("ME").agg(["mean", "min", "max", "last"])
print("\nmonthly resampled:\n", monthly.head())

# Rolling window
rolling_avg = ts.rolling(window=30).mean()
rolling_std = ts.rolling(window=30).std()
print("\n30-day rolling mean (last 5):\n", rolling_avg.tail())

# Shift and pct_change
ts_shifted = ts.shift(1)
daily_returns = ts.pct_change().dropna()
print("\ndaily returns stats:")
print(f"  mean:   {daily_returns.mean():.6f}")
print(f"  std:    {daily_returns.std():.6f}")
print(f"  sharpe: {daily_returns.mean() / daily_returns.std() * np.sqrt(252):.4f}")


# =============================================================================
# 8. Merge, join, concat
# =============================================================================

print("\n=== Merge/Join ===")

# Left table
orders = pd.DataFrame({
    "order_id": range(1, 8),
    "customer": ["Alice", "Bob", "Alice", "Charlie", "Bob", "Diana", "Alice"],
    "amount": [120, 85, 200, 150, 95, 310, 175],
})

# Right table
customers = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Eve"],
    "tier": ["gold", "silver", "gold", "bronze"],
    "signup_year": [2020, 2021, 2019, 2023],
})

# Inner, left, outer merges
inner = pd.merge(orders, customers, left_on="customer", right_on="name", how="inner")
left = pd.merge(orders, customers, left_on="customer", right_on="name", how="left")
print("inner merge:\n", inner)
print("\nleft merge (note NaNs):\n", left)

# Concat
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
concatenated = pd.concat([df1, df2], ignore_index=True)
print("\nconcatenated:\n", concatenated)


# =============================================================================
# 9. Missing data handling
# =============================================================================

print("\n=== Missing Data ===")

messy = pd.DataFrame({
    "x": [1, 2, np.nan, 4, np.nan, 6],
    "y": [np.nan, 2, 3, np.nan, 5, 6],
    "z": ["a", "b", None, "d", "e", None],
})
print("messy:\n", messy)
print("\nisnull counts:\n", messy.isnull().sum())

# Fill strategies
print("\nffill:\n", messy.ffill())
print("\nfillna(0):\n", messy.fillna({"x": 0, "y": messy["y"].mean()}))
print("\ndropna:\n", messy.dropna())
print("\ninterpolate x:\n", messy["x"].interpolate())


# =============================================================================
# 10. Advanced: window functions, rank, categoricals, pipe
# =============================================================================

print("\n=== Advanced ===")

# Rank
df["salary_rank"] = df["salary"].rank(ascending=False, method="dense").astype(int)
print("salary rankings:\n", df[["name", "salary", "salary_rank"]].sort_values("salary_rank"))

# Categorical
df["perf_tier"] = pd.cut(
    df["performance"],
    bins=[0, 3.0, 3.5, 4.0, 5.0],
    labels=["low", "mid", "high", "top"],
)
print("\nperformance tiers:\n", df["perf_tier"].value_counts())

# Pipe for method chaining
def add_seniority(frame):
    frame = frame.copy()
    frame["seniority"] = pd.cut(
        frame["years_exp"],
        bins=[0, 2, 5, 10, 100],
        labels=["junior", "mid", "senior", "staff"],
    )
    return frame

def top_n_by_group(frame, group_col, value_col, n=2):
    return (
        frame.sort_values(value_col, ascending=False)
        .groupby(group_col, sort=False)
        .head(n)
        .reset_index(drop=True)
    )

result = (
    df
    .pipe(add_seniority)
    .pipe(top_n_by_group, "department", "salary")
)
print("\ntop 2 earners per dept:\n", result[["name", "department", "salary", "seniority"]])

# Expanding window
expanding_max = df.groupby("department")["salary"].expanding().max()
print("\nexpanding max salary by dept:\n", expanding_max)


# =============================================================================
# 11. NumPy + Pandas interop
# =============================================================================

print("\n=== NumPy/Pandas Interop ===")

# DataFrame from NumPy array
np_data = np.random.randn(5, 3)
np_df = pd.DataFrame(np_data, columns=["feat_1", "feat_2", "feat_3"])
print("df from numpy:\n", np_df)

# Back to NumPy
arr_back = np_df.to_numpy()
print("back to numpy shape:", arr_back.shape)
print("dtypes match:", arr_back.dtype == np.float64)

# Vectorized NumPy functions on Series
df["log_salary"] = np.log(df["salary"])
df["salary_zscore"] = (df["salary"] - df["salary"].mean()) / df["salary"].std()
print("\nz-scores:\n", df[["name", "salary", "salary_zscore"]].sort_values("salary_zscore"))

# Correlation matrix
numeric_cols = df[["salary", "years_exp", "performance"]].copy()
corr = numeric_cols.corr()
print("\ncorrelation matrix:\n", corr)


# =============================================================================
# 12. Exception handling and edge cases
# =============================================================================

print("\n=== Edge Cases ===")

# KeyError handling
try:
    _ = df["nonexistent_column"]
except KeyError as e:
    print(f"caught KeyError: {e}")

# Index alignment in operations
s1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
s2 = pd.Series([10, 20, 30], index=["b", "c", "d"])
aligned_sum = s1 + s2
print("\naligned sum (NaN where no overlap):\n", aligned_sum)

# Empty DataFrame operations
empty = pd.DataFrame(columns=["a", "b", "c"])
print("empty describe:\n", empty.describe())

# Chained comparison (pandas specific)
in_range = df[(df["salary"] >= 70000) & (df["salary"] <= 95000)]
print(f"\nsalaries in 70-95k range: {len(in_range)} employees")

# MultiIndex
multi_idx = df.set_index(["department", "name"])
print("\nMultiIndex:\n", multi_idx.head())
print("\nxs('Engineering'):\n", multi_idx.xs("Engineering"))


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("ALL PANDAS/NUMPY TESTS PASSED")
print(f"DataFrame shape: {df.shape}")
print(f"Total columns: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print("=" * 60)
