import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "indicators.db"
HORIZON = 21  # trading days

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    "SELECT date, value FROM indicators WHERE indicator = 'gold' ORDER BY date",
    conn, parse_dates=["date"]
)
conn.close()

df = df.set_index("date")
gold = df["value"]

# 21-day forward return
r = (gold.shift(-HORIZON) - gold) / gold
r = r.dropna()

# Chronological 80/20 split (giong dung split se dung khi train that)
split_idx = int(len(r) * 0.8)
r_train = r.iloc[:split_idx]
r_test = r.iloc[split_idx:]

print(f"So dong r_t: total={len(r)}, train={len(r_train)}, test={len(r_test)}")
print(f"Train range: {r_train.index.min().date()} -> {r_train.index.max().date()}")
print(f"Test range: {r_test.index.min().date()} -> {r_test.index.max().date()}")

# --- Method 1: sigma thuong ---
mean_train = r_train.mean()
std_train = r_train.std()
print(f"\n[Sigma] mean_train={mean_train:.5f}, std_train={std_train:.5f}")

lo, hi = r_train.quantile(0.01), r_train.quantile(0.99)
r_train_trimmed = r_train[(r_train >= lo) & (r_train <= hi)]
std_train_trimmed = r_train_trimmed.std()
print(f"[Sigma] std_train sau khi trim 1% hai duoi: {std_train_trimmed:.5f} "
      f"(thay doi {(std_train_trimmed/std_train - 1)*100:+.1f}% so voi std goc)")

# --- Method 2: MAD chong outlier ---
median_train = r_train.median()
mad_train = (r_train - median_train).abs().median()
robust_scale = 1.4826 * mad_train
print(f"\n[MAD] median_train={median_train:.5f}, robust_scale={robust_scale:.5f}")

# --- Method 3: percentile 33/67 ---
p33, p67 = r_train.quantile(1/3), r_train.quantile(2/3)
print(f"\n[Percentile] p33={p33:.5f}, p67={p67:.5f}")

def label_counts(series, low, high):
    labels = pd.cut(series, bins=[-np.inf, low, high, np.inf], labels=["Giam", "Di_ngang", "Tang"])
    counts = labels.value_counts(normalize=True) * 100
    return counts.reindex(["Giam", "Di_ngang", "Tang"])

k = 0.5
print("\n=== Ty le nhan (%) tren TRAIN vs TEST theo tung phuong phap ===")
for name, low, high in [
    ("Sigma (mean +/- 0.5*std)", mean_train - k * std_train, mean_train + k * std_train),
    ("Robust MAD (median +/- 0.5*robust_scale)", median_train - k * robust_scale, median_train + k * robust_scale),
    ("Percentile (p33/p67)", p33, p67),
]:
    print(f"\n-- {name} --")
    print("Train:", dict(label_counts(r_train, low, high).round(1)))
    print("Test: ", dict(label_counts(r_test, low, high).round(1)))