"""Kiem tra doc lap VIF cho model production cuoi (TRAIN+VAL gop, vol=21) -
tai su dung dung logic giai_doan_4_refit_final.py, doi chieu 2 phuong phap:
(1) compute_vif goc (LinearRegression, co intercept mac dinh)
(2) statsmodels variance_inflation_factor (voi add_constant, chuan hoc thuat)
Script tam, khong phai 1 phan pipeline chinh thuc - chi de doi chieu 1 lan.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

import giai_doan_4_train_model as m

VOL_WINDOW = 21
HORIZON = 21

wide, base = m.load_series()
gold = wide["gold"]
r_t_full = (gold.shift(-HORIZON) - gold) / gold

features = m.build_features(base, VOL_WINDOW)
data = features.copy()
data["r_t"] = r_t_full
data = data.dropna()

combined = data[data.index <= m.VAL_END]
print(f"So dong TRAIN+VAL gop: {len(combined)} ({combined.index.min().date()} -> {combined.index.max().date()})")

feature_cols = [c for c in data.columns if c != "r_t"]
X = combined[feature_cols]
print(f"So feature truoc loc: {len(feature_cols)}")

# --- Phuong phap 1: compute_vif goc cua du an (LinearRegression) ---
vif_custom = m.compute_vif(X)
print("\n=== Phuong phap 1 (LinearRegression, code goc du an) ===")
print("Max VIF:", vif_custom.max(), "tai feature:", vif_custom.idxmax())
print("So feature VIF > 5.0:", (vif_custom > 5.0).sum())
print("Top 10:")
print(vif_custom.head(10))

# --- Phuong phap 2: statsmodels chuan, co them constant ---
Xc = add_constant(X)
vif_sm = pd.Series(
    {col: variance_inflation_factor(Xc.values, Xc.columns.get_loc(col)) for col in X.columns}
).sort_values(ascending=False)
print("\n=== Phuong phap 2 (statsmodels, add_constant) ===")
print("Max VIF:", vif_sm.max(), "tai feature:", vif_sm.idxmax())
print("So feature VIF > 5.0:", (vif_sm > 5.0).sum())
print("Top 10:")
print(vif_sm.head(10))

# --- Doi chieu 2 phuong phap tren cung feature, cung thu tu ---
diff = (vif_custom - vif_sm).abs()
print("\n=== Doi chieu chenh lech tuyet doi giua 2 phuong phap ===")
print("Max abs diff:", diff.max())
print("Mean abs diff:", diff.mean())

# --- Kiem tra dieu kien so (condition number) - phat hien da cong tuyen BAC CAO
# ma VIF tung feature co the bo sot neu co > 1 to hop tuyen tinh gan hoan hao cung luc ---
Xs = (X - X.mean()) / X.std()
cond_number = np.linalg.cond(Xs.values)
print(f"\nCondition number (tren feature da chuan hoa, TRAIN+VAL): {cond_number:.2f}")
print("(Kinh nghiem: >30 dang lo ngai da cong tuyen bac cao, >100 nghiem trong)")

eigvals = np.linalg.eigvalsh(Xs.corr().values)
print(f"Eigenvalue nho nhat cua ma tran tuong quan: {eigvals.min():.5f} (gan 0 = gan singular)")
