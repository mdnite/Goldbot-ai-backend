"""Kiem tra doc lap: interaction terms (dxy_ret x real_yield_diff / real_yield_diff x
fed_rate_diff / dxy_ret x fed_rate_diff) co THAT SU anh huong toi quyet dinh cuoi cung
cua model 4a tren 50 moc test hay khong - hay chi ton tai o dang he so khac 0 nhung
khong du manh de lat ket qua (giong finding "Di_ngang thang sat nut" da co trong
giai_doan_4_report.md). Script tam, doi chieu 1 lan, khong sua model/pipeline chinh.
"""
import json

import joblib
import numpy as np
import pandas as pd

import giai_doan_4_train_model as m
from giai_doan_4_backtest_eval import build_features_asof, SAMPLE_DATES_PATH, BACKTEST_RESULTS_PATH

MODEL_PATH = "giai_doan_4_trend_model.joblib"

bundle = joblib.load(MODEL_PATH)
model, scaler = bundle["model"], bundle["scaler"]
kept_features = bundle["kept_features"]
vol_window = bundle["vol_window"]
ma_window = bundle["ma_window"]
lags = bundle["lags"]
macro_pairs = [tuple(p) for p in bundle["macro_pairs"]]

print("Kept features (6, model production):", kept_features)
print("Classes:", model.classes_)
print("Multi_class mode:", model.multi_class if hasattr(model, "multi_class") else "?")

INTERACTION_FEATS = [f for f in kept_features if f.startswith("inter_")]
MAIN_FEATS = [f for f in kept_features if not f.startswith("inter_")]
print("\nMain-effect features:", MAIN_FEATS)
print("Interaction features:", INTERACTION_FEATS)

with open(SAMPLE_DATES_PATH, encoding="utf-8") as f:
    sample_dates = json.load(f)["sample_dates"]
with open(BACKTEST_RESULTS_PATH, encoding="utf-8") as f:
    backtest_results = {r["as_of_date"]: r for r in json.load(f)}

_, base = m.load_series()

rows = []
for d in sample_dates:
    feats = build_features_asof(base, d, vol_window, ma_window, lags, macro_pairs)
    if feats is None:
        continue
    rows.append((d, feats))

X = pd.DataFrame([f for _, f in rows], columns=kept_features)
X_scaled = scaler.transform(X)

coef = model.coef_          # (n_classes, n_features)
intercept = model.intercept_  # (n_classes,)
classes = list(model.classes_)
feat_idx = {f: i for i, f in enumerate(kept_features)}

# --- 1) Bien do dong gop logit: main-effect vs interaction, tren tung mocday, tung class ---
rows_report = []
for (d, _), xs in zip(rows, X_scaled):
    logits = intercept + coef @ xs  # (n_classes,)
    pred_idx = int(np.argmax(logits))
    sorted_idx = np.argsort(logits)[::-1]
    margin = logits[sorted_idx[0]] - logits[sorted_idx[1]]  # logit gap giua top1/top2

    # dong gop tuyet doi cua main vs interaction VAO logit cua class duoc du doan
    main_contrib = sum(abs(coef[pred_idx, feat_idx[f]] * xs[feat_idx[f]]) for f in MAIN_FEATS)
    inter_contrib = sum(abs(coef[pred_idx, feat_idx[f]] * xs[feat_idx[f]]) for f in INTERACTION_FEATS)

    rows_report.append({
        "as_of_date": d,
        "pred": classes[pred_idx],
        "logit_margin_top1_top2": margin,
        "main_contrib_abs": main_contrib,
        "inter_contrib_abs": inter_contrib,
        "inter_share_pct": 100 * inter_contrib / (main_contrib + inter_contrib) if (main_contrib + inter_contrib) > 0 else None,
    })

df = pd.DataFrame(rows_report)
print("\n=== Ty trong dong gop |logit| cua interaction terms so voi main-effect (tren class du doan) ===")
print(df[["inter_share_pct"]].describe())
print("\n5 moc co inter_share_pct CAO NHAT (interaction anh huong nhieu nhat):")
print(df.sort_values("inter_share_pct", ascending=False).head(5)[["as_of_date", "pred", "logit_margin_top1_top2", "inter_share_pct"]])
print("\n5 moc co inter_share_pct THAP NHAT:")
print(df.sort_values("inter_share_pct").head(5)[["as_of_date", "pred", "logit_margin_top1_top2", "inter_share_pct"]])

# --- 2) ABLATION: khoa (zero-out) 2 feature interaction (~= gia tri trung binh train, do
# da chuan hoa StandardScaler), xem du doan co doi khong tai bat ky moc nao trong 50 moc ---
X_scaled_ablated = X_scaled.copy()
for f in INTERACTION_FEATS:
    X_scaled_ablated[:, feat_idx[f]] = 0.0  # = gia tri trung binh cua feature do tren TRAIN+VAL (sau StandardScaler)

logits_orig = intercept + X_scaled @ coef.T
logits_ablated = intercept + X_scaled_ablated @ coef.T
pred_orig = np.array(classes)[np.argmax(logits_orig, axis=1)]
pred_ablated = np.array(classes)[np.argmax(logits_ablated, axis=1)]

n_changed = int((pred_orig != pred_ablated).sum())
print(f"\n=== ABLATION: tat 2 interaction feature (dua ve gia tri trung binh) ===")
print(f"So moc du doan THAY DOI nhan khi tat interaction: {n_changed}/{len(pred_orig)}")
if n_changed > 0:
    changed_dates = [d for (d, _), o, a in zip(rows, pred_orig, pred_ablated) if o != a]
    print("Cac moc thay doi:", changed_dates)

# --- 3) Phan phoi gia tri THO (chua chuan hoa) cua 2 interaction feature tren 50 moc test,
# doi chieu voi do lech chuan tren TRAIN+VAL (de xem gia tri test co "binh thuong"/nho khong) ---
print("\n=== Gia tri THO cua interaction feature tren 50 moc test (chua chuan hoa) ===")
for f in INTERACTION_FEATS:
    vals = X[f].values
    print(f"{f}: min={vals.min():.6f} max={vals.max():.6f} mean={vals.mean():.6f} std={vals.std():.6f}")
    print(f"   scaler mean (TRAIN+VAL)={scaler.mean_[feat_idx[f]]:.6f}, scaler std (TRAIN+VAL)={scaler.scale_[feat_idx[f]]:.6f}")
