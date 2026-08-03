"""Giai đoạn 4 - Model định lượng: sửa 3 vấn đề kỹ thuật đã xác nhận trong
trend_model.joblib (Giai đoạn 1.5), GIỮ NGUYÊN Logistic Regression (quyết định đã
khoá - không đổi thuật toán). Không sửa train_trend_model.py (đã chốt/report ở
Giai đoạn 1.5) - script này độc lập, ghi model mới ra file riêng.

3 vấn đề kỹ thuật được sửa:
1. Cô lập đặc trưng: thêm interaction terms thủ công giữa 3 chỉ báo macro
   (dxy_ret, real_yield_diff, fed_rate_diff - KHÔNG gồm gold_ret, KHÔNG dùng insight
   lỗi Fed rate từ audit LLM Giai đoạn 3 để thiết kế feature riêng vá đúng điểm đó).
2. Volatility trễ: thay Vol60 cố định bằng ứng viên cửa sổ ngắn hơn, chọn qua grid
   search trên validation (không chốt cứng trước).
3. Đa cộng tuyến: VIF thật (iterative, ngưỡng >5, tính bằng LinearRegression vì
   statsmodels chưa cài trong venv), không chỉ lọc pairwise corr như bản gốc.

3 tập tách biệt theo thời gian (không shuffle, seed cố định RANDOM_SEED):
- Train: đến TRAIN_END (~3009 dòng, chủ yếu regime range-bound 2010-2022)
- Validation: VAL_START -> VAL_END (~531 dòng, span cả 2 regime - chọn hyperparameter
  VÀ đánh giá tách riêng theo regime trong nội bộ validation)
- Test: KHÔNG đụng trong script này - xem giai_doan_4_backtest_eval.py, chạm 1 lần
  bằng đúng 50 mốc archive/giai_doan_3_llm_approach/giai_doan_3_sample_dates.json.

Grid round 1 (cố định trước khi chạy, tính cả lưới là 1 vòng trong giới hạn 5-6 vòng
chống overfit): 2 vol_window x 2 penalty x 4 C = 16 tổ hợp.
"""
import json
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DB_PATH = "indicators.db"
RESULTS_PATH = "giai_doan_4_train_results.json"

RANDOM_SEED = 42
LAGS = [1, 5, 10, 21, 60]
MA_WINDOW = 60  # không đổi - không phải vấn đề kỹ thuật bị nêu (chỉ Vol60 mới là vấn đề)
VOL_WINDOW_CANDIDATES = [10, 21]
K = 0.5  # bề rộng ngưỡng nhãn, giống Giai đoạn 1.5/3
VIF_THRESHOLD = 5.0

TRAIN_END = "2022-04-14"
VAL_START = "2022-04-18"
VAL_END = "2024-05-31"
REGIME_SPLIT_DATE = "2023-03-17"  # tách range-bound vs bull-market TRONG validation, khớp ranh giới train/test đã xác nhận ở Giai đoạn 1.5 (giai_doan_1_5_report.md)

PENALTY_GRID = ["l1", "l2"]
C_GRID = [0.01, 0.1, 1, 10]

MACRO_PAIRS = [
    ("dxy_ret", "real_yield_diff"),
    ("real_yield_diff", "fed_rate_diff"),
    ("dxy_ret", "fed_rate_diff"),
]


def load_series():
    conn = sqlite3.connect(DB_PATH)
    long_df = pd.read_sql_query("SELECT date, indicator, value FROM indicators", conn, parse_dates=["date"])
    conn.close()
    wide = long_df.pivot(index="date", columns="indicator", values="value").sort_index()
    wide = wide.dropna(subset=["gold", "dxy", "real_yield", "fed_rate"])
    base = {
        "gold_ret": wide["gold"].pct_change(),
        "dxy_ret": wide["dxy"].pct_change(),
        "real_yield_diff": wide["real_yield"].diff(),
        "fed_rate_diff": wide["fed_rate"].diff(),
    }
    return wide, base


def build_features(base, vol_window):
    """4 series gốc: lag[1,5,10,21,60] + ma60 + vol{vol_window} (thay Vol60 cố định).
    Cộng 6 interaction feature: lag1 + ma60 của 3 cặp macro (dxy_ret, real_yield_diff,
    fed_rate_diff) - đúng phương án đã chọn (không full lag/ma/vol cho interaction,
    tránh nổ chiều feature khi train ~3000 dòng)."""
    features = pd.DataFrame(index=base["gold_ret"].index)
    for name, s in base.items():
        for lag in LAGS:
            features[f"{name}_lag{lag}"] = s.shift(lag)
        features[f"{name}_ma{MA_WINDOW}"] = s.rolling(MA_WINDOW).mean()
        features[f"{name}_vol{vol_window}"] = s.rolling(vol_window).std()

    for a, b in MACRO_PAIRS:
        inter = base[a] * base[b]
        name = f"inter_{a}_x_{b}"
        features[f"{name}_lag1"] = inter.shift(1)
        features[f"{name}_ma{MA_WINDOW}"] = inter.rolling(MA_WINDOW).mean()

    return features


def compute_vif(X):
    """VIF_i = 1/(1-R^2) hồi quy feature i theo các feature còn lại (LinearRegression -
    statsmodels chưa cài trong venv, công thức tương đương)."""
    if X.shape[1] <= 1:
        return pd.Series({c: 1.0 for c in X.columns})
    vifs = {}
    for col in X.columns:
        y = X[col].values
        Xo = X.drop(columns=[col]).values
        r2 = LinearRegression().fit(Xo, y).score(Xo, y)
        vifs[col] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(vifs).sort_values(ascending=False)


def filter_by_vif(X, threshold):
    """Loại lặp: bỏ feature VIF cao nhất, tính lại toàn bộ, tới khi tất cả <= threshold."""
    X = X.copy()
    dropped = []
    vifs = compute_vif(X)
    while len(X.columns) > 1 and vifs.iloc[0] > threshold:
        worst = vifs.index[0]
        dropped.append((worst, float(vifs.iloc[0])))
        X = X.drop(columns=[worst])
        vifs = compute_vif(X)
    return X.columns.tolist(), dropped, vifs


def main():
    wide, base = load_series()
    gold = wide["gold"]
    horizon = 21
    r_t_full = (gold.shift(-horizon) - gold) / gold

    grid_results = []
    vif_reports = {}

    for vol_window in VOL_WINDOW_CANDIDATES:
        features = build_features(base, vol_window)
        data = features.copy()
        data["r_t"] = r_t_full
        data = data.dropna()

        train = data[data.index <= TRAIN_END]
        val = data[(data.index >= VAL_START) & (data.index <= VAL_END)]

        median_train = train["r_t"].median()
        mad_train = (train["r_t"] - median_train).abs().median()
        robust_scale = 1.4826 * mad_train
        low = median_train - K * robust_scale
        high = median_train + K * robust_scale

        def label_series(r, low=low, high=high):
            return pd.cut(r, bins=[-np.inf, low, high, np.inf], labels=["Giam", "Di_ngang", "Tang"]).astype(str)

        y_train = label_series(train["r_t"])
        y_val = label_series(val["r_t"])

        feature_cols = [c for c in data.columns if c != "r_t"]
        X_train_raw = train[feature_cols]
        X_val_raw = val[feature_cols]

        kept_features, dropped_vif, final_vifs = filter_by_vif(X_train_raw, VIF_THRESHOLD)
        vif_reports[vol_window] = {
            "n_before": len(feature_cols), "n_after": len(kept_features),
            "dropped": dropped_vif, "kept_features": kept_features,
            "final_vifs": final_vifs.to_dict(),
        }
        print(f"\n[vol_window={vol_window}] VIF filter: {len(feature_cols)} -> {len(kept_features)} feature "
              f"(loai {len(dropped_vif)}, nguong>{VIF_THRESHOLD})")
        for feat, v in dropped_vif:
            print(f"  loai {feat}: VIF={v:.2f}")

        X_train = X_train_raw[kept_features]
        X_val = X_val_raw[kept_features]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        pre_mask = val.index < REGIME_SPLIT_DATE
        post_mask = ~pre_mask

        for penalty in PENALTY_GRID:
            for C in C_GRID:
                model = LogisticRegression(
                    penalty=penalty, C=C, solver="saga", max_iter=5000,
                    random_state=RANDOM_SEED,
                )
                model.fit(X_train_scaled, y_train)
                y_val_pred = model.predict(X_val_scaled)
                val_acc = accuracy_score(y_val, y_val_pred)
                acc_pre = accuracy_score(y_val[pre_mask], y_val_pred[pre_mask]) if pre_mask.sum() else None
                acc_post = accuracy_score(y_val[post_mask], y_val_pred[post_mask]) if post_mask.sum() else None

                grid_results.append({
                    "vol_window": vol_window, "penalty": penalty, "C": C,
                    "n_features_kept": len(kept_features),
                    "val_acc": val_acc,
                    "val_acc_pre2023": acc_pre, "val_acc_post2023": acc_post,
                    "n_val_pre2023": int(pre_mask.sum()), "n_val_post2023": int(post_mask.sum()),
                    "label_low": float(low), "label_high": float(high),
                })

    grid_results.sort(key=lambda r: r["val_acc"], reverse=True)
    print("\n=== Round 1 grid search - top 5 theo val_acc (tong ca luoi tinh la 1 vong) ===")
    for r in grid_results[:5]:
        print(f"  vol={r['vol_window']:>2} penalty={r['penalty']:<2} C={r['C']:<5} "
              f"n_feat={r['n_features_kept']:>2} val_acc={r['val_acc']:.3f} "
              f"(pre2023={r['val_acc_pre2023']:.3f} n={r['n_val_pre2023']}, "
              f"post2023={r['val_acc_post2023']:.3f} n={r['n_val_post2023']})")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"grid_results": grid_results, "vif_reports": vif_reports}, f, ensure_ascii=False, indent=2)
    print(f"\nDa ghi toan bo ket qua vao {RESULTS_PATH}")


if __name__ == "__main__":
    main()
