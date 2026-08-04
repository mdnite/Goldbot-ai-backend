"""Giai đoạn 4a - refit CUỐI CÙNG (không phải 1 vòng tuning, không tính vào 5-6 vòng
đã dùng ở giai_doan_4_train_model.py). Dùng đúng hyperparameter đã chốt qua Round 1-3
(xem giai_doan_4_train_results_round1_ewma.json / _round2.json / _round3.json):
vol_window=21 (EWMA span), penalty=l1, C=0.01 - lý do chọn: post2023 accuracy cao nhất
trong nhóm model thật (không rỗng), gap pre/post-2023 nhỏ.

Toàn bộ preprocessing (ngưỡng nhãn, VIF filter, StandardScaler) được TÍNH LẠI trên tập
GỘP TRAIN+VALIDATION (không dùng lại threshold/scaler đã fit riêng trên TRAIN ở giai
đoạn tuning) - vì tập gộp này là "train cuối cùng" cho model production, test (50 mốc
archive/giai_doan_3_llm_approach/giai_doan_3_sample_dates.json) vẫn hoàn toàn tách
biệt, CHƯA đụng tới trong script này.

Không sửa giai_doan_4_train_model.py - tái dùng load_series/build_features/filter_by_vif
qua import, giữ 2 script độc lập đúng tiền lệ với train_trend_model.py.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import giai_doan_4_train_model as m

VOL_WINDOW = 21
PENALTY = "l1"
C = 0.01
HORIZON = 21
MODEL_PATH = "giai_doan_4_trend_model.joblib"


def main():
    wide, base = m.load_series()
    gold = wide["gold"]
    r_t_full = (gold.shift(-HORIZON) - gold) / gold

    features = m.build_features(base, VOL_WINDOW)
    data = features.copy()
    data["r_t"] = r_t_full
    data = data.dropna()

    combined = data[data.index <= m.VAL_END]
    print(f"So dong TRAIN+VALIDATION gop: {len(combined)} "
          f"({combined.index.min().date()} -> {combined.index.max().date()})")

    median_c = combined["r_t"].median()
    mad_c = (combined["r_t"] - median_c).abs().median()
    robust_scale = 1.4826 * mad_c
    low = median_c - m.K * robust_scale
    high = median_c + m.K * robust_scale
    print(f"Nguong nhan tinh lai tren TRAIN+VAL gop: low={low:.5f}, high={high:.5f}")

    y = pd.cut(combined["r_t"], bins=[-np.inf, low, high, np.inf],
               labels=["Giam", "Di_ngang", "Tang"]).astype(str)
    print("Phan phoi nhan tren TRAIN+VAL gop:")
    print((y.value_counts(normalize=True) * 100).round(1))

    feature_cols = [c for c in data.columns if c != "r_t"]
    X_raw = combined[feature_cols]

    kept, dropped, final_vifs = m.filter_by_vif(X_raw, m.VIF_THRESHOLD)
    print(f"\nVIF filter tren TRAIN+VAL gop: {len(feature_cols)} -> {len(kept)} "
          f"(loai {len(dropped)}, nguong>{m.VIF_THRESHOLD})")
    for feat, v in dropped:
        print(f"  loai {feat}: VIF={v:.2f}")

    X = X_raw[kept]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(penalty=PENALTY, C=C, solver="saga", max_iter=5000,
                                random_state=m.RANDOM_SEED)
    model.fit(X_scaled, y)

    train_acc = model.score(X_scaled, y)
    print(f"\nAccuracy in-sample tren chinh TRAIN+VAL (chi de tham khao/sanity check - "
          f"KHONG phai chi so danh gia model, vi day la du lieu da dung de fit): {train_acc:.3f}")

    coefs = model.coef_
    nonzero_mask = np.abs(coefs).max(axis=0) > 1e-8
    survived = [f for f, alive in zip(kept, nonzero_mask) if alive]
    print(f"\nFeature song sot (|coef|>0 o it nhat 1 class) tren model refit cuoi: "
          f"{len(survived)}/{len(kept)}")
    for f in survived:
        idx = kept.index(f)
        vals = ", ".join(f"{cls}={v:+.4f}" for cls, v in zip(model.classes_, coefs[:, idx]))
        print(f"  {f}: {vals}")

    artifact = {
        "model": model,
        "scaler": scaler,
        "kept_features": kept,
        "vol_window": VOL_WINDOW,
        "ma_window": m.MA_WINDOW,
        "lags": m.LAGS,
        "macro_pairs": m.MACRO_PAIRS,
        "horizon": HORIZON,
        "label_low": float(low),
        "label_high": float(high),
        "penalty": PENALTY,
        "C": C,
        "random_seed": m.RANDOM_SEED,
        "train_val_end": m.VAL_END,
        "n_rows_fit": len(combined),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nDa ghi model vao {MODEL_PATH}")


if __name__ == "__main__":
    main()
