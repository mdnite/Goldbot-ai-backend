"""Giai đoạn 4b - Thí nghiệm rolling-window retrain (walk-forward). Định nghĩa GỐC
của 4b (Random Forest/XGBoost) đã bị bỏ - xem CLAUDE.md/Roadmap.md mục Giai đoạn 4b
(quyết định 2026-08-07). Đây là arm mới: thay vì train 1 lần trên cửa sổ tĩnh cố định
(2010-2022), retrain định kỳ trên cửa sổ trượt gần nhất, xem có cải thiện Brier so với
static không.

RÀNG BUỘC CỨNG:
- KHÔNG đụng archive/giai_doan_3_llm_approach/giai_doan_3_sample_dates.json (50 mốc test
  khoá). Toàn bộ walk-forward chỉ chạy trong val_post (2023-03-17 -> 2024-05-31, n=303),
  đúng vùng đã dùng để tính static Brier 0.6276.
- KHÔNG dùng gate ĐẠT/KHÔNG ĐẠT tự động. Script chỉ báo cáo Brier tổng hợp + 95% CI
  (block bootstrap, block=21) cho từng arm, không tự kết luận thắng/thua.
- Feature CỐ ĐỊNH: rolling60 + interaction (34 feature trước VIF, y hệt cấu hình cho ra
  Brier static 0.6276 trong giai_doan_4_brier_diagnostic_results.json).
- Hyperparameter CỐ ĐỊNH: penalty=l1, C=0.02 (cấu hình thắng trong 96 tổ hợp) - dùng y hệt
  ở MỌI arm/fold, không tune lại, để cô lập đúng 1 biến: chiến lược train (window size).
- Ngưỡng nhãn CỐ ĐỊNH: load thẳng từ giai_doan_4_brier_diagnostic_results.json
  (label_threshold, tính trên biến thể tham chiếu ewma21+I - đã verify TRAIN row range
  giống hệt rolling60+I nên tái dùng hợp lệ), KHÔNG tính lại theo từng fold/window.

WALK-FORWARD:
- 5 arm: window size = 504/756/1260/2016 (dòng dữ liệu gần nhất, ~2/3/5/8 năm) và
  "expanding" (toàn bộ lịch sử đủ điều kiện, không giới hạn) - arm đối chứng để tách bạch
  "retrain định kỳ" khỏi "chỉ dùng dữ liệu gần nhất".
- Retrain theo quý: chia 303 dòng val_post thành các fold liên tiếp 63 dòng (fold cuối
  ngắn hơn), KHÔNG chồng KHÔNG hở - tổng đúng bằng 303.
- Tại mỗi fold, cutoff retrain T = ngày dự đoán ĐẦU TIÊN của fold đó.
- Embargo leak-safe: 1 dòng ngày d chỉ được dùng để train nếu d + 21 phiên (HORIZON) <= T
  - tính bằng vị trí THẬT trong lịch giao dịch (wide.index), không xấp xỉ theo ngày lịch.
  Lý do: nhãn r_t tại d dùng giá gold tại d+21; nếu d nằm trong (T-21, T], nhãn đó đã
  "nhìn thấy" giá sau T.
- VIF filter + StandardScaler: fit lại từ đầu trên MỖI cửa sổ train riêng (không tái dùng
  giữa các fold/arm).
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import giai_doan_4_train_model as m
import giai_doan_4_brier_diagnostic as bd

OUTPUT_PATH = "giai_doan_4b_rolling_window_results.json"
DIAG_RESULTS_PATH = "giai_doan_4_brier_diagnostic_results.json"

HORIZON = 21
FOLD_SIZE = 63  # ~1 quy (phien giao dich)
WINDOW_SIZES = [504, 756, 1260, 2016, "expanding"]
FIXED_PENALTY = "l1"
FIXED_C = 0.02
N_BOOTSTRAP = 2000
BLOCK_SIZE = 21
BOOTSTRAP_SEED = 42

BASELINE_CLASSES = ["Di_ngang", "Giam", "Tang"]  # thu tu co dinh, khop bd.BASELINE_CLASSES

STATIC_REFERENCE = {
    "config": "rolling60+interaction, l1, C=0.02 (thang trong 96 to hop)",
    "brier_score": 0.6276484986243936,
}


def align_proba(proba, model_classes, canonical_classes):
    """Sap lai cot proba ve dung thu tu canonical_classes. Neu 1 fold train khong
    thay du 3 nhan (fold nho, window nho), model.classes_ thieu cot - dien 0 cho
    nhan vang mat (dung 0 vi model chua bao gio gan xac suat cho nhan no chua thay)."""
    out = np.zeros((proba.shape[0], len(canonical_classes)))
    for j, c in enumerate(model_classes):
        out[:, canonical_classes.index(c)] = proba[:, j]
    return out


def per_row_brier(y_true, proba, canonical_classes):
    class_to_idx = {c: i for i, c in enumerate(canonical_classes)}
    y_onehot = np.zeros((len(y_true), len(canonical_classes)))
    for i, y in enumerate(y_true):
        y_onehot[i, class_to_idx[y]] = 1.0
    return np.sum((proba - y_onehot) ** 2, axis=1)


def block_bootstrap_ci(values, block_size, n_boot, seed):
    """Moving block bootstrap tren chuoi da sap theo thoi gian - ton trong tu tuong
    quan do nhan horizon=21 phien chong lap (giong huong xu ly n_eff o Giai doan 3),
    thay vi bootstrap i.i.d. thuong se phong dai do tin cay."""
    values = np.asarray(values)
    n = len(values)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        for _ in range(n_blocks):
            start = rng.integers(0, max_start + 1)
            idx.extend(range(start, start + block_size))
        idx = idx[:n]
        boot_means[b] = values[np.array(idx)].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    wide, base = m.load_series()
    gold = wide["gold"]
    all_dates = wide.index

    diag = json.load(open(DIAG_RESULTS_PATH, encoding="utf-8"))
    low = diag["label_threshold"]["low"]
    high = diag["label_threshold"]["high"]
    print(f"Nguong nhan tai su dung (tu {DIAG_RESULTS_PATH}): low={low:.6f} high={high:.6f} "
          f"(reference_variant={diag['label_threshold']['reference_variant']})")

    def label_series(r):
        return pd.cut(r, bins=[-np.inf, low, high, np.inf],
                       labels=["Giam", "Di_ngang", "Tang"]).astype(str)

    data = bd.build_labeled_data(base, gold, "rolling60", True)
    feature_cols = [c for c in data.columns if c != "r_t"]

    val_post = data[(data.index >= m.REGIME_SPLIT_DATE) & (data.index <= m.VAL_END)]
    assert len(val_post) == 303, f"val_post length mismatch: {len(val_post)} (ky vong 303)"
    print(f"val_post: n={len(val_post)}, range=[{val_post.index.min().date()} -> {val_post.index.max().date()}]")

    # vi tri that trong lich giao dich, dung cho embargo
    date_pos = {d: i for i, d in enumerate(all_dates)}

    def safe_avail_date(d):
        target = date_pos[d] + HORIZON
        return all_dates[target] if target < len(all_dates) else None

    avail = pd.Series({d: safe_avail_date(d) for d in data.index})

    # fold: chia val_post thanh cac khoi 63 dong lien tiep, khong chong khong ho
    val_dates = val_post.index
    folds = [val_dates[i:i + FOLD_SIZE] for i in range(0, len(val_dates), FOLD_SIZE)]
    total_fold_rows = sum(len(f) for f in folds)
    assert total_fold_rows == 303, f"fold partition sai: tong={total_fold_rows}"
    print(f"So fold: {len(folds)}, kich thuoc tung fold: {[len(f) for f in folds]}\n")

    arm_results = {}
    for window in WINDOW_SIZES:
        window_label = "expanding" if window == "expanding" else str(window)
        print(f"=== Arm: {window_label} ===")
        fold_logs = []
        all_y_true = []
        all_proba = []

        for fold_dates in folds:
            T = fold_dates[0]
            elig_mask = avail.notna() & (avail <= T)
            elig_rows = data.index[elig_mask]

            train_idx = elig_rows if window == "expanding" else elig_rows[-window:]
            train = data.loc[train_idx]
            X_train_raw = train[feature_cols]
            y_train = label_series(train["r_t"])

            kept, dropped, _ = m.filter_by_vif(X_train_raw, m.VIF_THRESHOLD)
            X_train = X_train_raw[kept]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = LogisticRegression(
                penalty=FIXED_PENALTY, C=FIXED_C, solver="saga", max_iter=5000,
                random_state=m.RANDOM_SEED,
            )
            model.fit(X_train_scaled, y_train)

            test_rows = data.loc[fold_dates]
            X_test = test_rows[feature_cols][kept]
            X_test_scaled = scaler.transform(X_test)
            proba_raw = model.predict_proba(X_test_scaled)
            proba = align_proba(proba_raw, list(model.classes_), BASELINE_CLASSES)
            y_test = label_series(test_rows["r_t"])

            all_y_true.extend(y_test.tolist())
            all_proba.append(proba)

            fold_logs.append({
                "fold_start": str(T.date()),
                "fold_end": str(fold_dates[-1].date()),
                "n_test": len(fold_dates),
                "n_train": len(train),
                "train_range": [str(train.index.min().date()), str(train.index.max().date())],
                "n_features_before_vif": len(feature_cols),
                "n_features_after_vif": len(kept),
                "classes_seen_in_train": sorted(y_train.unique().tolist()),
            })
            print(f"  fold [{T.date()} -> {fold_dates[-1].date()}] n_test={len(fold_dates)} "
                  f"n_train={len(train)} train_range=[{train.index.min().date()} -> {train.index.max().date()}] "
                  f"n_feat_after_vif={len(kept)}")

        proba_full = np.vstack(all_proba)
        brier_rows = per_row_brier(all_y_true, proba_full, BASELINE_CLASSES)
        aggregate_brier = float(np.mean(brier_rows))
        ci_lo, ci_hi = block_bootstrap_ci(brier_rows, BLOCK_SIZE, N_BOOTSTRAP, BOOTSTRAP_SEED)

        print(f"  -> Brier tong hop (n={len(brier_rows)}) = {aggregate_brier:.4f}, "
              f"95% CI (block bootstrap, block={BLOCK_SIZE}) = [{ci_lo:.4f}, {ci_hi:.4f}]\n")

        arm_results[window_label] = {
            "window_size": window if window != "expanding" else None,
            "n_predictions": len(brier_rows),
            "brier_score": aggregate_brier,
            "ci_95_block_bootstrap": [ci_lo, ci_hi],
            "n_bootstrap": N_BOOTSTRAP,
            "block_size": BLOCK_SIZE,
            "folds": fold_logs,
        }

    output = {
        "label_threshold": {"low": low, "high": high,
                             "source": DIAG_RESULTS_PATH,
                             "reference_variant": diag["label_threshold"]["reference_variant"]},
        "fixed_hyperparameters": {"penalty": FIXED_PENALTY, "C": FIXED_C,
                                    "feature_config": "rolling60+interaction"},
        "embargo_horizon": HORIZON,
        "fold_size": FOLD_SIZE,
        "n_folds": len(folds),
        "val_post_range": [str(val_post.index.min().date()), str(val_post.index.max().date())],
        "n_val_post": len(val_post),
        "static_reference": STATIC_REFERENCE,
        "baselines_reference": {k: v["brier_score"] for k, v in diag["baselines"].items()},
        "arms": arm_results,
        "note": "KHONG co gate DAT/KHONG DAT tu dong - bao cao so lieu thuan tuy, "
                "khong tu ket luan thang/thua. Chua dung den 50 moc test khoa.",
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Da ghi ket qua vao {OUTPUT_PATH}")

    print("\n=== TONG HOP ===")
    print(f"  Static (tham chieu, khong chay lai): {STATIC_REFERENCE['brier_score']:.4f} "
          f"({STATIC_REFERENCE['config']})")
    for k, v in diag["baselines"].items():
        print(f"  {k}: {v['brier_score']:.4f}")
    for label, r in arm_results.items():
        print(f"  Arm {label:<10}: Brier={r['brier_score']:.4f}  CI95=[{r['ci_95_block_bootstrap'][0]:.4f}, "
              f"{r['ci_95_block_bootstrap'][1]:.4f}]  (n={r['n_predictions']})")


if __name__ == "__main__":
    main()
