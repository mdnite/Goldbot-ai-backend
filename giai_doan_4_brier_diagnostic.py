"""Giai đoạn 4 - CHẨN ĐOÁN Brier score (KHÔNG PHẢI "4a-v2", không tạo model production
mới). Mục đích: đo riêng xem 2 trong 3 fix kỹ thuật gốc của Giai đoạn 4a - interaction
terms (cô lập đặc trưng) và EWMA volatility (volatility trễ) - có thật sự cải thiện chất
lượng DỰ ĐOÁN XÁC SUẤT hay không, tách khỏi ảnh hưởng của việc chọn C bằng accuracy (đã
xác nhận là nguyên nhân khiến Giai đoạn 4a suy biến về nhãn đa số - xem giai_doan_4_report.md
+ memory giai_doan_4_interaction_verification: 2 interaction feature còn sống có hệ số=0
cho đúng nhãn Di_ngang đang thắng, ablation xác nhận 0/50 mốc test bị ảnh hưởng).

RÀNG BUỘC CỨNG (không phá trong file này):
- KHÔNG đụng archive/giai_doan_3_llm_approach/giai_doan_3_sample_dates.json (50 mốc test khoá).
- KHÔNG refit trên TRAIN+VAL, KHÔNG ghi .joblib model mới.
- KHÔNG dùng accuracy để chọn/so sánh - chỉ Brier score đa lớp. Accuracy chỉ được dùng
  (ở giai_doan_4_train_model.py, không phải ở đây) để né vùng sụp đổ C<=0.005 lúc chọn lưới C.

THIẾT KẾ (đã user duyệt qua 2 vòng phản biện):
- 3 volatility method: ewma10 (span=10), ewma21 (span=21, cấu hình đã khoá 4a),
  rolling60 (Vol60 rolling std GỐC - y hệt train_trend_model.py dòng 43, Giai đoạn 1.5).
- 2 interaction option: with (28 base + 6 interaction, y hệt m.build_features),
  without (chỉ 28 base feature).
- -> 6 biến thể feature.
- Lưới C = [0.01, 0.02, 0.05, 0.1, 0.3, 1, 3, 10] (8 mức, log-spaced, né vùng sụp đổ
  C<=0.005 đã xác nhận ở Round 2 giai_doan_4_train_model.py).
- Penalty = [l1, l2] (2 mức - QUAN TRỌNG: KHÔNG cố định l1. l1 ở 4a là SẢN PHẨM của
  chính grid search bằng accuracy đang bị nghi ngờ, không phải constraint độc lập. l1 zero
  hoá hệ số (sparse, gây "vách đứng"), l2 chỉ co lại chứ không zero - nếu chỉ test l1,
  không phân biệt được "feature không có tín hiệu" với "l1 ở C thấp luôn cắt trụi bất kể
  feature nào". Test cả 2 để tách rời 2 khả năng này).
- -> 6 x 8 x 2 = 96 tổ hợp.

NGƯỠNG NHÃN - tính 1 LẦN DUY NHẤT, dùng CHUNG cho toàn bộ 96 tổ hợp (không tính lại theo
từng biến thể feature). Lý do: nhãn (Tăng/Giảm/Đi_ngang) là thuộc tính của target (forward
return 21 ngày của gold), về nguyên tắc KHÔNG phụ thuộc feature nào dùng để dự đoán nó -
nếu để ngưỡng trôi theo biến thể (do khác biệt warmup NaN giữa EWMA/rolling60), chênh lệch
Brier giữa các biến thể có thể do ngưỡng nhãn khác nhau, không phải do feature khác nhau -
1 biến gây nhiễu ngoài ý muốn. Script sẽ TỰ VERIFY (in ra train.index min/max/len cho cả
6 biến thể) rằng TRAIN row range thật sự giống hệt nhau trước khi tin ngưỡng dùng chung là
hợp lệ - không giả định suông. Lý thuyết đã lập luận trước: MA_WINDOW=60/LAGS (gồm lag60)
giữ cố định ở MỌI biến thể nên đây mới là ràng buộc NaN chặt nhất, rolling60/ewma và
có/không interaction đều không siết chặt hơn nó -> TRAIN range phải trùng nhau. Nếu dry-run
cho thấy KHÔNG trùng, script sẽ cảnh báo rõ, không âm thầm dùng ngưỡng sai.

Brier score tính trên validation post-2023 (index >= REGIME_SPLIT_DATE) - CHỈ vùng này,
KHÔNG toàn bộ validation - vì đây là vùng regime giống 50 mốc test thật (2024-06 -> 2026-06
đều nằm trong bull market post-2023), khớp đúng phạm vi user yêu cầu.
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import giai_doan_4_train_model as m

OUTPUT_PATH = "giai_doan_4_brier_diagnostic_results.json"
HORIZON = 21  # khớp m (HORIZON là biến cục bộ trong m.main(), không phải hằng số module)

VOL_METHODS = ["ewma10", "ewma21", "rolling60"]
INTERACTION_OPTIONS = [True, False]
C_GRID = [0.01, 0.02, 0.05, 0.1, 0.3, 1, 3, 10]
PENALTY_GRID = ["l1", "l2"]

REFERENCE_VOL_METHOD = "ewma21"       # cấu hình đã khoá Giai đoạn 4a
REFERENCE_INTERACTION = True

# Task 6: 3 baseline Brier tham chieu (KHONG dung de chon cau hinh nao trong 96 to hop -
# chi de biet muc tham chieu). Thu tu class CO DINH, khop dung thu tu alphabet sklearn
# dung noi bo (Di_ngang < Giam < Tang) - da xac nhan qua model.classes_ o cac lan chay truoc.
BASELINE_CLASSES = ["Di_ngang", "Giam", "Tang"]


def build_features_variant(base, vol_method, include_interaction):
    """4 chuỗi gốc: lag[1,5,10,21,60] + ma60 + volatility (EWMA span=10/21 hoặc rolling
    std 60 ngày cố định, tuỳ vol_method). Nếu include_interaction=True, cộng thêm 6
    interaction feature (lag1+ma60 x 3 cặp macro trong m.MACRO_PAIRS) - y hệt
    giai_doan_4_train_model.py::build_features, chỉ tham số hoá thêm vol_method."""
    features = pd.DataFrame(index=base["gold_ret"].index)
    for name, s in base.items():
        for lag in m.LAGS:
            features[f"{name}_lag{lag}"] = s.shift(lag)
        features[f"{name}_ma{m.MA_WINDOW}"] = s.rolling(m.MA_WINDOW).mean()
        if vol_method == "rolling60":
            features[f"{name}_vol60"] = s.rolling(60).std()
        else:
            span = int(vol_method.replace("ewma", ""))
            features[f"{name}_vol{span}"] = s.ewm(span=span, adjust=False).std()

    if include_interaction:
        for a, b in m.MACRO_PAIRS:
            inter = base[a] * base[b]
            name = f"inter_{a}_x_{b}"
            features[f"{name}_lag1"] = inter.shift(1)
            features[f"{name}_ma{m.MA_WINDOW}"] = inter.rolling(m.MA_WINDOW).mean()

    return features


def build_labeled_data(base, gold, vol_method, include_interaction):
    r_t_full = (gold.shift(-HORIZON) - gold) / gold
    features = build_features_variant(base, vol_method, include_interaction)
    data = features.copy()
    data["r_t"] = r_t_full
    return data.dropna()


def multiclass_brier(y_true, proba, classes):
    """BS = (1/N) * sum_i sum_k (p_ik - y_ik)^2 - công thức Brier đa lớp chuẩn (vector
    hoá qua one-hot). Khác sklearn.metrics.brier_score_loss (chỉ hỗ trợ binary)."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_onehot = np.zeros((len(y_true), len(classes)))
    for i, y in enumerate(y_true):
        y_onehot[i, class_to_idx[y]] = 1.0
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def variant_label(vol_method, include_interaction):
    return f"{vol_method}{'+I' if include_interaction else ''}"


def class_distribution(y, classes):
    """Vector ti le nhan (theo dung thu tu `classes`) - dung de tao baseline hang so."""
    counts = pd.Series(y).value_counts(normalize=True)
    return np.array([counts.get(c, 0.0) for c in classes])


def constant_proba_brier(y_true, proba_vector, classes):
    """Brier score khi du doan CUNG 1 vector xac suat hang so (khong doi theo tung
    dong) cho toan bo y_true - dung cho 3 baseline Task 6, khong fit model nao."""
    proba = np.tile(proba_vector, (len(y_true), 1))
    return multiclass_brier(y_true, proba, classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Chi verify TRAIN row range + VIF/shape cho 6 bien the, KHONG fit model, KHONG tinh Brier")
    args = parser.parse_args()

    wide, base = m.load_series()
    gold = wide["gold"]

    variants = list(itertools.product(VOL_METHODS, INTERACTION_OPTIONS))

    print("=== Buoc 1: verify TRAIN row range giong het nhau o ca 6 bien the ===")
    variant_data = {}
    train_ranges = {}
    for vol_method, include_interaction in variants:
        data = build_labeled_data(base, gold, vol_method, include_interaction)
        variant_data[(vol_method, include_interaction)] = data
        train = data[data.index <= m.TRAIN_END]
        train_ranges[(vol_method, include_interaction)] = (len(train), str(train.index.min().date()), str(train.index.max().date()))
        print(f"  {variant_label(vol_method, include_interaction):<12} "
              f"n_train={len(train)} range=[{train.index.min().date()} -> {train.index.max().date()}]")

    if len(set(train_ranges.values())) == 1:
        print("XAC NHAN: TRAIN row range GIONG HET nhau o ca 6 bien the -> dung 1 nguong nhan CHUNG la hop le.\n")
    else:
        print("[CANH BAO] TRAIN row range KHONG khop giua cac bien the - nguong nhan dung chung co the KHONG hop le!")
        for k, v in train_ranges.items():
            print(f"    {variant_label(*k)}: {v}")
        print()

    # Nguong nhan: tinh 1 LAN tren bien the tham chieu, dung CHUNG cho toan bo 96 to hop
    ref_data = variant_data[(REFERENCE_VOL_METHOD, REFERENCE_INTERACTION)]
    ref_train = ref_data[ref_data.index <= m.TRAIN_END]
    median_train = ref_train["r_t"].median()
    mad_train = (ref_train["r_t"] - median_train).abs().median()
    robust_scale = 1.4826 * mad_train
    low = median_train - m.K * robust_scale
    high = median_train + m.K * robust_scale
    print(f"=== Buoc 2: nguong nhan DUNG CHUNG (tinh tren bien the tham chieu "
          f"{variant_label(REFERENCE_VOL_METHOD, REFERENCE_INTERACTION)}) ===")
    print(f"  low={low:.5f}, high={high:.5f}\n")

    def label_series(r):
        return pd.cut(r, bins=[-np.inf, low, high, np.inf], labels=["Giam", "Di_ngang", "Tang"]).astype(str)

    # --- Task 6: 3 baseline Brier tham chieu, tinh tren bien the tham chieu (nhan chi phu
    # thuoc r_t + nguong dung chung, KHONG phu thuoc feature - da xac nhan TRAIN/VAL row
    # range giong het nhau o ca 6 bien the o Buoc 1, nen dung 1 bien the la du, khong can
    # lap qua ca 6). KHONG dung so lieu nay de chon cau hinh nao trong 96 to hop. ---
    ref_train_labels = label_series(ref_train["r_t"])
    ref_val = ref_data[(ref_data.index >= m.VAL_START) & (ref_data.index <= m.VAL_END)]
    ref_val_post = ref_val[ref_val.index >= m.REGIME_SPLIT_DATE]
    ref_val_post_labels = label_series(ref_val_post["r_t"])

    train_dist = class_distribution(ref_train_labels, BASELINE_CLASSES)
    val_post_dist = class_distribution(ref_val_post_labels, BASELINE_CLASSES)
    degenerate_dist = np.array([1.0 if c == "Di_ngang" else 0.0 for c in BASELINE_CLASSES])

    baseline1_brier = constant_proba_brier(ref_val_post_labels.values, train_dist, BASELINE_CLASSES)
    baseline2_brier = constant_proba_brier(ref_val_post_labels.values, val_post_dist, BASELINE_CLASSES)
    baseline3_brier = constant_proba_brier(ref_val_post_labels.values, degenerate_dist, BASELINE_CLASSES)

    baselines = {
        "baseline1_fair_train_distribution": {
            "description": "CONG BANG, out-of-sample - ti le nhan dong bang tu TRAIN, khong nhin truoc VAL",
            "class_order": BASELINE_CLASSES,
            "proba_vector": train_dist.tolist(),
            "brier_score": baseline1_brier,
        },
        "baseline2_hindsight_val_distribution": {
            "description": "KHONG CONG BANG (hindsight/tham khao phu) - ti le nhan tinh TRUC TIEP tren chinh VAL post-2023 dang cham, nhin truoc phan phoi that cua tap dang danh gia. Chi de biet can duoi ly thuyet, KHONG dung de so sanh cong bang.",
            "class_order": BASELINE_CLASSES,
            "proba_vector": val_post_dist.tolist(),
            "brier_score": baseline2_brier,
        },
        "baseline3_degenerate_all_Di_ngang": {
            "description": "Moc suy bien - luon doan 100% Di_ngang (p=1), dung hanh vi that model 4a goc da the hien tren 50 moc test khoa",
            "class_order": BASELINE_CLASSES,
            "proba_vector": degenerate_dist.tolist(),
            "brier_score": baseline3_brier,
        },
    }

    print("=== Buoc 2b (Task 6): 3 baseline Brier tham chieu tren VAL post-2023 "
          f"(n={len(ref_val_post)}) - KHONG dung de chon cau hinh nao trong 96 to hop ===")
    print(f"  Baseline1 (CONG BANG, TRAIN dist)      : proba={train_dist.round(4).tolist()} -> Brier={baseline1_brier:.4f}")
    print(f"  Baseline2 (hindsight, VAL dist, THAM KHAO PHU - KHONG cong bang): "
          f"proba={val_post_dist.round(4).tolist()} -> Brier={baseline2_brier:.4f}")
    print(f"  Baseline3 (suy bien, 100% Di_ngang)     : proba={degenerate_dist.tolist()} -> Brier={baseline3_brier:.4f}\n")

    if args.dry_run:
        print("=== --dry-run: kiem VIF + shape cho ca 6 bien the, KHONG fit model, KHONG tinh Brier ===")
        for vol_method, include_interaction in variants:
            data = variant_data[(vol_method, include_interaction)]
            train = data[data.index <= m.TRAIN_END]
            val = data[(data.index >= m.VAL_START) & (data.index <= m.VAL_END)]
            val_post = val[val.index >= m.REGIME_SPLIT_DATE]
            feature_cols = [c for c in data.columns if c != "r_t"]
            kept, dropped, _ = m.filter_by_vif(train[feature_cols], m.VIF_THRESHOLD)
            print(f"  {variant_label(vol_method, include_interaction):<12} "
                  f"n_train={len(train)} n_val={len(val)} n_val_post2023={len(val_post)} "
                  f"n_feat_before_vif={len(feature_cols)} n_feat_after_vif={len(kept)} n_dropped_vif={len(dropped)}")
        print("\n--dry-run: DUNG LAI o day, khong chay grid that.")
        return

    print(f"=== Buoc 3: chay {len(variants)} bien the x {len(PENALTY_GRID)} penalty x {len(C_GRID)} C "
          f"= {len(variants) * len(PENALTY_GRID) * len(C_GRID)} to hop ===")
    results = []
    for vol_method, include_interaction in variants:
        data = variant_data[(vol_method, include_interaction)]
        train = data[data.index <= m.TRAIN_END]
        val = data[(data.index >= m.VAL_START) & (data.index <= m.VAL_END)]
        val_post = val[val.index >= m.REGIME_SPLIT_DATE]

        feature_cols = [c for c in data.columns if c != "r_t"]
        X_train_raw = train[feature_cols]
        X_val_post_raw = val_post[feature_cols]

        kept, dropped, _ = m.filter_by_vif(X_train_raw, m.VIF_THRESHOLD)

        y_train = label_series(train["r_t"])
        y_val_post = label_series(val_post["r_t"])

        X_train = X_train_raw[kept]
        X_val_post = X_val_post_raw[kept]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_post_scaled = scaler.transform(X_val_post)

        for penalty in PENALTY_GRID:
            for C in C_GRID:
                model = LogisticRegression(
                    penalty=penalty, C=C, solver="saga", max_iter=5000,
                    random_state=m.RANDOM_SEED,
                )
                model.fit(X_train_scaled, y_train)
                proba = model.predict_proba(X_val_post_scaled)
                brier = multiclass_brier(y_val_post.values, proba, model.classes_)

                results.append({
                    "vol_method": vol_method,
                    "include_interaction": include_interaction,
                    "penalty": penalty,
                    "C": C,
                    "n_features_before_vif": len(feature_cols),
                    "n_features_after_vif": len(kept),
                    "n_val_post2023": len(val_post),
                    "brier_score": brier,
                })

    print(f"Da chay xong {len(results)} to hop.\n")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "label_threshold": {
                "low": float(low), "high": float(high),
                "reference_variant": variant_label(REFERENCE_VOL_METHOD, REFERENCE_INTERACTION),
            },
            "train_ranges_check": {variant_label(*k): v for k, v in train_ranges.items()},
            "baselines": baselines,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Da ghi ket qua vao {OUTPUT_PATH}")

    df = pd.DataFrame(results)
    df["variant"] = df.apply(lambda r: variant_label(r["vol_method"], r["include_interaction"]), axis=1)
    for penalty in PENALTY_GRID:
        print(f"\n=== Brier score (penalty={penalty}) - hang=C, cot=bien the (thap hon = tot hon) ===")
        sub = df[df["penalty"] == penalty]
        pivot = sub.pivot(index="C", columns="variant", values="brier_score")
        print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
