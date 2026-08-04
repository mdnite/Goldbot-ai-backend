"""Giai đoạn 4a - eval CUỐI CÙNG trên test, CHẠM 1 LẦN DUY NHẤT (đúng 50 mốc đã khoá
archive/giai_doan_3_llm_approach/giai_doan_3_sample_dates.json - tái dùng để so công
bằng với bot 46.0%/baseline 30.0%/baseline2 54.0%/trend_model.joblib cũ 24.0% đã có
trong giai_doan_3_report.md).

Model dùng: giai_doan_4_trend_model.joblib (refit trên TRAIN+VALIDATION gộp, xem
giai_doan_4_refit_final.py - vol_window=21 EWMA, penalty=l1, C=0.01).

Ground truth (actual_label) TÁI DÙNG NGUYÊN từ giai_doan_3_backtest_results.json
(outcome.label, đã đóng băng theo ngưỡng nhãn GỐC của Giai đoạn 1.5: low=-0.01912,
high=0.02326) - KHÔNG dùng ngưỡng nhãn riêng của model 4a (low=-0.01802, high=0.02304,
tính trên TRAIN+VAL gộp) để tính lại ground truth. Lý do: mục đích của việc tái dùng
đúng 50 mốc là so sánh công bằng với bot/baseline đã chấm bằng ngưỡng GỐC - nếu đổi
ngưỡng ground truth thì phá vỡ tính so sánh được. Ngưỡng của model 4a chỉ dùng để gán
nhãn lúc TRAIN, không dùng để định nghĩa "đúng/sai" lúc test.

GHI CHÚ PHƯƠNG PHÁP (không phải bug, không sửa): ngưỡng nhãn train của model 4a
(-0.01802/+0.02304) và ngưỡng ground truth gốc (-0.01912/+0.02326) rất gần nhau nhưng
KHÔNG giống hệt - do model 4a tính trên TRAIN+VAL gộp còn ground truth đóng băng từ
TRAIN 80% gốc của Giai đoạn 1.5. Chênh lệch nhỏ (~0.001-0.0002), chấp nhận được, không
điều chỉnh lại vì test đã CHẠM 1 LẦN DUY NHẤT - không quay lại sửa threshold sau khi
đã thấy kết quả.

Chạy `python giai_doan_4_backtest_eval.py --dry-run` để CHỈ kiểm tra đủ feature (leak-safe)
cho toàn bộ 50 mốc, KHÔNG tính accuracy/KHÔNG chạm ground truth - dùng để rà trước khi
chạy thật (giống --dry-run của giai_doan_3_run_backtest.py).

=== CÔNG THỨC KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (CHỐT TRƯỚC KHI CHẠY - KHÔNG ĐỔI SAU KHI THẤY
KẾT QUẢ) ===
So sánh accuracy model 4a với mốc tham chiếu LLM_REFERENCE_ACC = 0.46 (bot GoldBot,
Giai đoạn 3, đo trên ĐÚNG 50 mốc này).

Tiêu chí QUYẾT ĐỊNH chính (PRIMARY): khoảng tin cậy (CI) 95% CHO RIÊNG model 4a, theo
xấp xỉ chuẩn (Wald interval) - CÙNG công thức SE đã dùng trong giai_doan_3_report.md
mục 5 (SE = sqrt(p_hat*(1-p_hat)/n), CI = p_hat +/- 1.96*SE):
  wald_ci(p_hat, n) = (p_hat - 1.96*SE, p_hat + 1.96*SE)
Kết luận "CÓ Ý NGHĨA THỐNG KÊ" CHỈ KHI cận dưới CI > 0.46 (toàn bộ CI nằm trên 46%).
Tính CẢ 2 phiên bản n:
  - n=50 (danh nghĩa, số mốc thực tế)
  - n_eff=25 (hiệu dụng, do horizon 21 phiên chồng lấn giữa các mốc - ĐÃ xác nhận
    trong giai_doan_3_report.md mục 2: "n hiệu dụng ≈ 25 (≈ 515/21)", tái dùng nguyên
    số đó, không tính lại vì tập 50 mốc/cấu trúc chồng lấn không đổi).

Tiêu chí phụ (SECONDARY, không dùng để quyết định, chỉ đối chiếu/nhất quán với cách
Giai đoạn 3 đã trình bày "gap ≈ 1.2 SE"): two-proportion z-test so model 4a với 0.46,
dùng SE kết hợp KHÔNG GỘP (unpooled/Welch-style, giống hệt cách tính SE(bot)/SE(baseline1)
đã dùng ở giai_doan_3_report.md mục 5):
  SE1 = sqrt(p1*(1-p1)/n), SE2 = sqrt(p2*(1-p2)/n)  [p2=0.46, cùng n]
  SE_combined = sqrt(SE1^2 + SE2^2)
  z = (p1 - p2) / SE_combined
  |z| >= 1.96 coi là có ý nghĩa (~2 SE, đúng ngôn ngữ report cũ) - CHỈ để đối chiếu,
  KHÔNG phải tiêu chí quyết định (tiêu chí quyết định là CI một mẫu ở trên).

LƯU Ý PHƯƠNG PHÁP (không sửa, chỉ ghi nhận): model 4a và bot KHÔNG phải 2 mẫu độc lập
thật sự (đánh giá trên CÙNG 50 mốc) - two-proportion z-test giả định độc lập là xấp xỉ
đơn giản hoá, ĐÚNG NGUYÊN cách Giai đoạn 3 đã làm khi so bot vs baseline (không phải
lỗi mới của script này). Wald CI cũng là xấp xỉ chuẩn, có thể kém chính xác ở n nhỏ/p
cực trị - chấp nhận vì đây là công thức ĐÃ DÙNG xuyên suốt dự án, ưu tiên nhất quán
phương pháp hơn là đổi sang Wilson/Clopper-Pearson chính xác hơn giữa chừng.
"""
import argparse
import json
import re

import joblib
import numpy as np
import pandas as pd

import giai_doan_4_train_model as m

MODEL_PATH = "giai_doan_4_trend_model.joblib"
SAMPLE_DATES_PATH = "archive/giai_doan_3_llm_approach/giai_doan_3_sample_dates.json"
BACKTEST_RESULTS_PATH = "archive/giai_doan_3_llm_approach/giai_doan_3_backtest_results.json"
OLD_TREND_COMPARISON_PATH = "archive/giai_doan_3_llm_approach/giai_doan_3_trend_model_comparison.json"
OUTPUT_PATH = "giai_doan_4_backtest_eval_results.json"

BOT_LABEL_RE = re.compile(r"Kết luận chính:\s*(Tăng|Giảm|Đi ngang)")
VN_TO_ASCII_LABEL = {"Tăng": "Tang", "Giảm": "Giam", "Đi ngang": "Di_ngang"}

LLM_REFERENCE_ACC = 0.46  # bot GoldBot, Giai doan 3, do tren dung 50 moc nay
N_EFFECTIVE = 25  # tai dung nguyen tu giai_doan_3_report.md muc 2 (515 phien / 21 ~ 24.5)
Z_95 = 1.96


def wald_ci(p_hat, n, z=Z_95):
    """Khoang tin cay xap xi chuan (Wald), cung cong thuc SE da dung o
    giai_doan_3_report.md muc 5: SE = sqrt(p_hat*(1-p_hat)/n)."""
    se = (p_hat * (1 - p_hat) / n) ** 0.5
    return p_hat - z * se, p_hat + z * se


def two_proportion_z(p1, p2, n):
    """Two-proportion z-test, SE ket hop KHONG GOP (unpooled), dung n giong nhau cho
    ca 2 proportion - dung cach tinh y het giai_doan_3_report.md muc 5 (SE(bot)/SE(baseline1)
    roi cong RSS). Chi dung lam tieu chi PHU, khong quyet dinh."""
    se1 = (p1 * (1 - p1) / n) ** 0.5
    se2 = (p2 * (1 - p2) / n) ** 0.5
    se_combined = (se1 ** 2 + se2 ** 2) ** 0.5
    z = (p1 - p2) / se_combined if se_combined > 0 else float("inf")
    return z, se_combined


def build_features_asof(base, as_of_date, vol_window, ma_window, lags, macro_pairs):
    """Vector 34 feature tại as_of_date, CHỈ dùng dữ liệu <= as_of_date (leak-safe).
    `.loc[:ts]` lấy tới đúng dòng gần nhất <= ts - y hệt nguyên tắc
    giai_doan_3_compare_trend_model.py::build_features_asof, mở rộng thêm EWMA-vol
    (thay rolling std) và 6 interaction feature (khớp giai_doan_4_train_model.py::build_features).
    EWMA có tính nhân quả (chỉ phụ thuộc quá khứ) nên cắt chuỗi trước rồi tính ewm
    cho kết quả giống hệt tính trên chuỗi đầy đủ rồi lấy đúng điểm as_of_date."""
    ts = pd.Timestamp(as_of_date)
    feats = {}
    truncated = {}
    for name, s in base.items():
        s_upto = s.loc[:ts]
        truncated[name] = s_upto
        if s_upto.empty:
            return None
        for lag in lags:
            pos = len(s_upto) - 1 - lag
            feats[f"{name}_lag{lag}"] = s_upto.iloc[pos] if pos >= 0 else np.nan
        window = s_upto.tail(ma_window)
        feats[f"{name}_ma{ma_window}"] = window.mean() if len(window) >= ma_window else np.nan
        ewma_vol = s_upto.ewm(span=vol_window, adjust=False).std()
        feats[f"{name}_vol{vol_window}"] = ewma_vol.iloc[-1] if len(ewma_vol) else np.nan

    for a, b in macro_pairs:
        inter = truncated[a] * truncated[b]
        name = f"inter_{a}_x_{b}"
        pos1 = len(inter) - 1 - 1
        feats[f"{name}_lag1"] = inter.iloc[pos1] if pos1 >= 0 else np.nan
        w = inter.tail(ma_window)
        feats[f"{name}_ma{ma_window}"] = w.mean() if len(w) >= ma_window else np.nan

    if any(pd.isna(v) for v in feats.values()):
        return None
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Chi kiem tra du feature cho 50 moc, KHONG cham ground truth/accuracy")
    args = parser.parse_args()

    bundle = joblib.load(MODEL_PATH)
    model, scaler = bundle["model"], bundle["scaler"]
    kept_features = bundle["kept_features"]
    vol_window = bundle["vol_window"]
    ma_window = bundle["ma_window"]
    lags = bundle["lags"]
    macro_pairs = [tuple(p) for p in bundle["macro_pairs"]]

    with open(SAMPLE_DATES_PATH, encoding="utf-8") as f:
        sample_dates = json.load(f)["sample_dates"]

    _, base = m.load_series()

    rows, missing = [], []
    for d in sample_dates:
        feats = build_features_asof(base, d, vol_window, ma_window, lags, macro_pairs)
        if feats is None:
            missing.append(d)
            continue
        rows.append((d, feats))

    print(f"Feature check: {len(rows)}/{len(sample_dates)} moc du feature (leak-safe).")
    if missing:
        print(f"[canh bao] {len(missing)} moc THIEU feature: {missing}")

    if args.dry_run:
        print("\n--dry-run: DUNG LAI o day, KHONG tinh accuracy, KHONG cham ground truth.")
        return

    with open(BACKTEST_RESULTS_PATH, encoding="utf-8") as f:
        backtest_results = {r["as_of_date"]: r for r in json.load(f)}

    X = pd.DataFrame([f for _, f in rows], columns=kept_features)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    records = []
    n_model_correct = n_bot_correct = n_baseline1_correct = 0
    n_bot_parsed = 0
    actual_labels = []
    for (d, _), model_pred in zip(rows, preds):
        r = backtest_results.get(d)
        if r is None:
            continue
        actual = r["outcome"]["label"]
        actual_labels.append(actual)
        baseline1_pred = r["baseline_pred"]

        bm = BOT_LABEL_RE.search(r["bot_reply"])
        bot_pred = VN_TO_ASCII_LABEL[bm.group(1)] if bm else None
        if bot_pred is not None:
            n_bot_parsed += 1
            n_bot_correct += int(bot_pred == actual)

        n_model_correct += int(model_pred == actual)
        n_baseline1_correct += int(baseline1_pred == actual)

        records.append({
            "as_of_date": d,
            "actual_label": actual,
            "model_4a_pred": model_pred,
            "model_4a_correct": bool(model_pred == actual),
            "bot_pred": bot_pred,
            "bot_correct": None if bot_pred is None else bool(bot_pred == actual),
            "baseline1_pred": baseline1_pred,
            "baseline1_correct": bool(baseline1_pred == actual),
        })

    n = len(records)
    baseline2_label = pd.Series(actual_labels).value_counts().idxmax()
    n_baseline2_correct = sum(1 for a in actual_labels if a == baseline2_label)

    old_trend_ref = None
    try:
        with open(OLD_TREND_COMPARISON_PATH, encoding="utf-8") as f:
            old_trend_ref = json.load(f)["summary"]["trend_model_accuracy"]
    except FileNotFoundError:
        pass

    p_model = n_model_correct / n if n else None

    stat_test = None
    if p_model is not None:
        ci_n50_low, ci_n50_high = wald_ci(p_model, n)
        ci_neff_low, ci_neff_high = wald_ci(p_model, N_EFFECTIVE)
        z_n50, se_n50 = two_proportion_z(p_model, LLM_REFERENCE_ACC, n)
        z_neff, se_neff = two_proportion_z(p_model, LLM_REFERENCE_ACC, N_EFFECTIVE)
        stat_test = {
            "reference_llm_acc": LLM_REFERENCE_ACC,
            "n_nominal": n,
            "n_effective": N_EFFECTIVE,
            "primary_decision_rule": "CO Y NGHIA chi khi can duoi CI > reference_llm_acc",
            "ci95_n_nominal": [ci_n50_low, ci_n50_high],
            "ci95_n_nominal_significant": ci_n50_low > LLM_REFERENCE_ACC,
            "ci95_n_effective": [ci_neff_low, ci_neff_high],
            "ci95_n_effective_significant": ci_neff_low > LLM_REFERENCE_ACC,
            "secondary_two_proportion_z_n_nominal": z_n50,
            "secondary_two_proportion_z_n_effective": z_neff,
        }

    summary = {
        "n_samples_requested": len(sample_dates),
        "n_missing_features": len(missing),
        "missing_dates": missing,
        "n_scored": n,
        "model_4a_accuracy": p_model,
        "n_bot_labels_parsed": n_bot_parsed,
        "bot_accuracy": n_bot_correct / n_bot_parsed if n_bot_parsed else None,
        "baseline1_accuracy": n_baseline1_correct / n if n else None,
        "baseline2_label": baseline2_label,
        "baseline2_accuracy": n_baseline2_correct / n if n else None,
        "old_trend_model_1_5_accuracy_reference": old_trend_ref,
        "stat_significance_vs_llm_46pct": stat_test,
    }

    print(f"\n=== Ket qua tren {n}/{len(sample_dates)} moc ===")
    print(f"  Model 4a (refit)     : {n_model_correct}/{n} = {summary['model_4a_accuracy']:.3f}")
    print(f"  Bot GoldBot (LLM)    : {n_bot_correct}/{n_bot_parsed} = {summary['bot_accuracy']:.3f} (doi chieu 46.0%)")
    print(f"  Baseline1 (Di_ngang) : {n_baseline1_correct}/{n} = {summary['baseline1_accuracy']:.3f} (doi chieu 30.0%)")
    print(f"  Baseline2 (in-sample/hindsight, nhan='{baseline2_label}'): {n_baseline2_correct}/{n} = {summary['baseline2_accuracy']:.3f} (doi chieu 54.0%)")
    if old_trend_ref is not None:
        print(f"  trend_model.joblib cu (Giai doan 1.5): {old_trend_ref:.3f} (tham khao, khong tinh lai)")

    print(f"\n=== Kiem dinh y nghia thong ke: model 4a vs LLM {LLM_REFERENCE_ACC:.2f} ===")
    print(f"  [n={n} danh nghia] CI95 model 4a = [{ci_n50_low:.3f}, {ci_n50_high:.3f}] "
          f"-> {'CO Y NGHIA' if stat_test['ci95_n_nominal_significant'] else 'CHUA du y nghia'}")
    print(f"  [n_eff={N_EFFECTIVE}]      CI95 model 4a = [{ci_neff_low:.3f}, {ci_neff_high:.3f}] "
          f"-> {'CO Y NGHIA' if stat_test['ci95_n_effective_significant'] else 'CHUA du y nghia'}")
    print(f"  (phu, doi chieu) two-proportion z: n={n} -> z={z_n50:.2f}; n_eff={N_EFFECTIVE} -> z={z_neff:.2f}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)
    print(f"\nDa ghi ket qua vao {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
