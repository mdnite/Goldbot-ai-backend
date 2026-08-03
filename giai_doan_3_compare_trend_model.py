"""Giai đoạn 3 (optional, task #3) - So sánh dự đoán trend_model.joblib (Giai đoạn 1.5)
với bot GoldBot + 2 baseline, trên ĐÚNG 50 mốc backtest đã dùng cho giai_doan_3_backtest_results.json.

Input cho trend_model.joblib KHÁC hẳn market_data_text (dùng để feed LLM) - model cần
28 feature kỹ thuật hoá (lag/MA60/vol60 x 4 series), không phải 1 con số %change/diff.
Script này KHÔNG sửa train_trend_model.py - lặp lại đúng logic feature engineering của
nó (Bước 1-3) tại từng as_of_date, cắt dữ liệu leak-safe (chỉ dùng <= as_of_date).

Ground truth (outcome.label) và bot_reply đã có sẵn trong giai_doan_3_backtest_results.json,
không tính lại. Nhãn bot được trích từ dòng "Kết luận chính: [Tăng/Giảm/Đi ngang]" đã
code hoá trong giai_doan_3_system_prompt.txt (QUY TẮC SỐ 8) bằng regex, dùng để đối chiếu
lại đúng con số 46.0% (23/50) đã có trong giai_doan_3_report.md - nếu khớp, xác nhận cách
trích của script này nhất quán với cách đã chấm trước đó.
"""
import json
import re
import sqlite3

import joblib
import numpy as np
import pandas as pd

DB_PATH = "indicators.db"
MODEL_PATH = "trend_model.joblib"
SAMPLE_DATES_PATH = "giai_doan_3_sample_dates.json"
RESULTS_PATH = "giai_doan_3_backtest_results.json"
OUTPUT_PATH = "giai_doan_3_trend_model_comparison.json"

LAGS = [1, 5, 10, 21, 60]
LOOKBACK = 60

BOT_LABEL_RE = re.compile(r"Kết luận chính:\s*(Tăng|Giảm|Đi ngang)")
VN_TO_ASCII_LABEL = {"Tăng": "Tang", "Giảm": "Giam", "Đi ngang": "Di_ngang"}


def build_wide_series():
    """Đọc indicators.db, pivot EAV->wide, tính 4 series return/diff - y hệt
    Bước 1-2 của train_trend_model.py (không dropna riêng từng series - features
    dùng .shift()/.rolling() trực tiếp, đúng như lúc train)."""
    conn = sqlite3.connect(DB_PATH)
    long_df = pd.read_sql_query(
        "SELECT date, indicator, value FROM indicators", conn, parse_dates=["date"]
    )
    conn.close()
    wide = long_df.pivot(index="date", columns="indicator", values="value").sort_index()
    wide = wide.dropna(subset=["gold", "dxy", "real_yield", "fed_rate"])
    return {
        "gold_ret": wide["gold"].pct_change(),
        "dxy_ret": wide["dxy"].pct_change(),
        "real_yield_diff": wide["real_yield"].diff(),
        "fed_rate_diff": wide["fed_rate"].diff(),
    }


def build_features_asof(series, as_of_date):
    """Vector feature tại as_of_date, CHỈ dùng dữ liệu <= as_of_date (leak-safe).
    `.loc[:ts]` trên index đã sort = lấy tới đúng dòng gần nhất <= ts (giống hệt
    nguyên tắc _nearest_value dùng trong main.py/giai_doan_3_backtest.py), không
    đòi hỏi as_of_date phải khớp tuyệt đối 1 dòng trong wide."""
    ts = pd.Timestamp(as_of_date)
    feats = {}
    for name, s in series.items():
        s_upto = s.loc[:ts]
        if s_upto.empty:
            return None
        for lag in LAGS:
            pos = len(s_upto) - 1 - lag
            feats[f"{name}_lag{lag}"] = s_upto.iloc[pos] if pos >= 0 else np.nan
        window = s_upto.tail(LOOKBACK)
        if len(window) < LOOKBACK:
            feats[f"{name}_ma{LOOKBACK}"] = np.nan
            feats[f"{name}_vol{LOOKBACK}"] = np.nan
        else:
            feats[f"{name}_ma{LOOKBACK}"] = window.mean()
            feats[f"{name}_vol{LOOKBACK}"] = window.std()
    if any(pd.isna(v) for v in feats.values()):
        return None
    return feats


def main():
    bundle = joblib.load(MODEL_PATH)
    model, scaler, feature_names = bundle["model"], bundle["scaler"], bundle["feature_names"]

    with open(SAMPLE_DATES_PATH, encoding="utf-8") as f:
        sample_dates = json.load(f)["sample_dates"]
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = {r["as_of_date"]: r for r in json.load(f)}

    series = build_wide_series()

    rows, missing = [], []
    for d in sample_dates:
        feats = build_features_asof(series, d)
        if feats is None:
            missing.append(d)
            continue
        rows.append((d, feats))

    if missing:
        print(f"[canh bao] {len(missing)}/{len(sample_dates)} moc THIEU du lieu feature (< {LOOKBACK} phien lich su): {missing}")

    X = pd.DataFrame([f for _, f in rows], columns=feature_names)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    records = []
    n_trend_correct = n_bot_correct = n_baseline_correct = 0
    n_bot_parsed = 0
    for (d, _), trend_pred in zip(rows, preds):
        r = results.get(d)
        if r is None:
            continue
        actual = r["outcome"]["label"]
        baseline_pred = r["baseline_pred"]

        m = BOT_LABEL_RE.search(r["bot_reply"])
        bot_pred = VN_TO_ASCII_LABEL[m.group(1)] if m else None
        if bot_pred is not None:
            n_bot_parsed += 1
            n_bot_correct += int(bot_pred == actual)

        n_trend_correct += int(trend_pred == actual)
        n_baseline_correct += int(baseline_pred == actual)

        records.append({
            "as_of_date": d,
            "actual_label": actual,
            "trend_model_pred": trend_pred,
            "trend_model_correct": bool(trend_pred == actual),
            "bot_pred": bot_pred,
            "bot_correct": None if bot_pred is None else bool(bot_pred == actual),
            "baseline_pred": baseline_pred,
            "baseline_correct": bool(baseline_pred == actual),
        })

    n = len(records)
    summary = {
        "n_samples_requested": len(sample_dates),
        "n_missing_features": len(missing),
        "missing_dates": missing,
        "n_scored": n,
        "n_bot_labels_parsed": n_bot_parsed,
        "trend_model_accuracy": n_trend_correct / n if n else None,
        "bot_accuracy": n_bot_correct / n_bot_parsed if n_bot_parsed else None,
        "baseline_accuracy": n_baseline_correct / n if n else None,
    }

    print(f"\nSo sanh tren {n}/{len(sample_dates)} moc (thieu feature: {len(missing)}):")
    print(f"  trend_model.joblib : {n_trend_correct}/{n} = {summary['trend_model_accuracy']:.3f}")
    print(f"  bot GoldBot        : {n_bot_correct}/{n_bot_parsed} = {summary['bot_accuracy']:.3f} (doi chieu voi 46.0% da co trong report)")
    print(f"  baseline Di_ngang  : {n_baseline_correct}/{n} = {summary['baseline_accuracy']:.3f} (doi chieu voi 30.0% da co trong report)")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)
    print(f"\nDa ghi ket qua vao {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
