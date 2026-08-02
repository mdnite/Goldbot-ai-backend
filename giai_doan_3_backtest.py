"""Giai đoạn 3 (revised) - Backtest khả năng suy luận dự đoán ngắn hạn của GoldBot.
Đo SUY LUẬN về tương lai (khác Giai đoạn 2.5 - đo faithfulness với dữ liệu ĐÃ CHO).
Tool MỚI, KHÔNG đụng eval_script.py/eval_dataset.json/EVAL_REPORT.md của Giai đoạn 2.5.

Hằng số Bước 0 (đã xác nhận, đọc trực tiếp từ trend_model.joblib, KHÔNG suy diễn từ
văn xuôi báo cáo) - đóng băng cho toàn bộ Giai đoạn 3, dùng đúng bản ghi trong
giai_doan_1_5_report.md dù indicators.db đã được cập nhật (refetch) sau khi model train
nên recompute trực tiếp từ db hiện tại lệch nhẹ (bậc 10^-4) so với bản đã đóng băng.
"""
import sqlite3

DB_PATH = "indicators.db"
HORIZON = 21  # phiên giao dịch, khớp trend_model.joblib
LABEL_LOW = -0.01911504222191937
LABEL_HIGH = 0.023262414737611113
BASELINE_LABEL = "Di_ngang"  # nhãn phổ biến nhất trên TRAIN của Giai đoạn 1.5


def label_from_return(r):
    """Áp ngưỡng nhãn đã đóng băng (Bước 0) lên 1 giá trị forward return."""
    if r < LABEL_LOW:
        return "Giam"
    if r > LABEL_HIGH:
        return "Tang"
    return "Di_ngang"


def _nearest_value(cur, indicator, as_of_date):
    """Giá trị GẦN NHẤT của indicator tính đến as_of_date (<=) - giống hệt logic
    trong main.py get_market_data(), tái dùng nguyên văn để đảm bảo tương thích định dạng."""
    cur.execute(
        "SELECT date, value FROM indicators WHERE indicator=? AND date<=? ORDER BY date DESC LIMIT 1",
        (indicator, as_of_date),
    )
    return cur.fetchone()


def get_market_data_asof(as_of_date, db_path=DB_PATH, horizon=HORIZON):
    """Snapshot market_data CHỈ dùng dữ liệu tới as_of_date (<=) - chống leakage.
    Mỗi mốc backtest gọi hàm này với ngày cắt riêng, không bao giờ nhìn thấy dữ liệu
    sau as_of_date. Trả (text, raw) hoặc (None, None) nếu thiếu dữ liệu.
    `text` đúng định dạng get_market_data() trong main.py / market_data_text trong
    eval_dataset.json (Giai đoạn 2.5) - để tiêm vào prompt đúng format đã validate."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        row_now = _nearest_value(cur, "gold", as_of_date)
        if row_now is None:
            return None, None
        date_now, gold_now = row_now

        cur.execute(
            "SELECT date, value FROM indicators WHERE indicator='gold' AND date<=? ORDER BY date DESC LIMIT 1 OFFSET ?",
            (date_now, horizon),
        )
        row_past = cur.fetchone()
        if row_past is None:
            return None, None
        date_past, gold_past = row_past

        others = {}
        for indicator in ("dxy", "real_yield", "fed_rate"):
            r_now = _nearest_value(cur, indicator, date_now)
            r_past = _nearest_value(cur, indicator, date_past)
            if r_now is None or r_past is None:
                return None, None
            others[indicator] = (r_now[1], r_past[1])

        dxy_now, dxy_past = others["dxy"]
        real_yield_now, real_yield_past = others["real_yield"]
        fed_rate_now, fed_rate_past = others["fed_rate"]

        gold_pct = (gold_now - gold_past) / gold_past * 100
        dxy_pct = (dxy_now - dxy_past) / dxy_past * 100
        real_yield_diff = real_yield_now - real_yield_past
        fed_rate_diff = fed_rate_now - fed_rate_past

        text = (
            f"Tính đến ngày {date_now} (so với {date_past}, khoảng {horizon} phiên giao dịch trước):\n"
            f"- Vàng (gold futures): hiện tại {gold_now:.2f} USD/oz, thay đổi {gold_pct:+.2f}%\n"
            f"- Chỉ số USD (DXY): hiện tại {dxy_now:.2f}, thay đổi {dxy_pct:+.2f}%\n"
            f"- Lợi suất thực 10 năm (real yield): hiện tại {real_yield_now:.2f}%, thay đổi {real_yield_diff:+.2f} điểm %\n"
            f"- Lãi suất Fed (fed funds rate): hiện tại {fed_rate_now:.2f}%, thay đổi {fed_rate_diff:+.2f} điểm %"
        )
        raw = {
            "date_now": date_now, "date_past": date_past,
            "gold_now": gold_now, "gold_past": gold_past, "gold_pct": gold_pct,
            "dxy_now": dxy_now, "dxy_past": dxy_past, "dxy_pct": dxy_pct,
            "real_yield_now": real_yield_now, "real_yield_past": real_yield_past, "real_yield_diff": real_yield_diff,
            "fed_rate_now": fed_rate_now, "fed_rate_past": fed_rate_past, "fed_rate_diff": fed_rate_diff,
        }
        return text, raw
    finally:
        conn.close()


def get_actual_outcome(as_of_date, db_path=DB_PATH, horizon=HORIZON):
    """Nhãn THẬT đã xảy ra (Giam/Di_ngang/Tang) - forward return gold từ as_of_date
    đến horizon phiên SAU đó. Dùng dữ liệu TƯƠNG LAI so với as_of_date một cách CHỦ Ý
    (đây là ground truth để chấm, không phải dữ liệu tiêm vào prompt cho bot)."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        row_now = _nearest_value(cur, "gold", as_of_date)
        if row_now is None:
            return None
        date_now, gold_now = row_now

        cur.execute(
            "SELECT date, value FROM indicators WHERE indicator='gold' AND date>? ORDER BY date ASC LIMIT 1 OFFSET ?",
            (date_now, horizon - 1),
        )
        row_future = cur.fetchone()
        if row_future is None:
            return None
        date_future, gold_future = row_future

        r = (gold_future - gold_now) / gold_now
        return {
            "date_now": date_now, "date_future": date_future,
            "gold_now": gold_now, "gold_future": gold_future,
            "forward_return": r, "label": label_from_return(r),
        }
    finally:
        conn.close()


def _count_trading_rows_between(db_path, indicator, date_a, date_b):
    """Đếm số DÒNG giao dịch (indicator=... , date > date_a AND date <= date_b) trong
    indicators.db - dùng để xác minh horizon=21 là 21 DÒNG (phiên giao dịch thật có
    trong db), không phải +21 ngày lịch (weekend/holiday không có dòng trong bảng)."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM indicators WHERE indicator=? AND date>? AND date<=?",
            (indicator, date_a, date_b),
        )
        return cur.fetchone()[0]
    finally:
        conn.close()


if __name__ == "__main__":
    # Smoke test bắt buộc trước khi chạy full N=50 (Task #4/#7) - kiểm tra thủ công
    # 4 mốc: đầu/giữa/cuối-an-toàn/SÁT BIÊN TRÊN của khoảng hợp lệ. Biên trên thật sự
    # là 21 phiên giao dịch TRƯỚC ngày gold mới nhất có trong indicators.db (không phải
    # trước "hôm nay" theo lịch, vì db có thể chưa update tới hôm nay).
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT date FROM indicators WHERE indicator='gold' ORDER BY date DESC LIMIT 1 OFFSET ?", (HORIZON,))
    upper_bound_asof = cur.fetchone()[0]
    cur.execute("SELECT MAX(date) FROM indicators WHERE indicator='gold'")
    last_db_date = cur.fetchone()[0]
    conn.close()
    print(f"[info] Ngay gold moi nhat trong db: {last_db_date} | Bien tren as_of hop le "
          f"(={HORIZON} phien truoc do): {upper_bound_asof}\n")

    smoke_dates = ["2024-06-03", "2025-01-15", "2025-12-01", upper_bound_asof]
    for d in smoke_dates:
        text, raw = get_market_data_asof(d)
        print(f"=== as_of={d} ===")
        if text is None:
            print("  KHONG CO DU LIEU SNAPSHOT")
        else:
            print(text)
            print("  raw values (tuyet doi that, tu get_market_data_asof, khong dien giai):")
            for k, v in raw.items():
                print(f"    {k}: {v}")
            print("  [leak-check moi chi bao rieng, nhat la real_yield/fed_rate]:")
            conn2 = sqlite3.connect(DB_PATH)
            cur2 = conn2.cursor()
            for ind in ("dxy", "real_yield", "fed_rate"):
                r_now = _nearest_value(cur2, ind, raw["date_now"])
                r_past = _nearest_value(cur2, ind, raw["date_past"])
                print(f"    {ind}: date_now_thuc_te={r_now[0]} (<=date_now={raw['date_now']}: "
                      f"{r_now[0] <= raw['date_now']}) | date_past_thuc_te={r_past[0]} "
                      f"(<=date_past={raw['date_past']}: {r_past[0] <= raw['date_past']})")
            conn2.close()

        outcome = get_actual_outcome(d)
        if outcome is None:
            print("  KHONG CO OUTCOME THAT (chua du du lieu tuong lai - dung neu d = bien tren)")
        else:
            n_rows = _count_trading_rows_between(DB_PATH, "gold", outcome["date_now"], outcome["date_future"])
            print(f"  outcome that: {outcome['date_now']} -> {outcome['date_future']} | "
                  f"forward_return={outcome['forward_return']:+.4f} -> nhan={outcome['label']}")
            print(f"  [verify horizon] so DONG giao dich giua date_now va date_future (phai = {HORIZON}): {n_rows}")
        print()
