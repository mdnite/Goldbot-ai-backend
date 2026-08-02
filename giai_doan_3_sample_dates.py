"""Giai đoạn 3, Task #4 - Chọn N=50 mốc backtest.

Thiết kế đã xác nhận với người dùng:
- Khung: chỉ ngày giao dịch THẬT trong indicators.db (indicator='gold'), khoảng
  [2024-06-01, 2026-06-22] (biên trên = 21 phiên trước ngày dữ liệu mới nhất, xem
  giai_doan_3_backtest.py smoke test).
- 1 regime duy nhất trong khung này (Giai đoạn 1.5 chỉ định nghĩa 2 regime, ranh giới
  = điểm cắt train/test 2023-03-16/17 - đã kết thúc hơn 1 năm trước khi khung này bắt
  đầu) -> không phân tầng theo regime, chỉ cần phủ đều thời gian.
- min-gap lý tưởng = 21 phiên (bằng horizon) chỉ cho tối đa ~25 mốc không chồng lấn,
  không đủ N=50 -> hạ xuống min-gap ~10 phiên (= floor(515/50), mức lớn nhất còn đủ
  chỗ cho 50 điểm): chia 515 ngày hợp lệ thành 50 bin gần đều, lấy ngẫu nhiên 1
  ngày/bin, seed=42 cố định (khớp convention seed Ollama ở Giai đoạn 2.5).
- n hiệu dụng ~ 515/21 ~ 24-25 (cửa sổ outcome 21 phiên của các mốc liền kề chồng lấn
  đáng kể vì spacing thực tế ~10 < horizon 21) - ghi rõ trong output, dùng cho SE ở
  Task #7, KHÔNG dùng N=50 danh nghĩa để tính SE.
"""
import json
import random
import sqlite3

from giai_doan_3_backtest import DB_PATH, HORIZON

VALID_START = "2024-06-01"
N_SAMPLES = 50
SEED = 42
OUTPUT_PATH = "giai_doan_3_sample_dates.json"


def get_valid_trading_dates(db_path=DB_PATH, horizon=HORIZON, valid_start=VALID_START):
    """Toàn bộ ngày giao dịch gold trong [valid_start, upper_bound], upper_bound =
    horizon phiên trước ngày gold mới nhất có trong db."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT date FROM indicators WHERE indicator='gold' ORDER BY date DESC LIMIT 1 OFFSET ?",
            (horizon,),
        )
        upper_bound = cur.fetchone()[0]
        cur.execute(
            "SELECT date FROM indicators WHERE indicator='gold' AND date>=? AND date<=? ORDER BY date ASC",
            (valid_start, upper_bound),
        )
        dates = [row[0] for row in cur.fetchall()]
        return dates, upper_bound
    finally:
        conn.close()


def stratified_sample(dates, n_samples, seed):
    """Chia `dates` thành n_samples bin gần đều theo thời gian, lấy ngẫu nhiên đúng 1
    ngày/bin (seed cố định) - vừa random vừa tránh dồn cụm/spacing quá hẹp."""
    rng = random.Random(seed)
    n = len(dates)
    # np.array_split-style: bin đầu nhận phần dư để các bin gần đều nhau nhất có thể
    base, extra = divmod(n, n_samples)
    bins = []
    start = 0
    for i in range(n_samples):
        size = base + (1 if i < extra else 0)
        bins.append(dates[start:start + size])
        start += size
    sampled = sorted(rng.choice(b) for b in bins)
    return sampled, bins


def trading_day_gaps(dates_sorted, all_valid_dates):
    """Khoảng cách (số phiên giao dịch) giữa các mốc liền kề đã chọn, tính trên chỉ
    mục trong all_valid_dates (không phải hiệu ngày lịch)."""
    idx = {d: i for i, d in enumerate(all_valid_dates)}
    gaps = []
    for a, b in zip(dates_sorted, dates_sorted[1:]):
        gaps.append(idx[b] - idx[a])
    return gaps


if __name__ == "__main__":
    valid_dates, upper_bound = get_valid_trading_dates()
    print(f"[info] So ngay giao dich hop le trong [{VALID_START}, {upper_bound}]: {len(valid_dates)}")

    sampled, bins = stratified_sample(valid_dates, N_SAMPLES, SEED)
    print(f"[info] Da chon {len(sampled)} moc (seed={SEED}), bin size: "
          f"{[len(b) for b in bins][:5]}... (tong {len(bins)} bin)")

    gaps = trading_day_gaps(sampled, valid_dates)
    print(f"\n[check spacing] min gap={min(gaps)} phien, max gap={max(gaps)} phien, "
          f"trung binh={sum(gaps)/len(gaps):.1f} phien")

    effective_n = len(valid_dates) / HORIZON
    print(f"[n hieu dung] tong_phien_hop_le / horizon = {len(valid_dates)}/{HORIZON} = {effective_n:.1f} "
          f"(~{round(effective_n)}) - dung con so nay cho SE o Task #7, KHONG dung N={N_SAMPLES} danh nghia")

    print(f"\n[danh sach 50 moc]")
    for d in sampled:
        print(f"  {d}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "_meta": {
                "n_samples": N_SAMPLES,
                "seed": SEED,
                "valid_start": VALID_START,
                "valid_end": upper_bound,
                "n_valid_trading_days": len(valid_dates),
                "horizon": HORIZON,
                "effective_n": round(effective_n, 1),
                "min_gap_trading_days": min(gaps),
                "max_gap_trading_days": max(gaps),
                "mean_gap_trading_days": round(sum(gaps) / len(gaps), 2),
            },
            "sample_dates": sampled,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {OUTPUT_PATH}")
