"""Giai đoạn 4b - Verify độc lập cấu trúc fold walk-forward của
giai_doan_4b_rolling_window_experiment.py: embargo (train_max + 21 phien <= T) và
partition (khong chong, khong ho, tong dung 303) trên DỮ LIỆU THẬT, không suy diễn từ
ý định thiết kế.

Ghi chú: T, khoang du doan, va tap "elig_rows" (cac dong du dieu kien embargo tinh
den T) la GIONG HET nhau giua moi arm - window size chi quyet dinh CAT BAO NHIEU tu
elig_rows (elig_rows[-window:]), khong doi elig_rows hay T. Rieng train_max_date =
max(elig_rows) cung vi vay GIONG HET moi arm (ke ca expanding) tai cung 1 fold - da
xac nhan qua log chay that (fold 1: ca 5 arm deu train_range ket thuc 2023-02-15).
Vi vay chi can verify 1 lan, khong can lap lai cho tung arm.
"""
import json

import numpy as np
import pandas as pd

import giai_doan_4_train_model as m
import giai_doan_4_brier_diagnostic as bd

HORIZON = 21
FOLD_SIZE = 63

wide, base = m.load_series()
gold = wide["gold"]
all_dates = wide.index
date_pos = {d: i for i, d in enumerate(all_dates)}

diag = json.load(open("giai_doan_4_brier_diagnostic_results.json", encoding="utf-8"))
low = diag["label_threshold"]["low"]
high = diag["label_threshold"]["high"]

data = bd.build_labeled_data(base, gold, "rolling60", True)
val_post = data[(data.index >= m.REGIME_SPLIT_DATE) & (data.index <= m.VAL_END)]
assert len(val_post) == 303, f"val_post length={len(val_post)}, ky vong 303"


def safe_avail_date(d):
    target = date_pos[d] + HORIZON
    return all_dates[target] if target < len(all_dates) else None


avail = pd.Series({d: safe_avail_date(d) for d in data.index})

val_dates = val_post.index
folds = [val_dates[i:i + FOLD_SIZE] for i in range(0, len(val_dates), FOLD_SIZE)]

print(f"{'Fold':<5}{'T (retrain)':<13}{'train_max':<12}{'train_max+21<=T':<18}"
      f"{'predict_start':<15}{'predict_end':<13}{'n_train_2016':<13}{'n_predict':<10}")

rows = []
seen_dates = []
for i, fold_dates in enumerate(folds, 1):
    T = fold_dates[0]
    T_pos = date_pos[T]

    elig_mask = avail.notna() & (avail <= T)
    elig_rows = data.index[elig_mask]
    train_max = elig_rows.max()
    train_max_pos = date_pos[train_max]
    embargo_ok = (train_max_pos + HORIZON) <= T_pos

    # dung window=2016 lam vi du hien thi n_train (train_max/embargo khong doi theo window)
    n_train_2016 = min(len(elig_rows), 2016)

    row = {
        "fold": i,
        "T": str(T.date()),
        "train_max": str(train_max.date()),
        "embargo_ok": bool(embargo_ok),
        "train_max_pos_plus21": int(train_max_pos + HORIZON),
        "T_pos": int(T_pos),
        "predict_start": str(fold_dates[0].date()),
        "predict_end": str(fold_dates[-1].date()),
        "n_train_2016": int(n_train_2016),
        "n_predict": int(len(fold_dates)),
    }
    rows.append(row)
    seen_dates.extend(fold_dates.tolist())

    embargo_display = f"{embargo_ok} ({row['train_max_pos_plus21']}<={row['T_pos']})"
    print(f"{i:<5}{row['T']:<13}{row['train_max']:<12}{embargo_display:<18}"
          f"{row['predict_start']:<15}{row['predict_end']:<13}{row['n_train_2016']:<13}{row['n_predict']:<10}")

total_predict = sum(r["n_predict"] for r in rows)
no_dup = len(set(seen_dates)) == len(seen_dates)
matches_valpost_exact = sorted(seen_dates) == list(val_post.index)
all_embargo_ok = all(r["embargo_ok"] for r in rows)

# kiem tra khong chong / khong ho: fold sau phai bat dau NGAY SAU ngay cuoi fold truoc
# trong chinh day val_dates (khong co khoang trong / lap giua 2 fold lien ke)
no_gap_no_overlap = True
for i in range(1, len(folds)):
    prev_end_pos = list(val_dates).index(folds[i - 1][-1])
    cur_start_pos = list(val_dates).index(folds[i][0])
    if cur_start_pos != prev_end_pos + 1:
        no_gap_no_overlap = False

print(f"\nTong n_predict qua {len(folds)} fold = {total_predict} (ky vong 303): "
      f"{'DUNG' if total_predict == 303 else 'SAI'}")
print(f"Khong trung ngay giua cac fold: {no_dup}")
print(f"Khop CHINH XAC tap val_post (khong thieu, khong thua, dung thu tu): {matches_valpost_exact}")
print(f"Khong chong / khong ho giua fold lien ke (lien tuc trong val_dates): {no_gap_no_overlap}")
print(f"Embargo (train_max + {HORIZON} phien <= T) dung cho MOI fold: {all_embargo_ok}")
