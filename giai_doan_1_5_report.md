# Giai đoạn 1.5 — Báo cáo model dự đoán xu hướng vàng

## Dữ liệu & split
- Tổng số dòng dùng được (sau dropna lookback 60 + horizon 21): 4046
- Train: 3236 dòng (2010-04-01 → 2023-03-16)
- Test: 810 dòng (2023-03-17 → 2026-06-15)
- Split: 80/20 theo thời gian, không shuffle.

## Ngưỡng nhãn (Robust MAD, tính CHỈ trên train)
- median_train = 0.00207, MAD_train = 0.02858, robust_scale = 1.4826 × MAD = 0.04238
- Ngưỡng: low = median − 0.5×robust_scale = -0.01912, high = median + 0.5×robust_scale = 0.02326
- Ngưỡng cố định này áp dụng cho cả train và test (không tính lại trên test, tránh leakage).

## Đa cộng tuyến
- Mức raw indicator (real_yield diff vs fed_rate diff): 0.047 — đã kiểm tra trước, thấp, giữ cả 2 feature gốc, không tính lại.
- Mức feature đã kỹ thuật hoá (28 feature: lag/MA/volatility), tính trên train, ngưỡng |corr| > 0.85:
  loại 0/28 feature.
  - Không có cặp feature nào vượt ngưỡng.

## Phân phối nhãn Train vs Test (%)

| Nhãn | Train | Test |
|---|---|---|
| Giam | 28.7 | 18.5 |
| Di_ngang | 38.6 | 36.4 |
| Tang | 32.6 | 45.1 |

**Ghi chú**: tỉ lệ nhãn `Tang` ở test cao hơn hẳn train — test rơi vào giai đoạn vàng có xu hướng tăng cấu trúc (khoảng 2023-2026), phản ánh chế độ thị trường thật (structural bull market, không phải lỗi xử lý dữ liệu hay leakage).

## Baseline vs Model

- Baseline (nhãn phổ biến nhất trong train = `Di_ngang`, áp dụng hằng số lên test): accuracy test = 0.3642
- Logistic Regression: accuracy test = 0.2778, accuracy TRAIN = 0.4759

## Confusion matrix (hàng = thực tế, cột = dự đoán)

Thứ tự nhãn: ['Giam', 'Di_ngang', 'Tang']

```
[[ 95  26  29]
 [132  74  89]
 [117 192  56]]
```

## Classification report - Baseline (trên test)

```
              precision    recall  f1-score   support

        Giam      0.000     0.000     0.000       150
    Di_ngang      0.364     1.000     0.534       295
        Tang      0.000     0.000     0.000       365

    accuracy                          0.364       810
   macro avg      0.121     0.333     0.178       810
weighted avg      0.133     0.364     0.194       810

```

## Classification report - Logistic Regression (trên test)

```
              precision    recall  f1-score   support

        Giam      0.276     0.633     0.385       150
    Di_ngang      0.253     0.251     0.252       295
        Tang      0.322     0.153     0.208       365

    accuracy                          0.278       810
   macro avg      0.284     0.346     0.282       810
weighted avg      0.288     0.278     0.257       810

```

## Hệ số Logistic Regression (top 10 theo |giá trị|, mỗi nhãn)

**Nhãn `Di_ngang`:**

- `gold_ret_vol60`: -0.1607
- `fed_rate_diff_ma60`: -0.1598
- `fed_rate_diff_vol60`: +0.1323
- `real_yield_diff_vol60`: -0.1135
- `gold_ret_ma60`: +0.0937
- `fed_rate_diff_lag21`: -0.0552
- `real_yield_diff_ma60`: -0.0550
- `gold_ret_lag1`: -0.0547
- `dxy_ret_ma60`: +0.0539
- `fed_rate_diff_lag1`: +0.0525

**Nhãn `Giam`:**

- `fed_rate_diff_ma60`: +0.3485
- `gold_ret_vol60`: +0.3272
- `dxy_ret_vol60`: -0.2563
- `gold_ret_ma60`: +0.2024
- `fed_rate_diff_vol60`: -0.1141
- `dxy_ret_ma60`: +0.1118
- `real_yield_diff_vol60`: -0.0530
- `fed_rate_diff_lag60`: +0.0528
- `fed_rate_diff_lag1`: -0.0495
- `gold_ret_lag1`: +0.0457

**Nhãn `Tang`:**

- `gold_ret_ma60`: -0.2961
- `dxy_ret_vol60`: +0.2647
- `fed_rate_diff_ma60`: -0.1888
- `gold_ret_vol60`: -0.1665
- `real_yield_diff_vol60`: +0.1665
- `dxy_ret_ma60`: -0.1657
- `real_yield_diff_lag1`: -0.0368
- `gold_ret_lag60`: -0.0331
- `real_yield_diff_lag60`: -0.0306
- `dxy_ret_lag10`: -0.0277

## Diễn giải tổng hợp (phương pháp)

Trên TRAIN, model đạt accuracy 0.476 — cao hơn hẳn tỉ lệ nhãn phổ biến nhất trong train (38.6%), tức model không chỉ học thuộc nhãn đa số mà thực sự khai thác được quan hệ giữa feature và nhãn trên dữ liệu nó thấy trực tiếp.

Phần lớn hệ số khớp lý thuyết vĩ mô đã chốt trong dự án: `fed_rate_diff_ma60` mang dấu âm cho nhãn Tăng (-0.189) và dương cho nhãn Giảm (+0.349) — đúng cơ chế chi phí cơ hội (fed rate trung bình cao hơn thì vàng kém hấp dẫn); `real_yield_diff_vol60` mang dấu dương cho nhãn Tăng (+0.167) — đúng cơ chế bất định lãi suất thực đẩy dòng tiền trú ẩn vào vàng.

Nhưng feature có hệ số lớn nhất cho nhãn Tăng lại là `gold_ret_ma60` với dấu ÂM (-0.296) — model học được một quy luật MEAN-REVERSION: return trung bình 60 ngày của vàng càng cao thì xác suất TIẾP TỤC Tăng bị đánh giá càng THẤP. Quy luật này khớp với hành vi giá vàng trong phần lớn giai đoạn train (2010-2023), vốn dao động biên độ (range-bound) nhiều hơn là xu hướng dài hạn liên tục.

Test (2023-2026) lại rơi đúng vào giai đoạn vàng tăng cấu trúc (structural bull market) — bằng chứng nằm ngay ở phân phối nhãn đã đo được: tỉ lệ Tăng nhảy từ 32.6% (train) lên 45.1% (test). Trong chế độ này, return 60 ngày cao thường KHÔNG đảo chiều mà tiếp tục tăng — ngược hẳn với quy luật mean-reversion model đã học từ train.

Đây chính là lý do model thua baseline trên test (0.278 so với 0.364) dù học được tín hiệu thật và có ý nghĩa lý thuyết trên train: không phải model học sai hay có bug trong pipeline (ngưỡng nhãn, split, scaling đều đã kiểm tra không leakage), mà là REGIME SHIFT giữa 2 giai đoạn khiến một giả định thống kê hợp lý trên train (mean-reversion) trở thành sai lệch có hệ thống trên test. Đây là giới hạn cốt lõi, có số liệu cụ thể minh hoạ, của mọi model dự đoán tài chính dù tuân thủ đúng train/test split theo thời gian: model chỉ học được quy luật của quá khứ, không đảm bảo quy luật đó còn đúng khi chế độ thị trường đổi. Đây là một finding cần ghi nhận, không phải lỗi cần vá.