# GoldBot — Báo cáo Tổng hợp: Các Phương Án Dự Đoán Xu Hướng Đã Thử

**Phạm vi:** Giai đoạn 1.5 → Giai đoạn 4b (rolling-window retrain). Giai đoạn 4 (4a + 4b) đã đóng hoàn toàn (2026-08-07) — chưa chốt hướng đi tiếp theo cho bước sau.
**Mục đích:** Tổng hợp toàn bộ phương án đã thử cho bài toán dự đoán xu hướng giá vàng ngắn hạn, kết quả từng phương án, nguyên nhân thất bại/hạn chế, và toàn bộ phương pháp luận/thuật toán/thước đo đã áp dụng — dùng làm tài liệu tham chiếu khi quyết định hướng đi tiếp theo.
**Nguồn:** `giai_doan_1_5_report.md` (qua `GoldBot_Bao_Cao_Qua_Trinh.md`), `giai_doan_3_report.md`, `giai_doan_4_report.md`, `CLAUDE.md`/`Roadmap.md`, kết quả chẩn đoán Brier score, và kết quả thí nghiệm Giai đoạn 4b (`giai_doan_4b_rolling_window_experiment.py`, `giai_doan_4b_rolling_window_results.json`, `giai_doan_4b_verify_folds.py`).

---

## Tóm tắt nhanh

| Giai đoạn | Phương pháp | Kết quả chính | Trạng thái |
|---|---|---|---|
| 1.5 | Logistic Regression, 28 feature, Vol60 rolling std | Test acc 0.278 vs baseline 0.364 | THUA baseline — regime shift |
| 3 | LLM suy luận qua backtest N=50 (few-shot prompting) | 46.0% vs Baseline1 30.0% (chênh ~1.2 SE) | Cải thiện nhưng CHƯA đủ ý nghĩa thống kê — dừng đầu tư |
| 4a | Logistic Regression + VIF + interaction + EWMA | Test 30.0% (=Baseline1), thua LLM 46% và Baseline2 54% | KHÔNG ĐẠT — CI95 hoàn toàn dưới ngưỡng |
| Chẩn đoán Brier | Đo Brier score đa lớp, 96 tổ hợp feature×C×penalty trên validation | Chỉ 2/96 tổ hợp thắng baseline ngây thơ, biên rất mỏng | Đã dẫn tới Giai đoạn 4b bên dưới |
| 4b | Rolling-window retrain (walk-forward), 5 arm, feature/hyperparameter cố định | Cả 5 arm không tách biệt khỏi static (0.6276) hay Baseline1 (0.6341) | KHÔNG ĐẠT — ĐÃ ĐÓNG, Giai đoạn 4 đóng hoàn toàn |

---

## 1. Giai đoạn 1.5 — Logistic Regression gốc (`trend_model.joblib`)

### Thiết kế
- Horizon dự đoán: 21 phiên giao dịch (~1 tháng thị trường, không phải 30 ngày dương lịch).
- Lookback: 60 phiên (đủ bắt ít nhất 1 lần họp FOMC, ~45 ngày/lần).
- Nhãn 3 lớp (Tăng/Giảm/Đi_ngang), ngưỡng = median ± 0.5×1.4826×MAD, tính CHỈ trên train, đóng băng.
- 28 feature: 4 chuỗi gốc (`gold_ret`, `dxy_ret`, `real_yield_diff`, `fed_rate_diff`) × [lag 1,5,10,21,60 + MA60 + Vol60 rolling std].
- Lọc đa cộng tuyến: chỉ pairwise correlation (ngưỡng |corr|>0.85) — **chưa có VIF thật**.
- Split 80/20 theo thời gian (không shuffle). Baseline = nhãn phổ biến nhất của TRAIN áp lên test.
- Model: Logistic Regression (chọn thay Random Forest — ưu tiên diễn giải được hệ số).

### Kết quả

| Chỉ số | Giá trị |
|---|---|
| Tổng dòng dùng được | 4046 (từ 4156 gốc — mất 28 dòng do real_yield thiếu Columbus/Veterans Day, 1 dòng do off-by-one `shift(60)`) |
| Train | 3236 dòng (2010-04-01 → 2023-03-16) |
| Test | 810 dòng (2023-03-17 → 2026-06-15) |
| Ngưỡng nhãn (MAD) | low = −0.01912, high = +0.02326 |
| Đa cộng tuyến | 0/28 loại (pairwise corr) |
| Accuracy train | 0.4759 |
| Accuracy test | 0.2778 |
| Baseline accuracy (test) | 0.3642 |

### Nguyên nhân thất bại
Model học được tín hiệu thật trên train (47.6% > baseline train ~38.6%), nhiều hệ số khớp lý thuyết vĩ mô. Nhưng hệ số mạnh nhất cho nhãn Tăng là `gold_ret_ma60` (−0.296) — mã hoá quy luật **mean-reversion**, đúng với đặc tính range-bound của train (2010–2023). Test (2023–2026) là **regime khác hẳn** — bull market cấu trúc thật, tỉ lệ nhãn Tăng nhảy từ 32.6% (train) → 45.1% (test). Trong regime này, return cao KHÔNG đảo chiều — ngược hẳn quy luật model học được.

**Kết luận methodology:** finding hợp lệ về giới hạn của mọi model dự đoán tài chính (quy trình đúng, không leakage) — không phải bug. Ba vấn đề kỹ thuật được xác định để sửa ở giai đoạn sau: đa cộng tuyến (chỉ lọc pairwise, chưa VIF), feature cô lập (không có interaction), volatility trễ (rolling std cố định phản ứng chậm với đổi regime).

---

## 2. Giai đoạn 2 / 2.5 — Bối cảnh (không thuộc hướng dự đoán trực tiếp)

Giai đoạn 2 quyết định KHÔNG inject nhãn của `trend_model.joblib` vào RAG prompt (đã biết kém tin cậy hơn baseline), chỉ inject số liệu thô. Giai đoạn 2.5 đo Faithfulness (đọc đúng dữ liệu, không phải dự đoán) — 3 vòng benchmark tự động 17/25→20/25→21/25%, nhưng audit tay xác nhận **Faithfulness thật = 24/25 (96%)**, đa số "lỗi" là do chính công cụ chấm (`eval_script.py`), không phải bot. Không liên quan trực tiếp tới các phương án dự đoán xu hướng, nêu ở đây để đủ mạch giai đoạn.

---

## 3. Giai đoạn 3 — LLM suy luận qua backtest lịch sử

### Thiết kế
- N=50 mốc thời gian lịch sử (2024-06-01 → 2026-06-22), chia 50 bin gần đều trên 515 phiên hợp lệ, random 1 ngày/bin, seed=42.
- n hiệu dụng ≈ 25 (không phải 50 — do horizon 21 phiên chồng lấn giữa các mốc, không độc lập hoàn toàn).
- 1 câu hỏi cố định lặp lại ở 50 mốc, snapshot dữ liệu leak-safe tại từng `as_of_date`, gọi Ollama trực tiếp (`temperature=0.0, seed=42`).
- 2 baseline: Baseline 1 (đóng băng, nhãn đa số của TRAIN Giai đoạn 1.5 — công bằng, out-of-sample); Baseline 2 (nhãn đa số của chính 50 mốc test — hindsight, không công bằng, chỉ tham khảo).

### Kết quả

| | Đúng/Tổng | % |
|---|---|---|
| Bot GoldBot (LLM suy luận) | 23/50 | **46.0%** |
| Baseline 1 — đóng băng (công bằng) | 15/50 | 30.0% |
| Baseline 2 — hindsight (không công bằng) | 27/50 | 54.0% |
| `trend_model.joblib` (Giai đoạn 1.5, tham khảo) | 12/50 | 24.0% |

Chênh lệch bot–Baseline1 (16pp) ≈ **1.2 lần sai số chuẩn kết hợp** (n hiệu dụng=25) — chưa đạt ngưỡng ý nghĩa thống kê thông thường (~2 SE).

**Phân phối nhãn dự đoán lệch rõ so với thực tế:**

| Nhãn | Bot dự đoán | Thực tế |
|---|---|---|
| Tăng | 30/50 (60%) | 27/50 (54%) |
| Giảm | 16/50 (32%) | 8/50 (16%) |
| Đi ngang | 4/50 (8%) | 15/50 (30%) |

### Audit lỗi suy luận (rà tay toàn bộ 50/50 mốc)

| Loại lỗi | Số mốc mắc lỗi | Tỷ lệ |
|---|---|---|
| DXY sai chiều tác động | 3/50 | 6% |
| Real yield sai chiều tác động | 1/50 | 2% |
| Fed rate sai chiều tác động | 5/10 mốc có khẳng định rõ | ~50% |
| Đọc sai dấu số liệu thật (vi phạm QUY TẮC 5) | 1/50 | 2% |

Đặc biệt: model **không có "lý thuyết nội tại" nhất quán** về quan hệ Fed rate–vàng (đúng/sai gần như ngẫu nhiên), khác hẳn DXY/real yield (đúng áp đảo, sai rải rác).

### Quyết định
Dù có cải thiện tương đối so với baseline, chưa đạt ý nghĩa thống kê + có lỗi suy luận hệ thống + ưu tiên khả năng diễn giải/kiểm chứng cơ chế — quyết định **dừng đầu tư hướng LLM/prompt/LoRA**, chuyển sang sửa model định lượng hiện có. Toàn bộ code/dữ liệu archive tại `archive/giai_doan_3_llm_approach/`, không xoá.

---

## 4. Giai đoạn 4a — Cải tiến định lượng (thay thế hướng LoRA)

### Thiết kế — sửa 3 vấn đề kỹ thuật của Giai đoạn 1.5, giữ nguyên Logistic Regression

| Vấn đề gốc | Cách sửa |
|---|---|
| Đa cộng tuyến | VIF thật (iterative, ngưỡng >5.0, tính bằng LinearRegression) thay vì chỉ pairwise corr |
| Feature cô lập | +6 interaction feature (3 cặp macro × [lag1, MA60], không gồm `gold_ret`) → 34 feature |
| Volatility trễ | EWMA (`span=10/21`, `adjust=False`) thay Vol60 rolling std cố định |

### Quá trình tuning (validation 531 dòng: pre2023 n=228, post2023 n=303)

| Vòng | Lưới C | Best | val_acc | pre2023 | post2023 | Ghi chú |
|---|---|---|---|---|---|---|
| 1 | [0.01, 0.1, 1, 10] | vol=21, l1, C=0.01 | 0.473 | 0.461 | 0.482 | Best ở mép lưới → mở rộng |
| 2 | [0.001, 0.003, 0.005, 0.01] | C=0.01 (trùng R1) | 0.473 | 0.461 | 0.482 | Phát hiện "vách đứng": C≤0.005 sụp hoàn toàn (val_acc=0.414, 34/34 hệ số=0) |
| 3 | [0.02, 0.03, 0.05, 0.07] | vol=21, l1, C=0.02–0.03 | 0.484 | 0.535–0.557 | 0.446–0.429 | val_acc gộp cao hơn NHƯNG post2023 thấp hơn — loại vì 100% test thuộc post2023 |

**Hyperparameter chốt:** `vol=21 (EWMA), penalty=l1, C=0.01`. Refit trên TRAIN+VAL (n=3539, 2010-04-01→2024-05-31). Ground truth test dùng nguyên ngưỡng gốc Giai đoạn 1.5 (không tính lại) để so sánh công bằng.

### Kết quả test (chạm 1 lần duy nhất)

| | Đúng/Tổng | % |
|---|---|---|
| **Model 4a** | 15/50 | **30.0%** |
| Bot LLM (Giai đoạn 3) | 23/50 | 46.0% |
| Baseline 1 (đóng băng) | 15/50 | 30.0% |
| Baseline 2 (hindsight) | 27/50 | 54.0% |
| `trend_model.joblib` cũ | — | 24.0% |

**Kiểm định thống kê** (Wald CI 95% một mẫu, chốt công thức trước khi chạy test): CI95(n=50)=[0.173, 0.427]; CI95(n_eff=25)=[0.120, 0.480] — cả 2 **hoàn toàn dưới 0.46** → không có ý nghĩa thống kê → **KHÔNG ĐẠT**.

### Chẩn đoán nguyên nhân
Model dự đoán `Đi_ngang` cho TOÀN BỘ 50/50 mốc — trùng khớp Baseline1 tuyệt đối. Đã loại trừ bug scaler (khớp tay TRAIN+VAL chính xác) và outlier feature (chỉ 2/50 lệch nhẹ 1 feature). Cơ chế thật: `predict_proba` trung bình = [Đi_ngang 0.408, Giảm 0.307, Tăng 0.285], margin mỏng nhất chỉ **0.016** (mốc 2026-05-26) — regularization quá mạnh (chỉ 6/34 hệ số sống sót, đều nhỏ 0.03–0.12) khiến dự đoán bám sát nhãn nền. Model KHÔNG suy biến hoàn toàn (in-sample vẫn đoán Tăng 13.8%, Giảm 1.1%), nhưng tín hiệu quá yếu để thắng ở bất kỳ mốc nào trong 50 mốc test.

Câu hỏi còn treo (chưa giải thích dứt điểm, không chặn kết luận): 500 dòng in-sample cuối cùng vẫn đoán Tăng 24.6%, nhưng rớt về 0% ngay ở 50 mốc test kế tiếp.

### Đánh giá 3 vấn đề gốc

| Vấn đề | Trạng thái |
|---|---|
| Đa cộng tuyến | **Giải quyết dứt điểm** — VIF 0/34 loại ở mọi vòng, độc lập với kết quả test |
| Feature cô lập (interaction) | Chỉ sửa thiết kế — chưa có bằng chứng giải quyết vấn đề thật (tại thời điểm đó) |
| Volatility trễ (EWMA) | Chỉ sửa thiết kế — chưa có bằng chứng giải quyết vấn đề thật (tại thời điểm đó) |

**Quyết định loại bỏ:** Random Forest/XGBoost (Giai đoạn 4b định nghĩa gốc) — loại dứt khoát do mất tính diễn giải (lý do mạnh nhất), dù rủi ro overfit/extrapolation đã giảm nhẹ qua kiểm chứng feature không outlier.

---

## 5. Chẩn đoán Brier Score (đã dẫn tới Giai đoạn 4b — xem mục 6)

### Động lực
4a dùng accuracy để chọn C lúc tuning — accuracy dễ bị "lừa" bởi model suy biến về nhãn đa số. Trước khi triển khai "4a-v2" (đổi metric + ép C≥0.05), cần tách bạch: **interaction terms và EWMA có thật sự giúp không**, độc lập khỏi ảnh hưởng của việc chọn C bằng accuracy.

### Thiết kế thí nghiệm
- Thước đo: Brier score đa lớp — BS = (1/N)ΣᵢΣₖ(pᵢₖ−yᵢₖ)² (càng THẤP càng tốt).
- 6 biến thể feature: 3 volatility (`ewma10`, `ewma21` = cấu hình 4a đã khoá, `rolling60` = bản gốc Giai đoạn 1.5) × 2 interaction (có/không).
- 8 mức C log-spaced: [0.01, 0.02, 0.05, 0.1, 0.3, 1, 3, 10] × 2 penalty (l1, l2) = **96 tổ hợp**.
- Chạy hoàn toàn trên validation, khúc post-2023 (n=303) — **không chạm 50 mốc test khoá** (xác nhận cả về code lẫn về mặt thời gian: `VAL_END=2024-05-31` kết thúc trước mốc test sớm nhất 2024-06-01).
- Ngưỡng nhãn dùng chung 1 lần (verify bằng dry-run: TRAIN range trùng tuyệt đối ở cả 6 biến thể, n_train=3008) — low=−0.01873, high=+0.02302.

### Kết quả đầy đủ 96 tổ hợp (Brier score, thấp hơn = tốt hơn)

**Penalty = L1**

| C | ewma10 | ewma10+I | ewma21 | ewma21+I | rolling60 | rolling60+I |
|---|---|---|---|---|---|---|
| 0.01 | 0.6395 | 0.6361 | 0.6452 | 0.6394 | 0.6377 | **0.6290** |
| 0.02 | 0.6565 | 0.6432 | 0.6665 | 0.6494 | 0.6479 | **0.6276** |
| 0.05 | 0.6751 | 0.6523 | 0.6922 | 0.6689 | 0.6551 | 0.6377 |
| 0.10 | 0.6873 | 0.6579 | 0.7071 | 0.6731 | 0.6659 | 0.6424 |
| 0.30 | 0.6977 | 0.6652 | 0.7185 | 0.6802 | 0.6742 | 0.6509 |
| 1.00 | 0.7017 | 0.6688 | 0.7239 | 0.6839 | 0.6766 | 0.6535 |
| 3.00 | 0.7029 | 0.6699 | 0.7256 | 0.6850 | 0.6772 | 0.6542 |
| 10.00 | 0.7033 | 0.6702 | 0.7262 | 0.6854 | 0.6774 | 0.6545 |

**Penalty = L2**

| C | ewma10 | ewma10+I | ewma21 | ewma21+I | rolling60 | rolling60+I |
|---|---|---|---|---|---|---|
| 0.01 | 0.6860 | 0.6569 | 0.7021 | 0.6687 | 0.6632 | 0.6426 |
| 0.02 | 0.6932 | 0.6622 | 0.7119 | 0.6752 | 0.6683 | 0.6464 |
| 0.05 | 0.6988 | 0.6666 | 0.7198 | 0.6808 | 0.6730 | 0.6504 |
| 0.10 | 0.7010 | 0.6684 | 0.7230 | 0.6830 | 0.6751 | 0.6523 |
| 0.30 | 0.7026 | 0.6697 | 0.7252 | 0.6847 | 0.6767 | 0.6537 |
| 1.00 | 0.7032 | 0.6702 | 0.7261 | 0.6853 | 0.6772 | 0.6542 |
| 3.00 | 0.7034 | 0.6703 | 0.7263 | 0.6854 | 0.6774 | 0.6544 |
| 10.00 | 0.7034 | 0.6704 | 0.7264 | 0.6855 | 0.6775 | 0.6545 |

### Baseline tham chiếu (trên đúng VAL post-2023, n=303)

| Baseline | Vector xác suất [Đi_ngang, Giảm, Tăng] | Brier score |
|---|---|---|
| Baseline 1 — công bằng (tỉ lệ nhãn đóng băng từ TRAIN) | [0.3913, 0.2856, 0.3231] | **0.6341** |
| Baseline 2 — hindsight, không công bằng (tỉ lệ nhãn thật của VAL post-2023) | [0.5347, 0.1881, 0.2772] | 0.6019 |
| Baseline 3 — mốc suy biến (100% Đi_ngang, đúng hành vi 4a thật) | [1.0, 0.0, 0.0] | 0.9307 |

### Phát hiện chính

1. **Interaction terms — có bằng chứng thật, nhất quán.** Cả 48/48 cặp so sánh (có/không interaction, mọi C, mọi penalty) đều nghiêng về phía có interaction. Đây là bằng chứng thật đầu tiên cho fix này, khác với lúc 4a chỉ là "sửa thiết kế chưa chứng minh".

2. **EWMA volatility — bằng chứng đi NGƯỢC kỳ vọng.** `rolling60` (bản gốc, chính cái EWMA được sinh ra để thay thế) cho Brier thấp hơn (tốt hơn) `ewma21` (cấu hình đã khoá ở 4a) ở **mọi** mức C, cả 2 penalty. Toàn bộ họ `ewma21` — tức chính cấu hình 4a thật — nằm trong nhóm TỆ NHẤT của cả lưới 96 tổ hợp.

3. **Quy luật C — mâu thuẫn trực tiếp với kế hoạch "4a-v2" gốc.** Brier cải thiện đơn điệu khi C giảm, ở mọi biến thể. Nếu dùng Brier để chọn C thay accuracy, kết quả vẫn sẽ chọn lại đúng vùng C nhỏ (C=0.01 hoặc thấp hơn) — không phải C lớn hơn như kế hoạch "ép C≥0.05" dự định. Cách đọc hợp lý nhất: dưới điều kiện tín hiệu yếu + regime shift, mọi thước đo (kể cả Brier) đều "thưởng" cho việc thận trọng/bám gần nhãn nền — không phải vì Brier bị lừa như accuracy, mà vì thận trọng thật sự tốt hơn khi không có nhiều tín hiệu ổn định để khai thác.

4. **So với baseline ngây thơ — chỉ 2/96 tổ hợp thắng, biên rất mỏng.** Chỉ `rolling60+interaction` ở C=0.01 (0.6290) và C=0.02 (0.6276, thấp nhất toàn lưới) vượt qua Baseline 1 (0.6341) — chênh ~0.005–0.007 (~1%), CHƯA qua kiểm định ý nghĩa thống kê nào. Mọi tổ hợp còn lại — bao gồm TOÀN BỘ cấu hình `ewma21` (chính là 4a thật) — đều **tệ hơn việc không dùng model nào, chỉ đoán đúng tỉ lệ nhãn lịch sử**.

### Trạng thái (đã giải quyết)
Tại thời điểm chẩn đoán, có 3 phương án đang cân nhắc: (1) thử 1 lượt test cuối với `rolling60+interaction`, C nhỏ; (2) coi đây là bằng chứng đủ để dừng nhánh Logistic Regression, mở lại thảo luận Giai đoạn 4b; (3) xem lại framing bài toán. Quyết định thực tế đã đi hướng khác cả 3: **đổi định nghĩa Giai đoạn 4b** từ Random Forest/XGBoost (dự phòng gốc) sang rolling-window retrain, dùng nguyên cấu hình thắng của chẩn đoán này (`rolling60+interaction`, l1, C=0.02) làm nền — xem mục 6 cho thiết kế, kết quả, và lý do đóng giai đoạn.

---

## 6. Giai đoạn 4b — Rolling-window retrain (walk-forward) — ĐÃ ĐÓNG

### Bối cảnh và đổi định nghĩa
Roadmap gốc định nghĩa 4b = Random Forest/XGBoost (dự phòng nếu 4a không đạt — xem mục 4). Trước khi triển khai (quyết định 2026-08-07), định nghĩa này bị bỏ: tree-based model không extrapolate ra ngoài range feature đã thấy lúc train, đúng vào điểm yếu cốt lõi lặp lại xuyên suốt dự án (regime shift đẩy feature 2023-2026 ra ngoài range train 2010-2023 — xem mục 8.1). Đổi RF/XGBoost không giải quyết đúng vấn đề đang gặp. Định nghĩa mới đã chạy: train lại định kỳ (walk-forward) trên cửa sổ dữ liệu trượt gần nhất, thay vì train 1 lần trên cửa sổ tĩnh cố định (2010-2022).

### Thiết kế thí nghiệm
- Base feature CỐ ĐỊNH: `rolling60+interaction` — đúng cấu hình thắng trong 96 tổ hợp Brier (mục 5), cho Brier static = 0.6276.
- Hyperparameter CỐ ĐỊNH: `penalty=l1, C=0.02` — dùng y hệt ở mọi arm/fold, không tune lại. Chỉ đổi đúng 1 biến: chiến lược train (window size).
- Ngưỡng nhãn: tái dùng NGUYÊN VĂN từ chẩn đoán Brier (không tính lại theo fold/window).
- 5 arm: rolling window 504/756/1260/2016 phiên (~2/3/5/8 năm, dòng dữ liệu gần nhất) + `expanding` (toàn bộ lịch sử đủ điều kiện, không giới hạn) — arm đối chứng, thêm vào để tách bạch "retrain định kỳ" khỏi "chỉ dùng dữ liệu gần nhất" (nếu không có arm này, rolling-window thắng static sẽ không biết là do window ngắn hay do retrain nhiều lần).
- Walk-forward CHỈ trong val_post đã dùng để tính Brier static (2023-03-17 → 2024-05-31, n=303) — không mở rộng, không chạm 50 mốc test khoá.
- Retrain theo quý: 5 fold liên tiếp (~63 phiên/fold, fold cuối 51), không chồng không hở, tổng đúng 303 điểm dự đoán.
- Embargo leak-safe: dòng ngày d chỉ dùng để train nếu d+21 phiên (HORIZON, đúng vị trí lịch giao dịch thật) ≤ ngày retrain T.
- VIF filter + StandardScaler fit lại từ đầu trên MỖI cửa sổ train riêng.
- KHÔNG dùng gate ĐẠT/KHÔNG ĐẠT tự động — chỉ báo cáo Brier tổng hợp + 95% CI bằng **block bootstrap** (block=21, N=2000) — tôn trọng tự tương quan do nhãn horizon=21 phiên chồng lấn, tránh phóng đại độ tin cậy như bootstrap i.i.d. thường.

### Verify leak-safety độc lập
T, khoảng dự đoán, và tập dòng đủ điều kiện embargo (nên cả `train_max`) giống hệt nhau ở mọi arm — window size chỉ quyết định cắt bao nhiêu từ cuối tập đó. Verify trên dữ liệu thật (`giai_doan_4b_verify_folds.py`, không suy diễn từ ý định thiết kế): tổng n_predict qua 5 fold = 303 (đúng), không trùng ngày giữa fold, khớp chính xác val_post không thiếu không thừa, không chồng/không hở, embargo đúng cho MỌI fold (cả 5 fold đều khít sát biên `train_max+21 == T` — kết quả tất yếu của việc chọn dòng train gần T nhất còn hợp lệ, không phải dấu hiệu lỗi).

### Kết quả

| Arm | Brier | 95% CI (block bootstrap, block=21) |
|---|---|---|
| static (tham chiếu, không chạy lại) | 0.6276 | — |
| rolling 504 (~2 năm) | 0.6386 | [0.6108, 0.6760] |
| rolling 756 (~3 năm) | 0.6431 | [0.6099, 0.6905] |
| rolling 1260 (~5 năm) | **0.6076** | [0.5742, 0.6459] |
| rolling 2016 (~8 năm) | **0.6075** | [0.5657, 0.6611] |
| expanding | 0.6247 | [0.5830, 0.6772] |
| Baseline1 (công bằng, train dist) | 0.6341 | — |
| Baseline2 (hindsight) | 0.6019 | — |
| Baseline3 (suy biến, 100% Đi_ngang) | 0.9307 | — |

**Cả 5 arm ĐỀU KHÔNG tách biệt khỏi static (0.6276) hay Baseline1 (0.6341)** — 95% CI của mọi arm đều chứa cả 2 mốc tham chiếu này. Quan sát định hướng (KHÔNG phải kết luận thống kê, CI quá rộng để khẳng định): window ngắn (504/756) tệ hơn static; window dài (1260/2016) nhỉnh hơn nhưng không tách biệt; expanding nằm giữa.

### Tại sao không tách biệt được — 3 lý do kỹ thuật (không phải bug)
Đã verify fold-level (embargo, partition) đúng như thiết kế — đây là kết quả thật của thí nghiệm, không phải lỗi code.

1. **Cỡ mẫu quá nhỏ cho quyền lực thống kê (statistical power).** n=303 điểm dự đoán, nhưng vì nhãn horizon=21 phiên chồng lấn (đúng vấn đề n_eff đã gặp ở Giai đoạn 3), số "khối thông tin độc lập" thật chỉ ≈303/21≈14. Với 14 khối, CI95 cho Brier rộng tới ~0.06–0.09 — trong khi chênh lệch quan sát giữa các arm chỉ ~0.02–0.03. Đây là giới hạn cấu trúc của dữ liệu (chỉ có ~14 tháng vùng bull-market khả dụng trước mốc test khoá), không sửa được bằng thử thêm window size.
2. **Tiền đề gốc của 4b có dấu hiệu bị chính dữ liệu phản bác.** Giả thuyết "window ngắn = gần regime hiện tại hơn = ít ngoại suy hơn" ngụ ý window càng ngắn càng tốt. Nhưng quan sát thật: window ngắn nhất (504/756) lại là 2 arm TỆ NHẤT, tệ hơn cả static; chỉ window dài hơn (1260/2016, gần với 12 năm của static hơn) mới nhỉnh hơn. Cắt ngắn window không chỉ giảm độ "cũ" của dữ liệu train mà còn giảm độ phủ range của feature — đánh đổi bias-do-lệch-regime lấy variance-do-ít-dữ-liệu, và ở window ngắn cái giá variance có vẻ thắng.
3. **Arm `expanding` (đối chứng) cho thấy retrain định kỳ tự nó không phải chìa khoá.** Expanding retrain định kỳ nhưng không cắt window, kết quả gần sát static (0.6247 vs 0.6276). Nếu "retrain thường xuyên hơn" tự nó có giá trị, expanding phải khác biệt rõ so với static. Điều này gợi ý: chênh lệch quan sát được ở các arm khác chủ yếu đến từ việc CẮT window, không phải từ tần suất retrain.

### Kết luận & lý do không nên tiếp tục hướng này
Finding hợp lệ: không đủ bằng chứng để nói rolling-window retrain tốt hơn static-window. Không dùng lượt test-touch riêng của 4b trên 50 mốc khoá (không có candidate đủ mạnh để đáng thử). **Giai đoạn 4b ĐÃ ĐÓNG CHÍNH THỨC (2026-08-07)**, không mở vòng thử mới. Lý do không nên mở rộng thêm (thử thêm window size/cadence khác trong cùng thiết kế):

- **Rào cản là cỡ mẫu/cấu trúc dữ liệu, không phải tham số** (lý do 1) — chỉ có từng đó ngày giao dịch trong vùng bull-market trước mốc test khoá. Thử thêm window size hay cadence retrain khác không giải quyết được việc CI luôn rộng hơn hiệu ứng đang tìm; cần dữ liệu dài hơn hẳn mới thu hẹp được CI.
- **Tiền đề gốc yếu đi sau khi nhìn dữ liệu** (lý do 2) — đào sâu thêm vào cùng 1 tiền đề đã lung lay (window ngắn tốt hơn) có lợi tức kỳ vọng thấp.
- **Đã dùng đúng budget 5 arm đã khoá trước khi chạy** — mở thêm arm sau khi thấy 504/756 thua, 1260/2016 nhỉnh hơn sẽ là chọn lưới theo kết quả, đúng kiểu rủi ro mà nguyên tắc "chốt trước khi chạy" (mục 7) vốn dùng để tránh.
- **Nhìn toàn cảnh dự án — đây là điểm dữ liệu thứ 4 liên tiếp cùng 1 mẫu hình**: Giai đoạn 3 (LLM: chênh ~1.2 SE, chưa đủ ý nghĩa) → 4a (dưới cả baseline) → chẩn đoán Brier (chỉ 2/96 tổ hợp thắng baseline, biên ~1%) → 4b (5/5 arm không tách biệt). Mọi biến thể định lượng thử qua — feature khác, regularization khác, cửa sổ train khác — đều hội tụ về cùng 1 kết luận: chưa cấu hình nào tách biệt khỏi baseline một cách có ý nghĩa thống kê. Cách đọc hợp lý hơn "chưa tìm đúng tham số" là: tín hiệu ngắn hạn (21 ngày) từ đúng 4 chỉ báo vĩ mô này, với đúng khung Logistic Regression này, có thể không đủ mạnh để vượt nhiễu — một finding hợp lệ về giới hạn bài toán. Hướng có triển vọng hơn việc thử thêm 1 biến thể nữa trong cùng không gian thiết kế sẽ là xem lại framing bài toán (horizon khác, target khác) hoặc chấp nhận kết quả âm này làm kết luận chính thức — quyết định thuộc về bước lập kế hoạch tiếp theo, chưa chốt trong tài liệu này.

File tham chiếu: `giai_doan_4b_rolling_window_experiment.py` (script walk-forward), `giai_doan_4b_rolling_window_results.json` (kết quả đầy đủ + log từng fold), `giai_doan_4b_verify_folds.py` (verify độc lập embargo + partition fold).

---

## 7. Tổng hợp phương pháp luận / thuật toán đã áp dụng

**Thuật toán mô hình:**
- Logistic Regression đa lớp (multinomial), solver `saga` (hỗ trợ cả L1/L2, cần `random_state` cố định vì stochastic), penalty L1/L2.
- Random Forest/XGBoost — cân nhắc và loại bỏ dứt khoát (Giai đoạn 4a) vì mất tính diễn giải.
- LLM few-shot prompting (`qwen2.5` 7B qua Ollama) — cân nhắc và dừng đầu tư (Giai đoạn 3).

**Xử lý đa cộng tuyến:**
- Pairwise correlation (ngưỡng |corr|>0.85) — dùng ở Giai đoạn 1.5, sau xác nhận không đủ (bỏ sót đa cộng tuyến đa biến).
- VIF thật (iterative, ngưỡng >5.0, tính qua LinearRegression) — dùng từ Giai đoạn 4a, xác nhận giải quyết dứt điểm.

**Ngưỡng gán nhãn (đã cân nhắc 3 phương pháp ở Giai đoạn 1.5):** σ thuần (loại — bị outlier kéo lệch, đo được −8% khi trim 1%), Percentile (loại — không có ý nghĩa kinh tế, lợi thế cân bằng không chuyển sang test), MAD (chọn — `median ± 0.5×1.4826×MAD`, chống outlier, giữ ý nghĩa kinh tế).

**Volatility feature:** Vol60 rolling std cố định (gốc) vs EWMA (`span=10/21, adjust=False`) — EWMA được kỳ vọng phản ứng nhanh hơn với đổi regime, nhưng chẩn đoán Brier cho thấy bằng chứng ngược lại.

**Phương pháp kiểm định thống kê:**
- Wald CI 95% một mẫu: `p̂ ± 1.96×√(p̂(1−p̂)/n)` — tiêu chí chính khi chấm test.
- Two-proportion z-test không gộp: `z=(p₁−p₂)/√(SE1²+SE2²)` — tiêu chí phụ.
- Tính ở cả n danh nghĩa và n hiệu dụng (n_eff, điều chỉnh do horizon chồng lấn).
- Multiclass Brier score: `BS=(1/N)ΣᵢΣₖ(pᵢₖ−yᵢₖ)²` — dùng từ giai đoạn chẩn đoán Brier, đo chất lượng calibration của xác suất, không chỉ nhãn thắng cuối.
- Block bootstrap (dùng ở Giai đoạn 4b): resample theo khối liên tiếp (block=21, khớp horizon) thay vì bootstrap i.i.d. thường, để không phóng đại độ tin cậy khi các điểm dự đoán liền kề tự tương quan do nhãn horizon chồng lấn.

**Nguyên tắc thiết kế backtest/split:**
- Split theo thời gian, không shuffle (walk-forward).
- Backtest leak-safe (feature tại `as_of_date` chỉ dùng dữ liệu ≤ đúng thời điểm đó).
- Embargo/purge cho retrain định kỳ (dùng ở Giai đoạn 4b): dòng ngày d chỉ dùng để train tại điểm retrain T nếu `d + horizon ≤ T` (không phải `d ≤ T`) — vì nhãn tại d dùng giá tại `d+horizon`, nếu d nằm trong `(T-horizon, T]` thì nhãn đó đã "nhìn thấy" thông tin thuộc giai đoạn dự đoán tiếp theo.
- Baseline luôn lấy từ nhãn đa số của TRAIN (đóng băng, out-of-sample), tách bạch rõ với baseline hindsight/in-sample (chỉ tham khảo, không so ngang hàng).
- Test set chỉ chạm ĐÚNG 1 LẦN cho mỗi hướng tiếp cận thật sự khác biệt; validation được lặp lại thoải mái (giới hạn 5–6 vòng/hướng).
- Công thức kiểm định phải chốt TRƯỚC khi thấy kết quả test, không đổi sau khi thấy số. Áp dụng cả cho danh sách cấu hình cần thử (VD 5 arm window size ở Giai đoạn 4b) — chốt trước khi chạy, không mở thêm sau khi thấy kết quả.
- Redo sau thất bại: phải giữ log kết quả cũ, có lý do rõ ràng dựa trên bài học rút ra — không thử nhiều cấu hình trên test rồi chọn cái tốt nhất (p-hacking).
- Arm đối chứng (control arm) để tách bạch biến gây nhiễu: khi 1 thay đổi gộp nhiều biến (VD "rolling-window" gộp cả "cắt window" lẫn "retrain định kỳ"), thêm 1 arm chỉ đổi 1 biến (VD `expanding` — retrain định kỳ nhưng không cắt window) để biết chênh lệch quan sát được đến từ biến nào.

---

## 8. Bài học chung xuyên suốt các giai đoạn

1. **Regime shift là giới hạn cốt lõi, lặp lại ở mọi phương án đã thử.** Từ Giai đoạn 1.5 (mean-reversion học từ range-bound thất bại trên bull market) đến chẩn đoán Brier (mọi thước đo đều "thưởng" cho sự thận trọng vì thiếu tín hiệu ổn định) đến Giai đoạn 4b (rolling-window retrain — nỗ lực trực tiếp đối phó regime shift — cũng không tách biệt được khỏi static) — đây không phải lỗi của riêng 1 model hay 1 kỹ thuật, mà là đặc tính thật của dữ liệu trong khung thời gian đang xét.
2. **Sửa đúng vấn đề kỹ thuật không đảm bảo cải thiện kết quả cuối.** VIF, interaction terms, EWMA đều là các fix hợp lý về mặt kỹ thuật, nhưng chỉ VIF và interaction terms có bằng chứng thật chứng minh giá trị — EWMA, dù có lý do thiết kế chính đáng lúc đề xuất, lại cho bằng chứng ngược lại khi đo trực tiếp. Tương tự, "window ngắn hơn = ít ngoại suy hơn" (tiền đề của 4b) nghe hợp lý về lý thuyết nhưng bị chính dữ liệu phản bác (mục 6).
3. **Thước đo tuning (accuracy vs Brier) ít ảnh hưởng hơn kỳ vọng ban đầu** — cả 2 đều đồng thuận ưu tiên regularization mạnh trên dữ liệu này, cho thấy vấn đề gốc không chỉ nằm ở việc chọn sai thước đo.
4. **Tín hiệu khai thác được, nếu có, rất mỏng** — qua 144 tổ hợp đã thử (48 ở Giai đoạn 4a + 96 ở chẩn đoán Brier) cộng 5 arm walk-forward ở Giai đoạn 4b, chỉ một phần rất nhỏ (2/96) vượt qua được baseline ngây thơ với biên ~1% chưa kiểm định, và không arm 4b nào tách biệt khỏi baseline/static — chưa ở đâu tìm được biên đủ lớn để chắc chắn không phải nhiễu.
5. **Cỡ mẫu hiệu dụng (n_eff) là rào cản lặp lại, không phải chi tiết riêng của 1 giai đoạn.** Giai đoạn 3 (n=50 → n_eff≈25) và Giai đoạn 4b (n=303 → n_eff≈14, do cùng nguyên nhân: horizon 21 phiên chồng lấn) đều cho CI quá rộng để phân biệt các phương án — đây là giới hạn của chính cách đóng khung bài toán (horizon 21 ngày, dữ liệu bull-market post-2023 có hạn), không phải thứ 1 kỹ thuật tuning nào (window size, regularization, metric) có thể sửa được.
6. **4 lần thử độc lập (Giai đoạn 3, 4a, chẩn đoán Brier, 4b) đều hội tụ về cùng 1 kết luận** — không có cấu hình định lượng hay LLM nào tách biệt khỏi baseline một cách có ý nghĩa thống kê. Đây là bằng chứng tích luỹ đủ mạnh để coi là finding về giới hạn bài toán (dự đoán xu hướng vàng 21 ngày từ 4 chỉ báo vĩ mô này), không phải "chưa tìm đúng tham số" — tiếp tục tinh chỉnh trong cùng không gian thiết kế (feature/model/window) có lợi tức kỳ vọng thấp; hướng có triển vọng hơn là xem lại framing bài toán hoặc chấp nhận đây là kết luận chính thức.
