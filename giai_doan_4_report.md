# Giai đoạn 4a — Báo cáo (cải tiến model định lượng, thay LoRA)

## Subject

Sửa 3 vấn đề kỹ thuật đã xác nhận ở `trend_model.joblib` (Giai đoạn 1.5, thua baseline 0.278 vs 0.364 do regime shift — xem `giai_doan_1_5_report.md`):

1. **Cô lập đặc trưng**: feature gốc (lag/MA/vol) không tách biệt tương tác giữa các chỉ báo macro.
2. **Volatility trễ**: `Vol60` (rolling std, cửa sổ 60 ngày) phản ứng quá chậm với thay đổi regime.
3. **Đa cộng tuyến**: bản gốc chỉ lọc bằng pairwise correlation (ngưỡng |corr|>0.85), chưa kiểm VIF đúng chuẩn.

**GIỮ NGUYÊN Logistic Regression** — quyết định đã khoá, không đổi thuật toán ở vòng này (Random Forest/XGBoost là Giai đoạn 4b, dự phòng).

## Phương án đã cân nhắc

| Vấn đề | Phương án chọn | Phương án loại | Lý do |
|---|---|---|---|
| Volatility trễ | EWMA-Vol (span=10/21, `adjust=False`, tính trên chuỗi return/diff) | Giữ rolling std cố định (Vol60) hoặc chỉ đổi cửa sổ rolling std ngắn hơn | EWMA phản ứng nhanh hơn rolling std khi regime đổi — không cần đợi đủ N ngày để "xả" hết ảnh hưởng dữ liệu regime cũ ra khỏi cửa sổ, và tránh hiện tượng whipsaw (giật cục) khi 1 phiên biến động mạnh đột ngột rớt khỏi cửa sổ SMA cố định. Bản rolling-std đầu tiên đã chạy thử nhưng bị bỏ, chỉ còn bằng chứng ở commit git `61c58ff "Phase 4a (Pre-Votility Change)"`. |
| Đa cộng tuyến | VIF thật (iterative, ngưỡng >5.0, tính bằng LinearRegression vì statsmodels chưa cài) | Chỉ lọc pairwise corr như bản gốc Giai đoạn 1.5 | Pairwise corr bỏ sót đa cộng tuyến đa biến (1 feature là tổ hợp tuyến tính của NHIỀU feature khác cùng lúc, không chỉ 1 cặp). |
| `REGIME_SPLIT_DATE` | `2023-03-17` | `2023-01-01` (giá trị ban đầu trong bản nháp) | `2023-01-01` là làm tròn, lệch với ranh giới range-bound/bull-market THẬT đã xác nhận bằng số liệu ở `giai_doan_1_5_report.md` (kết thúc range-bound 2023-03-16, bắt đầu bull-market 2023-03-17). Đã sửa trước khi chạy grid. |
| Hyperparameter C (sau khi Round 1 cho best ở mép lưới) | Giữ `C=0.01` (Round 1) | Mở rộng xuống thấp hơn (Round 2: C≤0.005) — loại vì **sụp đổ hoàn toàn**: L1 zero hoá cả 34 hệ số, model chỉ còn dự đoán nhãn đa số cố định. Mở rộng lên cao hơn (Round 3: C=0.02-0.07) — loại vì tuy val_acc gộp (blended) cao hơn (0.484 vs 0.473), nhưng **post-2023 accuracy THẤP HƠN** (0.446-0.429 vs 0.482) và gap pre/post **đảo chiều hướng** (model nghiêng hẳn về pre-2023) — quan trọng: **toàn bộ 50 mốc test đều nằm trong regime post-2023** (test 2024-06→2026-06, sau `REGIME_SPLIT_DATE`=2023-03-17), nên post-2023 accuracy mới là con số liên quan trực tiếp tới test, không phải val_acc gộp. | Giữ C=0.01 vì đây là cấu hình có post-2023 accuracy cao nhất trong nhóm model hoạt động thật (không suy biến). |
| Ground truth cho test | Tái dùng NGUYÊN ngưỡng nhãn GỐC của Giai đoạn 1.5 (đóng băng, đã dùng xuyên suốt Giai đoạn 3: low=-0.01912, high=0.02326) | Tính lại ground truth bằng ngưỡng riêng của model 4a (low=-0.01802, high=0.02304, tính trên TRAIN+VAL gộp lúc refit) | Mục đích tái dùng đúng 50 mốc là so sánh CÔNG BẰNG với bot 46%/baseline 30%/baseline2 54% đã chấm bằng ngưỡng gốc — đổi ngưỡng ground truth sẽ phá vỡ tính so sánh được. Ngưỡng riêng của model 4a chỉ dùng để gán nhãn lúc TRAIN, không dùng để định nghĩa đúng/sai lúc test. |
| Công thức kiểm định thống kê | Wald CI 95% một mẫu (tiêu chí QUYẾT ĐỊNH chính) + two-proportion z-test (tiêu chí PHỤ, chỉ đối chiếu) — cả 2 tính với **n=50** (danh nghĩa) VÀ **n_eff=25** (hiệu dụng, tái dùng nguyên từ `giai_doan_3_report.md` do horizon 21 phiên chồng lấn giữa các mốc) | Wilson/Clopper-Pearson interval (chính xác hơn ở n nhỏ) | Ưu tiên nhất quán với công thức SE đã dùng xuyên suốt `giai_doan_3_report.md` mục 5, hơn là đổi sang công thức chính xác hơn giữa chừng dự án. Công thức đã CHỐT và ghi vào docstring `giai_doan_4_backtest_eval.py` TRƯỚC khi chạy test, không đổi sau khi thấy kết quả. |

## Process

**Feature engineering**: 28 feature gốc (4 chuỗi `gold_ret`/`dxy_ret`/`real_yield_diff`/`fed_rate_diff` × [lag 1,5,10,21,60 + MA60 + Vol{10,21}]) + 6 interaction feature (3 cặp macro `dxy_ret×real_yield_diff`, `real_yield_diff×fed_rate_diff`, `dxy_ret×fed_rate_diff` × [lag1, MA60]) = **34 feature**. Interaction KHÔNG gồm `gold_ret`, và KHÔNG dùng riêng insight lỗi Fed rate từ audit LLM Giai đoạn 3 để thiết kế feature vá đúng điểm đó.

**VIF**: kiểm tra ở MỌI lần chạy (Round 1, 2, 3, và lúc refit cuối trên TRAIN+VAL) — **luôn luôn 0/34 feature bị loại** (ngưỡng >5.0). Thêm interaction không gây đa cộng tuyến vượt ngưỡng.

**3 vòng tuning trên validation** (531 dòng, span=EWMA 10/21 × penalty L1/L2 × lưới C, 16 tổ hợp/vòng):

| Vòng | Lưới C | Best | val_acc | pre2023 | post2023 | Ghi chú |
|---|---|---|---|---|---|---|
| 1 | [0.01, 0.1, 1, 10] | vol=21, l1, C=0.01 | 0.473 | 0.461 | 0.482 | Best nằm ở mép dưới lưới → mở rộng round 2 |
| 2 | [0.001, 0.003, 0.005, 0.01] | vol=21, l1, C=0.01 (trùng round 1) | 0.473 | 0.461 | 0.482 | **Phát hiện "vách đứng"**: C≤0.005 (L1, cả 2 span) sụp đổ về ĐÚNG 1 kết quả (val_acc=0.414, toàn bộ 34 hệ số=0) — không phải vùng còn tốt hơn |
| 3 | [0.02, 0.03, 0.05, 0.07] | vol=21, l1, C=0.02/0.03 | 0.484 | 0.535-0.557 | 0.446-0.429 | val_acc gộp cao hơn NHƯNG post2023 thấp hơn round 1, gap đảo chiều (nghiêng pre2023) — loại vì test 100% thuộc post2023 |

**Hyperparameter chốt**: `vol_window=21 (EWMA span), penalty=l1, C=0.01`.

**Feature sống sót lúc tuning** (fit riêng trên TRAIN 3008 dòng, C=0.01/L1, vol=21) — 6/34:

| Feature | Nhãn có hệ số ≠0 |
|---|---|
| `gold_ret_ma60` | Tăng (-0.108) |
| `dxy_ret_vol21` | Tăng (+0.077) |
| `real_yield_diff_vol21` | Đi ngang (-0.068), Tăng (+0.047) |
| `inter_dxy_ret_x_real_yield_diff_ma60` | Đi ngang (+0.034) |
| `inter_real_yield_diff_x_fed_rate_diff_ma60` | Giảm (-0.118) |
| `inter_dxy_ret_x_fed_rate_diff_ma60` | Tăng (+0.012) |

**Feature sống sót SAU REFIT CUỐI** (fit trên TRAIN+VAL gộp 3539 dòng, cùng hyperparameter) — 6/34, KHÁC BIỆT so với lúc tuning:

| Feature | Nhãn có hệ số ≠0 |
|---|---|
| `gold_ret_ma60` | Tăng (-0.122) |
| `dxy_ret_vol21` | Đi ngang (-0.018), Tăng (+0.090) |
| `real_yield_diff_vol21` | Đi ngang (-0.019) |
| `fed_rate_diff_ma60` | **Giảm (+0.032) — MỚI, sống ĐỘC LẬP** |
| `inter_real_yield_diff_x_fed_rate_diff_ma60` | Giảm (-0.087) |
| `inter_dxy_ret_x_fed_rate_diff_ma60` | Tăng (+0.104) |

Khác biệt quan trọng nhất: `inter_dxy_ret_x_real_yield_diff_ma60` (sống lúc tuning) đã CHẾT sau refit, thay bằng `fed_rate_diff_ma60` sống ĐỘC LẬP (không chỉ qua interaction như lúc tuning) — đúng dấu lý thuyết chi phí cơ hội (Giảm: +0.032).

**Đính chính 2 claim đã tự audit lại trong quá trình làm việc (không dùng bản cũ sai)**:

(a) `gold_ret_ma60` là hệ số **lớn nhất CHO NHÃN TĂNG** (-0.296) ở Giai đoạn 1.5 — report xác định đây là feature mã hoá quy luật mean-reversion, cơ chế trực tiếp khiến model gốc thua baseline trên test do regime shift (`giai_doan_1_5_report.md` dòng 121-124). Đây **KHÔNG PHẢI** hệ số lớn nhất toàn model — hệ số có trị tuyệt đối lớn nhất toàn model là `fed_rate_diff_ma60: +0.3485` cho nhãn **Giảm** (dòng 92, cùng report).

(b) `fed_rate_diff_ma60` **CHỈ sống qua interaction** lúc tuning (train-only), nhưng sống **ĐỘC LẬP** trong model refit cuối (TRAIN+VAL gộp) — xem 2 bảng feature sống sót ở trên, không lấy nhầm finding của giai đoạn tuning làm finding của model production cuối cùng.

**Kiểm định thống kê** (công thức chốt trước khi chạy test, viết vào docstring `giai_doan_4_backtest_eval.py`):
- Tiêu chí chính: Wald CI 95% một mẫu cho model 4a — `CI = p̂ ± 1.96×√(p̂(1-p̂)/n)` — "có ý nghĩa" chỉ khi cận dưới CI > 0.46.
- Tiêu chí phụ: two-proportion z-test, SE kết hợp không gộp — `z = (p₁-p₂)/√(SE1²+SE2²)`.
- Cả 2 tính với n=50 (danh nghĩa) và n_eff=25 (hiệu dụng, tái dùng từ `giai_doan_3_report.md`).
- Công thức xác minh đúng bằng cách tái tạo số liệu ĐÃ BIẾT của Giai đoạn 3 (bot 46% vs baseline1 30%, n_eff=25 → z=1.18, khớp "≈1.2 lần SE kết hợp" đã ghi trong `giai_doan_3_report.md`).

## Kết quả

| | Đúng/Tổng | Tỷ lệ |
|---|---|---|
| **Model 4a (refit)** | 15/50 | **30.0%** |
| Bot GoldBot (LLM, Giai đoạn 3) | 23/50 | 46.0% |
| Baseline1 (`Di_ngang` cố định, out-of-sample) | 15/50 | 30.0% |
| Baseline2 (in-sample/hindsight) | 27/50 | 54.0% |
| `trend_model.joblib` cũ (Giai đoạn 1.5, tham khảo) | — | 24.0% |

**Kiểm định thống kê**: CI95(n=50) = [0.173, 0.427]; CI95(n_eff=25) = [0.120, 0.480] — cả 2 **KHÔNG có ý nghĩa thống kê** (nằm hoàn toàn DƯỚI 0.46, không chỉ chưa vượt). Two-proportion z (phụ): z=-1.67 (n=50), z=-1.18 (n_eff=25) — dấu âm cả 2 phiên bản, điểm ước lượng model 4a thấp hơn 46%.

**Rà soát nguyên nhân** (model 4a dự đoán `Di_ngang` cho TOÀN BỘ 50/50 mốc, trùng khớp Baseline1 từng mốc một):
- Đã loại trừ **bug scaler**: `scaler.mean_`/`scaler.scale_` khớp CHÍNH XÁC với tính tay trên TRAIN+VAL cho cả 6 feature sống sót.
- Đã loại trừ **outlier feature**: raw value của 6 feature ở 50 mốc test hầu như nằm trong khoảng train+val (chỉ 2/50 lệch nhẹ ở `gold_ret_ma60`, 5 feature còn lại 0/50 lệch).
- Cơ chế thật (từ `predict_proba`): `Di_ngang` luôn thắng nhưng RẤT SÁT NÚT (mean=0.408, min=0.370; có mốc chỉ chênh **0.016** với `Tăng`) — do regularization mạnh (C=0.01, chỉ 6/34 hệ số sống sót, đều nhỏ 0.03-0.12) khiến dự đoán bám sát tỉ lệ nhãn nền (`Di_ngang` có tỉ lệ train cao nhất, 38.6%). Model KHÔNG suy biến hoàn toàn — in-sample vẫn dự đoán Tăng 13.8%/Giảm 1.1% số lần — nhưng ở đúng 50 mốc test này, tín hiệu yếu không đủ mạnh để vượt lợi thế cấu trúc của `Di_ngang` ở bất kỳ mốc nào.

**Đánh giá 3 vấn đề kỹ thuật gốc đã giải quyết chưa**:

| Vấn đề | Đã giải quyết? |
|---|---|
| Đa cộng tuyến | **Giải quyết dứt điểm** — có xác minh độc lập (VIF thật 0/34 loại, nhất quán qua mọi lần kiểm: Round 1/2/3 và refit cuối), KHÔNG phụ thuộc vào kết quả test. |
| Cô lập đặc trưng (interaction terms) | **Chỉ mới sửa THIẾT KẾ, CHƯA có bằng chứng giải quyết được vấn đề thật** — model refit cuối vẫn không thích nghi được với regime mới trên test thật (dự đoán hằng số `Di_ngang`, không phản ứng được với đặc điểm riêng của giai đoạn 2024-2026). |
| Volatility trễ (EWMA) | **Chỉ mới sửa THIẾT KẾ, CHƯA có bằng chứng giải quyết được vấn đề thật** — cùng lý do trên, EWMA phản ứng nhanh hơn về mặt tính toán nhưng không giúp model tổng quát hoá tốt hơn qua regime shift trên test thật. |

## Kết luận

**KHÔNG ĐẠT** theo tiêu chí đã khoá trước khi chạy test (CLAUDE.md, mục "Quyết định đã chốt"): không vượt LLM 46% có ý nghĩa thống kê, không đạt Baseline2 54%. Test đã chạm ĐÚNG 1 LẦN — không retrain/test lại model này.

**Đính chính bổ sung (phiên 2026-08-05, sau khi báo cáo này đã viết xong)**: script chẩn đoán riêng `archive_old_phases/giai_doan_4_interaction_check.py` (không phải `giai_doan_4_backtest_eval.py`, không phải lượt chấm chính thức) đã dựng lại feature TẠI ĐÚNG 50 mốc `as_of_date` đã khoá (`archive/giai_doan_3_llm_approach/giai_doan_3_sample_dates.json`) để làm ablation (tắt 2 interaction feature còn sống, so `argmax` dự đoán trước/sau — kết quả 0/50 mốc đổi nhãn, chi tiết ở memory `giai_doan_4_interaction_verification.md`). Script này có `import` file chứa nhãn thật (`giai_doan_3_backtest_results.json`) nhưng biến đó KHÔNG được dùng ở bất kỳ đâu trong file — không đọc ground truth, không tính lại accuracy, nên KHÔNG ảnh hưởng tới kết luận KHÔNG ĐẠT ở trên. Nhưng về kỹ thuật đây LÀ một lượt dựng-lại-feature-và-suy-luận bổ sung tại đúng 50 mốc khoá, ngoài lượt chấm chính thức duy nhất — không được báo trước lúc thực hiện, chỉ phát hiện khi bị hỏi lại ở phiên sau. Xem quy tắc mới trong CLAUDE.md mục "Gotcha đã gặp thật" để không lặp lại.

## Việc còn lại

Hướng tiếp theo đã chốt: **"4a-v2"** — đổi metric tối ưu hoá lúc tuning sang F1-weighted/Brier score (thay vì accuracy thô), ép ràng buộc `C≥0.05` (tránh vùng regularization quá mạnh gây suy biến/bám nhãn nền đã phát hiện ở báo cáo này), giữ nguyên toàn bộ 34 feature đã thiết kế. **CHƯA triển khai** — còn đang bàn kế hoạch, chi tiết kỹ thuật sẽ viết ở báo cáo riêng khi bắt đầu, không mở rộng ở đây.
