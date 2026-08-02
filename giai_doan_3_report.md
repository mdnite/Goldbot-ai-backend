# Giai đoạn 3 — Báo cáo (đo suy luận dự đoán ngắn hạn)

## 1. Mục tiêu & phạm vi (đã revise so với Roadmap gốc)

Roadmap.md định nghĩa gốc Giai đoạn 3 là "format output (cấu trúc 4 mục + disclaimer)". Trong quá trình làm, mục tiêu đã được mở rộng thành: **đo và cải thiện khả năng SUY LUẬN dự đoán ngắn hạn** của LLM dựa trên dữ liệu định lượng thật injected từ Giai đoạn 2 — khác với Giai đoạn 2.5 (đo Faithfulness Rate: bot có đọc ĐÚNG dữ liệu đã cho hay không, không đo suy luận về tương lai).

Cấu trúc 4 mục (diễn biến / yếu tố vĩ mô / kịch bản / disclaimer) vẫn được giữ làm khung định dạng bắt buộc (QUY TẮC SỐ 8 trong `giai_doan_3_system_prompt.txt`), nhưng trọng tâm đo lường của giai đoạn này là **nội dung suy luận bên trong mục "Kịch bản"**, không chỉ hình thức.

## 2. Thiết kế backtest (hằng số Bước 0 — đóng băng, đọc trực tiếp từ `trend_model.joblib`)

- `horizon` = 21 phiên giao dịch (khớp `train_trend_model.py`, Giai đoạn 1.5).
- Ngưỡng nhãn: `label_low = -0.01912`, `label_high = 0.02326` (median ± 0.5×1.4826×MAD, tính trên TRAIN của Giai đoạn 1.5, đóng băng — không recompute dù `indicators.db` đã bị refetch sau khi model train).
- `baseline_label` (nhãn phổ biến nhất trên TRAIN của Giai đoạn 1.5) = `Di_ngang`.
- **Regime**: khung backtest hợp lệ (2024-06-01 → 2026-06-22, biên trên = 21 phiên trước ngày gold mới nhất trong `indicators.db` tại thời điểm audit) nằm hoàn toàn trong regime bull market (2023-2026) — không đủ đa dạng regime để phân tầng, chỉ có 1 stratum.
- **Sampling N=50**: 515 phiên giao dịch hợp lệ trong khung trên, chia 50 bin gần đều, random 1 ngày/bin, `seed=42` cố định. Min-gap trung bình ~10.2 phiên (4 cặp có gap < 5 phiên, chấp nhận không resample).
- **n hiệu dụng ≈ 25** (≈ 515/21), KHÔNG dùng N=50 danh nghĩa để tính sai số chuẩn — do các mốc chồng lấn cửa sổ 21 phiên nên không độc lập hoàn toàn với nhau.

## 3. Phương pháp đánh giá

- **1 câu hỏi cố định** lặp lại ở 50 mốc thời gian khác nhau (không phải 50 câu hỏi khác nhau): *"Xu hướng vàng ngắn hạn sắp tới thế nào, dựa trên các yếu tố vĩ mô hiện tại?"* (`giai_doan_3_run_backtest.py`, biến `QUESTION`).
- Mỗi mốc: snapshot dữ liệu leak-safe tính đến đúng `as_of_date` (`get_market_data_asof`), ground truth lấy từ return thật 21 phiên sau đó (`get_actual_outcome`), gọi Ollama trực tiếp (`temperature=0.0, seed=42`, bỏ qua FastAPI/RAG để cô lập biến số).
- **2 baseline đối chiếu**:
  1. **Baseline đóng băng (out-of-sample, chính)** = `Di_ngang` — nhãn phổ biến nhất học từ TRAIN của Giai đoạn 1.5, áp mù lên 50 mốc test này. Đây là baseline hợp lệ để so sánh công bằng với bot.
  2. **Baseline majority thật (in-sample/hindsight, tham khảo)** = nhãn phổ biến nhất tính trực tiếp từ chính 50 mốc đang chấm. **Không so ngang hàng nghiêm ngặt** với bot — bot suy luận không biết trước phân phối nhãn thật của tập test, còn baseline này thì có, nên luôn có lợi thế cấu trúc.

## 4. Việc đã làm trong phiên làm việc này

1. **Rà soát tổng quan project** — đọc lại CLAUDE.md, Roadmap.md, trạng thái các giai đoạn qua memory.
2. **Dọn file thừa**: chuyển `test_yfinance.py` (script thăm dò Giai đoạn 1, đã bị `update_indicators.py` thay thế hoàn toàn, không còn được reference) vào `Unused/test_yfinance.py` bằng `git mv`. `view_db.py` (dùng import `langchain_community` deprecated, khác `main.py`/`ingest_data.py`) — xem xét nhưng giữ nguyên theo yêu cầu, không sửa.
3. **Vòng smoke test 1** (`--limit 3`, prompt CŨ chưa sửa): phát hiện vấn đề — mục "Kịch bản" hedge liên tục kiểu "có thể đi ngang hoặc giảm nhẹ", không chốt một nhãn rõ ràng nào để đối chiếu với outcome/baseline.
4. **Sửa `giai_doan_3_system_prompt.txt`**: thêm yêu cầu câu đầu tiên của mục 3 phải là `"Kết luận chính: [Tăng/Giảm/Đi ngang] (tin cậy: thấp/trung bình/cao)"`, đồng bộ sửa cả 2 ví dụ few-shot minh hoạ cho khớp định dạng mới. Verify bằng `--dry-run` (không gọi Ollama) — `.format()` không lỗi placeholder.
5. **Vòng smoke test 2** (`--limit 3`, prompt MỚI): xác nhận "Kết luận chính" trích xuất sạch trên cả 3 mốc. Nhưng phát hiện thêm 2 lỗi suy luận không liên quan tới format — lỗi về chiều tác động lý thuyết của chỉ báo lên vàng.
6. **Quyết định**: chạy full N=50 ngay với prompt hiện tại, ghi nhận các lỗi suy luận trên làm finding thay vì tiếp tục vá thêm 1 vòng — theo đúng tiền lệ đã áp dụng ở Giai đoạn 2.5 (dừng vòng lặp sửa Grader ở 96% để tránh overfit lên đúng tập test nhỏ, thay vì cố vá tới 100%).
7. **Chạy full N=50 trên Colab**, resume-safe từ 3 mốc đã có sẵn (dùng đúng prompt cuối), hoàn tất 50/50.
8. **Kiểm tra file kết quả**: JSON hợp lệ, 50/50 mốc khớp chính xác thứ tự với `giai_doan_3_sample_dates.json`, không thiếu/thừa/trùng, 0 lỗi parse "Kết luận chính" trên toàn bộ 50 mốc.
9. **Rà tay toàn bộ 50 mốc** (mục "Yếu tố vĩ mô" của từng `bot_reply`, đối chiếu dấu số liệu thật trong `market_data_raw`) để tìm lỗi suy luận về chiều tác động — kết quả chi tiết ở mục 6.

## 5. Kết quả test N=50 (accuracy nhãn cuối)

| | Đúng/Tổng | Tỷ lệ |
|---|---|---|
| **Bot (GoldBot suy luận)** | 23/50 | **46.0%** |
| Baseline 1 — đóng băng (`Di_ngang`, out-of-sample) | 15/50 | 30.0% |
| Baseline 2 — majority thật (in-sample/hindsight) | 27/50 | 54.0% |

Baseline 1 = 30.0% khớp đúng với giá trị đã tính trước trong `giai_doan_3_baseline_check.json` — xác nhận đang chấm đúng trên cùng 50 mốc đã khoá.

**Sai số chuẩn (dùng n hiệu dụng ≈ 25, không dùng N=50 danh nghĩa)**:
- SE(bot, 46%) ≈ 10.0 điểm %
- SE(baseline 1, 30%) ≈ 9.2 điểm %
- Chênh lệch 16 điểm % giữa bot và baseline 1 ≈ 1.2 lần sai số chuẩn kết hợp — **chưa đạt ngưỡng ý nghĩa thống kê thông thường** (~2 SE cho mức tin cậy 95%) do mẫu hiệu dụng nhỏ. Bot có xu hướng tốt hơn baseline đóng băng trên tập này, nhưng không nên khẳng định chắc chắn với cỡ mẫu hiện tại.
- Bot vẫn thấp hơn Baseline 2 (54%), nhưng đây là so sánh không công bằng hoàn toàn (in-sample vs out-of-sample) như đã nêu ở mục 3.

**Phân phối nhãn dự đoán của bot lệch rõ rệt so với thực tế**:

| Nhãn | Bot dự đoán | Thực tế |
|---|---|---|
| Tăng | 30/50 (60%) | 27/50 (54%) |
| Giảm | 16/50 (32%) | 8/50 (16%) |
| Đi ngang | 4/50 (8%) | 15/50 (30%) |

Bot gần như không bao giờ chọn "Đi ngang" (chỉ 4/50) dù đây là nhãn baseline của chính hệ thống và chiếm tới 30% outcome thật — có xu hướng luôn nghiêng hẳn về một chiều (Tăng hoặc Giảm) thay vì thận trọng, dù QUY TẮC SỐ 3 yêu cầu "không khẳng định chắc chắn".

## 6. Finding về chất lượng suy luận — audit đầy đủ 50/50 mốc

Accuracy nhãn cuối (mục 5) **không phản ánh đủ** chất lượng suy luận, vì bot có thể ra đúng nhãn dù chuỗi lập luận bên trong sai (đúng nhờ may/thiên hướng chung), hoặc ngược lại. Đã rà tay toàn bộ mục "2. Yếu tố vĩ mô" của cả 50 `bot_reply`, đối chiếu dấu số liệu thật trong `market_data_raw` với chiều tác động bot khẳng định. Phát hiện 3 nhóm lỗi, một nhóm hoàn toàn mới so với 2 lỗi ban đầu tìm thấy ở vòng smoke test 3 mốc:

### 6.1 Lỗi chiều tác động DXY → vàng (bot nói DXY tăng "hỗ trợ" vàng — ngược lý thuyết) — 3/50 mốc

- **2024-06-20** và **2026-03-25**: cùng một câu gần như y hệt — *"Đồng USD mạnh lên có thể hỗ trợ giá vàng do nhu cầu mua vào để phòng ngừa rủi ro"*. DXY tăng (USD mạnh lên) về lý thuyết bất lợi cho vàng, không phải hỗ trợ — mâu thuẫn trực tiếp với QUY TẮC SỐ 5 và cả 2 ví dụ minh hoạ trong chính prompt. Việc lặp lại gần như nguyên văn ở 2 mốc cách nhau nhiều tháng cho thấy đây là một khuôn suy luận sai cố định, không phải lỗi ngẫu nhiên đơn lẻ.
- **2025-10-09**: lỗi nặng hơn — tự mâu thuẫn ngay trong một câu: *"Chỉ số USD (DXY) tăng 1.80% (giảm so với mức thay đổi +1.80%), đồng USD yếu đi, thường hỗ trợ giá vàng."* Vừa nói tăng vừa nói yếu đi trong cùng câu — câu vô nghĩa về mặt logic.

### 6.2 Lỗi chiều tác động lợi suất thực (real yield) → vàng — 1/50 mốc

- **2024-06-17** (đã phát hiện từ vòng smoke test): *"đồng USD mạnh lên và lợi suất thực giảm nhẹ - hai yếu tố lý thuyết đều bất lợi cho vàng"*. Sai: lợi suất thực giảm → chi phí cơ hội giữ vàng giảm → về lý thuyết là yếu tố THUẬN LỢI cho vàng, không phải bất lợi.

### 6.3 Lỗi chiều tác động lãi suất Fed → vàng — nhóm lỗi MỚI, tỷ lệ cao bất thường

Trong 15 mốc có `fed_rate_diff ≠ 0`, có 10 mốc bot đưa ra khẳng định rõ ràng về chiều tác động (5 mốc còn lại né tránh, chỉ nói "không phải yếu tố chi phối"). Trong 10 mốc có khẳng định:

- **5/10 ĐÚNG** (Fed giảm → hỗ trợ/giảm áp lực cho vàng, đúng cơ chế chi phí cơ hội tương tự real yield): 2024-09-24, 2024-11-08, 2024-12-06, 2025-11-03, 2025-11-18.
- **5/10 SAI** (ngược chiều — nói Fed giảm làm GIẢM sức hấp dẫn của vàng, đúng ra phải TĂNG): 2024-12-26, 2025-09-26, 2025-10-09, 2025-12-15, 2025-12-30. Câu lặp lại gần giống nhau: *"Lãi suất Fed giảm... có thể làm giảm sức hấp dẫn của các tài sản không sinh lãi như vàng."*

Tỷ lệ đúng/sai 50/50 trên cùng một loại quan hệ nhân quả (Fed rate giảm) cho thấy model **không có một "lý thuyết nội tại" nhất quán** về quan hệ Fed rate - vàng — khác hẳn DXY/real yield, nơi model đúng áp đảo và chỉ sai rải rác.

### 6.4 Lỗi đọc sai dấu số liệu thật (nghiêm trọng hơn 3 nhóm trên — vi phạm QUY TẮC SỐ 5, không chỉ là suy luận lý thuyết sai) — 1/50 mốc

- **2026-04-01**: số liệu thật `dxy_pct = +0.61%` (TĂNG), nhưng bot viết *"Đồng USD đang yếu đi một chút... tạo áp lực giảm giá đối với vàng"* — đọc ngược dấu số liệu hoàn toàn, đồng thời suy luận tiếp theo (USD yếu đi → áp lực giảm giá) cũng ngược lý thuyết ngay cả khi giả định tiền đề đúng. Đây là lỗi kép, và là loại lỗi QUY TẮC 5 được thiết kế ra để ngăn — cho thấy quy tắc đó không bảo vệ được 100% dù đã validate 96% faithfulness ở Giai đoạn 2.5 (lưu ý: Giai đoạn 2.5 đo faithfulness trên bộ câu hỏi khác, không phải 50 mốc backtest suy luận này).

### 6.5 Tổng hợp tỷ lệ lỗi theo loại

| Loại lỗi | Số mốc mắc lỗi | Tỷ lệ trên mẫu liên quan |
|---|---|---|
| DXY sai chiều tác động | 3/50 | 6% |
| Real yield sai chiều tác động | 1/50 | 2% |
| Fed rate sai chiều tác động | 5/10 mốc có khẳng định rõ | ~50% khi model chịu kết luận |
| Đọc sai dấu số liệu thật (QUY TẮC 5) | 1/50 | 2% |

**Kết luận methodology**: lỗi suy luận không phải hiện tượng hiếm như ước tính ban đầu (2/50 từ smoke test 3 mốc) — đặc biệt lỗi về Fed rate xuất hiện với tần suất rất cao (~50%) mỗi khi model chịu đưa ra khẳng định rõ ràng. Đây là finding hợp lệ về giới hạn suy luận của model 7B chạy CPU-only, không phải bug quy trình backtest — số liệu trích dẫn (QUY TẮC 7) hầu hết chính xác, vấn đề nằm ở việc GẮN đúng chiều tác động lý thuyết cho từng chỉ báo, đặc biệt là Fed rate — chỉ báo có tần suất thay đổi thấp nhất (step-function, ~8 lần/năm) nên model có ít "thấy" các trường hợp huấn luyện/ví dụ về chỉ báo này hơn DXY/real yield.

## 7. Việc còn mở / bước tiếp theo

- **Task tuỳ chọn**: so sánh thêm dự đoán của `trend_model.joblib` (28 feature kỹ thuật hoá) trên cùng 50 mốc — cần build riêng pipeline feature cho từng mốc, không tái dùng được trực tiếp text snapshot.
- **Cân nhắc**: có nên thêm 1 quy tắc riêng về chiều tác động Fed rate vào `giai_doan_3_system_prompt.txt` hay không, dựa trên tỷ lệ lỗi ~50% vừa phát hiện — nhưng lưu ý nguyên tắc đã chốt ở mục 4.6: N=50 chỉ chạy MỘT LẦN cho báo cáo chính thức, sửa thêm bây giờ đồng nghĩa phải chạy lại toàn bộ để đo lại, và có rủi ro overfit lên đúng tập 50 mốc này. Chưa quyết định — để ngỏ cho phiên sau.
- **Việc treo từ Giai đoạn 2.5** (không thuộc phạm vi Giai đoạn 3 nhưng vẫn chưa xử lý): `system_prompt_v2.txt` (QUY TẮC 5/6/7, đã validate 96% faithfulness) chưa được sync vào `ADVICE_PROMPT` trong `main.py` — bản production hiện vẫn chỉ có QUY TẮC 1-4.
- File kết quả liên quan: `giai_doan_3_backtest_results.json` (kết quả chính thức N=50), `giai_doan_3_backtest_results_smoketest_v1_prehedgefix.json` (bản smoke test 3 mốc TRƯỚC khi sửa prompt, giữ lại làm bằng chứng đối chiếu trước/sau).
