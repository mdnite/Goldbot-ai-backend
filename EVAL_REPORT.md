# Giai đoạn 2.5 — Faithfulness Eval Report

Mục tiêu: Faithfulness Rate ≥ 90% (≥23/25 câu) trên `eval_dataset.json`, chấm rule-based bằng `eval_script.py` chạy trên Colab (Ollama + model `GoldBot`, `temperature=0.0, seed=42` để loại nhiễu random sampling — biến số duy nhất giữa các vòng là `system_prompt_v2.txt`).

Quy tắc sửa `system_prompt_v2.txt` giữa các vòng: chỉ thêm/tinh chỉnh quy tắc TỔNG QUÁT, không hardcode số liệu/tên chỉ báo/nội dung của 25 câu test (tránh overfit lên chính bộ test).

---

## Vòng 1 (Baseline — `system_prompt_v2.txt` bản đầu)

- **Ngày chạy:** 2026-07-30
- **Môi trường:** Google Colab (Tesla T4 GPU, Ollama Server, `temperature=0.0`, `seed=42`)
- **Faithfulness Rate:** **17/25 (68.0%)**
- **Kết quả tổng quan:** PASS 17 câu, FAIL 8 câu (ID 4, 6, 7, 10, 11, 12, 22, 25)

### Danh sách câu sai & Rà soát nguyên nhân (Diagnosis):

1. **ID 4 (Nhóm 1 - Fed Rate change):**
   * *Question:* Lãi suất Fed gần đây thay đổi bao nhiêu?
   * *Bot Output:* Lãi suất Fed giữ nguyên 3.63%, không có biến động/thay đổi từ 2026-06-22 đến 2026-07-22.
   * *Lý do FAIL:* Grader tìm số exact/tolerance nhưng Bot không viết số `0` hoặc `0.00 điểm %` tường minh.
   * *Phân loại lỗi:* **BUG CỦA GRADER** (Thiếu fallback nhận diện cụm từ chỉ giá trị không đổi).

2. **ID 6, 7, 10 (Nhóm 2 - Direction Match Vàng & DXY):**
   * *Question:* ID 6 (Vàng tăng/giảm), ID 7 (DXY mạnh/yếu), ID 10 (Xu hướng ngắn hạn Vàng).
   * *Bot Output:* Bot đưa ra nhận định đúng (Vàng giảm nhẹ -0.84%, DXY tăng nhẹ +0.12%), nhưng phần diễn giải có nhắc kèm các chỉ báo phụ khác (VD: phân tích Vàng nhưng nhắc "DXY tăng").
   * *Lý do FAIL:* Grader tìm keyword trên TOÀN VĂN BẢN nên bị nhiễm từ khóa đối lập của chỉ báo phụ.
   * *Phân loại lỗi:* **BUG CỦA GRADER** (Chưa áp dụng Proximity Matching cho `direction_match`).

3. **ID 11 (Nhóm 3 - Comparison Vàng vs DXY):**
   * *Question:* Vàng và DXY di chuyển cùng chiều hay ngược chiều?
   * *Bot Output:* "...di chuyển theo **hướng khác nhau**... xu hướng **đi ngược nhau**..."
   * *Lý do FAIL:* Grader chỉ chờ từ exact match (`ngược chiều`, `nghịch chiều`), thiếu từ đồng nghĩa.
   * *Phân loại lỗi:* **BUG CỦA GRADER** (Bộ `accept_keywords` thiếu biến thể ngôn ngữ).

4. **ID 12 (Nhóm 3 - Comparison Real Yield vs Fed Rate):**
   * *Question:* Lợi suất thực và lãi suất Fed có cùng xu hướng không?
   * *Bot Output:* "...không đủ thông tin để đưa ra kết luận về xu hướng chung... mặc dù có sự tương đồng..."
   * *Lý do FAIL:* Grader bắt trúng cụm `cùng xu hướng` trong câu phủ định/phân tích phụ.
   * *Phân loại lỗi:* **BUG CỦA GRADER** (Grader "mù phủ định" - Negation Ignorance).

5. **ID 22, 25 (Nhóm 5 - Refusal Expected for Out-of-Scope Topics):**
   * *Question:* ID 22 (Bitcoin vs Vàng), ID 25 (Ethereum vs Vàng).
   * *Bot Output:* "Tôi không có đủ dữ liệu để phân tích...", "Tôi không thể đưa ra nhận định..." (Sau đó phân tích thêm dữ liệu Vàng/DXY có sẵn).
   * *Lý do FAIL:* Bot đã từ chối đúng bản chất nhưng danh sách từ khóa từ chối của Grader chưa bao quát hết mẫu câu diễn đạt thực tế của LLM, kết hợp việc Bot giải thích lan man.
   * *Phân loại lỗi:* **CẢ HAI** (Grader thiếu từ khóa từ chối thực tế & Bot bị lan man).

---

## Plan xử lý tiếp theo (Grader First Protocol)

Trước khi quyết định sửa Prompt (`system_prompt_v2.txt`) cho Vòng 2, tiến hành **Refactor Grader** (`eval_script.py` và `eval_dataset.json`) để loại bỏ toàn bộ lỗi đánh giá sai (False Failures):
1. Thêm fallback cho ID 4 khi con số kỳ vọng bằng 0.
2. Áp dụng Proximity Window cho ID 6, 7, 10.
3. Bổ sung từ đồng nghĩa cho ID 11.
4. Xử lý logic phủ định (Negation handling) cho ID 12.
5. Mở rộng bộ `refusal_keywords` cho ID 22, 25.

Sau khi sửa Grader, chạy lại **Re-benchmark Vòng 1** để thu được con số Faithfulness Rate thực sự chính xác.

---

## Vòng 2

- **Thay đổi so với vòng trước:** [CHỜ SAU KHỊ RE-BENCHMARK VÒNG 1 SỬA GRADER]
- **Faithfulness Rate:** [CHỜ]

## Kết luận

[CHỜ CẬP NHẬT KHI ĐẠT KẾT QUẢ CUỐI CÙNG]
