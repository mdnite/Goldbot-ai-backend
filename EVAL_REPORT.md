# Giai đoạn 2.5 — Faithfulness Eval Report

Mục tiêu: Faithfulness Rate ≥ 90% (≥23/25 câu) trên `eval_dataset.json`, chấm rule-based bằng `eval_script.py` chạy trên Colab (Ollama + model `GoldBot`, `temperature=0.0, seed=42` để loại nhiễu random sampling — biến số duy nhất giữa các vòng là `system_prompt_v2.txt`).

Quy tắc sửa `system_prompt_v2.txt` giữa các vòng: chỉ thêm/tinh chỉnh quy tắc TỔNG QUÁT, không hardcode số liệu/tên chỉ báo/nội dung của 25 câu test (tránh overfit lên chính bộ test).

## Vòng 1 (baseline — `system_prompt_v2.txt` bản đầu)

- Ngày chạy: [CHỜ KẾT QUẢ TỪ COLAB]
- Faithfulness Rate: [CHỜ]/25 ([CHỜ]%)
- Danh sách câu sai: [CHỜ]

## Vòng 2

- Thay đổi so với vòng trước: [CHỜ]
- Faithfulness Rate: [CHỜ]

## Kết luận

[Điền khi đạt ≥90% hoặc khi dừng vòng lặp vì lý do khác — ghi rõ vòng nào đạt ngưỡng và tổng số vòng đã chạy.]
