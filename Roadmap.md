# GoldBot — Roadmap (cố định)

Đây là định nghĩa các giai đoạn — KHÔNG phải tiến độ/trạng thái đang làm tới đâu. Xem Task trong Claude Code để biết đang làm tới đâu.

- Giai đoạn 0: chốt scope.
- Giai đoạn 1: data pipeline (yfinance + fredapi + SQLite).
- Giai đoạn 1.5: huấn luyện model dự đoán xu hướng (Tăng/Giảm/Đi ngang, kèm mức tin cậy) từ return/diff của 4 indicator — bắt buộc phải xong trước Giai đoạn 2 vì Giai đoạn 2 cần biết inject gì vào prompt (số thô hay output model), không thiết kế đúng được nếu thiếu bước này.
- Giai đoạn 2: kết nối RAG với dữ liệu định lượng thật.
- Giai đoạn 2.5: đo và tối ưu Faithfulness Rate (bot đọc đúng market_data — không đảo chiều, không bịa số/chỉ báo) qua vòng lặp eval_script.py → sửa system_prompt_v2.txt, dừng khi ≥90%. Không đo văn phong/chất lượng lời khuyên (để dành Giai đoạn 4).
- Giai đoạn 3: đo và cải thiện khả năng SUY LUẬN dự đoán ngắn hạn của LLM (đã revise so với định nghĩa gốc "format output" — cấu trúc 4 mục vẫn giữ làm khung bắt buộc nhưng chỉ là phương tiện, không phải mục tiêu đo). Backtest N=50 mốc thời gian, so 2 baseline (đóng băng từ Giai đoạn 1.5 + majority in-sample). Chi tiết đầy đủ: `giai_doan_3_report.md`.
- Giai đoạn 4: cải tiến model định lượng dự đoán xu hướng (Giai đoạn 1.5) — sửa 3 vấn đề kỹ thuật (cô lập đặc trưng/interaction terms, volatility trễ, đa cộng tuyến), giữ nguyên Logistic Regression. 4a = phương án chính (đã lên kế hoạch); 4b = Random Forest/XGBoost, dự phòng nếu 4a không đạt. Chi tiết đầy đủ: CLAUDE.md mục "Quyết định đã chốt". (Đã thay thế định nghĩa LoRA fine-tuning cũ — hướng LLM/prompt đã dừng đầu tư, xem `giai_doan_3_report.md` mục 10.)
