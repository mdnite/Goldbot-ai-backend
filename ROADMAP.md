# GoldBot — Roadmap (cố định)

Đây là định nghĩa các giai đoạn — KHÔNG phải tiến độ/trạng thái đang làm tới đâu. Xem Task trong Claude Code để biết đang làm tới đâu.

- Giai đoạn 0: chốt scope.
- Giai đoạn 1: data pipeline (yfinance + fredapi + SQLite).
- Giai đoạn 1.5: huấn luyện model dự đoán xu hướng (Tăng/Giảm/Đi ngang, kèm mức tin cậy) từ return/diff của 4 indicator — bắt buộc phải xong trước Giai đoạn 2 vì Giai đoạn 2 cần biết inject gì vào prompt (số thô hay output model), không thiết kế đúng được nếu thiếu bước này.
- Giai đoạn 2: kết nối RAG với dữ liệu định lượng thật.
- Giai đoạn 3: format output (cấu trúc 4 mục + disclaimer).
- Giai đoạn 4: LoRA fine-tuning (điều kiện, xem "Quyết định đã chốt" trong CLAUDE.md).
- Giai đoạn 5: chưa được định nghĩa nội dung cụ thể trong các phiên trước — hỏi lại người dùng nếu cần dùng tới.