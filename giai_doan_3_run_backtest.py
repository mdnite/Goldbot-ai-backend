"""Giai đoạn 3, Task #6/#7 - Chạy backtest suy luận N=50 mốc.

BẮT BUỘC CHẠY TRÊN COLAB (nơi Ollama + model GoldBot đang sống, giống hệt eval_script.py
Giai đoạn 2.5) - đã xác nhận không có Ollama server local (curl 127.0.0.1:11434 that bai).
Không đụng eval_script.py/eval_dataset.json/EVAL_REPORT.md (đo faithfulness, khác phạm vi).

Dùng --dry-run để kiểm thử toàn bộ pipeline (đọc sample dates, ghép market_data leak-safe,
build prompt) MÀ KHÔNG gọi Ollama thật - dùng để smoke test local trước khi mang lên Colab.
"""
import argparse
import json
import os

import requests

from giai_doan_3_backtest import get_market_data_asof, get_actual_outcome, BASELINE_LABEL

SAMPLE_DATES_PATH = "giai_doan_3_sample_dates.json"
SYSTEM_PROMPT_PATH = "giai_doan_3_system_prompt.txt"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "GoldBot"
# Deterministic inference - khớp convention eval_script.py Giai đoạn 2.5, biến số duy
# nhất được phép ảnh hưởng kết quả giữa các lần chạy là nội dung prompt, không phải
# random sampling.
OLLAMA_OPTIONS = {"temperature": 0.0, "seed": 42}
OUTPUT_PATH = "giai_doan_3_backtest_results.json"

QUESTION = "Xu hướng vàng ngắn hạn sắp tới thế nào, dựa trên các yếu tố vĩ mô hiện tại?"


def call_ollama(prompt):
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": OLLAMA_OPTIONS,
        },
        timeout=900,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def load_existing_results(output_path):
    """Resume: nếu file kết quả đã có từ lần chạy trước bị ngắt giữa chừng, đọc lại
    để bỏ qua các mốc đã xong - N=50 chỉ chạy MỘT LẦN cho báo cáo, nhưng nếu Colab bị
    ngắt kết nối giữa chừng thì resume không tính là "chạy lại", chỉ là hoàn tất nốt
    đúng 1 lần chạy đã bắt đầu."""
    if not os.path.exists(output_path):
        return []
    with open(output_path, encoding="utf-8") as f:
        return json.load(f)


def run_backtest(dates, dry_run=False):
    with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        prompt_template = f.read()

    results = load_existing_results(OUTPUT_PATH) if not dry_run else []
    done_dates = {r["as_of_date"] for r in results}

    for i, d in enumerate(dates):
        if d in done_dates:
            print(f"[{i+1}/{len(dates)}] {d}: DA CO KET QUA (resume) - bo qua")
            continue

        market_text, market_raw = get_market_data_asof(d)
        if market_text is None:
            print(f"[{i+1}/{len(dates)}] {d}: BO QUA - khong co market_data snapshot")
            continue
        outcome = get_actual_outcome(d)
        if outcome is None:
            print(f"[{i+1}/{len(dates)}] {d}: BO QUA - khong co outcome that")
            continue

        prompt = prompt_template.format(
            chat_history="Chưa có",
            context="Không có thông tin tham khảo bổ sung.",
            market_data=market_text,
            question=QUESTION,
        )

        if dry_run:
            print(f"[{i+1}/{len(dates)}] {d}: DRY-RUN OK - prompt {len(prompt)} ky tu, "
                  f"outcome that={outcome['label']}, baseline={BASELINE_LABEL}")
            continue

        reply = call_ollama(prompt)
        results.append({
            "as_of_date": d,
            "market_data_text": market_text,
            "market_data_raw": market_raw,
            "outcome": outcome,
            "baseline_pred": BASELINE_LABEL,
            "bot_reply": reply,
        })
        print(f"[{i+1}/{len(dates)}] {d}: OK (outcome that = {outcome['label']})")

        # Luu tang dan sau moi moc - tranh mat toan bo ket qua neu Colab ngat giua chung.
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Kiem thu toan bo pipeline KHONG goi Ollama that - dung de smoke test local.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Chi chay N moc dau tien (dung cho smoke test co goi Ollama that tren Colab).")
    args = parser.parse_args()

    with open(SAMPLE_DATES_PATH, encoding="utf-8") as f:
        sample = json.load(f)
    dates = sample["sample_dates"]
    if args.limit:
        dates = dates[:args.limit]

    print(f"Bat dau backtest N={len(dates)} moc, model={MODEL_NAME}, options={OLLAMA_OPTIONS}, "
          f"dry_run={args.dry_run}")
    results = run_backtest(dates, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\nHoan tat: {len(results)}/{len(dates)} moc co ket qua. Da luu {OUTPUT_PATH}")
