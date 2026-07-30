"""Giai đoạn 2.5 - chạy trên Colab (nơi Ollama + model GoldBot đang sống).
Đo Faithfulness Rate: % câu trả lời đọc ĐÚNG khối market_data (không đảo chiều, không bịa số/chỉ báo).
Chấm rule-based (keyword/number matching) trên văn xuôi tự do thật, không LLM-judge, không ép JSON output.
"""
import json
import re

import requests

EVAL_DATASET_PATH = "eval_dataset.json"
SYSTEM_PROMPT_PATH = "system_prompt_v2.txt"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "GoldBot"

# Số duy nhất được phép ảnh hưởng % Faithfulness giữa các vòng là system_prompt_v2.txt,
# không phải random sampling -> ép greedy decoding + seed cố định.
OLLAMA_OPTIONS = {"temperature": 0.0, "seed": 42}


def normalize_text(text):
    """Lowercase, xóa phân cách hàng nghìn, chuẩn hóa ký hiệu đơn vị - chạy trước khi match."""
    t = text.lower()
    # phân cách hàng nghìn: dấu phẩy/chấm kẹp giữa số và đúng 3 chữ số theo sau (không phải phần thập phân)
    t = re.sub(r"(?<=\d)[.,](?=\d{3}(\D|$))", "", t)
    t = t.replace("điểm %", "diem%").replace("điểm%", "diem%").replace("đ%", "diem%")
    t = t.replace("usd/oz", "usd_oz")
    t = re.sub(r"\s+", " ", t).strip()
    return t


NUMBER_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


def extract_numbers(normalized_text):
    return [float(m) for m in NUMBER_RE.findall(normalized_text)]


def extract_signed_numbers(normalized_text):
    """Chỉ trả số có dấu +/- TƯỜNG MINH trong văn bản - số không dấu (vd '0.84' viết
    kèm từ 'giảm') là số biên độ, không phải bằng chứng chiều, không đưa vào đây."""
    return [float(m.group()) for m in NUMBER_RE.finditer(normalized_text) if m.group()[0] in "+-"]


def resolve_keywords(question, field, keyword_sets):
    """Gộp keyword tường minh (accept_keywords/reject_keywords) + keyword tra theo set name."""
    words = list(question.get(field, []))
    for set_name in question.get(f"{field}_sets", []):
        words.extend(keyword_sets.get(set_name, []))
    return [normalize_text(w) for w in words]


def check_number_match(question, normalized_reply):
    expected = question["expected_value"]
    tolerance_type = question["tolerance_type"]
    tolerance = question["tolerance"]
    numbers = extract_numbers(normalized_reply)
    threshold = abs(expected) * tolerance if tolerance_type == "relative" else tolerance

    direction = question.get("expected_direction")
    if direction is None:
        # Giá trị tuyệt đối (Q1/Q3): không có khái niệm chiều, so khớp trực tiếp.
        magnitude_ok = any(abs(n - expected) <= threshold for n in numbers)
        return magnitude_ok, None if magnitude_ok else "không tìm thấy số khớp"

    # Giá trị thay đổi (Q2/Q4/Q5): tiếng Việt thường diễn đạt chiều bằng TỪ
    # ("giảm 0.84%") chứ không phải dấu "-" trong số -> so biên độ KHÔNG DẤU
    # trước, xác định chiều riêng (qua dấu số nếu có, hoặc qua từ khóa).
    magnitude_ok = any(abs(abs(n) - abs(expected)) <= threshold for n in numbers)

    if direction == "khong_doi":
        # Fallback: Bot thường diễn đạt "không đổi" thuần bằng từ (VD "giữ
        # nguyên 3.63%") mà không viết lại số biến động "0"/"0.00%" tường
        # minh -> chấp nhận từ khóa không đổi khi không có số khớp biên độ,
        # miễn không lẫn từ khóa tăng/giảm.
        reject_words = [normalize_text(w) for w in KEYWORD_SETS.get("tang", []) + KEYWORD_SETS.get("giam", [])]
        has_reject = any(w in normalized_reply for w in reject_words)
        if has_reject:
            return False, "từ khóa chỉ có thay đổi dù số liệu ~0"
        if magnitude_ok:
            return True, None
        accept_words = [normalize_text(w) for w in KEYWORD_SETS.get("khong_doi", [])]
        has_accept = any(w in normalized_reply for w in accept_words)
        return has_accept, None if has_accept else "không tìm thấy số khớp biên độ hoặc từ khóa không đổi"

    if not magnitude_ok:
        return False, "không tìm thấy số khớp biên độ"

    signed_numbers = extract_signed_numbers(normalized_reply)
    signed_match = any(abs(n - expected) <= threshold for n in signed_numbers)
    signed_reject = any(abs(n - (-expected)) <= threshold for n in signed_numbers)

    accept_words = [normalize_text(w) for w in KEYWORD_SETS.get(direction, [])]
    other_dirs = [d for d in ("tang", "giam") if d != direction]
    reject_words = []
    for d in other_dirs:
        reject_words.extend(normalize_text(w) for w in KEYWORD_SETS.get(d, []))

    has_accept = signed_match or any(w in normalized_reply for w in accept_words)
    has_reject = signed_reject or any(w in normalized_reply for w in reject_words)

    direction_ok = has_accept and not has_reject
    return direction_ok, None if direction_ok else "sai chiều/dấu so với số liệu"


PROXIMITY_WINDOW = 20  # ký tự, đủ chứa "đang"/"hiện" giữa tên chỉ báo và từ chỉ chiều


def _keyword_near_indicator(normalized_reply, indicator_keywords, target_words):
    """True nếu có từ trong target_words xuất hiện trong cửa sổ PROXIMITY_WINDOW
    ký tự quanh một lần xuất hiện bất kỳ của indicator_keywords - dùng để tránh
    nhiễu khi Bot nhắc nhiều chỉ báo trong cùng câu trả lời."""
    for kw in indicator_keywords:
        kw_norm = normalize_text(kw)
        for m in re.finditer(re.escape(kw_norm), normalized_reply):
            start = max(0, m.start() - PROXIMITY_WINDOW)
            end = m.end() + PROXIMITY_WINDOW
            window = normalized_reply[start:end]
            if any(w in window for w in target_words):
                return True
    return False


def _has_reversal(normalized_reply, indicator_keywords, wrong_direction_words):
    return _keyword_near_indicator(normalized_reply, indicator_keywords, wrong_direction_words)


def check_direction_match(question, normalized_reply):
    accept_words = resolve_keywords(question, "accept_keyword", KEYWORD_SETS)
    reject_words = resolve_keywords(question, "reject_keyword", KEYWORD_SETS)
    indicator_keywords = question.get("indicator_keywords")

    if indicator_keywords:
        # Chỉ tìm từ chỉ chiều GẦN tên chỉ báo đang hỏi - tránh bắt nhầm từ
        # chỉ chiều của một chỉ báo khác được nhắc kèm trong cùng câu trả lời.
        has_accept = _keyword_near_indicator(normalized_reply, indicator_keywords, accept_words)
        has_reject = _keyword_near_indicator(normalized_reply, indicator_keywords, reject_words)
    else:
        has_accept = any(w in normalized_reply for w in accept_words)
        has_reject = any(w in normalized_reply for w in reject_words)

    direction_ok = has_accept and not has_reject
    return direction_ok, None if direction_ok else "sai chiều hoặc không tìm thấy từ khóa gần tên chỉ báo"


NEGATION_WORDS = ["không", "chẳng"]  # đã bao trùm "không thể"/"không phải" (chứa "không" làm substring)
NEGATION_WINDOW = 40  # ký tự TRƯỚC cụm reject-keyword, đủ chứa 1 mệnh đề ngắn
                      # kiểu "không thể khẳng định ... có cùng xu hướng"


def _negated_before(normalized_reply, match_start):
    window = normalized_reply[max(0, match_start - NEGATION_WINDOW):match_start]
    return any(neg in window for neg in NEGATION_WORDS)


def check_comparison_match(question, normalized_reply):
    if question.get("mode") == "loose_no_reversal":
        for check in question.get("reversal_checks", []):
            wrong_words = [normalize_text(w) for w in KEYWORD_SETS.get(check["wrong_direction_set"], [])]
            if _has_reversal(normalized_reply, check["indicator_keywords"], wrong_words):
                return False, f"đảo chiều chỉ báo: {check['indicator_keywords'][0]}"
        return True, None

    reject_words = [normalize_text(w) for w in question.get("reject_keywords", [])]
    accept_words = [normalize_text(w) for w in question.get("accept_keywords", [])]

    # Bỏ qua một lần khớp reject-keyword nếu ngay trước nó có từ phủ định
    # (VD "không cùng xu hướng" không phải là khẳng định "cùng xu hướng") -
    # tránh grader "mù phủ định" (negation ignorance).
    has_reject = False
    for w in reject_words:
        for m in re.finditer(re.escape(w), normalized_reply):
            if not _negated_before(normalized_reply, m.start()):
                has_reject = True
                break
        if has_reject:
            break

    has_accept = any(w in normalized_reply for w in accept_words)
    direction_ok = has_accept and not has_reject
    return direction_ok, None if direction_ok else "không khớp accept/reject keyword (đã xét phủ định)"


def check_refusal_expected(question, normalized_reply, snapshot_numbers):
    refusal_words = [normalize_text(w) for w in KEYWORD_SETS.get("refusal", [])]
    has_refusal = any(w in normalized_reply for w in refusal_words)

    fabricated = False
    for topic in question.get("topic_keywords", []):
        topic_norm = normalize_text(topic)
        for m in re.finditer(re.escape(topic_norm), normalized_reply):
            window = normalized_reply[m.end(): m.end() + 40]
            for n in extract_numbers(window):
                if not any(abs(n - sv) <= max(0.5, abs(sv) * 0.02) for sv in snapshot_numbers):
                    fabricated = True

    if fabricated:
        return False, "bịa số gần từ khóa chủ đề"
    return has_refusal, None if has_refusal else "thiếu từ khóa từ chối"


def check_precision_trap(question, normalized_reply):
    known_value = question["known_value"]
    known_decimals = question["known_decimals"]
    int_prefix = str(int(known_value))
    pattern = re.compile(re.escape(int_prefix) + r"\.(\d+)")
    for m in pattern.finditer(normalized_reply):
        if len(m.group(1)) > known_decimals:
            return False, f"bịa thêm chữ số thập phân: {m.group(0)}"
    return True, None


def grade(question, reply_text, snapshot_numbers):
    normalized_reply = normalize_text(reply_text)
    check_type = question["check_type"]

    if check_type == "number_match":
        return check_number_match(question, normalized_reply)
    if check_type == "direction_match":
        return check_direction_match(question, normalized_reply)
    if check_type == "comparison_match":
        return check_comparison_match(question, normalized_reply)
    if check_type == "refusal_expected":
        return check_refusal_expected(question, normalized_reply, snapshot_numbers)
    if check_type == "precision_trap":
        return check_precision_trap(question, normalized_reply)
    raise ValueError(f"check_type không xác định: {check_type}")


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


def main():
    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        prompt_template = f.read()

    global KEYWORD_SETS
    KEYWORD_SETS = dataset["keyword_sets"]
    snapshot = dataset["market_data_snapshot"]
    snapshot_numbers = list(snapshot["raw_values"].values())

    results = []
    for q in dataset["questions"]:
        prompt = prompt_template.format(
            chat_history="Chưa có",
            context="Không có thông tin tham khảo bổ sung.",
            market_data=snapshot["market_data_text"],
            question=q["question"],
        )
        reply = call_ollama(prompt)
        passed, reason = grade(q, reply, snapshot_numbers)
        results.append({
            "id": q["id"], "group": q["group"], "question": q["question"],
            "reply": reply, "passed": passed, "reason": reason,
        })
        print(f"[{q['id']:02d}] {'PASS' if passed else 'FAIL'} - {q['question']}")

    n_pass = sum(r["passed"] for r in results)
    total = len(results)
    print(f"\n=== FAITHFULNESS RATE: {n_pass}/{total} ({n_pass/total:.1%}) ===\n")

    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"--- {len(failed)} câu sai (Prompt ID | Bot Output | Ground Truth/Lý do) ---\n")
        for r in failed:
            q = next(item for item in dataset["questions"] if item["id"] == r["id"])
            print(f"ID {r['id']} (nhóm {r['group']}): {r['question']}")
            print(f"  Bot output: {r['reply']}")
            print(f"  Lý do fail: {r['reason'] or 'không khớp ground truth/keyword mong đợi'}")
            print(f"  Ground truth config: {json.dumps({k: v for k, v in q.items() if not k.startswith('_')}, ensure_ascii=False)}")
            print()


if __name__ == "__main__":
    main()
