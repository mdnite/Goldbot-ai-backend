from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import os
import sqlite3

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

app = FastAPI(title="GoldBot API")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

@app.options("/api/chat")
async def preflight():
    return Response(status_code=200)

llm = OllamaLLM(model="GoldBot")
embedding_model = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
if os.path.exists("./chroma_db"):
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
else:
    retriever = None

class ChatRequest(BaseModel):
    session_id: str
    message: str

chat_sessions = {}

# Câu disclaimer CỐ ĐỊNH, phải khớp nguyên văn với rule 3 trong Modelfile.
# Không để model tự diễn giải lại - enforce bằng code ở dưới.
FIXED_DISCLAIMER = (
    "Đây là thông tin mang tính học thuật/kỹ thuật từ một bài tập nội bộ, "
    "không phải lời khuyên đầu tư. Mọi quyết định đầu tư đều do khách tự chịu trách nhiệm."
)

# --- GIAI ĐOẠN 2: DỮ LIỆU ĐỊNH LƯỢNG THẬT TỪ indicators.db ---
# Chỉ inject SỐ THÔ (%change/diff), KHÔNG dùng nhãn dự đoán của trend_model.joblib -
# model đó đã xác nhận thua baseline trên test (xem giai_doan_1_5_report.md), rủi ro
# trình bày sai cho khách hàng ngân hàng lớn hơn giá trị thêm vào.
DB_PATH = "indicators.db"
MARKET_DATA_HORIZON = 21  # phiên giao dịch, khớp HORIZON dùng trong train_trend_model.py


def _nearest_value(cur, indicator, as_of_date):
    """Giá trị GẦN NHẤT của indicator tính đến as_of_date (<=). Trả None nếu không có."""
    cur.execute(
        "SELECT date, value FROM indicators WHERE indicator=? AND date<=? ORDER BY date DESC LIMIT 1",
        (indicator, as_of_date),
    )
    return cur.fetchone()


def get_market_data():
    """%change (gold, dxy) / diff (real_yield, fed_rate) giữa mốc mới nhất và
    MARKET_DATA_HORIZON phiên trước, đọc từ indicators.db. None nếu lỗi/thiếu dữ liệu -
    caller phải xử lý None bằng cách bỏ qua khối số liệu, không crash chatbot."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        cur.execute(
            "SELECT date, value FROM indicators WHERE indicator='gold' ORDER BY date DESC LIMIT 1"
        )
        row_now = cur.fetchone()
        cur.execute(
            "SELECT date, value FROM indicators WHERE indicator='gold' ORDER BY date DESC LIMIT 1 OFFSET ?",
            (MARKET_DATA_HORIZON,),
        )
        row_past = cur.fetchone()
        if row_now is None or row_past is None:
            return None
        date_now, gold_now = row_now
        date_past, gold_past = row_past

        others = {}
        for indicator in ("dxy", "real_yield", "fed_rate"):
            r_now = _nearest_value(cur, indicator, date_now)
            r_past = _nearest_value(cur, indicator, date_past)
            if r_now is None or r_past is None:
                return None
            others[indicator] = (r_now[1], r_past[1])

        dxy_now, dxy_past = others["dxy"]
        real_yield_now, real_yield_past = others["real_yield"]
        fed_rate_now, fed_rate_past = others["fed_rate"]

        gold_pct = (gold_now - gold_past) / gold_past * 100
        dxy_pct = (dxy_now - dxy_past) / dxy_past * 100
        real_yield_diff = real_yield_now - real_yield_past
        fed_rate_diff = fed_rate_now - fed_rate_past

        return (
            f"Tính đến ngày {date_now} (so với {date_past}, khoảng {MARKET_DATA_HORIZON} phiên giao dịch trước):\n"
            f"- Vàng (gold futures): hiện tại {gold_now:.2f} USD/oz, thay đổi {gold_pct:+.2f}%\n"
            f"- Chỉ số USD (DXY): hiện tại {dxy_now:.2f}, thay đổi {dxy_pct:+.2f}%\n"
            f"- Lợi suất thực 10 năm (real yield): hiện tại {real_yield_now:.2f}%, thay đổi {real_yield_diff:+.2f} điểm %\n"
            f"- Lãi suất Fed (fed funds rate): hiện tại {fed_rate_now:.2f}%, thay đổi {fed_rate_diff:+.2f} điểm %"
        )
    except sqlite3.Error as e:
        print(f"[get_market_data] Lỗi truy vấn indicators.db: {e}")
        return None
    except Exception as e:
        print(f"[get_market_data] Lỗi không xác định khi lấy dữ liệu định lượng: {e}")
        return None
    finally:
        if conn is not None:
            conn.close()


# --- PHÂN LẬP PROMPT: CHIA LÀM 2 KỊCH BẢN RÕ RÀNG ---

# KỊCH BẢN 1: DÀNH CHO CÁC CÂU HỎI TRỰC TIẾP (ĐỊNH NGHĨA, CƠ CHẾ, QUY ĐỊNH...)
DIRECT_PROMPT = """
Bạn là GoldBot - trợ lý kiến thức và phân tích xu hướng thị trường vàng (bài tập nội bộ tại Saigonbank).
QUY TẮC SỐ 1: Hãy trả lời NGẮN GỌN, đi thẳng vào trọng tâm câu hỏi của khách dựa vào THÔNG TIN THAM KHẢO.
QUY TẮC SỐ 2: TUYỆT ĐỐI KHÔNG phân tích xu hướng, KHÔNG dự đoán tăng/giảm, KHÔNG dài dòng chèo kéo thêm - đây là câu hỏi khái niệm/quy định, chỉ cần trả lời đúng và đủ.
QUY TẮC SỐ 3: KHÔNG bịa số liệu giá vàng thời gian thực. Nếu nhắc đơn vị vàng, dùng đúng chuẩn Việt Nam (lượng/chỉ/phân, 1 lượng = 37,5g) và quốc tế (troy ounce ≈ 31,1035g), không quy đổi sai.

THÔNG TIN THAM KHẢO:
{context}

Câu hỏi của khách: {question}
"""

# KỊCH BẢN 2: DÀNH CHO CÁC CÂU NHỜ TƯ VẤN/NHẬN ĐỊNH XU HƯỚNG
# ĐÃ SỬA: bỏ hoàn toàn phần thời tiết/vị trí - không liên quan gì đến phân tích vàng,
# là phần leftover từ persona FreshBot gốc chưa được dọn.
ADVICE_PROMPT = """
Bạn là GoldBot - trợ lý kiến thức và phân tích xu hướng thị trường vàng (bài tập nội bộ tại Saigonbank).
QUY TẮC SỐ 1: Khách đang hỏi nhận định/xu hướng nên ĐƯỢC PHÉP giải thích, phân tích các yếu tố ảnh hưởng (USD, lãi suất, lạm phát, safe haven...) dựa vào THÔNG TIN THAM KHẢO, không cần ngắn gọn như câu hỏi thông thường.
QUY TẮC SỐ 2: TUYỆT ĐỐI KHÔNG bịa đặt thông tin không có trong THÔNG TIN THAM KHẢO. Được phép dùng số liệu trong khối DỮ LIỆU ĐỊNH LƯỢNG bên dưới để diễn giải xu hướng, nhưng KHÔNG bịa thêm bất kỳ số liệu giá vàng/thị trường thời gian thực nào khác ngoài khối đó.
QUY TẮC SỐ 3: KHÔNG khẳng định chắc chắn giá sẽ tăng/giảm hay khuyến nghị nên mua/bán - chỉ trình bày xu hướng có xác suất kèm mức độ không chắc chắn.
QUY TẮC SỐ 4: Giọng điệu chuyên nghiệp, rõ ràng - KHÔNG chào hỏi kiểu bán hàng, KHÔNG nhắc thời tiết hay chủ đề không liên quan.

LỊCH SỬ TRAO ĐỔI (Xem khách đã hỏi/quan tâm chủ đề gì):
{chat_history}

THÔNG TIN THAM KHẢO:
{context}

DỮ LIỆU ĐỊNH LƯỢNG (thời gian thực, do hệ thống cung cấp):
{market_data}

NHIỆM VỤ: Khách đang nhờ tư vấn/nhận định. Dựa vào LỊCH SỬ TRAO ĐỔI, THÔNG TIN THAM KHẢO và DỮ LIỆU ĐỊNH LƯỢNG, giải thích rõ ràng, đúng trọng tâm câu hỏi, tuân thủ QUY TẮC SỐ 2, 3, 4 ở trên.

Câu hỏi của khách: {question}
"""

@app.post("/api/chat")
async def chat(request_data: ChatRequest, request: Request):
    session_id = request_data.session_id
    user_msg = request_data.message

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    try:
        # Lấy kiến thức RAG
        context = ""
        if retriever:
            docs = retriever.invoke(user_msg)
            context = "\n".join([doc.page_content for doc in docs])

        # Lấy lịch sử chat
        recent_history = chat_sessions[session_id][-6:]
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        if not history_text: history_text = "Chưa có"

        # TÌM TỪ KHÓA BẰNG PYTHON (ĐÂY LÀ PHẦN LÕI CỦA GIẢI PHÁP)
        # Nếu khách dùng các từ này, mới chuyển sang mode Tư vấn
        advice_keywords = ["tư vấn", "nên mua", "nên bán", "có nên", "dự đoán", "dự báo", "xu hướng", "phân tích", "nhận định", "đầu tư", "tăng hay giảm", "ý kiến"]
        is_asking_advice = any(word in user_msg.lower() for word in advice_keywords)

        if is_asking_advice:
            # Mode Tư vấn: KHÔNG còn bơm vị trí/thời tiết (đã bỏ)
            # Giai đoạn 2: inject số thô 4 indicator, None -> báo rõ thiếu dữ liệu, không bịa.
            market_data = get_market_data() or "Không có dữ liệu định lượng khả dụng cho câu hỏi này."
            final_prompt = ADVICE_PROMPT.format(
                context=context,
                chat_history=history_text,
                market_data=market_data,
                question=user_msg
            )
        else:
            # Mode Trực tiếp: không liên quan thời tiết
            final_prompt = DIRECT_PROMPT.format(
                context=context,
                question=user_msg
            )

        # Gọi AI
        bot_reply = llm.invoke(final_prompt).strip()

        # ĐÃ THÊM: enforce disclaimer bằng code cho nhánh advice, không phụ thuộc
        # hoàn toàn vào việc model tự nhớ đúng nguyên văn qua prompt.
        if is_asking_advice and FIXED_DISCLAIMER not in bot_reply:
            bot_reply = f"{bot_reply}\n\n{FIXED_DISCLAIMER}"

        chat_sessions[session_id].append({"role": "Khách", "content": user_msg})
        chat_sessions[session_id].append({"role": "Bot", "content": bot_reply})

        return {"reply": bot_reply}

    except Exception as e:
        print(f"Lỗi AI: {e}")
        # ĐÃ SỬA: bỏ giọng điệu "Dạ...anh/chị" kiểu bán hàng, không khớp persona GoldBot
        return {"reply": "Xin lỗi, hệ thống đang gặp sự cố. Bạn vui lòng thử lại sau."}