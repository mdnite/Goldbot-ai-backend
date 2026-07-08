from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import os
import datetime
import pytz
import requests

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

app = FastAPI(title="Fruit Shop Bot API")

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

def get_location_from_ip(ip: str):
    if ip in ["127.0.0.1", "::1", "localhost"]:
        return 10.823, 106.629, "Hồ Chí Minh"
    try:
        res = requests.get(f"http://ip-api.com/json/{ip}", timeout=3).json()
        if res.get("status") == "success":
            return res["lat"], res["lon"], res["city"]
    except:
        pass
    return 10.823, 106.629, "Hồ Chí Minh"

def get_real_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=3)
        data = response.json()
        current = data.get("current_weather", {})
        temp = current.get("temperature", "không rõ")
        code = current.get("weathercode", 0)

        weather_desc = "quang mây, tạnh ráo"
        if code in [1, 2, 3]: weather_desc = "có mây"
        elif code in [45, 48]: weather_desc = "có sương mù"
        elif code in [51, 53, 55, 56, 57]: weather_desc = "mưa lất phất"
        elif code in [61, 63, 65, 66, 67]: weather_desc = "mưa rào"
        elif code in [80, 81, 82]: weather_desc = "mưa to"
        elif code in [95, 96, 99]: weather_desc = "mưa dông sấm chớp"
        return f"{temp}°C, trời {weather_desc}"
    except:
        return "không rõ thời tiết"

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

# --- PHÂN LẬP PROMPT: CHIA LÀM 2 KỊCH BẢN RÕ RÀNG ---

# KỊCH BẢN 1: DÀNH CHO CÁC CÂU HỎI TRỰC TIẾP (GIÁ CẢ, PHÍ SHIP...)
DIRECT_PROMPT = """
Bạn là GoldBot - Nhân viên bán vàng.
QUY TẮC SỐ 1: Hãy trả lời NGẮN GỌN, đi thẳng vào trọng tâm câu hỏi của khách dựa vào THÔNG TIN CỬA HÀNG.
QUY TẮC SỐ 2: TUYỆT ĐỐI KHÔNG phân tích, KHÔNG đề cập thời tiết, KHÔNG dài dòng chèo kéo thêm.

THÔNG TIN CỬA HÀNG:
{context}

Câu hỏi của khách: {question}
"""

# KỊCH BẢN 2: DÀNH CHO CÁC CÂU NHỜ TƯ VẤN (CÓ GỢI Ý, THỜI TIẾT)
ADVICE_PROMPT = """
Bạn là FreshBot - Nhân viên tư vấn trái cây cao cấp.
THÔNG TIN KHÁCH HÀNG:
- Vị trí: {location}
- Thời tiết: {current_weather}

LỊCH SỬ KHẨU VỊ (Xem khách thích ngọt hay chua):
{chat_history}

THÔNG TIN CỬA HÀNG:
{context}

NHIỆM VỤ: Khách đang nhờ tư vấn. Hãy chào hỏi, nhắc nhẹ đến thời tiết tại {location} để tạo sự thân thiện. Dựa vào LỊCH SỬ KHẨU VỊ, chọn 1-2 loại trái cây phù hợp nhất từ THÔNG TIN CỬA HÀNG để giới thiệu và mời khách.

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
        advice_keywords = ["tư vấn", "gợi ý", "chọn", "ngon", "nào", "gì", "thời tiết", "nóng", "mát", "chua", "ngọt", "ăn gì"]
        is_asking_advice = any(word in user_msg.lower() for word in advice_keywords)

        if is_asking_advice:
            # Mode Tư vấn: Bơm vị trí, thời tiết, lịch sử
            client_ip = request.client.host
            lat, lon, city = get_location_from_ip(client_ip)
            current_weather = get_real_weather(lat, lon)
            
            final_prompt = ADVICE_PROMPT.format(
                location=city,
                current_weather=current_weather,
                context=context,
                chat_history=history_text,
                question=user_msg
            )
        else:
            # Mode Trực tiếp: CẮT ĐỨT HOÀN TOÀN THỜI TIẾT KHỎI PROMPT
            final_prompt = DIRECT_PROMPT.format(
                context=context,
                question=user_msg
            )

        # Gọi AI
        bot_reply = llm.invoke(final_prompt).strip()

        chat_sessions[session_id].append({"role": "Khách", "content": user_msg})
        chat_sessions[session_id].append({"role": "Bot", "content": bot_reply})

        return {"reply": bot_reply}

    except Exception as e:
        print(f"Lỗi AI: {e}")
        return {"reply": "Dạ, hệ thống đang bận một chút, anh/chị thử lại sau nhé ạ."}