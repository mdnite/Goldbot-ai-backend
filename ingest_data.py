import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# SỬ DỤNG THƯ VIỆN MỚI NHẤT (Tránh cảnh báo màu đỏ trên Terminal)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Cấu hình chuẩn Tuning Level 2 (RAG Q&A) ---
PERSIST_DIRECTORY = "./chroma_db"
CHUNK_SIZE = 1000       # Tăng lên 1000 để chứa trọn vẹn 1 đoạn Hỏi-Đáp dài
CHUNK_OVERLAP = 150     # Tăng độ nối giữa các đoạn để không mất ngữ cảnh
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"
TEXT_FILE_PATH = "bank_data.txt"

# --- 1. Load mô hình Embedding (Chuyển chữ thành Vector) ---
print("📥 Đang tải mô hình ngôn ngữ (Embedding)...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# --- 2. Load Vector Database CŨ từ ổ cứng (Nếu có) ---
print(f"🔍 Đang tìm kiếm Database cũ tại thư mục '{PERSIST_DIRECTORY}'...")
if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )
    print(f"   ✅ Đã tải Database cũ. Đang có sẵn {vector_db._collection.count()} đoạn dữ liệu.")
else:
    print("   ⚠️ Không tìm thấy Database cũ. Hệ thống sẽ tạo một DB hoàn toàn mới.")
    vector_db = None

# --- 3. Đọc và băm nhỏ dữ liệu MỚI từ file txt ---
print(f"📄 Đang đọc dữ liệu mới từ file '{TEXT_FILE_PATH}'...")
loader = TextLoader(TEXT_FILE_PATH, encoding="utf-8")
new_documents = loader.load()

print(f"✂️ Đang băm nhỏ dữ liệu (Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
new_chunks = text_splitter.split_documents(new_documents)

# --- 4. Nạp dữ liệu mới vào Vector Database ---
if vector_db:
    print(f"➕ Đang nạp thêm {len(new_chunks)} đoạn văn bản mới vào Database cũ...")
    vector_db.add_documents(new_chunks)
    print("✅ Đã nạp thêm dữ liệu thành công.")
else:
    print("🆕 Đang khởi tạo Database mới và nạp dữ liệu...")
    vector_db = Chroma.from_documents(
        documents=new_chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )
    print("✅ Đã tạo mới Vector DB thành công.")

# Lưu ý: ChromaDB phiên bản mới tự động lưu (auto-persist) nên không cần gọi vector_db.persist() nữa.

# --- Báo cáo kết quả ---
final_count = vector_db._collection.count()
print(f"📊 HOÀN TẤT! Vector Database hiện tại chứa tổng cộng {final_count} đoạn kiến thức.")