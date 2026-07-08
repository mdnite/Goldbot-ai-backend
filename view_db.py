from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Kết nối lại với thư mục chroma_db trên máy
embedding_model = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# 2. Rút toàn bộ dữ liệu ra xem
data = vector_db.get()

print(f"\n✅ TỔNG SỐ ĐOẠN VĂN BẢN TRONG DATABASE: {len(data['documents'])} đoạn\n")

# 3. In từng đoạn ra màn hình
for i, doc in enumerate(data['documents']):
    print(f"--- ĐOẠN SỐ {i+1} ---")
    print(doc)
    print("-" * 30)