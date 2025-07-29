import os
from process_store_class import CVProcessor, FAISSVectorStore

cv_processor = CVProcessor()
vector_store = FAISSVectorStore()

cv_folder = "cv"

for filename in os.listdir(cv_folder):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(cv_folder, filename)
        print(f"🔍 Đang xử lý file: {file_path}")

        # 1. Đọc và chuyển đổi PDF sang markdown
        text = cv_processor.parse_cv_to_markdown(file_path)
        if not text.strip():
            print(f"⚠️ Bỏ qua file {filename} vì không đọc được nội dung.")
            continue

        # 2. Tách đoạn văn
        docs = cv_processor.chunk_text(text, source=filename)

        # 3. Tạo embedding
        embeddings = cv_processor.get_embeddings([doc.page_content for doc in docs])

        # 4. Thêm vào FAISS
        vector_store.add_documents(docs, embeddings)

# 5. Lưu FAISS và metadata
vector_store.save("cv_index.faiss", "cv_metadata.json")

