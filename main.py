import os
import argparse
from process_store_class import CVProcessor, FAISSVectorStore
from model_infer import CVChatBot

def build_vector_store(cv_folder: str = "cv", 
                      index_path: str = "cv_index.faiss", 
                      metadata_path: str = "cv_metadata.json"):
    """Xây dựng vector store từ thư mục CV"""
    
    print("🚀 Bắt đầu xử lý và embedding CV...")
    
    cv_processor = CVProcessor()
    vector_store = FAISSVectorStore()
    
    if not os.path.exists(cv_folder):
        print(f"❌ Không tìm thấy thư mục {cv_folder}")
        return False
    
    pdf_files = [f for f in os.listdir(cv_folder) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"❌ Không tìm thấy file PDF nào trong thư mục {cv_folder}")
        return False
    
    print(f"📁 Tìm thấy {len(pdf_files)} file PDF")
    
    processed_count = 0
    
    for filename in pdf_files:
        file_path = os.path.join(cv_folder, filename)
        print(f"🔍 Đang xử lý file: {filename}")
        
        try:
            # 1. Đọc và chuyển đổi PDF sang markdown
            text = cv_processor.parse_cv_to_markdown(file_path)
            if not text.strip():
                print(f"⚠️ Bỏ qua file {filename} vì không đọc được nội dung.")
                continue
            
            # 2. Tách đoạn văn
            docs = cv_processor.chunk_text(text, source=filename)
            print(f"   📝 Tạo {len(docs)} chunks")
            
            # 3. Tạo embedding
            embeddings = cv_processor.get_embeddings([doc.page_content for doc in docs])
            print(f"   🔗 Tạo embeddings thành công")
            
            # 4. Thêm vào FAISS
            vector_store.add_documents(docs, embeddings)
            processed_count += 1
            print(f"   ✅ Hoàn thành xử lý {filename}")
            
        except Exception as e:
            print(f"   ❌ Lỗi khi xử lý {filename}: {str(e)}")
            continue
    
    if processed_count > 0:
        # 5. Lưu FAISS và metadata
        print(f"💾 Lưu vector store...")
        vector_store.save(index_path, metadata_path)
        print(f"✅ Hoàn thành! Đã xử lý {processed_count}/{len(pdf_files)} file CV")
        return True
    else:
        print("❌ Không có file nào được xử lý thành công")
        return False

def test_chat():
    """Test chức năng chat"""
    chatbot = CVChatBot()
    
    # Hiển thị thông tin CV
    summary = chatbot.get_cv_summary()
    print(f"\n📊 Thông tin Vector Store:")
    print(f"   - Tổng số CV: {summary['total_cvs']}")
    print(f"   - Tổng số chunks: {summary['total_chunks']}")
    print(f"   - Files: {summary['cv_files']}")
    
    # Test queries
    test_queries = [
        "Có ứng viên nào có kinh nghiệm về Python không?",
        "Tìm ứng viên có kỹ năng quản lý dự án",
        "Ai có học vấn về công nghệ thông tin?",
        "Có ứng viên nào biết về machine learning không?"
    ]
    
    print("\n🧪 Test một số câu hỏi mẫu:")
    for query in test_queries:
        print(f"\n❓ {query}")
        result = chatbot.chat(query, top_k=3)
        print(f"💬 {result['answer'][:200]}...")

def interactive_chat():
    """Chế độ chat tương tác"""
    chatbot = CVChatBot()
    
    print("\n🤖 CV ChatBot - Chế độ tương tác")
    print("Nhập 'quit' hoặc 'exit' để thoát")
    print("Nhập 'info' để xem thông tin vector store")
    print("-" * 50)
    
    while True:
        query = input("\n❓ Bạn: ").strip()
        
        if query.lower() in ['quit', 'exit', 'thoát']:
            print("👋 Tạm biệt!")
            break
        
        if query.lower() == 'info':
            summary = chatbot.get_cv_summary()
            print(f"📊 Thông tin Vector Store:")
            print(f"   - Tổng số CV: {summary['total_cvs']}")
            print(f"   - Files: {summary['cv_files']}")
            continue
        
        if not query:
            continue
        
        print("🤔 Đang suy nghĩ...")
        result = chatbot.chat(query, top_k=5)
        print(f"🤖 Bot: {result['answer']}")

def main():
    parser = argparse.ArgumentParser(description="CV ChatBot System")
    parser.add_argument("--mode", choices=["build", "test", "chat", "ui"], 
                       default="ui", help="Chế độ chạy")
    parser.add_argument("--cv_folder", default="cv", 
                       help="Thư mục chứa CV PDF")
    
    args = parser.parse_args()
    
    if args.mode == "build":
        print("🔨 Chế độ: Xây dựng Vector Store")
        build_vector_store(args.cv_folder)
    
    elif args.mode == "test":
        print("🧪 Chế độ: Test ChatBot")
        test_chat()
    
    elif args.mode == "chat":
        print("💬 Chế độ: Chat tương tác")
        interactive_chat()
    
    elif args.mode == "ui":
        print("🌐 Chế độ: Streamlit UI")
        print("Chạy: streamlit run app.py")
        os.system("streamlit run app.py")

if __name__ == "__main__":
    main()