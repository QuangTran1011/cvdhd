# app.py

import streamlit as st
import os
import pandas as pd
from model_infer import CVChatBot
from main import build_vector_store
import time

# Page config
st.set_page_config(
    page_title="CV ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    
    .source-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chatbot():
    """Load chatbot with caching"""
    return CVChatBot()

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None

def main():
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 CV ChatBot</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        # Load chatbot
        if st.button("🔄 Tải lại ChatBot", type="primary"):
            st.session_state.chatbot = None
            st.cache_resource.clear()
            st.success("✅ Đã tải lại ChatBot!")
        
        # Build vector store
        st.subheader("🔨 Xây dựng Vector Store")
        cv_folder = st.text_input("Thư mục CV:", value="cv")
        
        if st.button("🚀 Xây dựng/Cập nhật Vector Store"):
            with st.spinner("Đang xử lý CV..."):
                success = build_vector_store(cv_folder)
                if success:
                    st.success("✅ Xây dựng Vector Store thành công!")
                    st.session_state.chatbot = None  # Reset chatbot to load new data
                else:
                    st.error("❌ Lỗi khi xây dựng Vector Store")
        
        st.markdown("---")
        
        # CV Upload
        st.subheader("📤 Upload CV mới")
        uploaded_file = st.file_uploader("Chọn file PDF", type="pdf")
        
        if uploaded_file is not None:
            if st.button("📥 Upload và xử lý"):
                try:
                    # Save uploaded file
                    os.makedirs(cv_folder, exist_ok=True)
                    file_path = os.path.join(cv_folder, uploaded_file.name)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.info(f"📁 Đã lưu {uploaded_file.name}")
                    st.info("💡 Hãy nhấn 'Xây dựng/Cập nhật Vector Store' để cập nhật dữ liệu")
                    
                except Exception as e:
                    st.error(f"Lỗi khi upload: {str(e)}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat với CV")
        
        # Load chatbot
        if st.session_state.chatbot is None:
            try:
                with st.spinner("Đang tải ChatBot..."):
                    st.session_state.chatbot = load_chatbot()
                st.success("✅ ChatBot đã sẵn sàng!")
            except Exception as e:
                st.error(f"❌ Lỗi khi tải ChatBot: {str(e)}")
                st.info("💡 Hãy đảm bảo đã xây dựng Vector Store trước")
                return
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 Bạn:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Bot:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_k = st.columns([3, 1])
            
            with col_input:
                user_input = st.text_area(
                    "Nhập câu hỏi của bạn:",
                    placeholder="VD: Tìm ứng viên có kinh nghiệm Python...",
                    height=100,
                    key="user_input"
                )
            
            with col_k:
                top_k = st.number_input("Top K:", min_value=1, max_value=20, value=5)
                submitted = st.form_submit_button("📤 Gửi", type="primary")
        
        if submitted and user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get bot response
            with st.spinner("🤔 Đang suy nghĩ..."):
                try:
                    result = st.session_state.chatbot.chat(user_input, top_k)
                    
                    # Add bot message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result["answer"]
                    })
                    
                    # Store sources for display
                    st.session_state.last_sources = result["sources"]
                    
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
            
            st.rerun()
    
    with col2:
        st.subheader("📊 Thông tin hệ thống")
        
        # Display CV summary
        if st.session_state.chatbot:
            try:
                summary = st.session_state.chatbot.get_cv_summary()
                
                # Metrics
                col_metric1, col_metric2 = st.columns(2)
                with col_metric1:
                    st.metric("📁 Số CV", summary["total_cvs"])
                with col_metric2:
                    st.metric("📝 Chunks", summary["total_chunks"])
                

                
            except Exception as e:
                st.error(f"Không thể tải thông tin: {str(e)}")
        
        # Display sources from last query
        if hasattr(st.session_state, 'last_sources') and st.session_state.last_sources:
            st.subheader("🔍 Nguồn thông tin (Top K Retrieval)")
            
            # Show sources overview first
            st.write(f"Tìm thấy **{len(st.session_state.last_sources)}** nguồn liên quan:")
            
            # Display each source in separate expandable sections
            for i, source in enumerate(st.session_state.last_sources, 1):
                # Source header with key info
                col_source, col_score = st.columns([3, 1])
                with col_source:
                    st.write(f"**📄 {source['metadata']['source']}** - Chunk {source['metadata']['chunk_id']}")
                with col_score:
                    st.metric("Score", f"{source['score']:.3f}")
                
                # Show preview content
                st.markdown(f"""
                <div class="source-card">
                    <div style="font-size: 0.9em; line-height: 1.4; margin-bottom: 10px;">
                        {source['content'][:250]}{'...' if len(source['content']) > 250 else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Full content button
                if len(source['content']) > 250:
                    if st.button(f"📖 Xem toàn bộ nội dung chunk {i}", key=f"show_full_{i}"):
                        st.text_area(
                            f"Nội dung đầy đủ chunk {i}:",
                            source['content'],
                            height=300,
                            key=f"full_content_{i}"
                        )
                
                st.markdown("---")

def show_analytics_page():
    """Analytics page"""
    st.subheader("📈 Phân tích CV")
    
    if st.session_state.chatbot:
        try:
            summary = st.session_state.chatbot.get_cv_summary()
            
            # Create analytics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Tổng số CV", summary["total_cvs"])
                st.metric("Tổng chunks", summary["total_chunks"])
                
                if summary["total_cvs"] > 0:
                    avg_chunks = summary["total_chunks"] / summary["total_cvs"]
                    st.metric("Trung bình chunks/CV", f"{avg_chunks:.1f}")
            
            with col2:
                # CV files distribution
                if summary["cv_files"]:
                    cv_data = []
                    for cv_file in summary["cv_files"]:
                        chunk_count = sum(1 for meta in st.session_state.chatbot.vector_store.metadata 
                                        if meta['metadata']['source'] == cv_file)
                        cv_data.append({"CV": cv_file, "Chunks": chunk_count})
                    
                    df = pd.DataFrame(cv_data)
                    st.bar_chart(df.set_index("CV"))
            
            # Sample queries
            st.subheader("🎯 Câu hỏi mẫu")
            sample_queries = [
                "Tìm ứng viên có kinh nghiệm Python",
                "Ai có kỹ năng quản lý dự án?",
                "Ứng viên nào biết về machine learning?",
                "Tìm người có kinh nghiệm làm việc tại công ty công nghệ",
                "Ai có bằng đại học ngành CNTT?"
            ]
            
            for query in sample_queries:
                if st.button(f"💡 {query}", key=f"sample_{hash(query)}"):
                    # Execute sample query
                    result = st.session_state.chatbot.chat(query, 3)
                    st.success("✅ Kết quả:")
                    st.write(result["answer"])
                    
        except Exception as e:
            st.error(f"Lỗi phân tích: {str(e)}")

def show_settings_page():
    """Settings page"""
    st.subheader("⚙️ Cài đặt nâng cao")
    
    # Model settings
    st.write("🤖 **Cài đặt Model**")
    
    col1, col2 = st.columns(2)
    with col1:
        embedding_model = st.selectbox(
            "Embedding Model:",
            ["mxbai-embed-large", "nomic-embed-text", "all-minilm"],
            index=0
        )
    
    with col2:
        generation_model = st.selectbox(
            "Generation Model:",
            ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
            index=0
        )
    
    # Vector store settings
    st.write("🗄️ **Cài đặt Vector Store**")
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size:", value=1000, min_value=200, max_value=2000)
    
    with col2:
        chunk_overlap = st.number_input("Chunk Overlap:", value=100, min_value=0, max_value=500)
    
    # File management
    st.write("📁 **Quản lý File**")
    
    if st.button("🗑️ Xóa tất cả CV"):
        if st.checkbox("Tôi xác nhận muốn xóa tất cả CV"):
            try:
                import shutil
                if os.path.exists("cv"):
                    shutil.rmtree("cv")
                if os.path.exists("cv_index.faiss"):
                    os.remove("cv_index.faiss")
                if os.path.exists("cv_metadata.json"):
                    os.remove("cv_metadata.json")
                st.success("✅ Đã xóa tất cả dữ liệu CV!")
                st.session_state.chatbot = None
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")
    


# Main app navigation
if __name__ == "__main__":
    # Navigation
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📈 Phân tích", "⚙️ Cài đặt"])
    
    with tab1:
        main()
    
    with tab2:
        show_analytics_page()
    
    with tab3:
        show_settings_page()