from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import shutil
from model_infer import CVChatBot
from process_store_class import CVProcessor, FAISSVectorStore

app = FastAPI(title="CV ChatBot API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot = None

def get_chatbot():
    global chatbot
    if chatbot is None:
        chatbot = CVChatBot()
    return chatbot

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    context: str
    sources: List[Dict[str, Any]]

class CVSummaryResponse(BaseModel):
    total_cvs: int
    cv_files: List[str]
    total_chunks: int

@app.get("/")
async def root():
    return {"message": "CV ChatBot API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint"""
    try:
        bot = get_chatbot()
        result = bot.chat(request.query, request.top_k)
        
        return ChatResponse(
            answer=result["answer"],
            context=result["context"], 
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/cv-summary", response_model=CVSummaryResponse)
async def get_cv_summary():
    """Get CV summary information"""
    try:
        bot = get_chatbot()
        summary = bot.get_cv_summary()
        
        return CVSummaryResponse(
            total_cvs=summary["total_cvs"],
            cv_files=summary["cv_files"],
            total_chunks=summary["total_chunks"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting CV summary: {str(e)}")

@app.post("/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    """Upload new CV file"""
    try:
        # Kiểm tra file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Tạo thư mục cv nếu chưa có
        cv_folder = "cv"
        os.makedirs(cv_folder, exist_ok=True)
        
        # Lưu file
        file_path = os.path.join(cv_folder, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process file và update vector store
        cv_processor = CVProcessor()
        
        # Parse CV
        text = cv_processor.parse_cv_to_markdown(file_path)
        if not text.strip():
            os.remove(file_path)  # Xóa file nếu không đọc được
            raise HTTPException(status_code=400, detail="Cannot parse CV content")
        
        # Chunk text
        docs = cv_processor.chunk_text(text, source=file.filename)
        
        # Generate embeddings
        embeddings = cv_processor.get_embeddings([doc.page_content for doc in docs])
        
        # Update vector store
        bot = get_chatbot()
        bot.vector_store.add_documents(docs, embeddings)
        bot.vector_store.save("cv_index.faiss", "cv_metadata.json")
        
        return {
            "message": f"Successfully uploaded and processed {file.filename}",
            "chunks_created": len(docs)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading CV: {str(e)}")

@app.delete("/cv/{filename}")
async def delete_cv(filename: str):
    """Delete CV file (note: requires rebuilding vector store)"""
    try:
        file_path = os.path.join("cv", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="CV file not found")
        
        os.remove(file_path)
        
        return {
            "message": f"Successfully deleted {filename}. Please rebuild vector store to update search index."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting CV: {str(e)}")

@app.post("/rebuild-index")
async def rebuild_index():
    """Rebuild vector store from CV folder"""
    try:
        from main import build_vector_store
        
        # Rebuild vector store
        success = build_vector_store()
        
        if success:
            # Reload chatbot
            global chatbot
            chatbot = CVChatBot()
            
            return {"message": "Successfully rebuilt vector store"}
        else:
            raise HTTPException(status_code=500, detail="Failed to rebuild vector store")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")

@app.get("/search")
async def search_cvs(q: str, top_k: int = 5):
    """Direct search endpoint"""
    try:
        bot = get_chatbot()
        context, sources = bot.search_relevant_context(q, top_k)
        
        return {
            "query": q,
            "context": context,
            "sources": sources,
            "total_results": len(sources)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)