import os
from typing import List, Dict, Any, Tuple
import json
from google import genai
from google.genai import types
from process_store_class import CVProcessor, FAISSVectorStore
from prompts import system_prompt, get_answer_prompt



class CVChatBot:
    def __init__(self, 
                 index_path: str = "cv_index.faiss",
                 metadata_path: str = "cv_metadata.json",
                 model_name: str = "gemini-2.0-flash-exp"):
        
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.google_api_key)
        self.model_name = model_name
        
        # Khởi tạo processor và vector store
        self.cv_processor = CVProcessor()
        self.vector_store = FAISSVectorStore()
        
        # Load vector store nếu có
        if not self.vector_store.load(index_path, metadata_path):
            print("⚠️ Không tìm thấy vector store. Vui lòng chạy script embedding trước.")
    
    def search_relevant_context(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """Tìm kiếm context liên quan từ vector store"""
        try:
            # Tạo embedding cho query
            query_embedding = self.cv_processor.get_embeddings([query])
            
            # Tìm kiếm
            results = self.vector_store.search(query_embedding, k=top_k)
            
            # Tạo context string
            context_parts = []
            for i, result in enumerate(results, 1):
                content = result['content']
                score = result['score']
                
                context_parts.append(f"(Score: {score:.3f}):\n{content}\n")
            
            context = "\n".join(context_parts)
            return context, results
            
        except Exception as e:
            print(f"Error searching context: {e}")
            return "", []
    
    def generate_answer(self, query: str, context: str) -> str:
        """Sinh câu trả lời từ Gemini"""
        try:
            prompt = get_answer_prompt(query, context)
            
            # Try different approaches based on API version
            try:
                # Method 1: Direct string content
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=f"{system_prompt}\n\n{prompt}",
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=2048,
                    )
                )
            except Exception:
                # Method 2: Using Part with text parameter
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Part(text=system_prompt),
                        types.Part(text=prompt)
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=2048,
                    )
                )
            
            return response.text
            
        except Exception as e:
            return f"Lỗi khi sinh câu trả lời: {str(e)}"
    
    def chat(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Main chat function"""
        if not query.strip():
            return {
                "answer": "Vui lòng nhập câu hỏi.",
                "context": "",
                "sources": []
            }
        
        # Tìm kiếm context
        context, sources = self.search_relevant_context(query, top_k)
        
        if not context:
            return {
                "answer": "Không tìm thấy thông tin liên quan trong các CV.",
                "context": "",
                "sources": []
            }
        
        # Sinh câu trả lời
        answer = self.generate_answer(query, context)
        
        return {
            "answer": answer,
            "context": context,
            "sources": sources
        }
    
    def get_cv_summary(self) -> Dict[str, Any]:
        """Lấy thông tin tổng quan về các CV"""
        if not self.vector_store.metadata:
            return {"total_cvs": 0, "cv_files": []}
        
        cv_files = set()
        for meta in self.vector_store.metadata:
            cv_files.add(meta['metadata']['source'])
        
        return {
            "total_cvs": len(cv_files),
            "cv_files": list(cv_files),
            "total_chunks": len(self.vector_store.metadata)
        }