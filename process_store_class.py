import os
import pathlib
import json
from typing import List, Dict, Any
import numpy as np

from dotenv import load_dotenv


import faiss
import ollama
from google import genai
from google.genai import types

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from prompts import parser_prompt

load_dotenv()

class CVProcessor:
    def __init__(self, embedding_model: str = "mxbai-embed-large"):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.embedding_model = embedding_model
        self.embedding_dim = 1024
        self.client = genai.Client(api_key=self.google_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def parse_cv_to_markdown(self, filepath: str) -> str:
        """Chuyển CV PDF thành markdown bằng Gemini"""
        try:
            file_path = pathlib.Path(filepath)
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[
                    types.Part.from_bytes(
                        data=file_path.read_bytes(),
                        mime_type='application/pdf',
                    ),
                    parser_prompt
                ],
            )
            return response.text
        except Exception as e:
            print(f"Error parsing CV: {e}")
            return ""

    def chunk_text(self, text: str, source: str) -> List[Document]:
        """Chia text thành nhiều đoạn nhỏ (chunk)"""
        chunks = self.text_splitter.split_text(text)
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_id": i,
                    "chunk_size": len(chunk)
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        return documents

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Sinh embedding từ văn bản bằng Ollama"""
        embeddings = []
        for text in texts:
            try:
                response = ollama.embed(
                    model=self.embedding_model,
                    input=text
                )
                embeddings.append(response['embeddings'][0])
            except Exception as e:
                print(f"Error generating embedding: {e}")
                embeddings.append([0.0] * self.embedding_dim)
        return np.array(embeddings, dtype=np.float32)


class FAISSVectorStore:
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Thêm document và embedding vào FAISS"""
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        for doc in documents:
            self.metadata.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Tìm kiếm văn bản gần giống"""
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "content": self.metadata[idx]["content"],
                    "metadata": self.metadata[idx]["metadata"],
                    "score": float(score)
                })
        return results

    def save(self, index_path: str, metadata_path: str):
        """Lưu index FAISS và metadata"""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str, metadata_path: str) -> bool:
        """Tải lại index FAISS và metadata"""
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            return True
        return False
