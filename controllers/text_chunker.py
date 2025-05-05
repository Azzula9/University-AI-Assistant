from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict
import os
import json
from datetime import datetime

class TextChunker:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )
    
    def chunk_text(self, raw_text: str) -> List[Dict]:
        """Split text into semantic chunks without technical metadata"""
        splitter = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type="gradient",
            breakpoint_threshold_amount=90
        )
        
        chunks = splitter.create_documents([raw_text])
        return [
            {
                "content": chunk.page_content.strip(),
                "chunk_num": i+1  # Only keep sequential numbering
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def save_chunks(self, chunks: List[Dict], original_filename: str) -> str:
        """Save simplified chunks to JSON"""
        os.makedirs("assets/chunks", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(original_filename)[0]
        output_path = os.path.join(
            "assets/chunks",
            f"chunks_{timestamp}_{base_name}.json"
        )
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
            
        return output_path