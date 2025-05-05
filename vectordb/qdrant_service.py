from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from config.qdrant_config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME
from config.cohere_config import COHERE_API_KEY
from langchain_cohere import CohereEmbeddings  
import os
from qdrant_client.http import models

class VectorDB:
    def __init__(self):
        self.embeddings = CohereEmbeddings(
            model="embed-english-v3.0", 
            cohere_api_key=COHERE_API_KEY
        )
        self.client = QdrantClient(
              url=QDRANT_URL, 
            api_key=QDRANT_API_KEY,
            https=True,
            timeout=30
 )
        self._ensure_collection()
        
    def _ensure_collection(self):
        
        try:
            collection_info = self.client.get_collection(COLLECTION_NAME)
            if collection_info.config.params.vectors.size != 1024:
                raise ValueError("Existing collection has wrong dimensions")
        except Exception:
            
            self.client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=1024, 
                    distance=models.Distance.COSINE
                )
            )

    
    def as_retriever(self):
        return self.client.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
    def store_chunks(self, chunks: list):
        """Store chunks in Qdrant Cloud"""
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [{"chunk_num": chunk["chunk_num"]} for chunk in chunks]

        Qdrant.from_texts(
            texts=documents,
            embedding=self.embeddings,
            metadatas=metadatas,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
            prefer_grpc=True,
            https=True  
        )
        return True