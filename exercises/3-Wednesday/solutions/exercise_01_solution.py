"""
Exercise 01: Local to Cloud Deployment - SOLUTION

Uses base class pattern to share embedding/search logic between backends.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict
from pathlib import Path
import shutil
import chromadb
from sentence_transformers import SentenceTransformer

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

LOCAL_DATA_DIR = Path("./chroma_local_test")
PINECONE_INDEX_NAME = "week4-exercise"
EMBEDDING_DIMENSION = 384

SAMPLE_DOCS = [
    {"text": "Python is the most popular language for machine learning.", "category": "ml"},
    {"text": "TensorFlow and PyTorch are leading deep learning frameworks.", "category": "ml"},
    {"text": "React and Vue are popular JavaScript frontend frameworks.", "category": "web"},
    {"text": "Docker containers simplify application deployment.", "category": "devops"},
    {"text": "Kubernetes orchestrates containerized applications at scale.", "category": "devops"},
    {"text": "PostgreSQL is a powerful open-source relational database.", "category": "database"},
    {"text": "Vector databases store embeddings for similarity search.", "category": "database"},
    {"text": "RAG combines retrieval and generation for better AI responses.", "category": "ml"},
]


# ============================================================================
# BASE CLASS - SHARED LOGIC
# ============================================================================

class VectorStoreBase(ABC):
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self._initialize()
    
    @abstractmethod
    def _initialize(self): pass
    
    @abstractmethod
    def _store(self, ids, embeddings, texts, metadatas): pass
    
    @abstractmethod
    def _query(self, embedding, n_results) -> List[dict]: pass
    
    @abstractmethod
    def _get_count(self) -> int: pass
    
    @abstractmethod
    def _get_all_data(self) -> dict: pass
    
    # Shared methods
    def _embed(self, texts):
        return self.embedder.encode(texts).tolist()
    
    def _generate_ids(self, count, prefix="doc"):
        return [f"{prefix}_{i}" for i in range(count)]
    
    def add_documents(self, texts, metadatas):
        embeddings = self._embed(texts)
        ids = self._generate_ids(len(texts))
        self._store(ids, embeddings, texts, metadatas)
        return len(texts)
    
    def search(self, query, n_results=5):
        return self._query(self._embed([query])[0], n_results)
    
    def count(self):
        return self._get_count()
    
    def get_all(self):
        return self._get_all_data()


# ============================================================================
# LOCAL VECTOR STORE - SOLUTION
# ============================================================================

class LocalVectorStore(VectorStoreBase):
    def __init__(self, collection_name="local_test", persist_dir=LOCAL_DATA_DIR):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        super().__init__()
    
    def _initialize(self):
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(self.collection_name)
    
    def _store(self, ids, embeddings, texts, metadatas):
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    
    def _query(self, embedding, n_results):
        results = self.collection.query(
            query_embeddings=[embedding], n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {"text": results["documents"][0][i], 
             "metadata": results["metadatas"][0][i],
             "score": 1 - results["distances"][0][i]}
            for i in range(len(results["documents"][0]))
        ]
    
    def _get_count(self):
        return self.collection.count() if self.collection else 0
    
    def _get_all_data(self):
        data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        return {"ids": data["ids"], "texts": data["documents"], 
                "metadatas": data["metadatas"], "embeddings": data["embeddings"]}


# ============================================================================
# CLOUD VECTOR STORE - SOLUTION
# ============================================================================

class CloudVectorStore(VectorStoreBase):
    def __init__(self, index_name=PINECONE_INDEX_NAME):
        self.index_name = index_name
        self.pc = None
        self.index = None
        super().__init__()
    
    def _initialize(self):
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not set")
        
        self.pc = Pinecone(api_key=api_key)
        existing = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name, dimension=EMBEDDING_DIMENSION, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            import time
            while not self.pc.describe_index(self.index_name).status.ready:
                time.sleep(1)
        
        self.index = self.pc.Index(self.index_name)
    
    def _store(self, ids, embeddings, texts, metadatas):
        vectors = [{"id": ids[i], "values": embeddings[i], 
                    "metadata": {**metadatas[i], "text": texts[i]}}
                   for i in range(len(ids))]
        
        for i in range(0, len(vectors), 100):
            self.index.upsert(vectors=vectors[i:i+100])
    
    def _query(self, embedding, n_results):
        results = self.index.query(vector=embedding, top_k=n_results, include_metadata=True)
        return [{"text": m.metadata.get("text", ""), 
                 "metadata": {k: v for k, v in m.metadata.items() if k != "text"},
                 "score": m.score}
                for m in results.matches]
    
    def _get_count(self):
        return self.index.describe_index_stats().total_vector_count if self.index else 0
    
    def _get_all_data(self):
        return {"ids": [], "texts": [], "metadatas": [], "embeddings": []}


# ============================================================================
# HELPERS
# ============================================================================

def migrate_to_cloud(local, cloud):
    data = local.get_all()
    if data.get("texts"):
        cloud.add_documents(data["texts"], data["metadatas"])
    return {"local_count": len(data.get("texts", [])), "cloud_count": cloud.count()}

def verify_migration(local, cloud, queries):
    for q in queries:
        l, c = local.search(q, 1), cloud.search(q, 1)
        if l and c and l[0].get("text") != c[0].get("text"):
            return False
    return True

def create_vector_store(env="local"):
    return LocalVectorStore() if env == "local" else CloudVectorStore()


# ============================================================================
# TEST
# ============================================================================

def run_tests():
    print("=" * 60)
    print("Exercise 01 SOLUTION: Base Class Pattern")
    print("=" * 60)
    
    shutil.rmtree(LOCAL_DATA_DIR, ignore_errors=True)
    
    local = LocalVectorStore()
    local.add_documents([d["text"] for d in SAMPLE_DOCS], [{"category": d["category"]} for d in SAMPLE_DOCS])
    print(f"[OK] Local: {local.count()} docs, search: {local.search('ml')[0]['text'][:40]}...")
    
    if os.environ.get("PINECONE_API_KEY"):
        cloud = CloudVectorStore()
        migrate_to_cloud(local, cloud)
        print(f"[OK] Cloud: {cloud.count()} docs, verified: {verify_migration(local, cloud, ['ml'])}")
    
    shutil.rmtree(LOCAL_DATA_DIR, ignore_errors=True)
    print("[DONE]")

if __name__ == "__main__":
    run_tests()
