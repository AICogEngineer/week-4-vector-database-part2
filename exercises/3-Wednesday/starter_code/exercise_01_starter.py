"""
Exercise 01: Local to Cloud Deployment with Pinecone - Starter Code

Build a local-to-cloud deployment workflow using a base class pattern.
You will implement BOTH the base class AND the backend-specific subclasses.

Prerequisites:
- pip install pinecone-client chromadb sentence-transformers
- Set PINECONE_API_KEY environment variable
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
    print("[Warning] pinecone-client not installed")

# ============================================================================
# CONFIGURATION
# ============================================================================

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
# TODO: IMPLEMENT BASE CLASS
# ============================================================================

class VectorStoreBase(ABC):
    """
    Base class with shared logic for all vector stores.
    
    TODO: Implement the shared methods that DON'T change between backends.
    The abstract methods are already defined - subclasses implement those.
    """
    
    def __init__(self):
        """
        TODO: Initialize the embedder and call _initialize().
        
        1. Create self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        2. Call self._initialize()
        """
        pass  # Remove and implement
    
    # ========================
    # ABSTRACT METHODS (subclasses implement these)
    # ========================
    
    @abstractmethod
    def _initialize(self):
        """Initialize the backend connection."""
        pass
    
    @abstractmethod
    def _store(self, ids: List[str], embeddings: List[List[float]], 
               texts: List[str], metadatas: List[dict]):
        """Store vectors in the backend."""
        pass
    
    @abstractmethod
    def _query(self, embedding: List[float], n_results: int) -> List[dict]:
        """Query backend. Return list of {text, metadata, score}."""
        pass
    
    @abstractmethod
    def _get_count(self) -> int:
        """Get total vector count."""
        pass
    
    @abstractmethod
    def _get_all_data(self) -> dict:
        """Export all data: {ids, texts, metadatas, embeddings}."""
        pass
    
    # ========================
    # TODO: IMPLEMENT SHARED METHODS
    # ========================
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        """
        TODO: Generate embeddings for texts.
        
        Use: self.embedder.encode(texts).tolist()
        """
        pass  # Remove and implement
    
    def _generate_ids(self, count: int, prefix: str = "doc") -> List[str]:
        """
        TODO: Generate unique IDs.
        
        Return: [f"{prefix}_{i}" for i in range(count)]
        """
        pass  # Remove and implement
    
    def add_documents(self, texts: List[str], metadatas: List[dict]) -> int:
        """
        TODO: Add documents using shared workflow.
        
        1. Generate embeddings with self._embed(texts)
        2. Generate IDs with self._generate_ids(len(texts))
        3. Store with self._store(ids, embeddings, texts, metadatas)
        4. Return count
        """
        pass  # Remove and implement
    
    def search(self, query: str, n_results: int = 5) -> List[dict]:
        """
        TODO: Search using shared workflow.
        
        1. Embed query: self._embed([query])[0]
        2. Query backend: self._query(embedding, n_results)
        3. Return results
        """
        pass  # Remove and implement
    
    def count(self) -> int:
        """Get count - delegates to _get_count()."""
        return self._get_count()
    
    def get_all(self) -> dict:
        """Get all data - delegates to _get_all_data()."""
        return self._get_all_data()


# ============================================================================
# TODO: IMPLEMENT LOCAL VECTOR STORE
# ============================================================================

class LocalVectorStore(VectorStoreBase):
    """Local Chroma backend - implement the abstract methods."""
    
    def __init__(self, collection_name: str = "local_test", persist_dir: Path = LOCAL_DATA_DIR):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        super().__init__()  # This calls _initialize()
    
    def _initialize(self):
        """
        TODO: Initialize Chroma.
        
        1. Create persist_dir with mkdir(parents=True, exist_ok=True)
        2. Create PersistentClient: chromadb.PersistentClient(path=str(persist_dir))
        3. Get/create collection
        """
        pass  # Remove and implement
    
    def _store(self, ids, embeddings, texts, metadatas):
        """
        TODO: Store in Chroma.
        
        self.collection.add(ids=ids, embeddings=embeddings, 
                            documents=texts, metadatas=metadatas)
        """
        pass  # Remove and implement
    
    def _query(self, embedding, n_results):
        """
        TODO: Query Chroma.
        
        1. results = self.collection.query(query_embeddings=[embedding], 
                                           n_results=n_results,
                                           include=["documents", "metadatas", "distances"])
        2. Return list of {text, metadata, score} where score = 1 - distance
        """
        pass  # Remove and implement
    
    def _get_count(self):
        return self.collection.count() if self.collection else 0
    
    def _get_all_data(self):
        """
        TODO: Export from Chroma.
        
        data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        Return {ids, texts, metadatas, embeddings}
        """
        pass  # Remove and implement


# ============================================================================
# TODO: IMPLEMENT CLOUD VECTOR STORE
# ============================================================================

class CloudVectorStore(VectorStoreBase):
    """Pinecone cloud backend - implement the abstract methods."""
    
    def __init__(self, index_name: str = PINECONE_INDEX_NAME):
        self.index_name = index_name
        self.pc = None
        self.index = None
        super().__init__()
    
    def _initialize(self):
        """
        TODO: Initialize Pinecone.
        
        1. Get API key: os.environ.get("PINECONE_API_KEY")
        2. Create client: Pinecone(api_key=api_key)
        3. Create index if needed (dimension=384, metric=cosine, serverless)
        4. Connect: self.index = self.pc.Index(index_name)
        """
        pass  # Remove and implement
    
    def _store(self, ids, embeddings, texts, metadatas):
        """
        TODO: Store in Pinecone.
        
        1. Build vectors: [{id, values, metadata}] - include text in metadata!
        2. Batch upsert: for i in range(0, len(vectors), 100): self.index.upsert(...)
        """
        pass  # Remove and implement
    
    def _query(self, embedding, n_results):
        """
        TODO: Query Pinecone.
        
        1. results = self.index.query(vector=embedding, top_k=n_results, include_metadata=True)
        2. Return list of {text, metadata, score}
        """
        pass  # Remove and implement
    
    def _get_count(self):
        return self.index.describe_index_stats().total_vector_count if self.index else 0
    
    def _get_all_data(self):
        return {"ids": [], "texts": [], "metadatas": [], "embeddings": []}


# ============================================================================
# HELPERS (PROVIDED)
# ============================================================================

def migrate_to_cloud(local_store, cloud_store):
    data = local_store.get_all()
    if data.get("texts"):
        cloud_store.add_documents(data["texts"], data["metadatas"])
    return {"local_count": len(data.get("texts", [])), "cloud_count": cloud_store.count()}

def verify_migration(local, cloud, queries):
    for q in queries:
        l, c = local.search(q, 1), cloud.search(q, 1)
        if l and c and l[0].get("text") != c[0].get("text"):
            return False
    return True

def create_vector_store(env="local"):
    return LocalVectorStore() if env == "local" else CloudVectorStore()


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_tests():
    print("=" * 60)
    print("Exercise 01: Implement Base Class + Backends")
    print("=" * 60)
    
    shutil.rmtree(LOCAL_DATA_DIR, ignore_errors=True)
    
    # Test base class and local store
    print("\n=== Phase 1: Local ===\n")
    try:
        local = LocalVectorStore()
        if local.collection is None:
            print("[TODO] LocalVectorStore._initialize not implemented")
            return
        
        count = local.add_documents(
            [d["text"] for d in SAMPLE_DOCS],
            [{"category": d["category"]} for d in SAMPLE_DOCS]
        )
        if count is None:
            print("[TODO] VectorStoreBase.add_documents not implemented")
            return
        print(f"[OK] Added {count} docs")
        
        results = local.search("machine learning")
        if not results:
            print("[TODO] VectorStoreBase.search not implemented")
            return
        print(f"[OK] Search: {results[0]['text'][:40]}...")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    # Test cloud
    print("\n=== Phase 2: Cloud ===\n")
    if not os.environ.get("PINECONE_API_KEY"):
        print("[SKIP] No PINECONE_API_KEY set")
        shutil.rmtree(LOCAL_DATA_DIR, ignore_errors=True)
        print("\n[DONE] Local tests passed!")
        return
    
    try:
        cloud = CloudVectorStore()
        if cloud.index is None:
            print("[TODO] CloudVectorStore._initialize not implemented")
            return
        print("[OK] Pinecone connected")
        
        stats = migrate_to_cloud(local, cloud)
        print(f"[OK] Migrated {stats['local_count']} docs")
        
        ok = verify_migration(local, cloud, ["machine learning"])
        print(f"[OK] Verified: {'PASS' if ok else 'FAIL'}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
    
    shutil.rmtree(LOCAL_DATA_DIR, ignore_errors=True)
    print("\n[DONE]")


if __name__ == "__main__":
    run_tests()
