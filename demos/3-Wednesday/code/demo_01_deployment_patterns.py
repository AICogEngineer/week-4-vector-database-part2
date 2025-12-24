"""
Demo 01: Local vs Cloud Deployment Patterns (with Pinecone)

This demo shows trainees how to:
1. Set up persistent local storage with Chroma
2. Connect to Pinecone cloud vector database
3. Build abstraction layers for deployment flexibility
4. Compare local vs cloud deployment patterns

Learning Objectives:
- Understand local vs cloud trade-offs
- Implement configuration management patterns
- Build vendor-agnostic vector database interfaces
- Use Pinecone for cloud vector storage

References:
- Written Content: 01-local-vs-cloud-storage.md
- Written Content: 02-cloud-deployment-considerations.md

Prerequisites:
- pip install pinecone-client chromadb sentence-transformers
- Set PINECONE_API_KEY environment variable (or set USE_PINECONE=False)
"""

import os
import chromadb
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path
import tempfile
import shutil

# ============================================================================
# PINECONE CONFIGURATION
# ============================================================================
# Set to True if you have a Pinecone API key and want to demo cloud features
# Set to False to run the demo with only local Chroma (shows mock for cloud)
USE_PINECONE = True

if USE_PINECONE:
    try:
        from pinecone import Pinecone, ServerlessSpec
        
        # Initialize Pinecone client
        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        if PINECONE_API_KEY:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            print("[Pinecone] Successfully initialized Pinecone client")
        else:
            print("[Pinecone] PINECONE_API_KEY not found in environment")
            print("[Pinecone] Set the environment variable or set USE_PINECONE=False")
            USE_PINECONE = False
    except ImportError:
        print("[Pinecone] pinecone-client not installed. Run: pip install pinecone-client")
        print("[Pinecone] Falling back to mock cloud implementation")
        USE_PINECONE = False

# ============================================================================
# PART 1: The Deployment Decision
# ============================================================================

print("\n" + "=" * 70)
print("PART 1: The Deployment Decision")
print("=" * 70)

print("""
WHEN TO USE EACH DEPLOYMENT MODEL:
─────────────────────────────────────────────────────────────────────

LOCAL (Chroma)                │  CLOUD (Pinecone)
─────────────────────────────│───────────────────────────────────────
✓ Prototyping & Development  │  ✓ Production at Scale
✓ Privacy-sensitive data     │  ✓ Multi-region deployment
✓ Low-latency requirements   │  ✓ Zero operational overhead
✓ Offline/Edge deployments   │  ✓ Automatic scaling
✓ Cost-sensitive projects    │  ✓ Built-in high availability
                             │
Challenges:                  │  Challenges:
• Manual scaling             │  • Vendor lock-in risk
• Backup management          │  • Network latency
• No built-in HA             │  • Ongoing costs
─────────────────────────────────────────────────────────────────────

DECISION FACTORS:
1. Scale        → Million+ vectors? Consider Pinecone
2. Privacy      → Sensitive data? Consider local Chroma
3. Team         → Limited ops? Consider Pinecone
4. Budget       → Tight budget? Consider local Chroma
5. Latency      → Sub-10ms? Consider local Chroma
""")

# ============================================================================
# PART 2: Local Persistent Storage (Chroma)
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Local Persistent Storage (Chroma)")
print("=" * 70)

# Create a temporary directory for demo purposes
DEMO_DB_PATH = Path(tempfile.mkdtemp()) / "chroma_db"

print(f"\n[Step 1] Setting up persistent storage at: {DEMO_DB_PATH}")

# Simple persistent client
print("\nCreating PersistentClient:")
print("-" * 50)

client = chromadb.PersistentClient(path=str(DEMO_DB_PATH))
print(f"  ✓ Created persistent client")
print(f"  ✓ Data stored at: {DEMO_DB_PATH}")

# Create a collection
collection = client.get_or_create_collection(
    name="demo_collection",
    metadata={"hnsw:space": "cosine"}
)
print(f"  ✓ Collection 'demo_collection' created")

# Add some data
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "Machine learning fundamentals",
        "Deep learning with neural networks",
        "Natural language processing basics"
    ],
    metadatas=[
        {"category": "ml"},
        {"category": "dl"},
        {"category": "nlp"}
    ]
)
print(f"  ✓ Added 3 documents")

# Verify persistence
print("\n[Step 2] Verifying persistence:")
print("-" * 50)

# Create a new client pointing to same path
client2 = chromadb.PersistentClient(path=str(DEMO_DB_PATH))
collection2 = client2.get_collection("demo_collection")
count = collection2.count()
print(f"  ✓ Reopened database, found {count} documents")

# ============================================================================
# PART 3: Cloud Storage (Pinecone)
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Cloud Storage (Pinecone)")
print("=" * 70)

if USE_PINECONE:
    print("\n[Pinecone Demo - LIVE CONNECTION]")
    print("-" * 50)
    
    # Index configuration
    INDEX_NAME = "week4-demo"
    DIMENSION = 384  # all-MiniLM-L6-v2 dimension
    
    # Check if index exists, create if not
    print(f"\n  Checking for index '{INDEX_NAME}'...")
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"  Creating new serverless index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"  ✓ Created index '{INDEX_NAME}'")
        print("  ⏳ Waiting for index to be ready...")
        
        # Wait for index to be ready
        import time
        while True:
            desc = pc.describe_index(INDEX_NAME)
            if desc.status.ready:
                break
            time.sleep(1)
        print("  ✓ Index is ready!")
    else:
        print(f"  ✓ Index '{INDEX_NAME}' already exists")
    
    # Connect to the index
    index = pc.Index(INDEX_NAME)
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"\n  Index Statistics:")
    print(f"    Total vectors: {stats.total_vector_count}")
    print(f"    Dimension: {DIMENSION}")
    print(f"    Metric: cosine")
    
    # Add sample vectors
    print("\n  Adding sample vectors...")
    
    # Sample embeddings (in production, use sentence-transformers)
    import random
    sample_vectors = [
        {"id": "pinecone-doc1", "values": [random.random() for _ in range(DIMENSION)], 
         "metadata": {"text": "Machine learning intro", "category": "ml"}},
        {"id": "pinecone-doc2", "values": [random.random() for _ in range(DIMENSION)], 
         "metadata": {"text": "Neural network basics", "category": "dl"}},
        {"id": "pinecone-doc3", "values": [random.random() for _ in range(DIMENSION)], 
         "metadata": {"text": "NLP fundamentals", "category": "nlp"}},
    ]
    
    index.upsert(vectors=sample_vectors)
    print(f"  ✓ Upserted 3 vectors")
    
    # Query the index
    print("\n  Querying index...")
    query_vector = [random.random() for _ in range(DIMENSION)]
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    
    print(f"  ✓ Query returned {len(results.matches)} results:")
    for match in results.matches:
        print(f"    - {match.id}: score={match.score:.4f}, category={match.metadata.get('category', 'N/A')}")
    
    # Show Pinecone dashboard info
    print(f"""
  ──────────────────────────────────────────────────────────────
  📊 View your index in the Pinecone Console:
     https://app.pinecone.io
     
  Index: {INDEX_NAME}
  Region: us-east-1 (AWS)
  ──────────────────────────────────────────────────────────────
    """)

else:
    print("\n[Pinecone Demo - MOCK MODE]")
    print("-" * 50)
    print("  ⚠️  Pinecone not configured. Using mock implementation.")
    print("  To enable: Set PINECONE_API_KEY environment variable")
    print("")
    print("  Simulating cloud operations:")
    print("    → Creating index 'week4-demo'... ✓")
    print("    → Upserting 3 vectors... ✓")
    print("    → Querying with top_k=3... ✓")
    print("    → Results: doc1 (0.95), doc2 (0.87), doc3 (0.72)")

# ============================================================================
# PART 4: Abstraction Layer Pattern
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Abstraction Layer Pattern")
print("=" * 70)

print("""
WHY ABSTRACT THE VECTOR DATABASE?
─────────────────────────────────
1. Swap backends without changing application code
2. Test with local Chroma, deploy with cloud Pinecone
3. Multi-vendor strategy (avoid lock-in)
4. Consistent interface across environments
""")


class VectorDBClient(ABC):
    """
    Abstract interface for vector database operations.
    
    Implement this interface for each backend you support.
    Application code uses this interface, not vendor-specific APIs.
    """
    
    @abstractmethod
    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None
    ) -> None:
        """Add documents with embeddings to the database."""
        pass
    
    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Query documents by embedding similarity."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return total document count."""
        pass


class ChromaDBClient(VectorDBClient):
    """
    Chroma implementation of VectorDBClient.
    
    Supports both in-memory and persistent modes.
    """
    
    def __init__(self, persist_path: Optional[str] = None, collection_name: str = "default"):
        self.collection_name = collection_name
        
        # Initialize client based on persistence
        if persist_path:
            self.client = chromadb.PersistentClient(path=persist_path)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None
    ) -> None:
        """Add documents to Chroma collection."""
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Query Chroma collection."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter,
            include=["metadatas", "distances"]
        )
    
    def delete(self, ids: List[str]) -> None:
        """Delete from Chroma collection."""
        self.collection.delete(ids=ids)
    
    def count(self) -> int:
        """Count documents in collection."""
        return self.collection.count()


class PineconeClient(VectorDBClient):
    """
    Pinecone implementation of VectorDBClient.
    
    Connects to Pinecone serverless for cloud vector storage.
    """
    
    def __init__(self, api_key: str, index_name: str, dimension: int = 384):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        
        # Get or create index
        existing = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        self.index = self.pc.Index(index_name)
    
    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None
    ) -> None:
        """Add documents to Pinecone index."""
        vectors = []
        for i, id_ in enumerate(ids):
            vec = {"id": id_, "values": embeddings[i]}
            if metadatas:
                vec["metadata"] = metadatas[i]
            vectors.append(vec)
        
        self.index.upsert(vectors=vectors)
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Query Pinecone index."""
        results = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            filter=filter,
            include_metadata=True
        )
        
        # Convert to common format
        return {
            "ids": [[m.id for m in results.matches]],
            "distances": [[1 - m.score for m in results.matches]],  # Convert similarity to distance
            "metadatas": [[m.metadata for m in results.matches]]
        }
    
    def delete(self, ids: List[str]) -> None:
        """Delete from Pinecone index."""
        self.index.delete(ids=ids)
    
    def count(self) -> int:
        """Count vectors in index."""
        stats = self.index.describe_index_stats()
        return stats.total_vector_count


@dataclass
class VectorDBConfig:
    """Configuration for vector database connection."""
    
    # Deployment type: 'local', 'local_persistent', 'pinecone'
    deployment: str = "local"
    
    # Local settings
    persist_path: Optional[str] = None
    collection_name: str = "default"
    
    # Pinecone settings
    pinecone_api_key: Optional[str] = None
    pinecone_index: Optional[str] = None
    dimension: int = 384


def create_vector_client(config: VectorDBConfig) -> VectorDBClient:
    """
    Factory function to create appropriate client based on config.
    
    This is the entry point for application code.
    """
    if config.deployment == "local":
        return ChromaDBClient(collection_name=config.collection_name)
    elif config.deployment == "local_persistent":
        return ChromaDBClient(persist_path=config.persist_path, collection_name=config.collection_name)
    elif config.deployment == "pinecone":
        if not config.pinecone_api_key:
            raise ValueError("pinecone_api_key required for Pinecone deployment")
        return PineconeClient(
            api_key=config.pinecone_api_key,
            index_name=config.pinecone_index or "default",
            dimension=config.dimension
        )
    else:
        raise ValueError(f"Unknown deployment type: {config.deployment}")


print("\nDemonstrating the abstraction layer:")
print("-" * 50)

# Local development
local_config = VectorDBConfig(deployment="local", collection_name="abstraction_demo")
local_client = create_vector_client(local_config)
print(f"\n  Local client created: {type(local_client).__name__}")

# Add test data
local_client.add_documents(
    ids=["test1", "test2"],
    embeddings=[[0.1] * 384, [0.2] * 384],
    metadatas=[{"type": "test"}, {"type": "test"}]
)
print(f"  Document count: {local_client.count()}")

if USE_PINECONE:
    # Pinecone cloud
    cloud_config = VectorDBConfig(
        deployment="pinecone",
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_index="week4-demo"
    )
    cloud_client = create_vector_client(cloud_config)
    print(f"\n  Pinecone client created: {type(cloud_client).__name__}")
    print(f"  Vector count: {cloud_client.count()}")

print("\n  ✓ Same interface works for both backends!")
print("  ✓ Application code doesn't change between local and cloud!")

# ============================================================================
# PART 5: Comparison Summary
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Chroma vs Pinecone Comparison")
print("=" * 70)

print("""
┌────────────────────┬─────────────────────────┬─────────────────────────┐
│ Feature            │ Chroma (Local)          │ Pinecone (Cloud)        │
├────────────────────┼─────────────────────────┼─────────────────────────┤
│ Setup              │ pip install chromadb    │ pip install pinecone    │
│                    │ No account needed       │ Account + API key       │
├────────────────────┼─────────────────────────┼─────────────────────────┤
│ Scaling            │ Single-node only        │ Auto-scales             │
│                    │ Manual sharding         │ Serverless option       │
├────────────────────┼─────────────────────────┼─────────────────────────┤
│ Latency            │ ~1-5ms (in-process)     │ ~20-100ms (network)     │
├────────────────────┼─────────────────────────┼─────────────────────────┤
│ Cost               │ Free (self-hosted)      │ Free tier + usage-based │
│                    │ Hardware/ops costs      │ ~$70/mo per 1M vectors  │
├────────────────────┼─────────────────────────┼─────────────────────────┤
│ Best For           │ Development, POCs       │ Production at scale     │
│                    │ Privacy-sensitive       │ Managed operations      │
└────────────────────┴─────────────────────────┴─────────────────────────┘

RECOMMENDATION:
  • Start with Chroma locally for development
  • Migrate to Pinecone when you need scale + managed ops
  • Use the abstraction layer to make migration painless
""")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO 1 COMPLETE: Deployment Patterns")
print("=" * 70)

print("""
Key Takeaways:

1. LOCAL DEPLOYMENT (Chroma)
   - Use for development and privacy-sensitive workloads
   - PersistentClient for data durability
   - Simple, no ops overhead, sub-5ms latency

2. CLOUD DEPLOYMENT (Pinecone)
   - Use for scale and managed operations
   - Serverless = no infrastructure management
   - Built-in replication and high availability

3. ABSTRACTION LAYER
   - VectorDBClient interface
   - Same code, different backends
   - Enables testing and migration

4. MIGRATION PATH
   - Develop with Chroma
   - Test with Pinecone free tier
   - Deploy to Pinecone production

Coming Next: Demo 2 covers performance benchmarking!
""")

# Cleanup local files
shutil.rmtree(DEMO_DB_PATH.parent, ignore_errors=True)
print(f"[Cleanup] Removed local demo database")

if USE_PINECONE:
    print(f"[Note] Pinecone index '{INDEX_NAME}' was NOT deleted (for inspection)")
    print(f"       Delete manually in console if desired: https://app.pinecone.io")

# ============================================================================
# INSTRUCTOR NOTES
# ============================================================================

print("\n" + "=" * 70)
print("INSTRUCTOR NOTES")
print("=" * 70)

print("""
Discussion Questions:
1. "What deployment would you choose for a healthcare application?"
2. "How would you handle a customer requiring both cloud and on-prem?"
3. "What's your backup strategy for local vs cloud deployments?"

Setup for Live Demo:
1. Create Pinecone account: https://www.pinecone.io (free tier available)
2. Set environment variable: PINECONE_API_KEY=your-key
3. Run demo with USE_PINECONE=True

Cost Notes (Pinecone):
- Free tier: 1 index, 100K vectors
- Starter: ~$70/month per 1M vectors
- Serverless: Pay only for what you use

Common Confusions:
- "Is local always faster?" → Yes for single-node, no at scale
- "Is Pinecone expensive?" → Free tier for learning, competitive for production
- "Do I need the abstraction layer?" → Not for prototypes, yes for production

If Running Short on Time:
- Skip live Pinecone demo, show mock output
- Focus on abstraction layer concept

If Trainees Are Advanced:
- Discuss Pinecone namespaces for multi-tenancy
- Cover metadata filtering differences
- Explore hybrid search (dense + sparse)
""")

print("\n" + "=" * 70)
