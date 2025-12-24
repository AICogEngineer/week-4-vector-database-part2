# Performance Considerations

## Learning Objectives
- Understand indexing strategies for vector databases
- Optimize query performance for production workloads
- Implement batching for efficient operations
- Manage resources effectively for scalable systems

## Why This Matters

A vector database that works in development can fail spectacularly in production. The difference between a 10ms query and a 500ms query compounds across millions of requests. Understanding performance optimization is essential for building responsive, cost-effective RAG systems.

## The Concept

### Performance Factors

```
QUERY PERFORMANCE
─────────────────
Index type (HNSW, IVF, Flat)     40% impact
Vector dimensions                 20% impact
Collection size                   20% impact
Hardware (memory, CPU)            15% impact
Network latency                    5% impact
```

### Indexing Strategies

Vector databases use specialized indexes for fast similarity search. Understanding these helps you tune for your use case.

#### Flat Index (Exact Search)

```python
# No index - brute force comparison
# Compares query to every vector

# Pros: 100% accurate
# Cons: O(n) complexity, slow for large collections
# Use when: < 10,000 vectors, accuracy critical
```

| Collection Size | Query Time |
|----------------|------------|
| 1,000 | ~1ms |
| 10,000 | ~10ms |
| 100,000 | ~100ms |
| 1,000,000 | ~1,000ms |

#### HNSW (Hierarchical Navigable Small World)

The most popular index for vector databases.

```python
import chromadb
from chromadb.config import Settings

# Configure HNSW parameters
client = chromadb.Client(Settings(
    anonymized_telemetry=False
))

collection = client.create_collection(
    name="optimized_collection",
    metadata={
        "hnsw:space": "cosine",        # Distance metric
        "hnsw:construction_ef": 200,    # Build quality (higher = slower build, better recall)
        "hnsw:search_ef": 100,          # Search quality (higher = slower search, better recall)
        "hnsw:M": 16                     # Connections per layer (higher = more memory, better recall)
    }
)
```

**HNSW Parameters Explained**:
- `M`: Connections per node (16 default). Higher = better recall, more memory
- `ef_construction`: Build-time quality (200 default). Higher = slower build, better index
- `ef_search`: Query-time quality. Higher = slower query, better recall

```python
# Trade-off examples
configs = {
    "speed_optimized": {
        "hnsw:M": 8,
        "hnsw:search_ef": 50
    },
    "balanced": {
        "hnsw:M": 16,
        "hnsw:search_ef": 100
    },
    "recall_optimized": {
        "hnsw:M": 32,
        "hnsw:search_ef": 200
    }
}
```

#### IVF (Inverted File Index)

Clusters vectors for faster search:

```python
# IVF divides vectors into clusters (cells)
# Only searches relevant clusters

# nlist: number of clusters
# nprobe: clusters to search at query time

# Example: 1M vectors with IVF
# nlist=1000 creates 1000 clusters of ~1000 vectors each
# nprobe=10 searches 10 clusters (10K vectors) instead of 1M
```

### Query Optimization

#### 1. Limit Result Count

```python
# Bad: Retrieve too many results
results = collection.query(
    query_embeddings=[embedding],
    n_results=1000  # Expensive!
)

# Good: Only get what you need
results = collection.query(
    query_embeddings=[embedding],
    n_results=10  # Just enough
)
```

#### 2. Use Metadata Filters

```python
# Bad: Retrieve all, filter in application
results = collection.query(query_embeddings=[embedding], n_results=1000)
filtered = [r for r in results if r["metadata"]["category"] == "tech"]

# Good: Filter at database level
results = collection.query(
    query_embeddings=[embedding],
    n_results=10,
    where={"category": "tech"}
)
```

#### 3. Reduce Include Fields

```python
# Bad: Include everything
results = collection.query(
    query_embeddings=[embedding],
    n_results=10,
    include=["documents", "embeddings", "metadatas", "distances"]
)

# Good: Only include what you need
results = collection.query(
    query_embeddings=[embedding],
    n_results=10,
    include=["documents", "metadatas"]  # Skip embeddings if not needed
)
```

### Batching Operations

Single operations have overhead. Batching amortizes this cost.

```python
import time


def benchmark_single_vs_batch(collection, documents, embeddings, metadatas):
    """Compare single inserts vs batch insert."""
    
    # Single inserts (slow)
    start = time.time()
    for i, (doc, emb, meta) in enumerate(zip(documents, embeddings, metadatas)):
        collection.add(
            ids=[f"single_{i}"],
            documents=[doc],
            embeddings=[emb],
            metadatas=[meta]
        )
    single_time = time.time() - start
    
    # Batch insert (fast)
    start = time.time()
    collection.add(
        ids=[f"batch_{i}" for i in range(len(documents))],
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    batch_time = time.time() - start
    
    print(f"Single inserts: {single_time:.2f}s")
    print(f"Batch insert: {batch_time:.2f}s")
    print(f"Speedup: {single_time/batch_time:.1f}x")


# Optimal batch sizes
BATCH_SIZE_RECOMMENDATIONS = {
    "embed_generation": 32,      # For embedding models
    "vector_insert": 1000,       # For database inserts
    "vector_query": 10,          # For parallel queries
}
```

#### Chunked Batch Processing

```python
def batch_insert(collection, items: list, batch_size: int = 1000):
    """Insert items in optimally-sized batches."""
    total = len(items)
    
    for i in range(0, total, batch_size):
        batch = items[i:i + batch_size]
        
        collection.add(
            ids=[item["id"] for item in batch],
            documents=[item["document"] for item in batch],
            embeddings=[item["embedding"] for item in batch],
            metadatas=[item["metadata"] for item in batch]
        )
        
        print(f"Inserted {min(i + batch_size, total)}/{total}")


def batch_query(collection, query_embeddings: list, batch_size: int = 10):
    """Query in batches for large query sets."""
    all_results = []
    
    for i in range(0, len(query_embeddings), batch_size):
        batch = query_embeddings[i:i + batch_size]
        
        results = collection.query(
            query_embeddings=batch,
            n_results=5
        )
        all_results.extend(results["documents"])
    
    return all_results
```

### Resource Management

#### Memory Optimization

```python
# Vector storage memory estimation
def estimate_memory_usage(
    num_vectors: int,
    dimensions: int,
    index_type: str = "hnsw"
) -> dict:
    """Estimate memory requirements."""
    
    # Base vector storage (float32 = 4 bytes)
    vector_bytes = num_vectors * dimensions * 4
    
    # Index overhead by type
    index_overhead = {
        "flat": 1.0,       # No additional overhead
        "hnsw": 1.5,       # ~50% overhead for graph structure
        "ivf": 1.2         # ~20% overhead for cluster centers
    }
    
    total_bytes = vector_bytes * index_overhead.get(index_type, 1.5)
    
    return {
        "vectors_mb": vector_bytes / (1024 * 1024),
        "total_mb": total_bytes / (1024 * 1024),
        "total_gb": total_bytes / (1024 * 1024 * 1024),
        "recommendation": get_instance_recommendation(total_bytes)
    }


def get_instance_recommendation(bytes_needed: int) -> str:
    """Recommend instance size based on memory needs."""
    gb_needed = bytes_needed / (1024 ** 3)
    
    if gb_needed < 4:
        return "t3.medium (4GB) - sufficient for development"
    elif gb_needed < 16:
        return "r5.large (16GB) - good for small production"
    elif gb_needed < 64:
        return "r5.2xlarge (64GB) - medium production workload"
    else:
        return "r5.4xlarge+ or distributed setup - large scale"


# Example usage
estimate = estimate_memory_usage(
    num_vectors=1_000_000,
    dimensions=1536,
    index_type="hnsw"
)
print(f"Estimated memory: {estimate['total_gb']:.2f} GB")
print(f"Recommendation: {estimate['recommendation']}")
```

#### Connection Pooling

```python
from contextlib import contextmanager
from queue import Queue
import threading


class VectorDBPool:
    """Connection pool for vector database clients."""
    
    def __init__(self, factory_fn, pool_size: int = 10):
        self.factory_fn = factory_fn
        self.pool = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        
        # Pre-create connections
        for _ in range(pool_size):
            self.pool.put(factory_fn())
    
    @contextmanager
    def get_client(self, timeout: float = 30):
        """Get a client from the pool."""
        client = self.pool.get(timeout=timeout)
        try:
            yield client
        finally:
            self.pool.put(client)
    
    def execute(self, operation, *args, **kwargs):
        """Execute an operation using a pooled client."""
        with self.get_client() as client:
            return getattr(client, operation)(*args, **kwargs)


# Usage
def create_chroma_client():
    import chromadb
    return chromadb.Client()

pool = VectorDBPool(create_chroma_client, pool_size=5)

# Use pooled connection
with pool.get_client() as client:
    collection = client.get_collection("documents")
    results = collection.query(query_embeddings=[[0.1, 0.2, 0.3]], n_results=5)
```

## Code Example

Complete performance-optimized vector store:

```python
import time
from dataclasses import dataclass
from typing import List, Optional
import chromadb


@dataclass
class PerformanceMetrics:
    """Track performance metrics."""
    total_queries: int = 0
    total_query_time_ms: float = 0
    total_inserts: int = 0
    total_insert_time_ms: float = 0
    
    @property
    def avg_query_time_ms(self) -> float:
        if self.total_queries == 0:
            return 0
        return self.total_query_time_ms / self.total_queries
    
    @property
    def avg_insert_time_ms(self) -> float:
        if self.total_inserts == 0:
            return 0
        return self.total_insert_time_ms / self.total_inserts


class OptimizedVectorStore:
    """Performance-optimized vector store wrapper."""
    
    def __init__(
        self,
        collection_name: str,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 100,
        batch_size: int = 1000
    ):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": hnsw_m,
                "hnsw:construction_ef": hnsw_ef_construction,
                "hnsw:search_ef": hnsw_ef_search
            }
        )
        self.batch_size = batch_size
        self.metrics = PerformanceMetrics()
    
    def add_batch(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[dict]
    ):
        """Add documents in optimized batches."""
        total = len(ids)
        
        for i in range(0, total, self.batch_size):
            end = min(i + self.batch_size, total)
            
            start_time = time.time()
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            self.metrics.total_inserts += end - i
            self.metrics.total_insert_time_ms += elapsed_ms
    
    def query_optimized(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[dict] = None,
        include_embeddings: bool = False
    ) -> dict:
        """Optimized query with minimal data transfer."""
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        
        start_time = time.time()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=include
        )
        elapsed_ms = (time.time() - start_time) * 1000
        
        self.metrics.total_queries += 1
        self.metrics.total_query_time_ms += elapsed_ms
        
        return results
    
    def get_performance_report(self) -> dict:
        """Get performance metrics report."""
        return {
            "total_queries": self.metrics.total_queries,
            "avg_query_time_ms": round(self.metrics.avg_query_time_ms, 2),
            "total_inserts": self.metrics.total_inserts,
            "avg_insert_time_ms": round(self.metrics.avg_insert_time_ms, 2),
            "collection_count": self.collection.count()
        }


# Usage and benchmarking
if __name__ == "__main__":
    import random
    
    # Create optimized store
    store = OptimizedVectorStore(
        collection_name="benchmark",
        hnsw_m=16,
        hnsw_ef_search=100,
        batch_size=500
    )
    
    # Generate test data
    num_docs = 5000
    dim = 384
    
    ids = [f"doc_{i}" for i in range(num_docs)]
    embeddings = [[random.random() for _ in range(dim)] for _ in range(num_docs)]
    documents = [f"Document content {i}" for i in range(num_docs)]
    metadatas = [{"index": i, "category": f"cat_{i % 10}"} for i in range(num_docs)]
    
    # Benchmark insert
    print("Inserting documents...")
    store.add_batch(ids, embeddings, documents, metadatas)
    
    # Benchmark queries
    print("Running queries...")
    for _ in range(100):
        query_emb = [random.random() for _ in range(dim)]
        store.query_optimized(query_emb, n_results=5)
    
    # Print report
    report = store.get_performance_report()
    print("\nPerformance Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
```

## Key Takeaways

1. **Choose the right index** - HNSW for most cases, tune M and ef parameters
2. **Batch operations** - single inserts are expensive, batch for efficiency
3. **Minimize data transfer** - only include fields you need
4. **Filter at database level** - metadata filters are faster than application filtering
5. **Monitor and measure** - track latency to identify bottlenecks
6. **Plan for scale** - estimate memory needs before deployment

## Additional Resources

- [HNSW Paper](https://arxiv.org/abs/1603.09320) - Original algorithm paper
- [Chroma Performance Tuning](https://docs.trychroma.com/usage-guide#changing-the-distance-function) - Official tuning guide
- [Vector Index Benchmarks](https://ann-benchmarks.com/) - Compare index algorithms
