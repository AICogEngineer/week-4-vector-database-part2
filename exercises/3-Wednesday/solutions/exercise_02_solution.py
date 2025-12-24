"""
Exercise 02: Performance Optimization - Solution

Complete implementation of vector database performance optimization.
"""

import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import statistics
import random
import string
import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_TEST_DOCUMENTS = 100
BATCH_SIZE = 50
CACHE_TTL_SECONDS = 300


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_test_documents(n: int) -> List[Dict]:
    """Generate n random test documents."""
    docs = []
    for i in range(n):
        words = [
            ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 10)))
            for _ in range(random.randint(20, 50))
        ]
        docs.append({
            "id": f"doc_{i:05d}",
            "content": ' '.join(words),
            "metadata": {"category": random.choice(["A", "B", "C"])}
        })
    return docs


def time_operation(func, *args, **kwargs) -> tuple:
    """Time a single operation, return (result, time_ms)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


@dataclass
class BenchmarkResult:
    """Benchmark results container."""
    name: str
    times_ms: List[float] = field(default_factory=list)
    
    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0
    
    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]


# ============================================================================
# SOLUTION IMPLEMENTATIONS
# ============================================================================

class EmbeddingCache:
    """
    Cache for computed embeddings with hit/miss tracking.
    """
    
    def __init__(self, embedder, ttl_seconds: int = 300):
        self.embedder = embedder
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict] = {}  # {hash: {"embedding": [...], "timestamp": ...}}
        self.hits = 0
        self.misses = 0
    
    def _hash(self, text: str) -> str:
        """Create hash key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired."""
        age = time.time() - entry.get("timestamp", 0)
        return age > self.ttl_seconds
    
    def embed(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available."""
        key = self._hash(text)
        
        # Check cache
        if key in self._cache and not self._is_expired(self._cache[key]):
            self.hits += 1
            return self._cache[key]["embedding"]
        
        # Cache miss - compute embedding
        self.misses += 1
        embedding = self.embedder.encode([text]).tolist()[0]
        
        # Store in cache
        self._cache[key] = {
            "embedding": embedding,
            "timestamp": time.time()
        }
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed with caching."""
        results = [None] * len(texts)
        to_embed = []
        to_embed_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            key = self._hash(text)
            
            if key in self._cache and not self._is_expired(self._cache[key]):
                self.hits += 1
                results[i] = self._cache[key]["embedding"]
            else:
                self.misses += 1
                to_embed.append(text)
                to_embed_indices.append(i)
        
        # Batch embed uncached texts
        if to_embed:
            new_embeddings = self.embedder.encode(to_embed).tolist()
            
            for idx, text, emb in zip(to_embed_indices, to_embed, new_embeddings):
                key = self._hash(text)
                self._cache[key] = {
                    "embedding": emb,
                    "timestamp": time.time()
                }
                results[idx] = emb
        
        return results
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
    
    def stats(self) -> Dict:
        """Return cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "cache_size": len(self._cache)
        }
    
    def clear(self):
        """Clear the cache."""
        self._cache = {}
        self.hits = 0
        self.misses = 0


class OptimizedIngestion:
    """
    Optimized document ingestion with batching.
    """
    
    def __init__(self, collection, cache: EmbeddingCache, batch_size: int = 50):
        self.collection = collection
        self.cache = cache
        self.batch_size = batch_size
    
    def add_documents(self, documents: List[Dict]) -> Dict:
        """Add documents with optimized batching."""
        start_time = time.perf_counter()
        total_docs = len(documents)
        
        # Process in batches
        for i in range(0, total_docs, self.batch_size):
            batch = documents[i:i + self.batch_size]
            
            ids = [d["id"] for d in batch]
            contents = [d["content"] for d in batch]
            metadatas = [d["metadata"] for d in batch]
            
            # Use cached embeddings
            embeddings = self.cache.embed_batch(contents)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "total_docs": total_docs,
            "batches": (total_docs + self.batch_size - 1) // self.batch_size,
            "total_time_ms": elapsed_ms,
            "docs_per_sec": total_docs / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
            "cache_stats": self.cache.stats()
        }


class OptimizedQueryEngine:
    """
    Optimized query engine with result caching.
    """
    
    def __init__(self, collection, cache: EmbeddingCache, warm_up: bool = True):
        self.collection = collection
        self.cache = cache
        self._query_cache: Dict[str, Any] = {}
        
        if warm_up:
            self._warm_up()
    
    def _warm_up(self):
        """Warm up the query engine."""
        if self.collection.count() > 0:
            # Run a dummy query to warm up
            try:
                dummy_emb = self.cache.embed("warmup query")
                self.collection.query(
                    query_embeddings=[dummy_emb],
                    n_results=1
                )
            except:
                pass
    
    def _query_cache_key(self, query_text: str, n_results: int) -> str:
        """Generate cache key for query."""
        return f"{query_text}::{n_results}"
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        use_cache: bool = True
    ) -> Dict:
        """Execute optimized query."""
        cache_key = self._query_cache_key(query_text, n_results)
        
        # Check query cache
        if use_cache and cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        # Get query embedding (uses embedding cache)
        query_embedding = self.cache.embed(query_text)
        
        # Query with minimal includes
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]  # Skip embeddings
        )
        
        # Cache result
        if use_cache:
            self._query_cache[cache_key] = results
        
        return results
    
    def clear_query_cache(self):
        """Clear the query result cache."""
        self._query_cache = {}


def benchmark_current_system(collection, embedder, test_docs: List[Dict]) -> Dict:
    """Benchmark the current (unoptimized) system."""
    results = {}
    
    # Single insert benchmark
    single_times = []
    for doc in test_docs[:10]:
        embedding = embedder.encode([doc["content"]]).tolist()[0]
        
        _, elapsed = time_operation(
            collection.add,
            ids=[doc["id"]],
            documents=[doc["content"]],
            embeddings=[embedding],
            metadatas=[doc["metadata"]]
        )
        single_times.append(elapsed)
    
    results["single_insert_avg_ms"] = f"{statistics.mean(single_times):.1f}ms"
    
    # Cold query benchmark
    query_emb = embedder.encode(["test query"]).tolist()
    
    # Fresh collection for cold query
    _, cold_time = time_operation(
        collection.query,
        query_embeddings=query_emb,
        n_results=5
    )
    results["cold_query_ms"] = f"{cold_time:.1f}ms"
    
    # Warm query benchmark
    warm_times = []
    for _ in range(10):
        _, elapsed = time_operation(
            collection.query,
            query_embeddings=query_emb,
            n_results=5
        )
        warm_times.append(elapsed)
    
    results["warm_query_avg_ms"] = f"{statistics.mean(warm_times):.1f}ms"
    
    return results


def benchmark_optimized_system(
    collection,
    cache: EmbeddingCache,
    test_docs: List[Dict]
) -> Dict:
    """Benchmark the optimized system."""
    results = {}
    
    # Optimized ingestion
    ingestion = OptimizedIngestion(collection, cache, batch_size=BATCH_SIZE)
    stats = ingestion.add_documents(test_docs)
    
    results["ingestion_docs_per_sec"] = f"{stats['docs_per_sec']:.1f}"
    results["ingestion_total_ms"] = f"{stats['total_time_ms']:.1f}ms"
    
    # Optimized queries
    query_engine = OptimizedQueryEngine(collection, cache)
    
    query_times = []
    test_queries = ["machine learning", "data processing", "neural network"]
    
    for query in test_queries:
        start = time.perf_counter()
        query_engine.query(query)
        elapsed = (time.perf_counter() - start) * 1000
        query_times.append(elapsed)
    
    results["query_avg_ms"] = f"{statistics.mean(query_times):.1f}ms"
    results["cache_hit_rate"] = cache.stats()["hit_rate"]
    
    return results


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_tests():
    """Test the performance optimization implementations."""
    print("=" * 60)
    print("Exercise 02: Performance Optimization - SOLUTION")
    print("=" * 60)
    
    print("\n[INFO] Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"[INFO] Generating {NUM_TEST_DOCUMENTS} test documents...")
    test_docs = generate_test_documents(NUM_TEST_DOCUMENTS)
    
    # Create test collections
    client = chromadb.Client()
    
    try:
        client.delete_collection("baseline_test")
    except:
        pass
    
    try:
        client.delete_collection("optimized_test")
    except:
        pass
    
    baseline_collection = client.create_collection("baseline_test")
    optimized_collection = client.create_collection("optimized_test")
    
    # Baseline benchmark
    print("\n=== Baseline System ===")
    baseline_metrics = benchmark_current_system(baseline_collection, embedder, test_docs[:20])
    
    for key, value in baseline_metrics.items():
        print(f"  {key}: {value}")
    
    # Optimized system
    print("\n=== Optimized System ===")
    
    cache = EmbeddingCache(embedder)
    
    # Ingestion test
    print("\n-- Optimized Ingestion --")
    ingestion = OptimizedIngestion(optimized_collection, cache, batch_size=BATCH_SIZE)
    ingestion_stats = ingestion.add_documents(test_docs)
    
    print(f"  Documents: {ingestion_stats['total_docs']}")
    print(f"  Batches: {ingestion_stats['batches']}")
    print(f"  Time: {ingestion_stats['total_time_ms']:.1f}ms")
    print(f"  Throughput: {ingestion_stats['docs_per_sec']:.1f} docs/sec")
    
    # Query test
    print("\n-- Optimized Queries --")
    query_engine = OptimizedQueryEngine(optimized_collection, cache)
    
    test_queries = [
        "machine learning basics",
        "data processing pipeline",
        "neural network training",
        "machine learning basics",  # Repeat for cache hit
    ]
    
    query_times = []
    for query in test_queries:
        start = time.perf_counter()
        result = query_engine.query(query, n_results=5)
        elapsed = (time.perf_counter() - start) * 1000
        query_times.append(elapsed)
        print(f"  Query: '{query[:20]}...' -> {len(result['documents'][0])} results in {elapsed:.1f}ms")
    
    # Cache stats
    print("\n-- Cache Statistics --")
    print(f"  Embedding cache: {cache.stats()}")
    print(f"  Query cache size: {len(query_engine._query_cache)}")
    
    # Summary
    print("\n=== Performance Summary ===")
    print(f"  Avg query latency: {statistics.mean(query_times):.1f}ms")
    print(f"  Embedding cache hit rate: {cache.hit_rate:.1%}")
    print(f"  Ingestion throughput: {ingestion_stats['docs_per_sec']:.1f} docs/sec")
    
    print("\n" + "=" * 60)
    print("[OK] Performance optimization complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
