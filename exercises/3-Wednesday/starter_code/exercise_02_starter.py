"""
Exercise 02: Performance Optimization - Starter Code

Analyze and optimize a slow-performing vector search system.

Instructions:
1. Implement each TODO function
2. Run this file to test your implementations
3. Check the expected output in the exercise guide
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
# HELPER FUNCTIONS (DO NOT MODIFY)
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
# TODO: IMPLEMENT THESE CLASSES
# ============================================================================

class EmbeddingCache:
    """
    Cache for computed embeddings.
    
    Features:
    - Cache by content hash
    - Track hit/miss statistics
    - TTL for entries (optional)
    """
    
    def __init__(self, embedder, ttl_seconds: int = 300):
        self.embedder = embedder
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
    
    def _hash(self, text: str) -> str:
        """Create hash key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed(self, text: str) -> List[float]:
        """
        Get embedding for text, using cache if available.
        
        Tasks:
        1. Check cache for existing embedding
        2. If found, increment hits and return
        3. If not, compute embedding, cache it, increment misses
        """
        # TODO: Implement this method
        
        pass  # Remove this and add your implementation
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embed with caching.
        
        Tasks:
        1. Check cache for each text
        2. Batch embed only uncached texts
        3. Update cache with new embeddings
        4. Return all embeddings in order
        """
        # TODO: Implement this method
        
        pass  # Remove this and add your implementation
    
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


class OptimizedIngestion:
    """
    Optimized document ingestion with batching.
    """
    
    def __init__(self, collection, cache: EmbeddingCache, batch_size: int = 50):
        self.collection = collection
        self.cache = cache
        self.batch_size = batch_size
    
    def add_documents(self, documents: List[Dict]) -> Dict:
        """
        Add documents with optimized batching.
        
        Tasks:
        1. Split into batches of batch_size
        2. For each batch:
           - Use cached embeddings first
           - Batch embed the rest
           - Add to collection
        3. Return statistics
        
        Returns:
            Stats dict with timing and throughput
        """
        # TODO: Implement this method
        
        pass  # Remove this and add your implementation


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
        """Warm up the query engine with a test query."""
        # TODO: Implement warm-up (optional)
        pass
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        use_cache: bool = True
    ) -> Dict:
        """
        Execute optimized query.
        
        Tasks:
        1. Check query cache if enabled
        2. Use embedding cache for query embedding
        3. Query with minimal includes
        4. Cache result if enabled
        
        Returns:
            Query results
        """
        # TODO: Implement this method
        
        pass  # Remove this and add your implementation
    
    def clear_query_cache(self):
        """Clear the query result cache."""
        self._query_cache = {}


def benchmark_current_system(collection, embedder, test_docs: List[Dict]) -> Dict:
    """
    Benchmark the current (unoptimized) system.
    
    Returns baseline metrics for comparison.
    """
    # TODO: Implement this function
    # Measure:
    # - Single insert time
    # - Query latency (cold and warm)
    
    pass  # Remove this and add your implementation


def benchmark_optimized_system(
    collection,
    cache: EmbeddingCache,
    test_docs: List[Dict]
) -> Dict:
    """
    Benchmark the optimized system.
    
    Returns metrics to compare against baseline.
    """
    # TODO: Implement this function
    
    pass  # Remove this and add your implementation


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_tests():
    """Test the performance optimization implementations."""
    print("=" * 60)
    print("Exercise 02: Performance Optimization")
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
    
    # Test baseline
    print("\n=== Baseline System ===")
    
    baseline_metrics = benchmark_current_system(baseline_collection, embedder, test_docs[:20])
    
    if baseline_metrics is None:
        print("[INFO] benchmark_current_system not implemented yet")
    else:
        for key, value in baseline_metrics.items():
            print(f"  {key}: {value}")
    
    # Test embedding cache
    print("\n=== Embedding Cache Test ===")
    
    cache = EmbeddingCache(embedder)
    
    # First pass
    test_texts = [d["content"] for d in test_docs[:10]]
    
    try:
        for text in test_texts:
            cache.embed(text)
        
        # Second pass (should hit cache)
        for text in test_texts[:5]:  # Repeat some
            cache.embed(text)
        
        if cache.hits > 0 or cache.misses > 0:
            print(f"  Hits: {cache.hits}, Misses: {cache.misses}")
            print(f"  Hit rate: {cache.hit_rate:.1%}")
        else:
            print("[INFO] EmbeddingCache.embed not implemented yet")
            
    except Exception as e:
        print(f"[ERROR] {e}")
    
    # Test optimized ingestion
    print("\n=== Optimized Ingestion Test ===")
    
    try:
        ingestion = OptimizedIngestion(optimized_collection, cache, batch_size=BATCH_SIZE)
        stats = ingestion.add_documents(test_docs)
        
        if stats is None:
            print("[INFO] OptimizedIngestion.add_documents not implemented yet")
        else:
            print(f"  Documents: {stats.get('total_docs', 0)}")
            print(f"  Time: {stats.get('total_time_ms', 0):.1f}ms")
            print(f"  Throughput: {stats.get('docs_per_sec', 0):.1f} docs/sec")
            
    except Exception as e:
        print(f"[ERROR] {e}")
    
    # Test optimized query
    print("\n=== Optimized Query Test ===")
    
    try:
        query_engine = OptimizedQueryEngine(optimized_collection, cache)
        
        # Run queries
        test_queries = ["machine learning", "data processing", "neural network"]
        
        for query in test_queries:
            result = query_engine.query(query, n_results=3)
            
            if result is None:
                print("[INFO] OptimizedQueryEngine.query not implemented yet")
                break
            else:
                print(f"  Query: '{query}' -> {len(result.get('documents', [[]])[0])} results")
                
    except Exception as e:
        print(f"[ERROR] {e}")
    
    print("\n" + "=" * 60)
    print("[OK] Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
