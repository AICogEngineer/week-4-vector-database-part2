# Optimization Strategies

## Learning Objectives
- Understand approximate nearest neighbor (ANN) algorithms and their trade-offs
- Learn quantization techniques for memory reduction
- Implement hybrid search combining vector and keyword approaches
- Optimize end-to-end RAG pipeline performance

## Why This Matters

As your vector database grows from thousands to millions of vectors, naive approaches fail. Query times increase, memory costs explode, and accuracy may degrade. Advanced optimization techniques let you maintain performance at scale while managing costs effectively.

## The Concept

### Approximate Nearest Neighbor (ANN)

Exact nearest neighbor search is O(n) complexity - checking every vector. ANN algorithms trade perfect accuracy for dramatic speedups.

```
EXACT SEARCH (Flat Index)
Accuracy: 100%
Speed: O(n)
Memory: O(n*d)

ANN (HNSW, IVF, etc.)
Accuracy: 95-99.9% (configurable)
Speed: O(log n) to O(sqrt(n))
Memory: O(n*d) + index overhead
```

#### HNSW Optimization

```python
import chromadb


# HNSW tuning for different use cases
HNSW_CONFIGS = {
    "high_recall": {
        # Maximum accuracy, slower queries
        "hnsw:M": 48,                   # More connections
        "hnsw:construction_ef": 400,    # Higher build quality
        "hnsw:search_ef": 256           # More candidates during search
    },
    "balanced": {
        # Good accuracy with reasonable speed
        "hnsw:M": 16,
        "hnsw:construction_ef": 200,
        "hnsw:search_ef": 100
    },
    "high_speed": {
        # Faster queries, slightly lower recall
        "hnsw:M": 8,
        "hnsw:construction_ef": 100,
        "hnsw:search_ef": 50
    }
}


def create_optimized_collection(client, name: str, optimization: str = "balanced"):
    """Create collection with optimized HNSW parameters."""
    config = HNSW_CONFIGS.get(optimization, HNSW_CONFIGS["balanced"])
    
    return client.create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            **config
        }
    )


# Trade-offs
# Higher M → Better recall, more memory, slower inserts
# Higher ef_construction → Better index quality, slower build
# Higher ef_search → Better recall, slower queries
```

#### Recall vs Speed Trade-off

```python
def benchmark_recall_speed(collection, queries, ground_truth, ef_values):
    """Benchmark recall vs speed for different ef values."""
    import time
    
    results = []
    
    for ef in ef_values:
        # Update search ef (if supported)
        # Note: Chroma doesn't support dynamic ef changes
        # This is conceptual - other DBs like Qdrant support this
        
        start = time.time()
        for query in queries:
            collection.query(query_embeddings=[query], n_results=10)
        total_time = time.time() - start
        
        # Measure recall
        # recall = calculate_recall(results, ground_truth)
        
        results.append({
            "ef": ef,
            "time_per_query_ms": (total_time / len(queries)) * 1000,
            # "recall": recall
        })
    
    return results

# Typical results:
# ef=50  → 5ms/query, 92% recall
# ef=100 → 10ms/query, 97% recall
# ef=200 → 20ms/query, 99% recall
# ef=400 → 40ms/query, 99.5% recall
```

### Quantization

Reduce memory usage by compressing vector representations.

```python
import numpy as np


class VectorQuantization:
    """Vector quantization for memory reduction."""
    
    @staticmethod
    def float32_to_float16(vectors: np.ndarray) -> np.ndarray:
        """
        Simple precision reduction.
        
        Memory: 50% reduction
        Accuracy: minimal loss for most use cases
        """
        return vectors.astype(np.float16)
    
    @staticmethod
    def scalar_quantization(vectors: np.ndarray, bits: int = 8) -> tuple:
        """
        Scalar quantization - map floats to integers.
        
        8-bit: 75% memory reduction, ~1% accuracy loss
        4-bit: 87.5% reduction, ~5% accuracy loss
        """
        # Calculate scaling factors
        v_min = vectors.min(axis=1, keepdims=True)
        v_max = vectors.max(axis=1, keepdims=True)
        
        scale = (2 ** bits - 1) / (v_max - v_min + 1e-10)
        
        # Quantize
        quantized = ((vectors - v_min) * scale).astype(np.uint8)
        
        return quantized, (v_min, v_max, scale)
    
    @staticmethod
    def dequantize(quantized: np.ndarray, params: tuple) -> np.ndarray:
        """Restore quantized vectors to float32."""
        v_min, v_max, scale = params
        return (quantized.astype(np.float32) / scale) + v_min


# Memory comparison for 1M vectors, 1536 dimensions
def memory_comparison():
    dim = 1536
    n_vectors = 1_000_000
    
    sizes = {
        "float32": n_vectors * dim * 4,        # 5.7 GB
        "float16": n_vectors * dim * 2,        # 2.9 GB
        "int8": n_vectors * dim * 1,           # 1.4 GB
        "int4": n_vectors * dim * 0.5,         # 0.7 GB (packed)
    }
    
    for dtype, bytes_used in sizes.items():
        gb = bytes_used / (1024 ** 3)
        print(f"{dtype}: {gb:.2f} GB")
```

### Hybrid Search

Combine vector similarity with keyword/BM25 search for better results.

```python
from typing import List, Dict
import re


class HybridSearcher:
    """Combine vector and keyword search."""
    
    def __init__(self, collection, keyword_weight: float = 0.3):
        self.collection = collection
        self.keyword_weight = keyword_weight  # 0 = pure vector, 1 = pure keyword
    
    def search(
        self,
        query: str,
        query_embedding: List[float],
        n_results: int = 10
    ) -> List[Dict]:
        """Hybrid search combining vector and keyword matching."""
        
        # Get more candidates than needed
        candidates_multiplier = 3
        n_candidates = n_results * candidates_multiplier
        
        # Vector search
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"]
        )
        
        # Score and re-rank with keyword matching
        keywords = self._extract_keywords(query)
        ranked_results = []
        
        for i, (doc, meta, dist) in enumerate(zip(
            vector_results["documents"][0],
            vector_results["metadatas"][0],
            vector_results["distances"][0]
        )):
            # Vector score (normalize distance to 0-1 similarity)
            vector_score = 1 / (1 + dist)
            
            # Keyword score
            keyword_score = self._keyword_match_score(doc, keywords)
            
            # Combined score
            combined_score = (
                (1 - self.keyword_weight) * vector_score +
                self.keyword_weight * keyword_score
            )
            
            ranked_results.append({
                "document": doc,
                "metadata": meta,
                "vector_score": vector_score,
                "keyword_score": keyword_score,
                "combined_score": combined_score
            })
        
        # Sort by combined score and return top n
        ranked_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return ranked_results[:n_results]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Simple keyword extraction (production: use NLP library)
        words = re.findall(r'\w+', query.lower())
        # Filter common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how'}
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _keyword_match_score(self, document: str, keywords: List[str]) -> float:
        """Calculate keyword match score."""
        if not keywords:
            return 0.0
        
        doc_lower = document.lower()
        matches = sum(1 for kw in keywords if kw in doc_lower)
        return matches / len(keywords)


# Advanced: Use BM25 for keyword scoring
class BM25Scorer:
    """BM25 scoring for keyword search."""
    
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        # Pre-compute document frequencies
        self.doc_count = len(documents)
        self.avg_doc_len = sum(len(d.split()) for d in documents) / len(documents)
        
        # Term document frequencies
        self.term_doc_freq = {}
        for doc in documents:
            unique_terms = set(doc.lower().split())
            for term in unique_terms:
                self.term_doc_freq[term] = self.term_doc_freq.get(term, 0) + 1
    
    def score(self, query: str, document: str) -> float:
        """Calculate BM25 score for a document against a query."""
        import math
        
        query_terms = query.lower().split()
        doc_terms = document.lower().split()
        doc_len = len(doc_terms)
        
        # Term frequencies in document
        term_freq = {}
        for term in doc_terms:
            term_freq[term] = term_freq.get(term, 0) + 1
        
        score = 0.0
        for term in query_terms:
            if term not in self.term_doc_freq:
                continue
            
            # IDF
            idf = math.log((self.doc_count - self.term_doc_freq[term] + 0.5) /
                          (self.term_doc_freq[term] + 0.5) + 1)
            
            # TF with normalization
            tf = term_freq.get(term, 0)
            tf_normalized = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            )
            
            score += idf * tf_normalized
        
        return score
```

### Pipeline Optimization

Optimize the entire RAG pipeline, not just the vector search.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable
import time


class OptimizedRAGPipeline:
    """End-to-end optimized RAG pipeline."""
    
    def __init__(
        self,
        embedder: Callable[[str], List[float]],
        vector_store,
        llm: Callable[[str], str],
        cache_enabled: bool = True
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        
        # Cache for embeddings and results
        self.embedding_cache = {} if cache_enabled else None
        self.result_cache = {} if cache_enabled else None
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching."""
        if self.embedding_cache is not None:
            if text in self.embedding_cache:
                return self.embedding_cache[text]
        
        embedding = self.embedder(text)
        
        if self.embedding_cache is not None:
            self.embedding_cache[text] = embedding
        
        return embedding
    
    async def query_async(
        self,
        query: str,
        n_results: int = 5
    ) -> dict:
        """Async query for better concurrency."""
        loop = asyncio.get_event_loop()
        
        # Check cache
        if self.result_cache is not None and query in self.result_cache:
            return self.result_cache[query]
        
        # Parallel: embedding and any other preprocessing
        embedding_future = loop.run_in_executor(
            self.executor,
            self._get_embedding,
            query
        )
        
        embedding = await embedding_future
        
        # Vector search
        search_future = loop.run_in_executor(
            self.executor,
            lambda: self.vector_store.query(
                query_embeddings=[embedding],
                n_results=n_results
            )
        )
        
        results = await search_future
        
        # Cache results
        if self.result_cache is not None:
            self.result_cache[query] = results
        
        return results
    
    def batch_query(self, queries: List[str], n_results: int = 5) -> List[dict]:
        """Batch query for efficiency."""
        # Generate all embeddings
        embeddings = [self._get_embedding(q) for q in queries]
        
        # Batch vector search
        results = self.vector_store.query(
            query_embeddings=embeddings,
            n_results=n_results
        )
        
        return results
    
    def query_with_reranking(
        self,
        query: str,
        n_results: int = 5,
        rerank_top_k: int = 20
    ) -> dict:
        """Query with cross-encoder reranking for higher quality."""
        # Get more candidates
        embedding = self._get_embedding(query)
        candidates = self.vector_store.query(
            query_embeddings=[embedding],
            n_results=rerank_top_k
        )
        
        # Rerank with more expensive model
        # In production, use a cross-encoder model
        reranked = self._rerank(query, candidates)
        
        # Return top n after reranking
        return {
            "documents": [reranked["documents"][:n_results]],
            "metadatas": [reranked["metadatas"][:n_results]],
            "distances": [reranked["scores"][:n_results]]
        }
    
    def _rerank(self, query: str, candidates: dict) -> dict:
        """Rerank candidates (placeholder for cross-encoder)."""
        # In production, use a cross-encoder like:
        # from sentence_transformers import CrossEncoder
        # model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # For now, return as-is
        return {
            "documents": candidates["documents"][0],
            "metadatas": candidates["metadatas"][0],
            "scores": candidates["distances"][0]
        }


class CacheManager:
    """LRU cache with TTL for query results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache = {}
        self.order = []
    
    def get(self, key: str):
        """Get item from cache."""
        if key not in self.cache:
            return None
        
        item, timestamp = self.cache[key]
        
        # Check TTL
        if time.time() - timestamp > self.ttl:
            self._remove(key)
            return None
        
        # Move to end (most recent)
        self.order.remove(key)
        self.order.append(key)
        
        return item
    
    def set(self, key: str, value):
        """Set item in cache."""
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove oldest
            oldest = self.order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = (value, time.time())
        self.order.append(key)
    
    def _remove(self, key: str):
        """Remove item from cache."""
        if key in self.cache:
            del self.cache[key]
            self.order.remove(key)
```

## Code Example

Complete optimization toolkit:

```python
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class OptimizationMetrics:
    """Track optimization effectiveness."""
    baseline_latency_ms: float = 0
    optimized_latency_ms: float = 0
    memory_before_mb: float = 0
    memory_after_mb: float = 0
    recall_before: float = 0
    recall_after: float = 0
    
    @property
    def latency_improvement(self) -> float:
        if self.baseline_latency_ms == 0:
            return 0
        return (self.baseline_latency_ms - self.optimized_latency_ms) / self.baseline_latency_ms
    
    @property
    def memory_reduction(self) -> float:
        if self.memory_before_mb == 0:
            return 0
        return (self.memory_before_mb - self.memory_after_mb) / self.memory_before_mb


class VectorDBOptimizer:
    """Apply and measure optimizations."""
    
    def __init__(self, collection):
        self.collection = collection
        self.metrics = OptimizationMetrics()
    
    def benchmark_baseline(self, test_queries: list, n_results: int = 10) -> float:
        """Measure baseline performance."""
        start = time.time()
        
        for query_embedding in test_queries:
            self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
        
        total_time = time.time() - start
        avg_ms = (total_time / len(test_queries)) * 1000
        
        self.metrics.baseline_latency_ms = avg_ms
        return avg_ms
    
    def apply_hnsw_optimization(self, profile: str = "balanced"):
        """Apply HNSW tuning (creates new collection)."""
        configs = {
            "high_recall": {"M": 32, "ef_search": 200},
            "balanced": {"M": 16, "ef_search": 100},
            "high_speed": {"M": 8, "ef_search": 50}
        }
        
        config = configs.get(profile, configs["balanced"])
        print(f"Applied HNSW optimization: {profile}")
        print(f"  M={config['M']}, ef_search={config['ef_search']}")
        
        # Note: Chroma doesn't support dynamic HNSW parameter changes
        # This would require creating a new collection with these parameters
    
    def enable_query_cache(self, cache_size: int = 1000) -> CacheManager:
        """Enable query result caching."""
        cache = CacheManager(max_size=cache_size)
        print(f"Enabled query cache: {cache_size} entries")
        return cache
    
    def suggest_optimizations(self) -> list:
        """Suggest optimizations based on metrics."""
        suggestions = []
        
        if self.metrics.baseline_latency_ms > 100:
            suggestions.append({
                "type": "latency",
                "suggestion": "Consider using high_speed HNSW profile",
                "expected_improvement": "50-70% latency reduction"
            })
        
        count = self.collection.count()
        if count > 100000:
            suggestions.append({
                "type": "memory",
                "suggestion": "Consider vector quantization",
                "expected_improvement": "50-75% memory reduction"
            })
        
        if count > 1000000:
            suggestions.append({
                "type": "architecture",
                "suggestion": "Consider sharding or distributed deployment",
                "expected_improvement": "Linear scaling with shards"
            })
        
        return suggestions


# Usage
if __name__ == "__main__":
    import chromadb
    import random
    
    # Setup
    client = chromadb.Client()
    collection = client.get_or_create_collection("optimization_test")
    
    optimizer = VectorDBOptimizer(collection)
    
    # Generate test data
    dim = 384
    test_queries = [[random.random() for _ in range(dim)] for _ in range(100)]
    
    # Benchmark
    baseline = optimizer.benchmark_baseline(test_queries)
    print(f"Baseline latency: {baseline:.2f} ms/query")
    
    # Get suggestions
    suggestions = optimizer.suggest_optimizations()
    print("\nOptimization Suggestions:")
    for s in suggestions:
        print(f"  [{s['type']}] {s['suggestion']}")
        print(f"    Expected: {s['expected_improvement']}")
```

## Key Takeaways

1. **ANN algorithms trade accuracy for speed** - tune parameters for your needs
2. **HNSW is the most common choice** - tune M and ef parameters carefully
3. **Quantization reduces memory** - acceptable accuracy loss for most use cases
4. **Hybrid search improves relevance** - combine vector + keyword for best results
5. **Cache frequently used queries** - dramatic speedup for repeated patterns
6. **Optimize the whole pipeline** - not just the vector search component

## Additional Resources

- [ANN Benchmarks](https://ann-benchmarks.com/) - Compare ANN algorithms
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320) - Technical deep dive
- [Product Quantization](https://www.pinecone.io/learn/product-quantization/) - Memory optimization technique
- [Hybrid Search Guide (Weaviate)](https://weaviate.io/blog/hybrid-search-explained) - Combining approaches
