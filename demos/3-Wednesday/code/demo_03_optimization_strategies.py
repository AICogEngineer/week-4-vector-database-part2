"""
Demo 03: Optimization Strategies

This demo shows trainees how to:
1. Optimize batch sizes for insertions
2. Tune HNSW index parameters
3. Implement embedding caching
4. Apply query optimization techniques

Learning Objectives:
- Apply practical optimization techniques
- Understand HNSW parameter trade-offs
- Build production-ready caching

References:
- Written Content: 04-vector-db-best-practices.md
- Written Content: 05-optimization-strategies.md
"""

import chromadb
from sentence_transformers import SentenceTransformer
import time
import hashlib
from typing import List, Dict, Optional, Any
from functools import lru_cache
import random
import string

# ============================================================================
# PART 1: Optimization Mindset
# ============================================================================

print("=" * 70)
print("PART 1: Optimization Mindset")
print("=" * 70)

print("""
THE OPTIMIZATION STACK (Bottom to Top)
─────────────────────────────────────────────────────────────────────

LAYER 4: APPLICATION
├─ Result caching
├─ Query batching
└─ Async processing

LAYER 3: PIPELINE
├─ Embedding caching  ← BIG WIN
├─ Chunking optimization
└─ Parallel embedding

LAYER 2: INDEX
├─ HNSW parameters
├─ Quantization (advanced)
└─ Index selection

LAYER 1: INFRASTRUCTURE
├─ Memory allocation
├─ Storage type
└─ Network optimization

Today we focus on Layers 2-4 (what YOU can control in code).
""")

# ============================================================================
# PART 2: Initialize Components
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Setting Up Test Environment")
print("=" * 70)

print("\n[Step 1] Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("  ✓ Loaded all-MiniLM-L6-v2")

print("\n[Step 2] Creating Chroma client...")
client = chromadb.Client()
print("  ✓ In-memory Chroma client ready")

# Generate test data
def generate_docs(n: int) -> tuple:
    """Generate n test documents."""
    docs, ids = [], []
    for i in range(n):
        words = [
            ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 8)))
            for _ in range(random.randint(30, 60))
        ]
        docs.append(' '.join(words))
        ids.append(f"doc_{i:05d}")
    return ids, docs

print("\n[Step 3] Generating test data...")
NUM_DOCS = 200
test_ids, test_docs = generate_docs(NUM_DOCS)
test_embeddings = embedder.encode(test_docs).tolist()
print(f"  ✓ Generated {NUM_DOCS} documents with embeddings")

# ============================================================================
# PART 3: Batch Size Optimization
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Batch Size Optimization")
print("=" * 70)

print("""
FINDING OPTIMAL BATCH SIZE:
───────────────────────────
Too small → Many round-trips, slow
Too large → Memory pressure, diminishing returns
Just right → Balanced throughput

Common sweet spots: 50-200 for most use cases
""")

def benchmark_batch_size(batch_size: int, ids, docs, embeddings) -> float:
    """Benchmark insertion with given batch size, return docs/sec."""
    # Fresh collection
    try:
        client.delete_collection("batch_test")
    except:
        pass
    
    collection = client.create_collection("batch_test")
    
    start = time.perf_counter()
    
    for i in range(0, len(ids), batch_size):
        end_idx = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end_idx],
            documents=docs[i:end_idx],
            embeddings=embeddings[i:end_idx]
        )
    
    elapsed = time.perf_counter() - start
    return len(ids) / elapsed


print("\n[Benchmark] Comparing batch sizes:")
print("-" * 50)

batch_sizes = [1, 10, 25, 50, 100, 200]
results = {}

for bs in batch_sizes:
    throughput = benchmark_batch_size(bs, test_ids, test_docs, test_embeddings)
    results[bs] = throughput
    bar = "█" * int(throughput / 50)
    print(f"  Batch {bs:3d}: {throughput:7.1f} docs/sec {bar}")

optimal = max(results, key=results.get)
print(f"\n  → Optimal batch size: {optimal} ({results[optimal]:.1f} docs/sec)")

# ============================================================================
# PART 4: HNSW Parameter Tuning
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: HNSW Parameter Tuning")
print("=" * 70)

print("""
HNSW KEY PARAMETERS:
────────────────────────────────────────────────────────────────────

M (connections per node)
├─ Default: 16
├─ Higher → Better recall, more memory, slower build
└─ Lower  → Faster build, less memory, lower recall

ef_construction (build-time search width)
├─ Default: 100
├─ Higher → Better graph quality, slower indexing
└─ Lower  → Faster indexing, potentially worse quality

ef_search (query-time search width)
├─ Default: 10
├─ Higher → Better recall, slower queries
└─ Lower  → Faster queries, lower recall
""")

def create_collection_with_hnsw(name: str, M: int, ef_construction: int) -> Any:
    """Create collection with specific HNSW parameters."""
    try:
        client.delete_collection(name)
    except:
        pass
    
    return client.create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:M": M,
            "hnsw:construction_ef": ef_construction
        }
    )


print("\n[Experiment] Comparing HNSW configurations:")
print("-" * 50)

configs = [
    {"M": 8, "ef": 50},    # Fast build, lower quality
    {"M": 16, "ef": 100},  # Default
    {"M": 32, "ef": 200},  # High quality
]

for cfg in configs:
    # Create with config
    coll = create_collection_with_hnsw(
        f"hnsw_test_{cfg['M']}_{cfg['ef']}", 
        cfg['M'], 
        cfg['ef']
    )
    
    # Measure build time
    start = time.perf_counter()
    coll.add(
        ids=test_ids,
        documents=test_docs,
        embeddings=test_embeddings
    )
    build_time = (time.perf_counter() - start) * 1000
    
    # Measure query time (warm)
    query_emb = embedder.encode(["test query"]).tolist()
    coll.query(query_embeddings=query_emb, n_results=5)  # Warm up
    
    start = time.perf_counter()
    for _ in range(20):
        coll.query(query_embeddings=query_emb, n_results=5)
    query_time = (time.perf_counter() - start) * 1000 / 20
    
    print(f"  M={cfg['M']:2d}, ef={cfg['ef']:3d}: "
          f"build={build_time:6.1f}ms, query={query_time:.2f}ms")

print("""
RECOMMENDATIONS:
────────────────
• Start with defaults (M=16, ef_construction=100)
• Increase M if recall is low
• Increase ef_construction for better index quality
• Tune ef_search at query time for speed/quality balance
""")

# ============================================================================
# PART 5: Embedding Caching
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Embedding Caching")
print("=" * 70)

print("""
EMBEDDING IS EXPENSIVE!
───────────────────────
Typical embedding: 10-50ms per text
For 10,000 queries/day at 20ms each = 200 seconds of compute

CACHING STRATEGY:
1. Hash the input text
2. Check cache before embedding
3. Store new embeddings in cache
4. Significant speedup for repeated queries!
""")


class EmbeddingCache:
    """
    Simple embedding cache using dictionary.
    
    Production alternatives:
    - Redis for distributed caching
    - LRU cache with size limit
    - Disk-based cache for persistence
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
        self._cache: Dict[str, List[float]] = {}
        self.hits = 0
        self.misses = 0
    
    def _hash_text(self, text: str) -> str:
        """Create hash key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed(self, text: str) -> List[float]:
        """Get embedding, using cache if available."""
        key = self._hash_text(text)
        
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        
        self.misses += 1
        embedding = self.embedder.encode([text]).tolist()[0]
        self._cache[key] = embedding
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed with caching."""
        results = []
        to_embed = []
        to_embed_indices = []
        
        for i, text in enumerate(texts):
            key = self._hash_text(text)
            if key in self._cache:
                self.hits += 1
                results.append(self._cache[key])
            else:
                self.misses += 1
                to_embed.append(text)
                to_embed_indices.append(i)
                results.append(None)  # Placeholder
        
        # Batch embed uncached
        if to_embed:
            new_embeddings = self.embedder.encode(to_embed).tolist()
            for idx, text, emb in zip(to_embed_indices, to_embed, new_embeddings):
                key = self._hash_text(text)
                self._cache[key] = emb
                results[idx] = emb
        
        return results
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
    
    def stats(self) -> str:
        return f"Hits: {self.hits}, Misses: {self.misses}, Hit Rate: {self.hit_rate:.1%}"


print("\n[Demo] Embedding cache in action:")
print("-" * 50)

cache = EmbeddingCache(embedder)

# First pass - all misses
queries = [
    "machine learning fundamentals",
    "deep learning neural networks",
    "natural language processing",
    "machine learning fundamentals",  # Repeat
    "computer vision basics",
    "deep learning neural networks",   # Repeat
]

print("\n  First pass (mixed new and repeated queries):")
for i, q in enumerate(queries):
    start = time.perf_counter()
    _ = cache.embed(q)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"    '{q[:30]}...': {elapsed:.2f}ms")

print(f"\n  {cache.stats()}")

# Second pass - all hits
print("\n  Second pass (all cached):")
cache2_start = time.perf_counter()
for q in queries:
    _ = cache.embed(q)
cache2_elapsed = (time.perf_counter() - cache2_start) * 1000

print(f"    All 6 queries: {cache2_elapsed:.2f}ms (was ~{6*20}ms without cache)")
print(f"\n  {cache.stats()}")

# ============================================================================
# PART 6: Query Optimization
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Query Optimization Techniques")
print("=" * 70)

print("""
QUERY OPTIMIZATION CHECKLIST:
─────────────────────────────
✓ Request only needed fields (include parameter)
✓ Limit result count (n_results)
✓ Use metadata filters to reduce search space
✓ Cache frequent queries
✓ Use pre-filtering when possible
""")

# Setup test collection
try:
    client.delete_collection("query_opt_test")
except:
    pass

opt_collection = client.create_collection(
    name="query_opt_test",
    metadata={"hnsw:space": "cosine"}
)

# Add with metadata
metadatas = [{"category": random.choice(["A", "B", "C"])} for _ in test_ids]
opt_collection.add(
    ids=test_ids,
    documents=test_docs,
    embeddings=test_embeddings,
    metadatas=metadatas
)

query_emb = embedder.encode(["test query"]).tolist()

print("\n[Comparison] Include parameter impact:")
print("-" * 50)

# Full include
start = time.perf_counter()
for _ in range(50):
    opt_collection.query(
        query_embeddings=query_emb,
        n_results=10,
        include=["documents", "metadatas", "distances", "embeddings"]
    )
full_time = (time.perf_counter() - start) * 1000 / 50

# Minimal include
start = time.perf_counter()
for _ in range(50):
    opt_collection.query(
        query_embeddings=query_emb,
        n_results=10,
        include=["documents"]  # Only what we need
    )
minimal_time = (time.perf_counter() - start) * 1000 / 50

print(f"  Full include (all fields):   {full_time:.2f}ms")
print(f"  Minimal include (docs only): {minimal_time:.2f}ms")
print(f"  Savings: {(1 - minimal_time/full_time)*100:.0f}%")

print("\n[Comparison] Metadata pre-filtering:")
print("-" * 50)

# Without filter
start = time.perf_counter()
for _ in range(50):
    opt_collection.query(
        query_embeddings=query_emb,
        n_results=10
    )
no_filter_time = (time.perf_counter() - start) * 1000 / 50

# With filter (reduces search space)
start = time.perf_counter()
for _ in range(50):
    opt_collection.query(
        query_embeddings=query_emb,
        n_results=10,
        where={"category": "A"}
    )
filter_time = (time.perf_counter() - start) * 1000 / 50

print(f"  No filter (all docs):        {no_filter_time:.2f}ms")
print(f"  With filter (category=A):   {filter_time:.2f}ms")

# ============================================================================
# PART 7: Production Optimization Checklist
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: Production Optimization Checklist")
print("=" * 70)

checklist = """
BEFORE GOING TO PRODUCTION:
═══════════════════════════════════════════════════════════════════════

□ BATCH OPERATIONS
  ├─ Insertions use optimal batch size (50-100)
  ├─ Updates are batched, not one-by-one
  └─ Deletes are batched when possible

□ CACHING
  ├─ Embedding cache implemented
  ├─ Query result cache for frequent searches
  └─ Cache invalidation strategy defined

□ INDEX CONFIGURATION
  ├─ HNSW parameters tuned for use case
  ├─ Space metric matches embedding model
  └─ Persistence configured (if local)

□ QUERY OPTIMIZATION
  ├─ Only necessary fields in include
  ├─ n_results limited appropriately
  └─ Metadata filters used effectively

□ MONITORING
  ├─ Latency percentiles tracked
  ├─ Cache hit rates monitored
  └─ Index size monitored

═══════════════════════════════════════════════════════════════════════
"""

print(checklist)

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO 3 COMPLETE: Optimization Strategies")
print("=" * 70)

print("""
Key Takeaways:

1. BATCH SIZE MATTERS
   - Find your optimal batch size (usually 50-100)
   - Single operations are very inefficient

2. HNSW TUNING
   - M controls connections (memory vs recall)
   - ef_construction affects index quality
   - Start with defaults, tune based on needs

3. EMBEDDING CACHING IS HUGE
   - 10-100x speedup for repeated queries
   - Implement at least basic caching

4. QUERY OPTIMIZATION
   - Request only what you need
   - Use filters to reduce search space
   - Cache frequent query results

5. MEASURE EVERYTHING
   - Optimization without metrics is guessing
   - Track before/after for every change

Week 4 Complete! You now have production-ready RAG skills.
""")

# ============================================================================
# INSTRUCTOR NOTES
# ============================================================================

print("\n" + "=" * 70)
print("INSTRUCTOR NOTES")
print("=" * 70)

print("""
Discussion Questions:
1. "What would you optimize first in your system?"
2. "How would you handle cache invalidation?"
3. "When is it okay to NOT optimize?"

Hands-on Exercise:
- Have trainees implement a simple result cache
- Compare before/after performance

Common Confusions:
- "Should I always use highest M?" → No, balance with memory
- "Is caching always worth it?" → Depends on query patterns
- "What about GPU acceleration?" → Out of scope, but mention

If Running Short on Time:
- Skip HNSW parameter deep dive
- Focus on caching and batch size

If Trainees Are Advanced:
- Discuss quantization (PQ, SQ)
- Cover distributed caching with Redis
- Explore async/parallel embedding
""")

print("\n" + "=" * 70)
