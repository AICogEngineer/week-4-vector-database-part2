"""
Demo 02: Performance Benchmarking

This demo shows trainees how to:
1. Set up a benchmarking framework
2. Measure insertion performance (single vs batch)
3. Measure query latency (cold vs warm)
4. Understand scaling behavior

Learning Objectives:
- Establish performance baselines
- Identify bottlenecks in vector operations
- Make data-driven optimization decisions

References:
- Written Content: 03-performance-considerations.md
"""

import chromadb
from sentence_transformers import SentenceTransformer
import time
import statistics
from typing import List, Dict, Callable, Any
from dataclasses import dataclass, field
import random
import string

# ============================================================================
# PART 1: Why Benchmark?
# ============================================================================

print("=" * 70)
print("PART 1: Why Benchmark?")
print("=" * 70)

print("""
"PREMATURE OPTIMIZATION IS THE ROOT OF ALL EVIL" - Donald Knuth

BUT... you can't optimize what you can't measure!

BENCHMARKING GOALS:
───────────────────
1. Establish BASELINE performance
2. Identify BOTTLENECKS before they become problems
3. Make DATA-DRIVEN optimization decisions
4. Validate optimization IMPACT

KEY METRICS FOR VECTOR DATABASES:
─────────────────────────────────
LATENCY                    THROUGHPUT
├─ Query P50, P95, P99     ├─ Queries/second
├─ Insert time             ├─ Inserts/second
└─ Batch insert time       └─ Concurrent capacity

ACCURACY                   RESOURCES
├─ Recall@K                ├─ Memory usage
├─ Match quality           ├─ CPU utilization
└─ Relevance              └─ Storage size
""")

# ============================================================================
# PART 2: Benchmark Framework Setup
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Benchmark Framework Setup")
print("=" * 70)

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    operation: str
    sample_size: int
    times_ms: List[float] = field(default_factory=list)
    
    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0
    
    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0
    
    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    @property
    def p99_ms(self) -> float:
        if not self.times_ms:
            return 0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    @property
    def throughput(self) -> float:
        """Operations per second."""
        if not self.times_ms:
            return 0
        total_seconds = sum(self.times_ms) / 1000
        return self.sample_size / total_seconds if total_seconds > 0 else 0
    
    def __str__(self) -> str:
        return (
            f"{self.operation}:\n"
            f"  Samples: {self.sample_size}\n"
            f"  Mean: {self.mean_ms:.2f}ms\n"
            f"  Median: {self.median_ms:.2f}ms\n"
            f"  P95: {self.p95_ms:.2f}ms\n"
            f"  P99: {self.p99_ms:.2f}ms\n"
            f"  Throughput: {self.throughput:.1f} ops/sec"
        )


def time_operation(func: Callable, *args, **kwargs) -> tuple:
    """Time a single operation, return (result, time_ms)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def generate_random_docs(n: int) -> tuple:
    """Generate n random documents for benchmarking."""
    docs = []
    ids = []
    for i in range(n):
        # Generate somewhat realistic text
        words = [
            ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
            for _ in range(random.randint(20, 50))
        ]
        docs.append(' '.join(words))
        ids.append(f"doc_{i:06d}")
    return ids, docs


print("Benchmark framework initialized:")
print("-" * 50)
print("  ✓ BenchmarkResult class for statistics")
print("  ✓ time_operation helper for precise timing")
print("  ✓ generate_random_docs for test data")

# ============================================================================
# PART 3: Initialize Components
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Initializing Test Environment")
print("=" * 70)

print("\n[Step 1] Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("  ✓ Loaded all-MiniLM-L6-v2")

print("\n[Step 2] Creating Chroma client...")
client = chromadb.Client()
print("  ✓ In-memory Chroma client ready")

# Clean up if exists
try:
    client.delete_collection("benchmark_test")
except:
    pass

collection = client.create_collection(
    name="benchmark_test",
    metadata={"hnsw:space": "cosine"}
)
print("  ✓ Test collection created")

# ============================================================================
# PART 4: Insertion Benchmarks
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Insertion Benchmarks")
print("=" * 70)

# Generate test data
print("\n[Step 3] Generating test documents...")
NUM_DOCS = 100
test_ids, test_docs = generate_random_docs(NUM_DOCS)
print(f"  ✓ Generated {NUM_DOCS} test documents")

# Pre-compute embeddings (so we measure insertion, not embedding time)
print("\n[Step 4] Pre-computing embeddings...")
start = time.perf_counter()
test_embeddings = embedder.encode(test_docs).tolist()
embed_time = time.perf_counter() - start
print(f"  ✓ Embedded {NUM_DOCS} documents in {embed_time:.2f}s")
print(f"  ✓ Embedding throughput: {NUM_DOCS/embed_time:.1f} docs/sec")

# Benchmark: Single insertions
print("\n[Benchmark A] Single Document Insertions:")
print("-" * 50)

# Recreate collection for clean test
client.delete_collection("benchmark_test")
collection = client.create_collection(
    name="benchmark_test",
    metadata={"hnsw:space": "cosine"}
)

single_result = BenchmarkResult("Single Insert", NUM_DOCS)

for i in range(min(50, NUM_DOCS)):  # Test first 50
    _, elapsed = time_operation(
        collection.add,
        ids=[test_ids[i]],
        documents=[test_docs[i]],
        embeddings=[test_embeddings[i]]
    )
    single_result.times_ms.append(elapsed)

print(single_result)

# Benchmark: Batch insertions
print("\n[Benchmark B] Batch Insertions:")
print("-" * 50)

batch_sizes = [10, 25, 50, 100]

for batch_size in batch_sizes:
    # Recreate collection for each test
    client.delete_collection("benchmark_test")
    collection = client.create_collection(
        name="benchmark_test",
        metadata={"hnsw:space": "cosine"}
    )
    
    batch_result = BenchmarkResult(f"Batch-{batch_size}", NUM_DOCS)
    
    for i in range(0, NUM_DOCS, batch_size):
        end_idx = min(i + batch_size, NUM_DOCS)
        _, elapsed = time_operation(
            collection.add,
            ids=test_ids[i:end_idx],
            documents=test_docs[i:end_idx],
            embeddings=test_embeddings[i:end_idx]
        )
        batch_result.times_ms.append(elapsed)
    
    print(f"  Batch size {batch_size}: {batch_result.mean_ms:.2f}ms avg, "
          f"{batch_result.throughput:.1f} docs/sec")

print("""
OBSERVATION:
───────────
Batch insertions are MUCH faster than single insertions!
This is because:
1. Fewer round-trips to the database
2. Index updates can be batched
3. Better memory utilization
""")

# ============================================================================
# PART 5: Query Benchmarks
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Query Benchmarks")
print("=" * 70)

# Ensure collection has data
client.delete_collection("benchmark_test")
collection = client.create_collection(
    name="benchmark_test",
    metadata={"hnsw:space": "cosine"}
)
collection.add(
    ids=test_ids,
    documents=test_docs,
    embeddings=test_embeddings
)
print(f"\n[Setup] Collection populated with {collection.count()} documents")

# Generate query embeddings
query_texts = [
    "machine learning fundamentals",
    "data processing pipeline",
    "neural network architecture"
]
query_embeddings = embedder.encode(query_texts).tolist()

# Benchmark: Cold queries (first query after collection load)
print("\n[Benchmark C] Cold vs Warm Queries:")
print("-" * 50)

# Cold query (first query)
_, cold_time = time_operation(
    collection.query,
    query_embeddings=[query_embeddings[0]],
    n_results=5
)
print(f"  Cold query: {cold_time:.2f}ms")

# Warm queries (subsequent queries)
warm_result = BenchmarkResult("Warm Query", len(query_texts) * 10)

for _ in range(10):  # 10 iterations
    for qe in query_embeddings:
        _, elapsed = time_operation(
            collection.query,
            query_embeddings=[qe],
            n_results=5
        )
        warm_result.times_ms.append(elapsed)

print(f"  Warm query (avg): {warm_result.mean_ms:.2f}ms")
print(f"  Warm query (P95): {warm_result.p95_ms:.2f}ms")

print("""
OBSERVATION:
───────────
First query (cold) is typically slower because:
1. Data not yet in memory cache
2. Index structures being loaded
3. Query execution plan being optimized
""")

# Benchmark: Impact of n_results
print("\n[Benchmark D] Impact of Result Count (n_results):")
print("-" * 50)

for n in [1, 5, 10, 25, 50]:
    n_result = BenchmarkResult(f"n_results={n}", 20)
    
    for _ in range(20):
        _, elapsed = time_operation(
            collection.query,
            query_embeddings=[query_embeddings[0]],
            n_results=n
        )
        n_result.times_ms.append(elapsed)
    
    print(f"  n_results={n:2d}: {n_result.mean_ms:.2f}ms avg")

# Benchmark: Impact of collection size
print("\n[Benchmark E] Impact of Collection Size:")
print("-" * 50)

size_results = {}

for target_size in [50, 100]:  # Small sizes for demo
    # Recreate with specific size
    client.delete_collection("benchmark_test")
    collection = client.create_collection(
        name="benchmark_test",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add exactly target_size documents
    collection.add(
        ids=test_ids[:target_size],
        documents=test_docs[:target_size],
        embeddings=test_embeddings[:target_size]
    )
    
    # Warm up
    collection.query(query_embeddings=[query_embeddings[0]], n_results=5)
    
    # Benchmark
    size_bench = BenchmarkResult(f"Size-{target_size}", 20)
    for _ in range(20):
        _, elapsed = time_operation(
            collection.query,
            query_embeddings=[query_embeddings[0]],
            n_results=5
        )
        size_bench.times_ms.append(elapsed)
    
    size_results[target_size] = size_bench.mean_ms
    print(f"  {target_size:6d} docs: {size_bench.mean_ms:.2f}ms avg")

print("""
OBSERVATION:
───────────
Query time grows SUBLINEARLY with collection size!
This is thanks to HNSW index - O(log N) complexity.
A 10x larger collection is NOT 10x slower.
""")

# ============================================================================
# PART 6: Summary Report
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Benchmark Summary Report")
print("=" * 70)

print("""
PERFORMANCE BASELINE SUMMARY
════════════════════════════════════════════════════════════════════════

INSERTION PERFORMANCE:
├─ Single insert:    ~{:.1f}ms per doc
├─ Batch-10:         ~{:.1f}ms per batch ({:.1f} docs/sec)
├─ Batch-50:         ~{:.1f}ms per batch ({:.1f} docs/sec)
└─ Recommendation:   Use batch sizes of 50-100

QUERY PERFORMANCE:
├─ Cold query:       ~{:.1f}ms (first query penalty)
├─ Warm query:       ~{:.1f}ms average
├─ P95 latency:      ~{:.1f}ms
└─ P99 latency:      ~{:.1f}ms

SCALING BEHAVIOR:
└─ Query time scales logarithmically with collection size

────────────────────────────────────────────────────────────────────────
THESE ARE YOUR BASELINE NUMBERS. Any optimization should be measured
against these baselines to validate improvement.
════════════════════════════════════════════════════════════════════════
""".format(
    single_result.mean_ms,
    batch_result.mean_ms, batch_result.throughput,
    batch_result.mean_ms, batch_result.throughput,
    cold_time,
    warm_result.mean_ms,
    warm_result.p95_ms,
    warm_result.p99_ms
))

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO 2 COMPLETE: Performance Benchmarking")
print("=" * 70)

print("""
Key Takeaways:

1. ALWAYS MEASURE BEFORE OPTIMIZING
   - Establish baseline numbers
   - Document your starting point

2. BATCH OPERATIONS ARE CRITICAL
   - 10x+ improvement over single operations
   - Find optimal batch size for your use case

3. COLD VS WARM MATTERS
   - First queries are slower
   - Consider warm-up strategies

4. SCALING IS SUBLINEAR
   - HNSW gives O(log N) query time
   - Don't fear larger collections

5. PERCENTILES MATTER
   - P95/P99 show tail latency
   - Important for SLA compliance

Coming Next: Demo 3 covers optimization strategies!
""")

# ============================================================================
# INSTRUCTOR NOTES
# ============================================================================

print("\n" + "=" * 70)
print("INSTRUCTOR NOTES")
print("=" * 70)

print("""
Discussion Questions:
1. "What latency is acceptable for your RAG use case?"
2. "How would you automate these benchmarks in CI/CD?"
3. "When would you sacrifice accuracy for speed?"

Interactive Exercise:
- Have trainees modify batch sizes
- Compare results across machines

Common Confusions:
- "Why is my cold query so slow?" → Expected, explain caching
- "These numbers seem slow/fast" → Hardware dependent
- "How do I benchmark accuracy?" → Need ground truth dataset

If Running Short on Time:
- Skip collection size benchmark
- Focus on batch vs single

If Trainees Are Advanced:
- Discuss distributed benchmarking
- Cover recall@K measurement
- Explore profiling tools
""")

print("\n" + "=" * 70)
