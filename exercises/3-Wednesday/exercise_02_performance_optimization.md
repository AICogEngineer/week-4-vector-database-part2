# Exercise 02: Performance Optimization

## Overview

You've inherited a vector search system that's too slow for production. Your task: analyze the bottlenecks and apply optimization techniques to bring it up to spec.

## Learning Objectives

- Benchmark vector database operations
- Identify performance bottlenecks
- Apply batch processing optimizations
- Implement embedding caching
- Tune HNSW index parameters

## The Scenario

The current system has these problems:
1. Document ingestion takes too long (single inserts)
2. Query latency exceeds the 100ms SLA
3. Cold starts are unacceptable
4. Memory usage keeps growing

Your goal: Achieve <50ms query latency with efficient resource usage.

## Your Tasks

### Task 1: Baseline Benchmark (15 min)

Implement `benchmark_current_system()`:
- Measure single insert time
- Measure batch insert time (various sizes)
- Measure cold query time
- Measure warm query time

Document your baseline numbers before any optimization.

### Task 2: Batch Optimization (20 min)

Implement `OptimizedIngestion`:
- `add_documents()`: Accept list of documents
- Batch into optimal sizes (experiment to find best)
- Pre-compute embeddings in batches
- Report throughput statistics

### Task 3: Embedding Cache (25 min)

Implement `EmbeddingCache`:
- Cache computed embeddings by content hash
- Track hit/miss statistics
- Implement TTL (time-to-live) for cache entries
- Provide cache warming functionality

### Task 4: Query Optimization (20 min)

Implement `OptimizedQueryEngine`:
- Minimize data transferred (only needed fields)
- Implement query result caching
- Use appropriate n_results limits
- Handle warm-up queries on initialization

### Task 5: Before/After Comparison (10 min)

Run benchmarks on both original and optimized systems:
- Compare ingestion throughput
- Compare query latency
- Measure cache hit rates
- Document improvements

## Definition of Done

- [_] Baseline metrics documented
- [_] Batch ingestion 5x+ faster than single
- [_] Query latency under 50ms (warm)
- [_] Embedding cache achieving >50% hit rate
- [_] Before/after comparison shows improvement

## Testing Your Solution

```bash
cd exercises/3-Wednesday/starter_code
python exercise_02_starter.py
```

Expected output:
```
=== Performance Optimization Exercise ===

=== Baseline Metrics ===
Single insert:    45.2ms avg
Batch-100 insert: 8.7ms avg (100 docs)
Cold query:       85.3ms
Warm query:       12.4ms

=== Optimized System Metrics ===
Batch ingestion:  523.1 docs/sec (was 22.1)
Query latency:    8.2ms avg (was 12.4ms)
Cache hit rate:   67.3%

=== Improvement Summary ===
Ingestion:  23.7x faster
Query:      1.5x faster
Cache:      Saving 67% of embedding calls

[OK] Performance optimization complete!
```

## Stretch Goals (Optional)

1. Add async embedding for parallel processing
2. Implement connection pooling
3. Add query result prefetching for common queries
