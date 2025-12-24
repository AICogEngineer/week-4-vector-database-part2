# Interview Questions: Week 4 - Vector Databases (Part 2)

This question bank prepares you for technical interviews covering document preprocessing, chunking strategies, metadata filtering, deployment patterns, and performance optimization.

---

## Beginner (Foundational) - 70%

### Q1: What is the purpose of text preprocessing before generating embeddings?

**Keywords:** Noise removal, Normalization, Quality, Clean data, Consistency

<details>
<summary>Click to Reveal Answer</summary>

Text preprocessing removes noise (HTML tags, special characters, inconsistent whitespace) and normalizes content before embedding. This ensures the embedding model captures the actual semantic meaning rather than irrelevant artifacts. Clean input leads to higher quality embeddings and better retrieval results.

</details>

---

### Q2: What is "chunking" in the context of RAG systems?

**Keywords:** Splitting, Segments, Token limits, Retrieval granularity, Context

<details>
<summary>Click to Reveal Answer</summary>

Chunking is the process of splitting large documents into smaller, manageable segments for embedding and retrieval. It's necessary because:

1. Embedding models have token limits (typically 512 tokens)
2. Smaller chunks enable more precise retrieval
3. It controls how much context is returned for each match

Common strategies include fixed-size, sentence-based, and recursive chunking.

</details>

---

### Q3: What is "chunk overlap" and why is it used?

**Keywords:** Boundary, Context preservation, Redundancy, Split content, Continuity

<details>
<summary>Click to Reveal Answer</summary>

Chunk overlap means adjacent chunks share some content at their boundaries. For example, with 10% overlap, the end of chunk 1 appears at the beginning of chunk 2. This prevents important information from being split across chunks and lost during retrieval. Without overlap, a relevant sentence at a chunk boundary might be split and neither chunk would rank highly for a query.

</details>

---

### Q4: What are the main chunking strategies and when would you use each?

**Keywords:** Fixed-size, Sentence-based, Recursive, Semantic, Structure

<details>
<summary>Click to Reveal Answer</summary>

1. **Fixed-size**: Split at exact character/token counts. Simple but ignores structure. Good for uniform content.

2. **Sentence-based**: Split at sentence boundaries. Preserves complete thoughts. Good for prose content.

3. **Recursive**: Try multiple separators in priority (paragraphs → sentences → characters). Respects document hierarchy. Good for structured documents.

4. **Semantic**: Group content by meaning using embeddings. Most sophisticated. Good for topic-based retrieval.

</details>

---

### Q5: What is metadata filtering in vector databases?

**Keywords:** Structured filters, Where clause, Pre-filtering, Hybrid search

<details>
<summary>Click to Reveal Answer</summary>

Metadata filtering allows you to combine semantic similarity search with structured attribute filters. For example, "find documents similar to X where category='tutorial' AND version='2.0'". The filter is typically applied before the vector search to reduce the search space, making queries faster and more precise.

</details>

---

### Q6: What Chroma operators are available for metadata filtering?

**Keywords:** $eq, $gt, $lt, $in, $and, $or, Comparison

<details>
<summary>Click to Reveal Answer</summary>

Chroma supports:
- **Equality**: `{"field": "value"}` or `{"field": {"$eq": "value"}}`
- **Comparison**: `$gt`, `$gte`, `$lt`, `$lte` for numeric/date fields
- **Set membership**: `$in` for matching any value in a list
- **Logical**: `$and`, `$or` for combining multiple conditions
- **Negation**: `$ne` for not equal

Example: `{"$and": [{"category": "tutorial"}, {"version": {"$gte": "2.0"}}]}`

</details>

---

### Q7: What is the difference between `chromadb.Client()` and `chromadb.PersistentClient()`?

**Keywords:** In-memory, Disk storage, Persistence, Data survival, Ephemeral

<details>
<summary>Click to Reveal Answer</summary>

- `Client()`: Creates an in-memory database. Data is lost when the program ends. Best for testing and experimentation.

- `PersistentClient(path="./data")`: Stores data on disk at the specified path. Data survives program restarts. Best for development and production.

The API is identical after creation - only the storage backend differs.

</details>

---

### Q8: What does Unicode normalization (NFKC) do in text preprocessing?

**Keywords:** Character forms, Equivalence, Decomposition, Compatibility, Consistent encoding

<details>
<summary>Click to Reveal Answer</summary>

Unicode normalization converts different representations of the same character to a canonical form. For example:
- "é" (single character) vs "e" + combining accent
- Full-width "Ａ" to regular "A"
- Ligatures like "ﬁ" to "fi"

NFKC (Normalization Form Compatibility Composition) applies both decomposition and composition, ensuring consistent text for embedding.

</details>

---

### Q9: Why should `<script>` and `<style>` tags be removed before embedding HTML documents?

**Keywords:** Non-content, Noise, JavaScript code, CSS, Irrelevant

<details>
<summary>Click to Reveal Answer</summary>

Script and style tags contain code, not content. Including them would:
1. Pollute embeddings with programming code unrelated to document meaning
2. Waste embedding space on irrelevant tokens
3. Return irrelevant matches when users search for topic-related terms that happen to appear in code

Always remove these tags entirely (including their contents) before extracting text.

</details>

---

### Q10: What is the recommended chunk size for most RAG applications?

**Keywords:** 100-500 tokens, 400-2000 characters, Balance, Context, Precision

<details>
<summary>Click to Reveal Answer</summary>

Most RAG systems use chunks of 100-500 tokens (roughly 400-2000 characters) with 10-20% overlap. This range balances:

- **Too small**: Loses context, returns fragments
- **Too large**: Dilutes meaning, hits token limits, reduces precision

The optimal size depends on content type and query patterns. Start with 500 characters and adjust based on retrieval quality testing.

</details>

---

### Q11: What is the `include` parameter in Chroma queries used for?

**Keywords:** Response fields, Performance, Documents, Embeddings, Metadatas

<details>
<summary>Click to Reveal Answer</summary>

The `include` parameter controls which fields are returned in query results:

```python
collection.query(
    query_texts=["..."],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)
```

Options: `"documents"`, `"metadatas"`, `"embeddings"`, `"distances"`

Only requesting needed fields improves performance by reducing data transfer. Skip `"embeddings"` unless you specifically need the vectors.

</details>

---

### Q12: What is batch ingestion and why is it faster than single-document inserts?

**Keywords:** Bulk operations, Overhead reduction, Efficiency, Batch embeddings, Parallelization

<details>
<summary>Click to Reveal Answer</summary>

Batch ingestion adds multiple documents in a single operation:

```python
collection.add(
    ids=["id1", "id2", "id3"],
    documents=["doc1", "doc2", "doc3"],
    metadatas=[{...}, {...}, {...}]
)
```

It's faster because:
1. Reduces per-operation overhead (connection, transaction)
2. Enables batch embedding (one model call for many texts)
3. Allows database to optimize bulk writes

Typical improvement: 5-50x faster than individual inserts.

</details>

---

### Q13: What is an embedding cache and what problem does it solve?

**Keywords:** Avoid recomputation, Hash key, Hit rate, Repeated queries, Performance

<details>
<summary>Click to Reveal Answer</summary>

An embedding cache stores computed embeddings indexed by content hash. When the same text is embedded again, the cached result is returned instead of recomputing.

Benefits:
1. Avoid expensive model inference for repeated content
2. Speed up re-indexing and updates
3. Reduce costs for API-based embedding models

Cache hit rates of 30-70% are common in systems with repeated patterns.

</details>

---

### Q14: How do you store dates in Chroma metadata for range queries?

**Keywords:** ISO strings, Comparison operators, String sorting, Timestamps

<details>
<summary>Click to Reveal Answer</summary>

Two approaches work:

1. **ISO date strings**: Store as "2024-01-15" - Chroma string comparison works because ISO format sorts chronologically
   ```python
   {"created_date": {"$gt": "2024-01-01"}}
   ```

2. **Unix timestamps**: Store as integers for numeric comparison
   ```python
   {"created_timestamp": {"$gt": 1704067200}}
   ```

ISO strings are more readable; timestamps are more precise and timezone-agnostic.

</details>

---

### Q15: What is the HNSW algorithm and why is it used in vector databases?

**Keywords:** Approximate Nearest Neighbor, Graph-based, Fast search, Trade-off, Scalability

<details>
<summary>Click to Reveal Answer</summary>

HNSW (Hierarchical Navigable Small World) is an algorithm for approximate nearest neighbor search. It builds a multi-layer graph of vectors for fast traversal.

Why it's used:
- **Speed**: O(log n) search time vs O(n) for brute force
- **Scalability**: Handles millions of vectors efficiently
- **Quality**: Near-perfect recall with proper tuning

Trade-off: It's "approximate" - may miss the true nearest neighbor occasionally, but is orders of magnitude faster.

</details>

---

### Q16: What backup information must be preserved for a Chroma database?

**Keywords:** Documents, IDs, Metadata, Regenerate embeddings, Schema

<details>
<summary>Click to Reveal Answer</summary>

Essential to backup:
1. **Document text**: The original content
2. **Document IDs**: Unique identifiers
3. **Metadata**: All structured attributes

Embeddings can be regenerated from documents using the same model, so they're optional in backups (though storing them can speed up restores).

Always document which embedding model was used - changing models requires re-embedding everything.

</details>

---

### Q17: What is multi-tenant isolation in vector databases?

**Keywords:** Tenant ID, Data separation, Metadata filtering, Security, SaaS

<details>
<summary>Click to Reveal Answer</summary>

Multi-tenant isolation ensures each tenant (customer/organization) can only access their own data. Common approaches:

1. **Metadata filtering**: Add `tenant_id` to every document and filter on every query
   ```python
   where={"tenant_id": "customer_123"}
   ```

2. **Separate collections**: One collection per tenant (more isolation, harder to manage)

3. **Separate instances**: Different database instances (maximum isolation, highest cost)

Metadata filtering is most common for SaaS applications.

</details>

---

## Intermediate (Application) - 25%

### Q18: Your RAG system often misses relevant content because answers span multiple chunks. How would you diagnose and fix this?

**Keywords:** Chunk boundaries, Overlap, Size adjustment, Multiple retrieval

**Hint:** Think about where information gets split.

<details>
<summary>Click to Reveal Answer</summary>

Diagnosis:
1. Review the source documents for missed queries
2. Check if relevant content falls at chunk boundaries
3. Analyze if chunks are too small (fragmenting information)

Solutions:
1. **Increase overlap**: From 10% to 20-30% to capture more boundary content
2. **Increase chunk size**: Larger chunks contain more complete information
3. **Retrieve more chunks**: Increase k (n_results) to get adjacent chunks
4. **Use recursive chunking**: Respects natural paragraph boundaries

Test changes with a benchmark of known good queries.

</details>

---

### Q19: You're building a documentation search system. Users want to search within specific product versions. How would you design the metadata schema?

**Keywords:** Version field, Filtering, Schema design, Future-proofing

**Hint:** Consider what other filters users might want later.

<details>
<summary>Click to Reveal Answer</summary>

Recommended metadata schema:
```python
{
    "product": "ProductName",          # String - filter by product
    "version": "2.1",                  # String - filter by version
    "major_version": 2,                # Integer - for >= queries
    "doc_type": "tutorial",            # String - tutorial/reference/api
    "language": "en",                  # String - internationalization
    "updated_date": "2024-01-15",      # ISO string - for recency
    "source_url": "https://..."        # For citations
}
```

Use strings for versions (semantic versioning doesn't sort numerically well) but add a numeric `major_version` for range queries like "version >= 2".

</details>

---

### Q20: Query latency in your vector database varies wildly - sometimes 20ms, sometimes 500ms. What could cause this and how would you investigate?

**Keywords:** Cold cache, Warming, GC pauses, Load spikes, Profiling

**Hint:** Consider what happens on the first query vs subsequent queries.

<details>
<summary>Click to Reveal Answer</summary>

Possible causes and investigation:

1. **Cold cache/first query**: First query loads model/index into memory
   - Solution: Warm-up queries on startup

2. **Garbage collection pauses**: Python/JVM GC can cause spikes
   - Solution: Monitor GC metrics, tune heap settings

3. **Embedding computation variation**: Some texts take longer to embed
   - Solution: Cache embeddings, set timeouts

4. **Database contention**: Concurrent writes during reads
   - Solution: Separate read/write replicas

5. **Variable query complexity**: More filters = more processing
   - Solution: Profile specific slow queries

Use APM tools to trace latency through each component.

</details>

---

### Q21: You need to migrate documents from one Chroma collection to another with a different embedding model. What's the process?

**Keywords:** Re-embedding, Model change, Batch processing, Validation

**Hint:** Can you copy embeddings directly?

<details>
<summary>Click to Reveal Answer</summary>

You cannot copy embeddings between different models - they're incompatible. The migration process:

1. **Export source data**: Get documents, IDs, and metadata
   ```python
   data = source_collection.get(include=["documents", "metadatas"])
   ```

2. **Re-embed with new model**: Generate new embeddings for all documents

3. **Import to target**: Add to new collection with new embeddings
   ```python
   target_collection.add(
       ids=data["ids"],
       documents=data["documents"],
       metadatas=data["metadatas"]
   )
   ```

4. **Validate**: Run test queries to compare result quality

5. **Switch over**: Update application to use new collection

</details>

---

### Q22: Your startup's RAG system works well locally but needs to deploy to production. What factors should influence your choice between local Chroma vs. a managed cloud service?

**Keywords:** Scale, Operations, Cost, SLAs, Team size

**Hint:** Think about both technical and business factors.

<details>
<summary>Click to Reveal Answer</summary>

Decision factors:

**Choose local/self-managed Chroma when:**
- Document count under 100K-1M
- Single-node performance is sufficient
- Team has ops expertise
- Cost sensitivity is high
- Data must stay on-premises

**Choose managed cloud (Pinecone, Weaviate Cloud) when:**
- Need horizontal scaling
- High availability/SLA requirements
- No ops team bandwidth
- Rapid growth expected
- Need managed backups/monitoring

**Hybrid approach:**
- Start with local PersistentClient
- Monitor performance metrics
- Migrate to managed when growth justifies cost

</details>

---

### Q23: Users report that searches return old versions of documents even after updates. What's likely happening and how do you fix it?

**Keywords:** Stale data, Update process, Embedding regeneration, Cache invalidation

**Hint:** What happens when you update document content?

<details>
<summary>Click to Reveal Answer</summary>

Likely causes:

1. **Embeddings not regenerated**: Updating document text requires new embeddings. If you only update metadata, the old semantic meaning persists.

2. **Using add() instead of upsert()**: `add()` fails silently if update logic catches the error

3. **Query cache returning stale results**: Application-level caching

4. **Wrong collection**: Querying old collection after migration

Fix:
```python
# Correct update process
collection.upsert(
    ids=["doc_123"],
    documents=["Updated content..."],  # New content
    metadatas=[{"version": "2.0"}]     # New metadata
)
# Upsert automatically regenerates embeddings
```

Clear any application caches after updates.

</details>

---

## Advanced (Deep Dive) - 5%

### Q24: Explain the trade-offs of the HNSW parameters M and ef_construction and how you would tune them for a production system.

**Keywords:** Connections per node, Build time, Recall, Memory, Index quality

<details>
<summary>Click to Reveal Answer</summary>

**M (connections per node):**
- Higher M = more connections per node in the graph
- Trade-offs:
  - More memory usage (linear with M)
  - Better recall (finds true nearest neighbors more often)
  - Slower build time
  - Slightly slower queries (more edges to traverse)
- Typical range: 12-48; default 16

**ef_construction (search width during build):**
- Higher ef = more candidates considered during index building
- Trade-offs:
  - Much slower index build time (linear with ef)
  - Better index quality = better search recall
  - No runtime memory impact
- Typical range: 100-500; default 100-200

**Production tuning approach:**
1. Start with defaults (M=16, ef=100)
2. Measure recall on benchmark queries
3. If recall is low, increase ef_construction first (free at query time)
4. If still low, increase M (adds memory cost)
5. Balance against your latency SLA and hardware budget

</details>

---

### Q25: Design a high-availability RAG system that handles 1000 queries/second across multiple regions. What are the key architectural components and trade-offs?

**Keywords:** Replication, Load balancing, Eventual consistency, Edge deployment, Cache layers

<details>
<summary>Click to Reveal Answer</summary>

**Architecture Components:**

1. **Global Load Balancer**: Route queries to nearest region

2. **Regional Clusters**: 
   - Multiple read replicas per region
   - One write coordinator
   - Async replication between regions

3. **Caching Layers**:
   - CDN for static responses
   - Query result cache (Redis)
   - Embedding cache (avoid re-computing common queries)

4. **Embedding Service**:
   - Separate microservice
   - GPU-accelerated nodes
   - Horizontally scaled

5. **Consistency Trade-offs**:
   - Accept eventual consistency (seconds lag between regions)
   - Use write-through cache for updates
   - Version documents to detect staleness

6. **LLM Tier**:
   - Pool API connections
   - Implement circuit breakers
   - Stream responses to reduce perceived latency

**Key Trade-offs**:
- Consistency vs. latency (cross-region sync)
- Cost vs. availability (redundancy)
- Freshness vs. cache hit rate

At 1000 QPS, budget for: ~$10-50K/month cloud infrastructure.

</details>

---

## Quick Reference

| Difficulty | Count | Percentage |
|------------|-------|------------|
| Beginner | 17 | 68% |
| Intermediate | 6 | 24% |
| Advanced | 2 | 8% |
| **Total** | **25** | 100% |

---

## Tips for Interview Success

1. **Use the Keywords**: Interviewers listen for specific terms. Work them naturally into your answers.
2. **Draw Diagrams**: For architecture questions, sketch the data flow before explaining.
3. **Discuss Trade-offs**: Senior engineers consider multiple options and their pros/cons.
4. **Give Concrete Examples**: "In my project, I set chunk size to 500 because..." is better than theory.
5. **Admit Gaps Confidently**: "I haven't worked with that specific tool, but the concept is similar to..."
