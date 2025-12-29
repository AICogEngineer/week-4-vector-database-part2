"""
Demo 03: Metadata Storage and Filtering

This demo shows trainees how to:
1. Store metadata alongside vectors in Chroma
2. Filter queries with metadata
3. Combine semantic search with structured filters

Prerequisites:
- pip install chromadb sentence-transformers
"""

import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================================
# PART 1: Why Metadata Matters
# ============================================================================

print("=" * 70)
print("PART 1: Why Metadata Matters")
print("=" * 70)

print("""
VECTORS ALONE: "What's similar to my query?"

VECTORS + METADATA: "What's similar AND from this document/category/date?"

Metadata enables PRECISION RETRIEVAL!
""")

# ============================================================================
# PART 2: Setup and Sample Data
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Setup and Sample Data")
print("=" * 70)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()

try:
    client.delete_collection("metadata_demo")
except:
    pass

collection = client.create_collection("metadata_demo")

# Simple documents with metadata
docs = [
    {
        "id": "ml_1",
        "content": "Machine learning enables computers to learn from data.",
        "metadata": {"source": "ml_guide.pdf", "category": "tutorial", "year": 2024}
    },
    {
        "id": "ml_2", 
        "content": "Supervised learning uses labeled data for training models.",
        "metadata": {"source": "ml_guide.pdf", "category": "tutorial", "year": 2024}
    },
    {
        "id": "dl_1",
        "content": "Deep learning uses neural networks with many layers.",
        "metadata": {"source": "deep_learning.md", "category": "reference", "year": 2024}
    },
    {
        "id": "nlp_1",
        "content": "NLP enables computers to understand human language.",
        "metadata": {"source": "nlp_guide.html", "category": "tutorial", "year": 2023}
    },
    {
        "id": "rag_1",
        "content": "RAG combines retrieval with generation for better answers.",
        "metadata": {"source": "rag_patterns.pdf", "category": "reference", "year": 2024}
    },
]

# Add to collection
contents = [d["content"] for d in docs]
embeddings = embedder.encode(contents).tolist()

collection.add(
    ids=[d["id"] for d in docs],
    documents=contents,
    embeddings=embeddings,
    metadatas=[d["metadata"] for d in docs]
)

print(f"✓ Added {len(docs)} documents with metadata")
print(f"\nSample metadata: {docs[0]['metadata']}")

# ============================================================================
# PART 3: Basic Filtering
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Basic Filtering")
print("=" * 70)

query = "How do machines learn?"
query_emb = embedder.encode([query]).tolist()

print(f'\nQuery: "{query}"')

# No filter
print("\n[1] No filter:")
results = collection.query(query_embeddings=query_emb, n_results=3)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['source']}] {doc[:40]}...")

# Filter by category
print("\n[2] Filter: category = 'tutorial'")
results = collection.query(
    query_embeddings=query_emb,
    n_results=3,
    where={"category": "tutorial"}
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['category']}] {doc[:40]}...")

# Filter by year
print("\n[3] Filter: year > 2023")
results = collection.query(
    query_embeddings=query_emb,
    n_results=3,
    where={"year": {"$gt": 2023}}
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['year']}] {doc[:40]}...")

# ============================================================================
# PART 4: Filter Operators
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Filter Operators")
print("=" * 70)

print("""
CHROMA OPERATORS:
  $eq   - Equal (default)     {"field": "value"} or {"field": {"$eq": "value"}}
  $ne   - Not equal           {"field": {"$ne": "value"}}
  $gt   - Greater than        {"field": {"$gt": 10}}
  $gte  - Greater or equal    {"field": {"$gte": 10}}
  $lt   - Less than           {"field": {"$lt": 10}}
  $lte  - Less or equal       {"field": {"$lte": 10}}
  $in   - In list             {"field": {"$in": ["a", "b"]}}
  $nin  - Not in list         {"field": {"$nin": ["a", "b"]}}
""")

# $in example
print("[4] Filter: category IN ['tutorial', 'reference']")
results = collection.query(
    query_embeddings=query_emb,
    n_results=5,
    where={"category": {"$in": ["tutorial", "reference"]}}
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['category']}] {doc[:40]}...")

# ============================================================================
# PART 5: AND/OR Logic
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: AND/OR Logic")
print("=" * 70)

print("""
LOGICAL OPERATORS:
  $and - All conditions must match
  $or  - Any condition can match
""")

# AND
print("[5] AND: category='tutorial' AND year=2024")
results = collection.query(
    query_embeddings=query_emb,
    n_results=3,
    where={
        "$and": [
            {"category": "tutorial"},
            {"year": 2024}
        ]
    }
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['category']}, {meta['year']}] {doc[:35]}...")

# OR
print("\n[6] OR: source ends with .pdf OR .md")
results = collection.query(
    query_embeddings=query_emb,
    n_results=5,
    where={
        "$or": [
            {"source": {"$in": ["ml_guide.pdf", "rag_patterns.pdf"]}},
            {"source": "deep_learning.md"}
        ]
    }
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['source']}] {doc[:35]}...")

# ============================================================================
# QUICK REFERENCE
# ============================================================================

print("\n" + "=" * 70)
print("QUICK REFERENCE")
print("=" * 70)

print("""
# Basic equality
where={"category": "tutorial"}

# Comparison
where={"year": {"$gt": 2023}}

# In list
where={"source": {"$in": ["a.pdf", "b.pdf"]}}

# AND (all must match)
where={"$and": [{"category": "tutorial"}, {"year": 2024}]}

# OR (any can match)
where={"$or": [{"category": "tutorial"}, {"category": "reference"}]}

# Nested
where={"$and": [{"year": 2024}, {"$or": [{"category": "tutorial"}, {"category": "reference"}]}]}
""")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO 3 COMPLETE: Metadata Filtering")
print("=" * 70)

print("""
Key Takeaways:

1. METADATA = precision retrieval (filter by source, category, date, etc.)
2. OPERATORS: $eq, $ne, $gt, $lt, $in, $nin
3. LOGIC: $and, $or for complex queries
4. ALWAYS use metadata in production RAG!
""")

print("=" * 70)
