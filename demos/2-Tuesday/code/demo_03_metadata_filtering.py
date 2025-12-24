"""
Demo 03: Metadata Storage and Filtering

This demo shows trainees how to:
1. Design effective metadata schemas
2. Store metadata alongside vectors in Chroma
3. Build filtered queries with metadata
4. Combine semantic search with structured filters

Learning Objectives:
- Understand the role of metadata in vector databases
- Design schemas for different use cases
- Master Chroma's filtering syntax
- Build production-ready hybrid queries

References:
- Written Content: 03-storing-metadata-vectors.md
- Written Content: 04-filtering-querying-metadata.md
"""

import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

# ============================================================================
# PART 1: Why Metadata Matters
# ============================================================================

print("=" * 70)
print("PART 1: Why Metadata Matters")
print("=" * 70)

print("""
VECTORS ALONE can only answer: "What's similar to my query?"

VECTORS + METADATA can answer:
- "What's similar AND from this document?"
- "What's similar AND from the last month?"
- "What's similar AND in the 'tutorials' category?"

Metadata transforms semantic search into PRECISION RETRIEVAL!

EXAMPLE:
─────────
Vector: [0.23, -0.45, 0.67, ...]  →  "Some ML content"

Vector + Metadata:
  [0.23, -0.45, 0.67, ...]  →  "Some ML content"
                             + source: "ml_guide.pdf"
                             + category: "tutorial"  
                             + date: "2024-01-15"
                             + page: 42
""")

# ============================================================================
# PART 2: Designing a Metadata Schema
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Designing a Metadata Schema")
print("=" * 70)

print("""
SCHEMA DESIGN PRINCIPLES:
─────────────────────────
1. Think about your QUERY PATTERNS first
2. Keep it FLAT (no nesting in most DBs)
3. Use appropriate DATA TYPES
4. Avoid NULL values (use empty strings instead)
""")

# Schema examples
MINIMAL_SCHEMA = {
    "source": "str - filename or URL",
    "chunk_id": "str - unique identifier"
}

STANDARD_SCHEMA = {
    "source": "str - filename or URL",
    "source_type": "str - pdf, html, md, etc.",
    "chunk_index": "int - position in document",
    "total_chunks": "int - chunks in document",
    "created_at": "str - ISO datetime",
    "category": "str - content category"
}

RICH_SCHEMA = {
    # Source tracking
    "source": "str",
    "source_type": "str",
    "source_url": "str",
    
    # Chunk info
    "chunk_id": "str",
    "chunk_index": "int",
    "total_chunks": "int",
    
    # Content info
    "section_heading": "str",
    "page_number": "int",
    "word_count": "int",
    
    # Temporal
    "document_date": "str",
    "indexed_at": "str",
    
    # Categorical
    "category": "str",
    "subcategory": "str",
    "language": "str",
    
    # Quality
    "confidence": "float"
}

print("Schema Examples:")
print("-" * 50)
print("\nMINIMAL (for prototyping):")
for k, v in MINIMAL_SCHEMA.items():
    print(f"  {k}: {v}")

print("\nSTANDARD (recommended):")
for k, v in STANDARD_SCHEMA.items():
    print(f"  {k}: {v}")

print("\nRICH (enterprise):")
print(f"  {len(RICH_SCHEMA)} fields including source, chunk, content, temporal, categorical, and quality metadata")

# ============================================================================
# PART 3: Setting Up Chroma with Metadata
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Storing Documents with Metadata")
print("=" * 70)

# Initialize embedder and Chroma
print("\n[Step 1] Initializing components...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()

# Clean up if exists
try:
    client.delete_collection("metadata_demo")
except:
    pass

collection = client.create_collection(
    name="metadata_demo",
    metadata={"hnsw:space": "cosine"}
)
print("  ✓ Chroma collection created")

# Sample documents with rich metadata
SAMPLE_DOCUMENTS = [
    {
        "id": "ml_basics_1",
        "content": "Machine learning is a subset of AI that enables computers to learn from data.",
        "metadata": {
            "source": "ml_guide.pdf",
            "source_type": "pdf",
            "category": "tutorial",
            "subcategory": "basics",
            "chunk_index": 0,
            "page_number": 1,
            "language": "en",
            "year": 2024
        }
    },
    {
        "id": "ml_basics_2",
        "content": "Supervised learning uses labeled data to train models for classification and regression.",
        "metadata": {
            "source": "ml_guide.pdf",
            "source_type": "pdf",
            "category": "tutorial",
            "subcategory": "supervised",
            "chunk_index": 1,
            "page_number": 5,
            "language": "en",
            "year": 2024
        }
    },
    {
        "id": "dl_intro_1",
        "content": "Deep learning uses neural networks with many layers to learn complex patterns.",
        "metadata": {
            "source": "deep_learning.md",
            "source_type": "markdown",
            "category": "reference",
            "subcategory": "neural_networks",
            "chunk_index": 0,
            "page_number": 1,
            "language": "en",
            "year": 2024
        }
    },
    {
        "id": "nlp_guide_1",
        "content": "Natural language processing enables computers to understand human language.",
        "metadata": {
            "source": "nlp_handbook.html",
            "source_type": "html",
            "category": "tutorial",
            "subcategory": "nlp",
            "chunk_index": 0,
            "page_number": 1,
            "language": "en",
            "year": 2023
        }
    },
    {
        "id": "rag_systems_1",
        "content": "RAG systems combine retrieval with generation for better question answering.",
        "metadata": {
            "source": "rag_patterns.pdf",
            "source_type": "pdf",
            "category": "reference",
            "subcategory": "rag",
            "chunk_index": 0,
            "page_number": 1,
            "language": "en",
            "year": 2024
        }
    },
    {
        "id": "python_ml_1",
        "content": "Python is the most popular language for machine learning due to its rich ecosystem.",
        "metadata": {
            "source": "python_guide.md",
            "source_type": "markdown",
            "category": "tutorial",
            "subcategory": "tools",
            "chunk_index": 0,
            "page_number": 1,
            "language": "en",
            "year": 2023
        }
    }
]

print("\n[Step 2] Generating embeddings...")
contents = [doc["content"] for doc in SAMPLE_DOCUMENTS]
embeddings = embedder.encode(contents).tolist()
print(f"  ✓ Generated {len(embeddings)} embeddings")

print("\n[Step 3] Adding documents with metadata...")
collection.add(
    ids=[doc["id"] for doc in SAMPLE_DOCUMENTS],
    documents=contents,
    embeddings=embeddings,
    metadatas=[doc["metadata"] for doc in SAMPLE_DOCUMENTS]
)
print(f"  ✓ Stored {len(SAMPLE_DOCUMENTS)} documents")

# Verify storage
print("\n[Step 4] Verifying stored metadata...")
stored = collection.get(ids=["ml_basics_1"], include=["metadatas"])
print(f"  Sample metadata for 'ml_basics_1':")
for key, value in stored["metadatas"][0].items():
    print(f"    {key}: {value}")

# ============================================================================
# PART 4: Basic Metadata Filtering
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Basic Metadata Filtering")
print("=" * 70)

query_text = "How do machines learn from data?"
query_embedding = embedder.encode([query_text]).tolist()

print(f"\nQuery: \"{query_text}\"")
print("-" * 50)

# No filter (baseline)
print("\n[Test 1] No filter (all results):")
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"  [{meta['source']}] {doc[:50]}... (dist: {dist:.4f})")

# Filter by category
print("\n[Test 2] Filter: category = 'tutorial':")
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
    where={"category": "tutorial"},
    include=["documents", "metadatas", "distances"]
)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"  [{meta['category']}] {doc[:50]}... (dist: {dist:.4f})")

# Filter by source type
print("\n[Test 3] Filter: source_type = 'pdf':")
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
    where={"source_type": "pdf"},
    include=["documents", "metadatas", "distances"]
)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"  [{meta['source_type']}] {doc[:50]}... (dist: {dist:.4f})")

# ============================================================================
# PART 5: Comparison Operators
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Comparison Operators")
print("=" * 70)

print("""
CHROMA FILTER OPERATORS:
─────────────────────────
$eq   - Equal (default)
$ne   - Not equal
$gt   - Greater than
$gte  - Greater than or equal
$lt   - Less than
$lte  - Less than or equal
$in   - In list
$nin  - Not in list
""")

# Greater than
print("\n[Test 4] Filter: year > 2023 (documents from 2024):")
results = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
    where={"year": {"$gt": 2023}},
    include=["documents", "metadatas"]
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [Year {meta['year']}] {doc[:40]}...")

# In list
print("\n[Test 5] Filter: source_type IN ['pdf', 'markdown']:")
results = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
    where={"source_type": {"$in": ["pdf", "markdown"]}},
    include=["documents", "metadatas"]
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['source_type']}] {doc[:40]}...")

# Page number comparison
print("\n[Test 6] Filter: page_number > 1:")
results = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
    where={"page_number": {"$gt": 1}},
    include=["documents", "metadatas"]
)

if results["documents"][0]:
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  [Page {meta['page_number']}] {doc[:40]}...")
else:
    print("  (No results - all our demo docs are on page 1 or 5)")

# ============================================================================
# PART 6: Complex Queries with AND/OR
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Complex Queries with AND/OR")
print("=" * 70)

print("""
LOGICAL OPERATORS:
──────────────────
$and - All conditions must match
$or  - Any condition can match
""")

# AND query
print("\n[Test 7] AND: category='tutorial' AND year=2024:")
results = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
    where={
        "$and": [
            {"category": "tutorial"},
            {"year": 2024}
        ]
    },
    include=["documents", "metadatas"]
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['category']}, {meta['year']}] {doc[:40]}...")

# OR query
print("\n[Test 8] OR: category='tutorial' OR category='reference':")
results = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
    where={
        "$or": [
            {"category": "tutorial"},
            {"category": "reference"}
        ]
    },
    include=["documents", "metadatas"]
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['category']}] {doc[:40]}...")

# Combined AND/OR
print("\n[Test 9] Complex: (category='tutorial' OR category='reference') AND year=2024:")
results = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
    where={
        "$and": [
            {"year": 2024},
            {"$or": [
                {"category": "tutorial"},
                {"category": "reference"}
            ]}
        ]
    },
    include=["documents", "metadatas"]
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['category']}, {meta['year']}] {doc[:40]}...")

# ============================================================================
# PART 7: Building a Query Helper
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: Query Helper Class")
print("=" * 70)

class FilteredQueryBuilder:
    """
    Fluent interface for building filtered queries.
    
    Usage:
        builder = FilteredQueryBuilder(collection, embedder)
        results = (builder
            .where_equals("category", "tutorial")
            .where_in("year", [2023, 2024])
            .limit(5)
            .search("machine learning"))
    """
    
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder
        self.filters = []
        self._n_results = 5
    
    def where_equals(self, field: str, value) -> 'FilteredQueryBuilder':
        """Add equality filter."""
        self.filters.append({field: value})
        return self
    
    def where_in(self, field: str, values: list) -> 'FilteredQueryBuilder':
        """Add IN filter."""
        self.filters.append({field: {"$in": values}})
        return self
    
    def where_greater_than(self, field: str, value) -> 'FilteredQueryBuilder':
        """Add greater than filter."""
        self.filters.append({field: {"$gt": value}})
        return self
    
    def limit(self, n: int) -> 'FilteredQueryBuilder':
        """Set result limit."""
        self._n_results = n
        return self
    
    def _build_where(self) -> Optional[Dict]:
        """Build the where clause."""
        if not self.filters:
            return None
        if len(self.filters) == 1:
            return self.filters[0]
        return {"$and": self.filters}
    
    def search(self, query: str) -> Dict:
        """Execute the search."""
        embedding = self.embedder.encode([query]).tolist()
        where = self._build_where()
        
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=self._n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Reset for next query
        self.filters = []
        
        return results
    
    def reset(self) -> 'FilteredQueryBuilder':
        """Reset filters for new query."""
        self.filters = []
        return self


print("Using the FilteredQueryBuilder:")
print("-" * 50)

builder = FilteredQueryBuilder(collection, embedder)

print("\n[Fluent Query] Tutorials from 2024 about learning:")
results = (builder
    .where_equals("category", "tutorial")
    .where_in("year", [2024])
    .limit(3)
    .search("how do machines learn"))

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  [{meta['source']}] {doc[:50]}...")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO 3 COMPLETE: Metadata Storage and Filtering")
print("=" * 70)

print("""
Key Takeaways:

1. METADATA ENABLES HYBRID SEARCH
   - Combine semantic similarity with structured filters
   - Much more precise than vectors alone

2. DESIGN SCHEMAS FOR YOUR QUERIES
   - Think about what you'll filter by
   - Keep it flat, avoid null values
   - Include source, temporal, and categorical fields

3. CHROMA FILTER SYNTAX
   - Equality: {"field": "value"}
   - Comparison: {"field": {"$gt": value}}
   - Logical: {"$and": [...]} or {"$or": [...]}

4. BUILD QUERY HELPERS
   - Fluent API makes complex queries readable
   - Encapsulates common patterns

Production RAG systems ALWAYS use metadata filtering!
""")

# ============================================================================
# INSTRUCTOR NOTES
# ============================================================================

print("\n" + "=" * 70)
print("INSTRUCTOR NOTES")
print("=" * 70)

print("""
Discussion Questions:
1. "What metadata would you add for your documents?"
2. "When would you use OR vs AND?"
3. "How would you handle user permissions with metadata?"

Interactive Exercise:
- Have trainees design a schema for their use case
- Build queries together based on their requirements

Common Confusions:
- "Can I nest metadata?" → No, keep it flat
- "What about array values?" → Most DBs don't support well
- "How do I update metadata?" → collection.update() method

If Running Short on Time:
- Skip the complex AND/OR section
- Focus on basic filtering

If Trainees Are Advanced:
- Discuss metadata indexing performance
- Mention vector + keyword hybrid search
""")

print("\n" + "=" * 70)
