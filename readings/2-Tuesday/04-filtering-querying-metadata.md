# Filtering and Querying with Metadata

## Learning Objectives
- Master metadata filtering syntax in vector databases
- Combine semantic search with structured queries
- Build advanced queries using logical operators
- Optimize queries for performance

## Why This Matters

Pure semantic search returns the most similar vectors. But real applications need more control:

- "Find similar documents, but **only from the last month**"
- "Show me relevant product descriptions **in the Electronics category**"
- "Search technical documentation **authored by the AI team**"

Metadata filtering transforms broad similarity search into precise, targeted retrieval. This is essential for production RAG systems where users expect filtered, relevant results.

## The Concept

### The Power of Hybrid Queries

```
SEMANTIC ONLY:
Query: "machine learning algorithms"
Result: All ML-related content (could be 1000s of chunks)

SEMANTIC + METADATA FILTER:
Query: "machine learning algorithms" WHERE category = "tutorials" AND year = 2024
Result: Recent tutorial content about ML (focused, relevant subset)
```

### Chroma Filter Syntax

Chroma uses a `where` clause for metadata filtering with JSON-like syntax.

#### Basic Equality
```python
# Single field equality
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"category": "documentation"}
)

# Multiple field equality (implicit AND)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={
        "category": "documentation",
        "language": "en"
    }
)
```

#### Comparison Operators
```python
# Greater than
where={"page_number": {"$gt": 10}}

# Less than
where={"chunk_index": {"$lt": 5}}

# Greater than or equal
where={"confidence": {"$gte": 0.8}}

# Less than or equal
where={"year": {"$lte": 2023}}

# Not equal
where={"status": {"$ne": "draft"}}
```

#### Logical Operators
```python
# AND - all conditions must match
where={
    "$and": [
        {"category": "technical"},
        {"year": {"$gte": 2023}}
    ]
}

# OR - any condition can match
where={
    "$or": [
        {"category": "tutorial"},
        {"category": "guide"}
    ]
}

# Combined AND/OR
where={
    "$and": [
        {"language": "en"},
        {"$or": [
            {"category": "tutorial"},
            {"category": "documentation"}
        ]}
    ]
}
```

#### String Operations
```python
# Contains (if supported)
where={"title": {"$contains": "machine learning"}}

# In list
where={"category": {"$in": ["tutorial", "guide", "documentation"]}}

# Not in list
where={"status": {"$nin": ["draft", "archived"]}}
```

### Document Filtering with `where_document`

Filter on document content in addition to metadata:

```python
# Filter documents containing specific text
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"category": "technical"},
    where_document={"$contains": "neural network"}
)
```

### Query Patterns

#### Pattern 1: Category-Scoped Search
```python
def search_in_category(
    collection,
    query_embedding: list[float],
    category: str,
    n_results: int = 5
) -> dict:
    """Search only within a specific category."""
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"category": category},
        include=["documents", "metadatas", "distances"]
    )
```

#### Pattern 2: Time-Bounded Search
```python
from datetime import datetime, timedelta

def search_recent(
    collection,
    query_embedding: list[float],
    days: int = 30,
    n_results: int = 5
) -> dict:
    """Search only documents from the last N days."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"created_at": {"$gte": cutoff}},
        include=["documents", "metadatas", "distances"]
    )
```

#### Pattern 3: Multi-Source Search
```python
def search_across_sources(
    collection,
    query_embedding: list[float],
    sources: list[str],
    n_results: int = 5
) -> dict:
    """Search across multiple source documents."""
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"source": {"$in": sources}},
        include=["documents", "metadatas", "distances"]
    )
```

#### Pattern 4: Exclude Certain Content
```python
def search_excluding(
    collection,
    query_embedding: list[float],
    exclude_categories: list[str],
    n_results: int = 5
) -> dict:
    """Search excluding certain categories."""
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"category": {"$nin": exclude_categories}},
        include=["documents", "metadatas", "distances"]
    )
```

#### Pattern 5: Confidence-Filtered Search
```python
def search_high_confidence(
    collection,
    query_embedding: list[float],
    min_confidence: float = 0.8,
    n_results: int = 5
) -> dict:
    """Only return high-confidence content."""
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"confidence_score": {"$gte": min_confidence}},
        include=["documents", "metadatas", "distances"]
    )
```

## Code Example

Complete metadata-aware query system:

```python
import chromadb
from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum


class FilterOperator(Enum):
    EQ = "$eq"      # Equal
    NE = "$ne"      # Not equal
    GT = "$gt"      # Greater than
    GTE = "$gte"    # Greater than or equal
    LT = "$lt"      # Less than
    LTE = "$lte"    # Less than or equal
    IN = "$in"      # In list
    NIN = "$nin"    # Not in list


@dataclass
class MetadataFilter:
    """Represents a single metadata filter."""
    field: str
    operator: FilterOperator
    value: Any
    
    def to_dict(self) -> dict:
        if self.operator == FilterOperator.EQ:
            return {self.field: self.value}
        return {self.field: {self.operator.value: self.value}}


class QueryBuilder:
    """Fluent interface for building filtered queries."""
    
    def __init__(self, collection):
        self.collection = collection
        self.filters: list[MetadataFilter] = []
        self.logic: str = "$and"
        self._n_results: int = 5
        self._include: list[str] = ["documents", "metadatas", "distances"]
    
    def where_equals(self, field: str, value: Any) -> 'QueryBuilder':
        """Add equality filter."""
        self.filters.append(MetadataFilter(field, FilterOperator.EQ, value))
        return self
    
    def where_in(self, field: str, values: list) -> 'QueryBuilder':
        """Add IN filter."""
        self.filters.append(MetadataFilter(field, FilterOperator.IN, values))
        return self
    
    def where_not_in(self, field: str, values: list) -> 'QueryBuilder':
        """Add NOT IN filter."""
        self.filters.append(MetadataFilter(field, FilterOperator.NIN, values))
        return self
    
    def where_greater_than(self, field: str, value: Any) -> 'QueryBuilder':
        """Add greater than filter."""
        self.filters.append(MetadataFilter(field, FilterOperator.GT, value))
        return self
    
    def where_less_than(self, field: str, value: Any) -> 'QueryBuilder':
        """Add less than filter."""
        self.filters.append(MetadataFilter(field, FilterOperator.LT, value))
        return self
    
    def where_gte(self, field: str, value: Any) -> 'QueryBuilder':
        """Add greater than or equal filter."""
        self.filters.append(MetadataFilter(field, FilterOperator.GTE, value))
        return self
    
    def where_lte(self, field: str, value: Any) -> 'QueryBuilder':
        """Add less than or equal filter."""
        self.filters.append(MetadataFilter(field, FilterOperator.LTE, value))
        return self
    
    def use_or_logic(self) -> 'QueryBuilder':
        """Use OR logic to combine filters."""
        self.logic = "$or"
        return self
    
    def use_and_logic(self) -> 'QueryBuilder':
        """Use AND logic to combine filters."""
        self.logic = "$and"
        return self
    
    def limit(self, n: int) -> 'QueryBuilder':
        """Set result limit."""
        self._n_results = n
        return self
    
    def include(self, *fields) -> 'QueryBuilder':
        """Specify what to include in results."""
        self._include = list(fields)
        return self
    
    def _build_where_clause(self) -> Optional[dict]:
        """Build the where clause from filters."""
        if not self.filters:
            return None
        
        if len(self.filters) == 1:
            return self.filters[0].to_dict()
        
        return {
            self.logic: [f.to_dict() for f in self.filters]
        }
    
    def execute(self, query_embedding: list[float]) -> dict:
        """Execute the query."""
        where_clause = self._build_where_clause()
        
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": self._n_results,
            "include": self._include
        }
        
        if where_clause:
            kwargs["where"] = where_clause
        
        return self.collection.query(**kwargs)
    
    def get_all(self) -> dict:
        """Get all documents matching filters (no embedding query)."""
        where_clause = self._build_where_clause()
        
        kwargs = {
            "limit": self._n_results,
            "include": self._include
        }
        
        if where_clause:
            kwargs["where"] = where_clause
        
        return self.collection.get(**kwargs)


class FilteredVectorStore:
    """Vector store with fluent query building."""
    
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
    
    def query(self) -> QueryBuilder:
        """Start building a query."""
        return QueryBuilder(self.collection)
    
    def simple_search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        **filters
    ) -> dict:
        """Simple search with keyword filters."""
        where = filters if filters else None
        
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )


# Usage Examples
store = FilteredVectorStore("knowledge_base")

# Example embedding (placeholder)
query_embedding = [0.1, 0.2, 0.3]

# Simple filtered search
results = store.simple_search(
    query_embedding,
    n_results=10,
    category="documentation"
)

# Fluent query building
results = (
    store.query()
    .where_equals("category", "technical")
    .where_gte("confidence_score", 0.8)
    .where_in("language", ["en", "es"])
    .limit(5)
    .execute(query_embedding)
)

# Complex query with OR logic
results = (
    store.query()
    .where_equals("language", "en")
    .use_or_logic()
    .where_equals("category", "tutorial")
    .where_equals("category", "guide")
    .limit(10)
    .execute(query_embedding)
)

# Get documents by filter only (no semantic search)
docs = (
    store.query()
    .where_equals("source_type", "pdf")
    .where_greater_than("page_number", 0)
    .limit(100)
    .get_all()
)
```

### Performance Considerations

```python
# Indexable vs Non-indexable fields
# Chroma creates indexes for filterable metadata

# GOOD: Uses index
where={"category": "tutorial"}

# GOOD: Uses index
where={"year": {"$gte": 2023}}

# SLOWER: String contains (may not use index)
where_document={"$contains": "specific phrase"}

# TIP: Pre-filter to reduce search space
# Do metadata filtering BEFORE semantic search when possible
```

## Key Takeaways

1. **Metadata filtering enables precise retrieval** - go beyond pure semantic search
2. **Use comparison operators** for numeric filtering ($gt, $lt, $gte, $lte)
3. **Combine filters with logical operators** ($and, $or) for complex queries
4. **Build fluent query interfaces** for cleaner application code
5. **Consider performance** - indexed fields filter faster
6. **Pre-filter large collections** to improve search speed

## Additional Resources

- [Chroma Query Documentation](https://docs.trychroma.com/usage-guide#querying-a-collection) - Official query syntax
- [Pinecone Metadata Filtering](https://docs.pinecone.io/docs/metadata-filtering) - Alternative syntax patterns
- [Vector Database Query Optimization](https://www.pinecone.io/learn/vector-search-performance/) - Performance tuning
