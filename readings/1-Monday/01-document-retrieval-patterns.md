# Document Retrieval Patterns

## Learning Objectives
- Understand advanced document retrieval patterns beyond basic similarity search
- Learn context window optimization techniques for LLM consumption
- Master semantic chunking boundaries to preserve meaning
- Implement multi-document retrieval strategies for comprehensive answers

## Why This Matters

Welcome to **Week 4: Vector Databases (Part 2)** and our epic: *"From Search to Systems: Building Production-Ready RAG Pipelines."* Last week, you learned the fundamentals of vector databases, similarity search, and introductory RAG concepts. Now we advance to building systems that work reliably in production.

Basic similarity search returns the most similar documents to a query. But what happens when:
- A single chunk doesn't contain enough context?
- The answer spans multiple documents?
- Your LLM's context window is limited?

These real-world challenges require **advanced retrieval patterns** that go beyond simple top-K similarity search. Companies building production RAG systems invest heavily in retrieval optimization because **retrieval quality directly determines answer quality**.

## The Concept

### Beyond Basic Top-K Retrieval

In Week 3, you learned to retrieve the top-K most similar chunks. While effective for simple queries, this approach has limitations:

```python
# Basic retrieval (Week 3 approach)
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=3
)
```

Production systems need more sophisticated patterns.

### Pattern 1: Context Window Optimization

LLMs have finite context windows (4K, 8K, 32K, 128K tokens). Your retrieval strategy must optimize how you fill this window.

**The Challenge**: Retrieve enough context for accurate answers without exceeding limits or including noise.

**Strategies**:

1. **Token-Aware Retrieval**: Track token counts during retrieval
```python
def retrieve_within_token_limit(collection, query, max_tokens=3000):
    """Retrieve chunks until token limit is reached."""
    results = collection.query(query_texts=[query], n_results=20)
    
    selected_chunks = []
    current_tokens = 0
    
    for doc in results['documents'][0]:
        doc_tokens = len(doc.split()) * 1.3  # Rough token estimate
        if current_tokens + doc_tokens <= max_tokens:
            selected_chunks.append(doc)
            current_tokens += doc_tokens
        else:
            break
    
    return selected_chunks
```

2. **Relevance Thresholding**: Only include chunks above a similarity threshold
```python
def retrieve_with_threshold(collection, query, threshold=0.7):
    """Only return chunks with similarity above threshold."""
    results = collection.query(
        query_texts=[query],
        n_results=10,
        include=["documents", "distances"]
    )
    
    # Filter by distance (lower is better for L2, convert to similarity)
    filtered = [
        doc for doc, dist in zip(results['documents'][0], results['distances'][0])
        if (1 - dist) >= threshold  # Approximate similarity conversion
    ]
    return filtered
```

### Pattern 2: Semantic Chunking Boundaries

Not all chunk boundaries are equal. Splitting mid-sentence or mid-paragraph breaks semantic coherence.

**Bad Chunking** (fixed-size):
```
Chunk 1: "Machine learning algorithms learn from data. They can"
Chunk 2: "identify patterns that humans might miss. The most common"
```

**Good Chunking** (semantic boundaries):
```
Chunk 1: "Machine learning algorithms learn from data. They can identify patterns that humans might miss."
Chunk 2: "The most common types include supervised learning, unsupervised learning, and reinforcement learning."
```

**Implementation Strategy**:
```python
def find_semantic_boundaries(text, target_size=500):
    """Split text at semantic boundaries (paragraphs, sentences)."""
    # First, try paragraph boundaries
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) <= target_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
```

### Pattern 3: Multi-Document Retrieval

Complex questions often require synthesizing information across multiple sources.

**Single-Document Retrieval**:
- Query: "What is Python?"
- Returns: One chunk about Python

**Multi-Document Retrieval**:
- Query: "Compare Python and JavaScript for data science"
- Needs: Chunks about Python + Chunks about JavaScript + Comparison context

**Strategies**:

1. **Query Decomposition**: Break complex queries into sub-queries
```python
def multi_query_retrieval(collection, complex_query):
    """Decompose query and retrieve for each component."""
    # In production, use an LLM to decompose
    sub_queries = decompose_query(complex_query)
    
    all_results = []
    seen_ids = set()
    
    for sub_query in sub_queries:
        results = collection.query(query_texts=[sub_query], n_results=3)
        for doc, id in zip(results['documents'][0], results['ids'][0]):
            if id not in seen_ids:
                all_results.append(doc)
                seen_ids.add(id)
    
    return all_results
```

2. **Contextual Retrieval**: Retrieve surrounding chunks for context
```python
def retrieve_with_context(collection, query, context_window=1):
    """Retrieve matching chunks plus surrounding context."""
    results = collection.query(query_texts=[query], n_results=3)
    
    expanded_results = []
    for chunk_id in results['ids'][0]:
        # Get the chunk and its neighbors
        chunk_num = int(chunk_id.split('_')[-1])
        neighbor_ids = [
            f"doc_{chunk_num + i}" 
            for i in range(-context_window, context_window + 1)
        ]
        neighbors = collection.get(ids=neighbor_ids)
        expanded_results.extend(neighbors['documents'])
    
    return expanded_results
```

### Pattern 4: Hybrid Retrieval

Combine semantic search with keyword search for better coverage.

```python
def hybrid_retrieval(collection, query, keyword_weight=0.3):
    """Combine semantic and keyword-based retrieval."""
    # Semantic search
    semantic_results = collection.query(
        query_texts=[query],
        n_results=10
    )
    
    # Keyword filter (using metadata or separate index)
    keywords = extract_keywords(query)
    keyword_results = collection.get(
        where={"$or": [{"content": {"$contains": kw}} for kw in keywords]}
    )
    
    # Combine and re-rank results
    combined = merge_and_rank(
        semantic_results, 
        keyword_results, 
        keyword_weight
    )
    
    return combined
```

## Code Example

Here's a practical implementation combining multiple patterns:

```python
import chromadb
from typing import List, Dict

class AdvancedRetriever:
    def __init__(self, collection):
        self.collection = collection
        self.max_tokens = 4000
    
    def retrieve(
        self, 
        query: str, 
        n_results: int = 5,
        similarity_threshold: float = 0.6,
        include_context: bool = True
    ) -> List[str]:
        """
        Advanced retrieval with multiple optimization patterns.
        
        Args:
            query: The search query
            n_results: Maximum number of results
            similarity_threshold: Minimum similarity score
            include_context: Whether to include surrounding chunks
        
        Returns:
            List of retrieved document chunks
        """
        # Step 1: Initial retrieval with extra candidates
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results * 2,  # Get extras for filtering
            include=["documents", "distances", "metadatas"]
        )
        
        # Step 2: Filter by similarity threshold
        filtered_docs = []
        filtered_ids = []
        for doc, dist, id in zip(
            results['documents'][0], 
            results['distances'][0],
            results['ids'][0]
        ):
            similarity = 1 / (1 + dist)  # Convert distance to similarity
            if similarity >= similarity_threshold:
                filtered_docs.append(doc)
                filtered_ids.append(id)
        
        # Step 3: Add context if requested
        if include_context:
            filtered_docs = self._expand_with_context(filtered_ids)
        
        # Step 4: Trim to token limit
        final_docs = self._trim_to_token_limit(filtered_docs)
        
        return final_docs[:n_results]
    
    def _expand_with_context(self, chunk_ids: List[str]) -> List[str]:
        """Add surrounding chunks for context."""
        # Implementation depends on chunk ID structure
        pass
    
    def _trim_to_token_limit(self, docs: List[str]) -> List[str]:
        """Ensure total tokens don't exceed limit."""
        selected = []
        current_tokens = 0
        
        for doc in docs:
            doc_tokens = len(doc.split()) * 1.3
            if current_tokens + doc_tokens <= self.max_tokens:
                selected.append(doc)
                current_tokens += doc_tokens
        
        return selected

# Usage
client = chromadb.Client()
collection = client.get_collection("my_docs")
retriever = AdvancedRetriever(collection)
results = retriever.retrieve("How do neural networks learn?")
```

## Key Takeaways

1. **Context window optimization** ensures you maximize information within LLM limits
2. **Semantic chunking boundaries** preserve meaning across splits
3. **Multi-document retrieval** handles complex queries requiring multiple sources
4. **Hybrid retrieval** combines semantic and keyword search for better recall
5. **Production systems layer multiple patterns** for robust retrieval

## Additional Resources

- [Advanced RAG Techniques (LlamaIndex)](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/) - Production optimization strategies
- [Retrieval-Augmented Generation for Knowledge-Intensive Tasks (Paper)](https://arxiv.org/abs/2005.11401) - The foundational RAG research paper
- [Chunking Strategies Guide (Pinecone)](https://www.pinecone.io/learn/chunking-strategies/) - Comprehensive chunking comparison
