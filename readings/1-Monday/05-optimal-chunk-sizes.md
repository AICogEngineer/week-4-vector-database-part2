# Optimal Chunk Sizes

## Learning Objectives
- Understand how chunk size affects embedding quality and retrieval
- Learn to determine optimal chunk sizes for different embedding models
- Apply experimental methodology to find the best chunk size for your use case
- Balance trade-offs between precision and context completeness

## Why This Matters

Chunk size is one of the most impactful hyperparameters in a RAG system, yet many tutorials gloss over it with a simple "use 500 characters." The reality is more nuanced:

- Too small: Lose context, retrieve fragments
- Too large: Lose precision, retrieve noise
- Just right: Depends on your model, data, and use case

Finding the optimal chunk size can dramatically improve your RAG system's performance. This reading gives you the framework to find it systematically rather than guessing.

## The Concept

### Why Chunk Size Matters

```
Query: "How does backpropagation work?"

SMALL CHUNKS (100 chars):
Retrieved: "Backpropagation is an"  
Problem: Fragment, no useful information

LARGE CHUNKS (2000 chars):
Retrieved: "Chapter 5: Neural Networks [contains backpropagation 
           explanation mixed with 10 other topics]"
Problem: Noisy, answer buried in irrelevant content

OPTIMAL CHUNKS (400 chars):
Retrieved: "Backpropagation is the algorithm used to train neural 
           networks. It works by computing gradients of the loss 
           function with respect to each weight, then updating 
           weights to minimize loss. The process flows backward 
           from output to input layers."
Result: Complete, focused, useful
```

### Factors Affecting Optimal Chunk Size

#### 1. Embedding Model Constraints

Every embedding model has limits:

| Model | Max Tokens | Recommended Chunk Size |
|-------|-----------|----------------------|
| OpenAI text-embedding-3-small | 8,191 | 500-1000 tokens |
| all-MiniLM-L6-v2 | 256 | 200-256 tokens |
| all-mpnet-base-v2 | 384 | 300-384 tokens |
| BERT-base | 512 | 400-512 tokens |
| E5-large | 512 | 400-512 tokens |

**Token vs Character Estimation**:
```python
def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ~ 4 characters or 0.75 words."""
    return len(text) // 4  # Character-based
    # OR
    return int(len(text.split()) * 1.3)  # Word-based
```

#### 2. Content Type

Different content needs different chunking:

| Content Type | Suggested Size | Reasoning |
|-------------|---------------|-----------|
| Code documentation | 200-400 tokens | Functions are often self-contained |
| Legal documents | 400-600 tokens | Clauses need full context |
| Academic papers | 500-800 tokens | Arguments span paragraphs |
| Chat logs | 100-200 tokens | Conversations are turn-based |
| Product descriptions | 150-300 tokens | Each product is distinct |

#### 3. Query Patterns

How users query affects optimal size:

- **Factual questions** ("What year was X founded?"): Smaller chunks (higher precision)
- **Conceptual questions** ("Explain how X works"): Larger chunks (more context)
- **Comparative questions** ("Compare X and Y"): Medium chunks (balanced)

### Experimental Methodology

The best way to find optimal chunk size is experimentation. Here's a systematic approach:

#### Step 1: Create an Evaluation Dataset

```python
evaluation_set = [
    {
        "query": "How does gradient descent work?",
        "expected_source": "chapter3.md",
        "expected_keywords": ["gradient", "learning rate", "update"]
    },
    {
        "query": "What are the benefits of neural networks?",
        "expected_source": "chapter5.md", 
        "expected_keywords": ["pattern recognition", "flexibility"]
    },
    # Add 10-20 representative queries
]
```

#### Step 2: Test Multiple Chunk Sizes

```python
import chromadb
from sentence_transformers import SentenceTransformer

def evaluate_chunk_size(
    documents: list[str],
    chunk_size: int,
    eval_set: list[dict],
    overlap: int = 50
) -> dict:
    """Evaluate retrieval quality for a given chunk size."""
    
    # Setup
    client = chromadb.Client()
    collection = client.create_collection(f"test_{chunk_size}")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Chunk and embed documents
    all_chunks = []
    all_metadata = []
    for doc in documents:
        chunks = chunk_document(doc["content"], chunk_size, overlap)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({"source": doc["source"]})
    
    embeddings = embedder.encode(all_chunks).tolist()
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata
    )
    
    # Evaluate
    results = {
        "chunk_size": chunk_size,
        "total_chunks": len(all_chunks),
        "precision_at_1": 0,
        "precision_at_3": 0,
        "keyword_recall": 0,
    }
    
    for eval_item in eval_set:
        query_embedding = embedder.encode([eval_item["query"]]).tolist()
        retrieved = collection.query(
            query_embeddings=query_embedding,
            n_results=3,
            include=["documents", "metadatas"]
        )
        
        # Check if correct source in top-1
        if retrieved["metadatas"][0][0]["source"] == eval_item["expected_source"]:
            results["precision_at_1"] += 1
        
        # Check if correct source in top-3
        sources = [m["source"] for m in retrieved["metadatas"][0]]
        if eval_item["expected_source"] in sources:
            results["precision_at_3"] += 1
        
        # Check keyword recall
        top_doc = retrieved["documents"][0][0].lower()
        found_keywords = sum(
            1 for kw in eval_item["expected_keywords"] 
            if kw.lower() in top_doc
        )
        results["keyword_recall"] += found_keywords / len(eval_item["expected_keywords"])
    
    # Average scores
    n = len(eval_set)
    results["precision_at_1"] /= n
    results["precision_at_3"] /= n
    results["keyword_recall"] /= n
    
    # Cleanup
    client.delete_collection(f"test_{chunk_size}")
    
    return results
```

#### Step 3: Compare Results

```python
def find_optimal_chunk_size(
    documents: list[dict],
    eval_set: list[dict],
    chunk_sizes: list[int] = [100, 200, 300, 400, 500, 750, 1000]
) -> dict:
    """Find the optimal chunk size through experimentation."""
    
    results = []
    for size in chunk_sizes:
        print(f"Testing chunk size: {size}")
        result = evaluate_chunk_size(documents, size, eval_set)
        results.append(result)
        print(f"  Precision@1: {result['precision_at_1']:.2f}")
        print(f"  Precision@3: {result['precision_at_3']:.2f}")
        print(f"  Keyword Recall: {result['keyword_recall']:.2f}")
    
    # Find best by combined score
    for r in results:
        r["combined_score"] = (
            r["precision_at_1"] * 0.4 + 
            r["precision_at_3"] * 0.3 + 
            r["keyword_recall"] * 0.3
        )
    
    best = max(results, key=lambda x: x["combined_score"])
    print(f"\nOptimal chunk size: {best['chunk_size']}")
    
    return {
        "all_results": results,
        "optimal_size": best["chunk_size"],
        "optimal_metrics": best
    }
```

### Quick Start Guidelines

If you can't run experiments, use these evidence-based defaults:

| Embedding Model | Conservative | Balanced | Aggressive |
|-----------------|-------------|----------|------------|
| **OpenAI** | 256 tokens | 512 tokens | 1024 tokens |
| **all-MiniLM-L6-v2** | 128 tokens | 200 tokens | 256 tokens |
| **E5/BGE** | 256 tokens | 384 tokens | 512 tokens |

**General Rules**:
- Start with 80% of model's max token limit
- Use 10-20% overlap between chunks
- Adjust based on content type

### Overlap Considerations

Overlap helps maintain context between chunks:

```
Without Overlap:
Chunk 1: "...neural networks learn through backpropagation."
Chunk 2: "This algorithm calculates gradients..."

With 50-token Overlap:
Chunk 1: "...neural networks learn through backpropagation."
Chunk 2: "...through backpropagation. This algorithm calculates gradients..."
```

**Overlap Guidelines**:
- 10-20% of chunk size is typical
- More overlap = better context continuity
- More overlap = more storage and computation

```python
def calculate_overlap(chunk_size: int, overlap_percent: float = 0.15) -> int:
    """Calculate overlap based on percentage of chunk size."""
    return int(chunk_size * overlap_percent)

# Example: 500 char chunks with 15% overlap = 75 char overlap
```

## Code Example

Complete chunk size optimization workflow:

```python
from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class ChunkSizeResult:
    """Results from chunk size evaluation."""
    chunk_size: int
    num_chunks: int
    precision_at_1: float
    precision_at_3: float
    avg_chunk_length: float
    score: float


class ChunkSizeOptimizer:
    """Find optimal chunk size for your RAG system."""
    
    def __init__(
        self,
        documents: list[dict],
        eval_queries: list[dict],
        embedding_model_max_tokens: int = 256
    ):
        self.documents = documents
        self.eval_queries = eval_queries
        self.max_tokens = embedding_model_max_tokens
        self.results: list[ChunkSizeResult] = []
    
    def recommend_sizes_to_test(self) -> list[int]:
        """Recommend chunk sizes based on model limits."""
        max_chars = self.max_tokens * 4  # Rough token-to-char
        
        return [
            int(max_chars * 0.25),  # 25% of max
            int(max_chars * 0.50),  # 50% of max
            int(max_chars * 0.75),  # 75% of max
            int(max_chars * 0.90),  # 90% of max (near max)
            int(max_chars * 0.50) + 100,  # Slightly above middle
        ]
    
    def quick_estimate(self, sample_size: int = 3) -> int:
        """Quick estimate using content analysis."""
        # Sample some documents
        sample_docs = self.documents[:sample_size]
        
        # Extract paragraph lengths
        para_lengths = []
        for doc in sample_docs:
            paragraphs = re.split(r'\n\n+', doc["content"])
            para_lengths.extend([len(p) for p in paragraphs if p.strip()])
        
        if not para_lengths:
            return int(self.max_tokens * 4 * 0.5)  # Default to 50% of max
        
        # Use median paragraph length as starting point
        sorted_lengths = sorted(para_lengths)
        median_length = sorted_lengths[len(sorted_lengths) // 2]
        
        # Clamp to model limits
        max_chars = self.max_tokens * 4
        recommended = min(median_length, int(max_chars * 0.9))
        recommended = max(recommended, 100)  # At least 100 chars
        
        return recommended
    
    def get_optimal_size(self) -> int:
        """Return the best chunk size after evaluation."""
        if not self.results:
            return self.quick_estimate()
        
        best = max(self.results, key=lambda x: x.score)
        return best.chunk_size
    
    def summary(self) -> str:
        """Generate summary of optimization results."""
        if not self.results:
            estimate = self.quick_estimate()
            return f"Quick estimate (no evaluation): {estimate} characters"
        
        lines = ["Chunk Size Optimization Results:\n"]
        lines.append(f"{'Size':<8} {'Chunks':<8} {'P@1':<8} {'P@3':<8} {'Score':<8}")
        lines.append("-" * 40)
        
        for r in sorted(self.results, key=lambda x: x.chunk_size):
            lines.append(
                f"{r.chunk_size:<8} {r.num_chunks:<8} "
                f"{r.precision_at_1:.2f}    {r.precision_at_3:.2f}    {r.score:.2f}"
            )
        
        best = max(self.results, key=lambda x: x.score)
        lines.append(f"\nRecommended: {best.chunk_size} characters")
        
        return "\n".join(lines)


# Usage
optimizer = ChunkSizeOptimizer(
    documents=[
        {"content": "Your document content...", "source": "doc1.md"}
    ],
    eval_queries=[
        {"query": "Test query", "expected_source": "doc1.md"}
    ],
    embedding_model_max_tokens=256  # all-MiniLM-L6-v2
)

# Get quick estimate
quick_size = optimizer.quick_estimate()
print(f"Quick estimate: {quick_size} characters")

# Get recommended sizes to test
sizes_to_test = optimizer.recommend_sizes_to_test()
print(f"Recommended test sizes: {sizes_to_test}")
```

## Key Takeaways

1. **Chunk size is a critical hyperparameter** - don't use arbitrary defaults
2. **Respect embedding model limits** - stay under max token count
3. **Content type matters** - code needs different sizing than prose
4. **Experiment when possible** - use evaluation datasets to find optimal size
5. **Use overlap for context continuity** - 10-20% is a good starting point
6. **Start conservative** - it's easier to increase size than decrease

## Additional Resources

- [OpenAI Embedding Best Practices](https://platform.openai.com/docs/guides/embeddings/use-cases) - Official guidance on embedding usage
- [Sentence Transformers Documentation](https://www.sbert.net/) - Details on model limits and recommendations
- [Evaluating Retrieval Quality (MTEB)](https://huggingface.co/spaces/mteb/leaderboard) - Benchmark for embedding models
