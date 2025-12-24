"""
Exercise 02: Chunking Strategy Experiment - Solution

Complete implementation of all chunking strategies with analysis.
"""

import re
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# ============================================================================
# SAMPLE DOCUMENT
# ============================================================================

SAMPLE_DOCUMENT = """
# Introduction to Machine Learning

Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data. Unlike traditional programming where rules are explicitly coded, machine learning algorithms discover patterns in data and use them to make predictions or decisions.

## Types of Machine Learning

There are three main categories of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

### Supervised Learning

Supervised learning uses labeled training data to learn a mapping between inputs and outputs. Common applications include image classification, spam detection, and price prediction. The algorithm learns from examples where the correct answer is known.

For example, to build an email spam detector, you would train the model on thousands of emails that are already labeled as "spam" or "not spam". The model learns patterns that distinguish spam from legitimate email.

### Unsupervised Learning

Unsupervised learning finds patterns in data without labeled examples. Clustering and dimensionality reduction are common techniques. These methods are useful when you don't know what patterns exist in your data.

K-means clustering groups similar data points together. Principal Component Analysis reduces high-dimensional data to fewer dimensions while preserving important information.

### Reinforcement Learning

Reinforcement learning trains agents to make sequences of decisions by rewarding desired behaviors. The agent learns through trial and error, receiving positive or negative feedback based on its actions.

Game-playing AI and robotics often use reinforcement learning. AlphaGo, which defeated world champions at the game of Go, used reinforcement learning combined with deep neural networks.

## Neural Networks

Neural networks are computing systems inspired by biological neural networks in the brain. They consist of layers of interconnected nodes that process information.

### Deep Learning

Deep learning uses neural networks with many layers. These deep networks can learn hierarchical representations of data. Early layers might detect simple features like edges, while deeper layers recognize complex patterns like faces or objects.

Convolutional Neural Networks are specialized for image processing. Recurrent Neural Networks handle sequential data like text or time series. Transformers have revolutionized natural language processing.

### Training Process

Training a neural network involves feeding it data and adjusting its parameters to minimize prediction errors. This process uses an algorithm called backpropagation combined with gradient descent optimization.

The learning rate controls how quickly the model updates its parameters. Too high a rate causes unstable training; too low makes training slow. Modern optimizers like Adam adapt the learning rate automatically.

## Practical Considerations

Building effective machine learning systems requires careful attention to data quality, feature engineering, and model selection.

### Data Quality

High-quality training data is essential for good model performance. Data should be representative of the real-world distribution the model will encounter. Biased or noisy data leads to poor generalization.

### Model Selection

Different algorithms suit different problems. Linear models work well for simple relationships. Decision trees handle non-linear patterns. Neural networks excel at complex tasks but require more data and compute.

### Evaluation

Always evaluate models on held-out test data that wasn't used during training. Common metrics include accuracy, precision, recall, and F1 score for classification; mean squared error and R-squared for regression.

Cross-validation provides more robust performance estimates by training and testing on multiple data splits.
"""

TEST_QUERIES = [
    "How do neural networks learn?",
    "What is the difference between supervised and unsupervised learning?",
    "How should I evaluate a machine learning model?",
]

# ============================================================================
# SOLUTION IMPLEMENTATIONS
# ============================================================================

def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into fixed-size chunks with overlap.
    
    Uses a sliding window approach where we step forward by
    (chunk_size - overlap) characters each iteration.
    """
    chunks = []
    
    # Handle edge case
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    
    # Calculate step size
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size // 2  # Fallback if overlap >= chunk_size
    
    # Slide through the text
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        start += step
        
        # Break if we've captured all remaining text
        if end >= len(text):
            break
    
    return chunks


def chunk_by_sentences(text: str, max_chunk_size: int = 500, overlap_sentences: int = 1) -> List[str]:
    """
    Split text by sentences, grouping until max size is reached.
    
    Ensures we never break mid-sentence for cleaner context.
    """
    # Split into sentences
    # This pattern splits on .!? followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk_sentences = []
    current_size = 0
    
    for i, sentence in enumerate(sentences):
        sentence_size = len(sentence)
        
        # Check if adding this sentence would exceed max size
        if current_size + sentence_size + 1 > max_chunk_size and current_chunk_sentences:
            # Save current chunk
            chunks.append(' '.join(current_chunk_sentences))
            
            # Start new chunk with overlap
            if overlap_sentences > 0 and len(current_chunk_sentences) >= overlap_sentences:
                # Keep last N sentences for overlap
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                current_size = sum(len(s) for s in current_chunk_sentences) + len(current_chunk_sentences) - 1
            else:
                current_chunk_sentences = []
                current_size = 0
        
        # Add sentence to current chunk
        current_chunk_sentences.append(sentence)
        current_size += sentence_size + 1  # +1 for space
    
    # Don't forget the last chunk
    if current_chunk_sentences:
        chunks.append(' '.join(current_chunk_sentences))
    
    return chunks


def chunk_recursive(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Recursively split text using multiple separators.
    
    Priority:
    1. Paragraphs (double newline)
    2. Sentences
    3. Fixed size (last resort)
    """
    
    def split_recursively(text: str, separators: List[str]) -> List[str]:
        """Helper function to recursively split text."""
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []
        
        if not separators:
            # Last resort: fixed-size chunking
            return chunk_fixed_size(text, chunk_size, overlap)
        
        # Try current separator
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == '\n\n':
            parts = text.split('\n\n')
        elif separator == '. ':
            parts = re.split(r'(?<=[.!?])\s+', text)
        else:
            parts = text.split(separator)
        
        parts = [p.strip() for p in parts if p.strip()]
        
        # If splitting didn't help, try next separator
        if len(parts) <= 1:
            return split_recursively(text, remaining_separators)
        
        # Process each part
        chunks = []
        current_chunk = ""
        
        for part in parts:
            # If this part alone is too big, recursively split it
            if len(part) > chunk_size:
                # Save current chunk first
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Recursively split the large part
                sub_chunks = split_recursively(part, remaining_separators)
                chunks.extend(sub_chunks)
                continue
            
            # Try to add to current chunk
            test_chunk = current_chunk + ("\n\n" if current_chunk else "") + part
            
            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                # Save current and start new
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    # Define separator priority
    separators = ['\n\n', '. ']
    
    return split_recursively(text, separators)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def measure_retrieval_quality(chunks: List[str], queries: List[str], embedder) -> List[Tuple[float, str]]:
    """Measure how well chunks match test queries."""
    if not chunks:
        return [(0.0, "No chunks provided")]
    
    client = chromadb.Client()
    try:
        client.delete_collection("temp_test")
    except:
        pass
    
    collection = client.create_collection("temp_test")
    
    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )
    
    results = []
    for query in queries:
        query_emb = embedder.encode([query]).tolist()
        result = collection.query(
            query_embeddings=query_emb,
            n_results=1
        )
        
        if result["documents"][0]:
            score = 1 - result["distances"][0][0]
            best_chunk = result["documents"][0][0][:100] + "..."
            results.append((score, best_chunk))
        else:
            results.append((0.0, "No match found"))
    
    return results


# ============================================================================
# TEST AND ANALYSIS
# ============================================================================

def run_experiment():
    """Run chunking experiment and compare strategies."""
    print("=" * 60)
    print("Exercise 02: Chunking Strategy Experiment - SOLUTION")
    print("=" * 60)
    
    print("\n[INFO] Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    strategies = [
        ("Fixed (500 chars, 50 overlap)", lambda: chunk_fixed_size(SAMPLE_DOCUMENT, 500, 50)),
        ("Sentence-based", lambda: chunk_by_sentences(SAMPLE_DOCUMENT, 500, 1)),
        ("Recursive", lambda: chunk_recursive(SAMPLE_DOCUMENT, 500, 50)),
    ]
    
    all_results = {}
    
    print("\n=== Chunking Strategy Comparison ===\n")
    
    for name, chunker in strategies:
        print(f"Strategy: {name}")
        
        chunks = chunker()
        
        print(f"  Chunks: {len(chunks)}")
        if chunks:
            avg_size = sum(len(c) for c in chunks) / len(chunks)
            min_size = min(len(c) for c in chunks)
            max_size = max(len(c) for c in chunks)
            print(f"  Avg size: {avg_size:.0f} chars (min: {min_size}, max: {max_size})")
            print(f"  Sample: \"{chunks[0][:60]}...\"")
        
        all_results[name] = chunks
        print()
    
    # Quality comparison
    print("\n=== Retrieval Quality Test ===\n")
    
    quality_scores = {name: [] for name in all_results.keys()}
    
    for query in TEST_QUERIES:
        print(f"Query: \"{query}\"")
        
        for name, chunks in all_results.items():
            results = measure_retrieval_quality(chunks, [query], embedder)
            score, match = results[0]
            quality_scores[name].append(score)
            print(f"  {name}: {score:.3f} - \"{match[:50]}...\"")
        
        print()
    
    # Summary
    print("\n=== Quality Summary ===\n")
    
    for name, scores in quality_scores.items():
        avg_score = sum(scores) / len(scores)
        print(f"{name}: {avg_score:.3f} average similarity")
    
    best_strategy = max(quality_scores.keys(), key=lambda k: sum(quality_scores[k]))
    print(f"\n[INFO] Best performing strategy: {best_strategy}")
    
    print("\n" + "=" * 60)
    print("[OK] Experiment complete!")
    print("=" * 60)
    
    # Analysis
    print("""
=== ANALYSIS ===

Observations from this experiment:

1. FIXED-SIZE CHUNKING
   - Pros: Simple, predictable chunk sizes
   - Cons: Can break mid-sentence, mid-paragraph
   - Best for: Uniform content, when size matters more than meaning

2. SENTENCE-BASED CHUNKING
   - Pros: Preserves complete thoughts, natural boundaries
   - Cons: Variable chunk sizes, may be too granular
   - Best for: Conversational content, Q&A systems

3. RECURSIVE CHUNKING
   - Pros: Respects document structure, handles varied content
   - Cons: More complex, may create very small/large chunks
   - Best for: Mixed content types, technical documentation

RECOMMENDATION:
For most RAG applications, start with recursive chunking with 
chunk_size=500 and overlap=50. Adjust based on:
- Your embedding model's optimal input length
- The nature of your queries (short vs. long context)
- The structure of your documents

Always measure retrieval quality on YOUR data with YOUR queries!
""")


if __name__ == "__main__":
    run_experiment()
