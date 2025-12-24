"""
Exercise 02: Chunking Strategy Experiment - Starter Code

Implement and compare different chunking strategies for RAG.

Instructions:
1. Implement each chunking function
2. Run quality comparison
3. Document your findings
"""

import re
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# ============================================================================
# SAMPLE DOCUMENT (DO NOT MODIFY)
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
# TODO: IMPLEMENT THESE FUNCTIONS
# ============================================================================

def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into fixed-size chunks with overlap.
    
    Args:
        text: Input document text
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
        
    Example:
        For text="ABCDEFGHIJ", chunk_size=5, overlap=2:
        -> ["ABCDE", "DEFGH", "GHIJ"]
    """
    # TODO: Implement this function
    # Hint: Use a sliding window approach
    # Step by (chunk_size - overlap) each iteration
    
    pass  # Remove this and add your implementation


def chunk_by_sentences(text: str, max_chunk_size: int = 500, overlap_sentences: int = 1) -> List[str]:
    """
    Split text by sentences, grouping until max size is reached.
    
    Args:
        text: Input document text
        max_chunk_size: Maximum characters per chunk
        overlap_sentences: Number of sentences to overlap between chunks
        
    Returns:
        List of text chunks (each containing complete sentences)
    """
    # TODO: Implement this function
    # Step 1: Split into sentences (use regex on .!? followed by space)
    # Step 2: Group sentences until reaching max_chunk_size
    # Step 3: Handle overlap by including previous sentences
    
    # Hint for sentence splitting:
    # sentences = re.split(r'(?<=[.!?])\s+', text)
    
    pass  # Remove this and add your implementation


def chunk_recursive(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Recursively split text using multiple separators.
    
    Priority order:
    1. Try splitting by paragraphs (double newline)
    2. If chunks still too large, split by sentences
    3. If still too large, split by fixed size
    
    Args:
        text: Input document text
        chunk_size: Target maximum chunk size
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # TODO: Implement this function
    # This is the most complex - take it step by step
    
    # Approach:
    # 1. Split by paragraphs first
    # 2. For each paragraph chunk:
    #    - If it's small enough, keep it
    #    - If too large, split it further (by sentences or fixed)
    # 3. Merge small adjacent chunks if they fit together
    
    pass  # Remove this and add your implementation


# ============================================================================
# HELPER FUNCTIONS (PROVIDED)
# ============================================================================

def measure_retrieval_quality(chunks: List[str], queries: List[str], embedder) -> List[Tuple[float, str]]:
    """
    Measure how well chunks match test queries.
    Returns best matching chunk and score for each query.
    """
    if not chunks:
        return [(0.0, "No chunks provided")]
    
    # Create temporary collection
    client = chromadb.Client()
    try:
        client.delete_collection("temp_test")
    except:
        pass
    
    collection = client.create_collection("temp_test")
    
    # Embed and store chunks
    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )
    
    # Query and get best matches
    results = []
    for query in queries:
        query_emb = embedder.encode([query]).tolist()
        result = collection.query(
            query_embeddings=query_emb,
            n_results=1
        )
        
        if result["documents"][0]:
            score = 1 - result["distances"][0][0]  # Convert distance to similarity
            best_chunk = result["documents"][0][0][:100] + "..."
            results.append((score, best_chunk))
        else:
            results.append((0.0, "No match found"))
    
    return results


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_experiment():
    """Run chunking experiment and compare strategies."""
    print("=" * 60)
    print("Exercise 02: Chunking Strategy Experiment")
    print("=" * 60)
    
    print("\n[INFO] Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test each strategy
    strategies = [
        ("Fixed (500 chars, 50 overlap)", lambda: chunk_fixed_size(SAMPLE_DOCUMENT, 500, 50)),
        ("Sentence-based", lambda: chunk_by_sentences(SAMPLE_DOCUMENT, 500, 1)),
        ("Recursive", lambda: chunk_recursive(SAMPLE_DOCUMENT, 500, 50)),
    ]
    
    all_results = {}
    
    print("\n=== Chunking Strategy Comparison ===\n")
    
    for name, chunker in strategies:
        print(f"Strategy: {name}")
        
        try:
            chunks = chunker()
            
            if chunks is None:
                print("  [ERROR] Not implemented yet\n")
                continue
            
            print(f"  Chunks: {len(chunks)}")
            if chunks:
                avg_size = sum(len(c) for c in chunks) / len(chunks)
                print(f"  Avg size: {avg_size:.0f} chars")
                print(f"  Sample: \"{chunks[0][:60]}...\"")
            
            all_results[name] = chunks
            
        except Exception as e:
            print(f"  [ERROR] {e}")
        
        print()
    
    # Run quality comparison if we have results
    if all_results:
        print("\n=== Retrieval Quality Test ===\n")
        
        for query in TEST_QUERIES:
            print(f"Query: \"{query}\"")
            
            for name, chunks in all_results.items():
                if chunks:
                    results = measure_retrieval_quality(chunks, [query], embedder)
                    score, match = results[0]
                    print(f"  {name}: {score:.2f} - \"{match[:50]}...\"")
            
            print()
    
    print("=" * 60)
    print("[OK] Experiment complete!")
    print("=" * 60)
    
    # TODO: Add your analysis here
    # Which strategy worked best? Why?
    # YOUR ANALYSIS:
    # 
    # 


if __name__ == "__main__":
    run_experiment()
