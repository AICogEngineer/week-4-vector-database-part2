# Exercise 02: Chunking Strategy Experiment

## Overview

Chunking strategy significantly impacts RAG retrieval quality. In this exercise, you'll implement multiple chunking approaches, measure their effectiveness, and document which strategy works best for different content types.

## Learning Objectives

- Implement fixed-size, sentence-based, and recursive chunking
- Understand the role of chunk overlap
- Measure retrieval quality across strategies
- Make data-driven chunking decisions

## The Scenario

You're optimizing a technical documentation RAG system. The team needs to know:
1. Which chunking strategy retrieves the most relevant context?
2. What chunk size balances context window limits with retrieval quality?
3. How does overlap affect results?

## Your Tasks

### Task 1: Fixed-Size Chunking (20 min)

Implement `chunk_fixed_size()`:
- Split text into chunks of `chunk_size` characters
- Support configurable `overlap` between chunks
- Handle edge cases (last chunk, very short documents)

Parameters:
- `text`: Input document
- `chunk_size`: Target size (default 500)
- `overlap`: Overlap in characters (default 50)

### Task 2: Sentence-Based Chunking (25 min)

Implement `chunk_by_sentences()`:
- Group sentences until reaching target size
- Never break mid-sentence
- Support overlap by sentence count

> **Hint**: Use regex to split on sentence boundaries: `.!?` followed by space.

### Task 3: Recursive Chunking (30 min)

Implement `chunk_recursive()`:
- Try splitting by paragraphs first
- If chunks are too large, split by sentences
- If still too large, split by fixed size
- Pass chunk_size and overlap parameters

This mimics how production splitters work!

### Task 4: Quality Comparison (15 min)

Use the provided `measure_retrieval_quality()` function:
1. Chunk the sample document with each strategy
2. Run the test queries
3. Compare which strategy returns more relevant chunks
4. Document your findings in comments

## Sample Document

The starter code includes a technical article about machine learning. Use this for all experiments.

## Definition of Done

- [_] Three chunking functions implemented
- [_] Each function handles overlap correctly
- [_] Quality comparison run on all strategies
- [_] Findings documented in code comments
- [_] Console output shows chunk counts and sample chunks

## Testing Your Solution

```bash
cd exercises/1-Monday/starter_code
python exercise_02_starter.py
```

Expected output format:
```
=== Chunking Strategy Comparison ===

Strategy: Fixed (500 chars, 50 overlap)
  Chunks: 12
  Avg size: 487 chars
  Sample: "Machine learning models require..."

Strategy: Sentence-based
  Chunks: 8
  Avg size: 623 chars
  Sample: "Deep learning is a subset of machine learning..."

Strategy: Recursive
  Chunks: 10
  Avg size: 534 chars
  Sample: "Neural networks consist of layers..."

=== Retrieval Quality Test ===
Query: "How do neural networks learn?"
  Best match (Fixed): 0.82 - "...backpropagation algorithm..."
  Best match (Sentence): 0.87 - "Neural networks learn through..."
  Best match (Recursive): 0.85 - "...training process involves..."

[OK] Experiment complete! See analysis below.
```

## Stretch Goals (Optional)

1. Add semantic chunking using embeddings
2. Visualize chunk boundaries on the original document
3. Test with multiple document types (code, prose, mixed)
