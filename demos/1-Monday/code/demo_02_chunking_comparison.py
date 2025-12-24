"""
Demo 02: Chunking Strategy Comparison

This demo shows trainees how to:
1. Apply different chunking strategies to the same document
2. Compare chunk quality and retrieval effectiveness
3. Understand trade-offs between strategies
4. Choose the right strategy for their use case

Learning Objectives:
- Implement fixed-size, sentence-based, and semantic chunking
- Measure impact of chunking on retrieval quality
- Select appropriate chunking for different document types

References:
- Written Content: 04-chunking-strategies-embeddings.md
- Written Content: 05-optimal-chunk-sizes.md
"""

import re
from typing import List
from dataclasses import dataclass

# ============================================================================
# PART 1: The Chunking Dilemma
# ============================================================================

print("=" * 70)
print("PART 1: The Chunking Dilemma")
print("=" * 70)

print("""
THE PROBLEM:
- Embedding models have max token limits (~256-512 tokens)
- Documents are usually much longer
- We MUST split them, but HOW we split matters!

           TOO SMALL              JUST RIGHT              TOO LARGE
        ┌───────────┐         ┌───────────────┐        ┌─────────────────┐
        │ "Machine" │         │ "Machine      │        │ [entire chapter]│
        └───────────┘         │  learning is  │        └─────────────────┘
         Fragment!            │  a subset..." │          Too much noise
         No context           └───────────────┘
                               Complete thought
""")

# Sample document for all examples
SAMPLE_DOCUMENT = """
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. This technology has transformed numerous industries.

Supervised Learning

In supervised learning, the algorithm learns from labeled training data. The model makes predictions based on input features and compares them against known correct outputs. Common applications include email spam detection, image classification, and price prediction.

Unsupervised Learning

Unsupervised learning works with unlabeled data. The algorithm tries to find hidden patterns or structures in the data. Clustering and dimensionality reduction are key techniques in this category. Applications include customer segmentation and anomaly detection.

Deep Learning

Deep learning is a specialized subset of machine learning that uses neural networks with many layers. These deep neural networks can learn complex patterns from large amounts of data. They power modern AI applications like natural language processing and computer vision.

The key advantage of deep learning is feature learning - the model automatically discovers the representations needed for classification or detection. This eliminates the need for manual feature engineering.
"""

print(f"\nSample document ({len(SAMPLE_DOCUMENT.split())} words):")
print("-" * 50)
print(SAMPLE_DOCUMENT[:500] + "...")
print("-" * 50)

# ============================================================================
# PART 2: Fixed-Size Chunking
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Fixed-Size Chunking")
print("=" * 70)

print("""
FIXED-SIZE CHUNKING:
- Split text every N characters (or words/tokens)
- Simple and predictable
- But: May break mid-sentence or mid-word!
""")

def fixed_size_chunking(text: str, chunk_size: int = 200, overlap: int = 0) -> List[str]:
    """
    Split text into fixed-size chunks with optional overlap.
    
    Args:
        text: Input text to chunk
        chunk_size: Number of characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        # Move start, accounting for overlap
        start = end - overlap
    
    return chunks

print("\n[Step 1] Fixed-size chunking WITHOUT overlap...")

chunks_no_overlap = fixed_size_chunking(SAMPLE_DOCUMENT, chunk_size=200, overlap=0)

print(f"\nGenerated {len(chunks_no_overlap)} chunks:")
for i, chunk in enumerate(chunks_no_overlap[:3]):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(f"  '{chunk[:80]}...'")

print("\n[!] PROBLEM: Look at chunk boundaries!")
print("    Chunks may end mid-sentence or even mid-word.")

print("\n[Step 2] Fixed-size chunking WITH overlap...")

chunks_with_overlap = fixed_size_chunking(SAMPLE_DOCUMENT, chunk_size=200, overlap=50)

print(f"\nGenerated {len(chunks_with_overlap)} chunks (with 50-char overlap):")
print("\nCompare chunk 1 end and chunk 2 start:")
print(f"  Chunk 1 ends with:    '...{chunks_with_overlap[0][-50:]}'")
print(f"  Chunk 2 starts with:  '{chunks_with_overlap[1][:50]}...'")

print("""
OBSERVATION: 
- Overlap helps maintain context between chunks
- But we still break at arbitrary positions
- More chunks means more storage and slower search
""")

# ============================================================================
# PART 3: Sentence-Based Chunking
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Sentence-Based Chunking")
print("=" * 70)

print("""
SENTENCE-BASED CHUNKING:
- Split at sentence boundaries
- Group sentences until size limit
- Preserves complete thoughts!
""")

def sentence_chunking(text: str, max_chunk_size: int = 300) -> List[str]:
    """
    Split text at sentence boundaries, grouping sentences into chunks.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
    
    Returns:
        List of text chunks (each containing complete sentences)
    """
    # Split into sentences (simple regex - production should use better tokenization)
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed limit, save current chunk
        if current_length + sentence_length > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length + 1  # +1 for space
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

print("\n[Step 3] Sentence-based chunking...")

sentence_chunks = sentence_chunking(SAMPLE_DOCUMENT, max_chunk_size=300)

print(f"\nGenerated {len(sentence_chunks)} chunks:")
for i, chunk in enumerate(sentence_chunks[:3]):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(f"  '{chunk[:100]}...'")
    # Check if it ends with sentence-ending punctuation
    ends_properly = chunk.strip()[-1] in '.!?'
    print(f"  [Ends with complete sentence: {'Yes' if ends_properly else 'No'}]")

print("""
OBSERVATION:
- Each chunk contains complete sentences
- More natural reading
- BUT: Variable chunk sizes (some short, some long)
- Doesn't respect paragraph/section boundaries
""")

# ============================================================================
# PART 4: Semantic Chunking
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Semantic Chunking")
print("=" * 70)

print("""
SEMANTIC CHUNKING:
- Split at meaningful boundaries (paragraphs, sections)
- Respects document structure
- Each chunk is a coherent unit of meaning
""")

def semantic_chunking(
    text: str, 
    min_chunk_size: int = 100,
    max_chunk_size: int = 500
) -> List[str]:
    """
    Split text at semantic boundaries (paragraphs, sections).
    
    Args:
        text: Input text to chunk
        min_chunk_size: Minimum characters per chunk
        max_chunk_size: Maximum characters per chunk
    
    Returns:
        List of text chunks following semantic boundaries
    """
    # Split by paragraph (double newline)
    paragraphs = re.split(r'\n\n+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_size = len(para)
        
        # If single paragraph exceeds max, split by sentences
        if para_size > max_chunk_size:
            # Save current chunk first
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Split large paragraph into sentence chunks
            para_sentences = sentence_chunking(para, max_chunk_size=max_chunk_size)
            chunks.extend(para_sentences)
            continue
        
        # If adding paragraph exceeds max, start new chunk
        if current_size + para_size > max_chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(para)
        current_size += para_size
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

print("\n[Step 4] Semantic chunking...")

semantic_chunks = semantic_chunking(SAMPLE_DOCUMENT, min_chunk_size=100, max_chunk_size=400)

print(f"\nGenerated {len(semantic_chunks)} chunks:")
for i, chunk in enumerate(semantic_chunks):
    # Check if this chunk starts with a header
    is_section = chunk.strip().split('\n')[0].istitle() or len(chunk.strip().split('\n')[0].split()) <= 4
    print(f"\nChunk {i+1} ({len(chunk)} chars) {'[SECTION START]' if is_section else ''}:")
    print(f"  '{chunk[:80].replace(chr(10), ' ')}...'")

print("""
OBSERVATION:
- Chunks follow document structure
- Headers stay with their content
- Each chunk is a meaningful unit
- Best for structured documents (docs, articles)
""")

# ============================================================================
# PART 5: Recursive Chunking
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Recursive Chunking")
print("=" * 70)

print("""
RECURSIVE CHUNKING:
- Try multiple separators in order
- Falls back gracefully
- Used by LangChain's default splitter
""")

def recursive_chunking(
    text: str,
    chunk_size: int = 300,
    separators: List[str] = None
) -> List[str]:
    """
    Recursively split text using a hierarchy of separators.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size
        separators: List of separators to try, in order
    
    Returns:
        List of text chunks
    """
    if separators is None:
        separators = ['\n\n', '\n', '. ', ' ']
    
    def split_recursive(text: str, sep_index: int = 0) -> List[str]:
        # Base case: text is small enough
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []
        
        # Base case: no more separators
        if sep_index >= len(separators):
            # Force split at chunk_size
            return [text[i:i+chunk_size].strip() 
                    for i in range(0, len(text), chunk_size) 
                    if text[i:i+chunk_size].strip()]
        
        separator = separators[sep_index]
        parts = text.split(separator)
        
        # If separator doesn't help, try next one
        if len(parts) == 1:
            return split_recursive(text, sep_index + 1)
        
        # Group parts into chunks
        chunks = []
        current = []
        current_len = 0
        
        for part in parts:
            part_len = len(part) + len(separator)
            
            if current_len + part_len > chunk_size and current:
                chunks.append(separator.join(current))
                current = []
                current_len = 0
            
            current.append(part)
            current_len += part_len
        
        if current:
            chunks.append(separator.join(current))
        
        # Recursively process chunks that are still too large
        result = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                result.extend(split_recursive(chunk, sep_index + 1))
            elif chunk.strip():
                result.append(chunk.strip())
        
        return result
    
    return split_recursive(text)

print("\n[Step 5] Recursive chunking...")

recursive_chunks = recursive_chunking(SAMPLE_DOCUMENT, chunk_size=300)

print(f"\nGenerated {len(recursive_chunks)} chunks:")
for i, chunk in enumerate(recursive_chunks[:4]):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(f"  '{chunk[:80].replace(chr(10), ' ')}...'")

# ============================================================================
# PART 6: Side-by-Side Comparison
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Strategy Comparison")
print("=" * 70)

@dataclass
class ChunkingResult:
    strategy: str
    num_chunks: int
    avg_size: float
    min_size: int
    max_size: int
    complete_sentences: float  # Percentage ending with .!?

def analyze_chunks(chunks: List[str], strategy: str) -> ChunkingResult:
    """Analyze chunk quality metrics."""
    sizes = [len(c) for c in chunks]
    complete = sum(1 for c in chunks if c.strip()[-1] in '.!?' if c.strip())
    
    return ChunkingResult(
        strategy=strategy,
        num_chunks=len(chunks),
        avg_size=sum(sizes) / len(sizes) if sizes else 0,
        min_size=min(sizes) if sizes else 0,
        max_size=max(sizes) if sizes else 0,
        complete_sentences=complete / len(chunks) * 100 if chunks else 0
    )

# Generate all chunk types
all_results = {
    "Fixed (no overlap)": fixed_size_chunking(SAMPLE_DOCUMENT, 200, 0),
    "Fixed (with overlap)": fixed_size_chunking(SAMPLE_DOCUMENT, 200, 50),
    "Sentence-based": sentence_chunking(SAMPLE_DOCUMENT, 300),
    "Semantic": semantic_chunking(SAMPLE_DOCUMENT, 100, 400),
    "Recursive": recursive_chunking(SAMPLE_DOCUMENT, 300),
}

print("\n" + "-" * 80)
print(f"{'Strategy':<22} | {'Chunks':>6} | {'Avg Size':>8} | {'Min':>5} | {'Max':>5} | {'Complete':>8}")
print("-" * 80)

for strategy, chunks in all_results.items():
    result = analyze_chunks(chunks, strategy)
    print(f"{result.strategy:<22} | {result.num_chunks:>6} | {result.avg_size:>8.1f} | {result.min_size:>5} | {result.max_size:>5} | {result.complete_sentences:>7.1f}%")

print("-" * 80)

print("""
ANALYSIS:
- Fixed chunking: Consistent size but often incomplete sentences
- Sentence-based: 100% complete sentences, variable sizes
- Semantic: Follows structure, good for documentation
- Recursive: Balanced approach, adapts to content
""")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO 2 COMPLETE: Chunking Strategy Comparison")
print("=" * 70)

print("""
Key Takeaways:

1. FIXED-SIZE is simple but breaks arbitrarily
   - Use for: Prototyping, when size consistency matters
   - Add overlap to improve context continuity

2. SENTENCE-BASED preserves grammar
   - Use for: General text, articles, prose
   - Variable sizes may be a concern

3. SEMANTIC respects document structure
   - Use for: Technical docs, structured content
   - Best for documentation and guides

4. RECURSIVE is flexible and adaptive
   - Use for: Unknown document types
   - Good default choice (used by LangChain)

5. CHOOSE BASED ON YOUR CONTENT
   - No single "best" strategy
   - Test on your actual data!

Coming Next: Demo 3 will build a complete RAG pipeline!
""")

# ============================================================================
# INSTRUCTOR NOTES
# ============================================================================

print("\n" + "=" * 70)
print("INSTRUCTOR NOTES")
print("=" * 70)

print("""
Discussion Questions:
1. "Which strategy would you use for code documentation?"
2. "What happens to retrieval quality with very small chunks?"
3. "How would you handle a document that mixes prose and code?"

Interactive Exercise:
- Have trainees predict results before running comparison
- Ask them to suggest other documents to test

Common Confusions:
- "Which strategy is best?" → Depends on content and use case!
- "What chunk size should I use?" → Tomorrow's topic, start with 300-500
- "Does overlap always help?" → Helps context, but increases storage

If Running Short on Time:
- Skip recursive chunking (Part 5)
- Focus on comparison table (Part 6)

If Trainees Are Advanced:
- Discuss embedding model token limits
- Mention chunk size optimization techniques
""")

print("\n" + "=" * 70)
