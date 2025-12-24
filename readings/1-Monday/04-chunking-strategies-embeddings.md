# Chunking Strategies for Embeddings

## Learning Objectives
- Understand why chunking strategy matters for RAG system performance
- Learn the four main chunking approaches: fixed-size, sentence-based, semantic, and recursive
- Evaluate trade-offs between chunk granularity and context preservation
- Select appropriate chunking strategies for different use cases

## Why This Matters

Chunking is where many RAG systems succeed or fail. The way you break documents into pieces directly affects:
- **Retrieval precision**: Will the right chunks be found?
- **Context completeness**: Will retrieved chunks have enough information?
- **Embedding quality**: Will semantic meaning be preserved in each chunk?

There's no universal "best" chunking strategy. Each approach has trade-offs, and production systems often combine multiple strategies. Understanding these options lets you make informed decisions for your specific use case.

## The Concept

### The Chunking Dilemma

Embedding models have input limits (typically 256-512 tokens). Documents are usually much longer. We must split them, but how we split matters enormously.

**Too Small**:
```
Chunk: "Machine learning is"
```
Not enough context. This could mean anything.

**Too Large**:
```
Chunk: [1000 words about machine learning, deep learning, neural networks, 
        and also some unrelated content about company history]
```
Too much noise. Retrieval becomes imprecise.

**Just Right**:
```
Chunk: "Machine learning is a subset of artificial intelligence that enables 
        computers to learn from data. It uses algorithms to identify patterns 
        and make predictions without being explicitly programmed."
```
Enough context to be meaningful, focused enough for precise retrieval.

### Strategy 1: Fixed-Size Chunking

The simplest approach: split text into chunks of N characters or tokens.

**Pros**:
- Simple to implement
- Predictable chunk sizes
- Easy to reason about

**Cons**:
- Breaks mid-sentence or mid-word
- Loses semantic boundaries
- Context can be awkwardly split

```python
def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.
    
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
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk.strip())
        
        # Move start position, accounting for overlap
        start = end - overlap
    
    return chunks


# Example
text = "Machine learning is transforming how we build software. It enables systems to learn from data and improve over time..."
chunks = fixed_size_chunking(text, chunk_size=100, overlap=20)
```

**When to Use**: Simple documents, uniform content, prototyping.

### Strategy 2: Sentence-Based Chunking

Split at sentence boundaries, grouping sentences to reach target size.

**Pros**:
- Preserves complete sentences
- More natural reading
- Better semantic coherence

**Cons**:
- Variable chunk sizes
- Very long sentences can exceed limits
- Doesn't respect paragraph or section boundaries

```python
import re

def sentence_chunking(text: str, sentences_per_chunk: int = 5) -> list[str]:
    """
    Split text into chunks at sentence boundaries.
    
    Args:
        text: Input text to chunk
        sentences_per_chunk: Number of sentences per chunk
    
    Returns:
        List of text chunks
    """
    # Simple sentence splitting (production should use better tokenization)
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        current_chunk.append(sentence)
        
        if len(current_chunk) >= sentences_per_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


# Example
text = "AI is changing everything. Companies are investing heavily. The technology is maturing rapidly. New applications emerge daily. This trend will continue."
chunks = sentence_chunking(text, sentences_per_chunk=2)
# Result: ["AI is changing everything. Companies are investing heavily.", 
#          "The technology is maturing rapidly. New applications emerge daily.", 
#          "This trend will continue."]
```

**When to Use**: General documents, articles, content without clear structure.

### Strategy 3: Semantic Chunking

Split at meaningful boundaries like paragraphs, sections, or topic changes.

**Pros**:
- Preserves topical coherence
- Respects document structure
- Best for structured documents

**Cons**:
- Highly variable chunk sizes
- Requires understanding document structure
- More complex to implement

```python
def semantic_chunking(
    text: str, 
    min_chunk_size: int = 100,
    max_chunk_size: int = 1000
) -> list[str]:
    """
    Split text at semantic boundaries (paragraphs, sections).
    
    Args:
        text: Input text to chunk
        min_chunk_size: Minimum characters per chunk
        max_chunk_size: Maximum characters per chunk
    
    Returns:
        List of text chunks
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
        
        # If single paragraph exceeds max, we need to split it
        if para_size > max_chunk_size:
            # Save current chunk first
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Split large paragraph by sentences
            sub_chunks = sentence_chunking(para, sentences_per_chunk=3)
            chunks.extend(sub_chunks)
        
        # If adding paragraph exceeds max, start new chunk
        elif current_size + para_size > max_chunk_size:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        
        # Otherwise, add to current chunk
        else:
            current_chunk.append(para)
            current_size += para_size
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks
```

**When to Use**: Technical documentation, structured articles, markdown files.

### Strategy 4: Recursive Chunking

Apply multiple splitting strategies in sequence, from coarse to fine.

**Pros**:
- Tries to maintain structure at every level
- Handles varied document types
- Used by LangChain's default splitter

**Cons**:
- More complex
- May still produce uneven chunks
- Requires tuning separators

```python
def recursive_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: list[str] = None
) -> list[str]:
    """
    Recursively split text using a hierarchy of separators.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        separators: List of separators to try, in order
    
    Returns:
        List of text chunks
    """
    if separators is None:
        separators = ['\n\n', '\n', '. ', ' ', '']
    
    chunks = []
    
    def split_recursive(text: str, sep_index: int = 0) -> list[str]:
        # If text is small enough, return it
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        # If no more separators, force split
        if sep_index >= len(separators):
            return fixed_size_chunking(text, chunk_size, overlap)
        
        separator = separators[sep_index]
        
        # Split by current separator
        if separator:
            parts = text.split(separator)
        else:
            # Empty separator means character-level split
            parts = list(text)
        
        # If separator doesn't help, try next one
        if len(parts) == 1:
            return split_recursive(text, sep_index + 1)
        
        # Group parts into chunks
        result = []
        current = []
        current_len = 0
        
        for part in parts:
            part_with_sep = part + separator if separator else part
            
            if current_len + len(part_with_sep) <= chunk_size:
                current.append(part)
                current_len += len(part_with_sep)
            else:
                if current:
                    combined = separator.join(current) if separator else ''.join(current)
                    result.append(combined)
                current = [part]
                current_len = len(part_with_sep)
        
        if current:
            combined = separator.join(current) if separator else ''.join(current)
            result.append(combined)
        
        # Recursively process any chunks that are still too large
        final_result = []
        for chunk in result:
            if len(chunk) > chunk_size:
                final_result.extend(split_recursive(chunk, sep_index + 1))
            else:
                if chunk.strip():
                    final_result.append(chunk.strip())
        
        return final_result
    
    return split_recursive(text)
```

**When to Use**: General-purpose, when you don't know document structure.

### Comparison Table

| Strategy | Chunk Size | Semantic Coherence | Complexity | Best For |
|----------|-----------|-------------------|------------|----------|
| Fixed-Size | Uniform | Low | Simple | Prototyping, simple text |
| Sentence | Variable | Medium | Medium | Articles, general text |
| Semantic | Variable | High | Medium | Structured documents |
| Recursive | Variable | High | Complex | Unknown document types |

## Code Example

Here's a complete chunking utility that lets you choose strategies:

```python
from enum import Enum
from typing import Callable
import re


class ChunkingStrategy(Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


class DocumentChunker:
    """Flexible document chunking with multiple strategies."""
    
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 500,
        overlap: int = 50
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        self._strategies: dict[ChunkingStrategy, Callable] = {
            ChunkingStrategy.FIXED: self._fixed_chunk,
            ChunkingStrategy.SENTENCE: self._sentence_chunk,
            ChunkingStrategy.SEMANTIC: self._semantic_chunk,
            ChunkingStrategy.RECURSIVE: self._recursive_chunk,
        }
    
    def chunk(self, text: str) -> list[str]:
        """Chunk text using the configured strategy."""
        strategy_fn = self._strategies[self.strategy]
        chunks = strategy_fn(text)
        
        # Post-process: add overlap between chunks
        if self.overlap > 0 and self.strategy != ChunkingStrategy.FIXED:
            chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _fixed_chunk(self, text: str) -> list[str]:
        """Fixed-size chunking."""
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + self.chunk_size].strip())
            start += self.chunk_size - self.overlap
        return [c for c in chunks if c]
    
    def _sentence_chunk(self, text: str) -> list[str]:
        """Sentence-based chunking."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        current_len = 0
        
        for sentence in sentences:
            if current_len + len(sentence) > self.chunk_size and current:
                chunks.append(' '.join(current))
                current = []
                current_len = 0
            current.append(sentence)
            current_len += len(sentence)
        
        if current:
            chunks.append(' '.join(current))
        return chunks
    
    def _semantic_chunk(self, text: str) -> list[str]:
        """Paragraph-based semantic chunking."""
        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        current = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if current_len + len(para) > self.chunk_size and current:
                chunks.append('\n\n'.join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += len(para)
        
        if current:
            chunks.append('\n\n'.join(current))
        return chunks
    
    def _recursive_chunk(self, text: str) -> list[str]:
        """Recursive chunking with multiple separators."""
        separators = ['\n\n', '\n', '. ', ' ']
        return self._recursive_split(text, separators)
    
    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        if not separators:
            return self._fixed_chunk(text)
        
        sep = separators[0]
        parts = text.split(sep)
        
        if len(parts) == 1:
            return self._recursive_split(text, separators[1:])
        
        chunks = []
        current = []
        current_len = 0
        
        for part in parts:
            if current_len + len(part) > self.chunk_size and current:
                chunks.append(sep.join(current))
                current = []
                current_len = 0
            current.append(part)
            current_len += len(part)
        
        if current:
            chunks.append(sep.join(current))
        
        # Recursively handle oversized chunks
        result = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                result.extend(self._recursive_split(chunk, separators[1:]))
            elif chunk.strip():
                result.append(chunk.strip())
        
        return result
    
    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap context from previous chunk."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_suffix = chunks[i-1][-self.overlap:]
            overlapped.append(prev_suffix + " " + chunks[i])
        
        return overlapped


# Usage
chunker = DocumentChunker(
    strategy=ChunkingStrategy.RECURSIVE,
    chunk_size=500,
    overlap=50
)

text = """Your long document here..."""
chunks = chunker.chunk(text)
```

## Key Takeaways

1. **Chunking strategy directly impacts retrieval quality** - choose carefully
2. **Fixed-size is simple but loses semantic coherence**
3. **Sentence-based preserves grammar but ignores structure**
4. **Semantic chunking respects document hierarchy**
5. **Recursive chunking is flexible and works for unknown formats**
6. **Overlap helps maintain context between chunks**

## Additional Resources

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) - Production-ready splitter implementations
- [Chunking for RAG (Greg Kamradt)](https://www.youtube.com/watch?v=8OJC21T2SL4) - Deep dive video on chunking strategies
- [Pinecone Chunking Guide](https://www.pinecone.io/learn/chunking-strategies/) - Practical comparison of approaches
