# Text Splitting Techniques

## Learning Objectives
- Master different text splitter implementations: character, recursive, and token-based
- Understand overlap strategies and their impact on retrieval
- Learn to maintain context continuity across chunk boundaries
- Select the right splitter for your document types

## Why This Matters

Yesterday we explored chunking *strategies*. Today we dive into the *implementation* details of text splitters. The difference between a good and bad splitter can mean the difference between:

- "What is machine learning?" returning a relevant explanation
- "What is machine learning?" returning "earning rate of 0.01 is typically..."

Understanding these techniques gives you fine-grained control over how your documents are segmented for optimal retrieval.

## The Concept

### Text Splitter Types

There are three primary families of text splitters, each with distinct characteristics:

```
CHARACTER-BASED          RECURSIVE              TOKEN-BASED
----------------         ---------              -----------
Split by char count      Split hierarchically   Split by token count
Simple, predictable      Structure-aware        LLM-aligned
May break mid-word       Falls back gracefully  Accurate limits
```

### Character-Based Splitting

The simplest approach: split at character boundaries.

```python
class CharacterTextSplitter:
    """Split text by character count."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def split_text(self, text: str) -> list[str]:
        """Split text into chunks."""
        # First, split by separator
        splits = text.split(self.separator)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            
            # If adding this split exceeds limit, save current chunk
            if current_length + split_length > self.chunk_size and current_chunk:
                chunks.append(self.separator.join(current_chunk))
                
                # Calculate overlap - keep last portion of current chunk
                overlap_text = self.separator.join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    overlap_text = overlap_text[-self.chunk_overlap:]
                current_chunk = [overlap_text]
                current_length = len(overlap_text)
            
            current_chunk.append(split)
            current_length += split_length + len(self.separator)
        
        # Add final chunk
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))
        
        return chunks


# Usage
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(document_text)
```

**When to Use**: Simple documents, when exact character limits matter.

### Recursive Character Text Splitting

The most popular approach: try multiple separators in order of preference.

```python
class RecursiveCharacterTextSplitter:
    """Split text recursively using multiple separators."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def split_text(self, text: str) -> list[str]:
        """Recursively split text."""
        return self._split_text(text, self.separators)
    
    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Internal recursive splitting."""
        final_chunks = []
        
        # Get the current separator to try
        separator = separators[0] if separators else ""
        new_separators = separators[1:] if len(separators) > 1 else []
        
        # Split by current separator
        splits = text.split(separator) if separator else list(text)
        
        # Merge small splits, recurse on large ones
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split)
            
            # If single split is too large, recurse with next separator
            if split_len > self.chunk_size:
                # First, flush current chunk
                if current_chunk:
                    merged = separator.join(current_chunk)
                    final_chunks.append(merged)
                    current_chunk = []
                    current_length = 0
                
                # Recurse on oversized split
                if new_separators:
                    sub_chunks = self._split_text(split, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    # No more separators, force split
                    for i in range(0, len(split), self.chunk_size - self.chunk_overlap):
                        final_chunks.append(split[i:i + self.chunk_size])
            
            # Check if adding this split would exceed limit
            elif current_length + split_len + len(separator) > self.chunk_size:
                if current_chunk:
                    merged = separator.join(current_chunk)
                    final_chunks.append(merged)
                    
                    # Handle overlap
                    overlap_chunk = self._get_overlap(current_chunk, separator)
                    current_chunk = [overlap_chunk] if overlap_chunk else []
                    current_length = len(overlap_chunk) if overlap_chunk else 0
                
                current_chunk.append(split)
                current_length += split_len + len(separator)
            
            else:
                current_chunk.append(split)
                current_length += split_len + len(separator)
        
        # Don't forget the last chunk
        if current_chunk:
            merged = separator.join(current_chunk)
            final_chunks.append(merged)
        
        return [chunk for chunk in final_chunks if chunk.strip()]
    
    def _get_overlap(self, chunks: list[str], separator: str) -> str:
        """Get overlap text from previous chunks."""
        combined = separator.join(chunks)
        if len(combined) <= self.chunk_overlap:
            return combined
        return combined[-self.chunk_overlap:]


# Language-specific separators
MARKDOWN_SEPARATORS = [
    "\n## ",    # H2 headers
    "\n### ",   # H3 headers
    "\n#### ",  # H4 headers
    "\n\n",     # Paragraphs
    "\n",       # Lines
    " ",        # Words
    ""          # Characters
]

PYTHON_SEPARATORS = [
    "\nclass ",     # Class definitions
    "\ndef ",       # Function definitions
    "\n\n",         # Double newlines
    "\n",           # Single newlines
    " ",            # Spaces
    ""              # Characters
]

# Usage for Markdown
md_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=MARKDOWN_SEPARATORS
)
```

**When to Use**: Most documents, especially structured content.

### Token-Based Splitting

Split by tokens rather than characters for alignment with LLM limits.

```python
from typing import Callable


class TokenTextSplitter:
    """Split text by token count."""
    
    def __init__(
        self,
        chunk_size: int = 256,  # Tokens, not characters
        chunk_overlap: int = 50,
        tokenizer: Callable[[str], list] = None,
        detokenizer: Callable[[list], str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default to simple whitespace tokenization
        self.tokenizer = tokenizer or (lambda x: x.split())
        self.detokenizer = detokenizer or (lambda x: ' '.join(x))
    
    def split_text(self, text: str) -> list[str]:
        """Split text into token-based chunks."""
        tokens = self.tokenizer(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.detokenizer(chunk_tokens)
            chunks.append(chunk_text)
            
            start = end - self.chunk_overlap
        
        return chunks


# Using tiktoken for accurate OpenAI token counting
def create_tiktoken_splitter(
    encoding_name: str = "cl100k_base",  # GPT-4 encoding
    chunk_size: int = 256,
    chunk_overlap: int = 50
) -> TokenTextSplitter:
    """Create a splitter using tiktoken for accurate token counts."""
    import tiktoken
    
    encoding = tiktoken.get_encoding(encoding_name)
    
    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=encoding.encode,
        detokenizer=lambda tokens: encoding.decode(tokens)
    )


# Usage
splitter = create_tiktoken_splitter(chunk_size=256, chunk_overlap=50)
chunks = splitter.split_text(document_text)
```

**When to Use**: When precise token limits matter (API costs, context windows).

### Overlap Strategies

Overlap preserves context that might be split across boundaries.

#### Fixed Overlap
```python
def fixed_overlap_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split with fixed character overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Move back by overlap amount
    return chunks
```

#### Percentage Overlap
```python
def percentage_overlap_split(
    text: str, 
    chunk_size: int, 
    overlap_percent: float = 0.1
) -> list[str]:
    """Split with percentage-based overlap."""
    overlap = int(chunk_size * overlap_percent)
    return fixed_overlap_split(text, chunk_size, overlap)
```

#### Semantic Overlap
```python
def semantic_overlap_split(
    text: str,
    chunk_size: int,
    sentences_overlap: int = 2
) -> list[str]:
    """Overlap by complete sentences for better context."""
    import re
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    
    current_sentences = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > chunk_size and current_sentences:
            chunks.append(' '.join(current_sentences))
            
            # Keep last N sentences for overlap
            current_sentences = current_sentences[-sentences_overlap:]
            current_length = sum(len(s) for s in current_sentences)
        
        current_sentences.append(sentence)
        current_length += len(sentence)
    
    if current_sentences:
        chunks.append(' '.join(current_sentences))
    
    return chunks
```

### Preserving Context Across Chunks

Beyond overlap, there are techniques to maintain context:

#### 1. Adding Chunk Headers
```python
def split_with_headers(
    text: str, 
    splitter, 
    document_title: str
) -> list[str]:
    """Add context headers to each chunk."""
    chunks = splitter.split_text(text)
    
    contextualized_chunks = []
    for i, chunk in enumerate(chunks):
        header = f"[Document: {document_title} | Part {i+1}/{len(chunks)}]\n\n"
        contextualized_chunks.append(header + chunk)
    
    return contextualized_chunks
```

#### 2. Parent-Child Chunking
```python
def hierarchical_split(
    text: str,
    parent_size: int = 2000,
    child_size: int = 400
) -> dict:
    """Create parent-child chunk relationships."""
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size)
    
    parents = parent_splitter.split_text(text)
    
    hierarchy = {}
    for i, parent in enumerate(parents):
        children = child_splitter.split_text(parent)
        hierarchy[f"parent_{i}"] = {
            "content": parent,
            "children": {f"child_{j}": c for j, c in enumerate(children)}
        }
    
    return hierarchy
```

## Code Example

A complete text splitter implementation combining all techniques:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re


class SplitType(Enum):
    CHARACTER = "character"
    RECURSIVE = "recursive"
    TOKEN = "token"
    SEMANTIC = "semantic"


@dataclass
class SplitResult:
    """Result of text splitting with metadata."""
    chunks: list[str]
    total_chunks: int
    avg_chunk_size: float
    overlap_used: int


class TextSplitter(ABC):
    """Abstract base class for text splitters."""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split text into chunks."""
        pass
    
    def split_with_metadata(self, text: str) -> SplitResult:
        """Split text and return metadata."""
        chunks = self.split_text(text)
        return SplitResult(
            chunks=chunks,
            total_chunks=len(chunks),
            avg_chunk_size=sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
            overlap_used=self.chunk_overlap
        )


class UniversalTextSplitter:
    """Unified interface for all splitting strategies."""
    
    def __init__(
        self,
        split_type: SplitType = SplitType.RECURSIVE,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[list[str]] = None,
        add_context_headers: bool = False
    ):
        self.split_type = split_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.add_context_headers = add_context_headers
    
    def split(
        self, 
        text: str, 
        document_name: str = "document"
    ) -> list[str]:
        """Split text using configured strategy."""
        
        if self.split_type == SplitType.CHARACTER:
            chunks = self._character_split(text)
        elif self.split_type == SplitType.RECURSIVE:
            chunks = self._recursive_split(text)
        elif self.split_type == SplitType.TOKEN:
            chunks = self._token_split(text)
        elif self.split_type == SplitType.SEMANTIC:
            chunks = self._semantic_split(text)
        else:
            chunks = self._recursive_split(text)
        
        if self.add_context_headers:
            chunks = self._add_headers(chunks, document_name)
        
        return chunks
    
    def _character_split(self, text: str) -> list[str]:
        """Simple character-based splitting."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap
        return [c for c in chunks if c]
    
    def _recursive_split(self, text: str) -> list[str]:
        """Recursive splitting with fallback separators."""
        separators = self.separators or ["\n\n", "\n", ". ", " ", ""]
        return self._recursive_helper(text, separators)
    
    def _recursive_helper(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        if not separators:
            return self._character_split(text)
        
        sep = separators[0]
        splits = text.split(sep) if sep else list(text)
        
        chunks = []
        current = []
        current_len = 0
        
        for split in splits:
            if current_len + len(split) > self.chunk_size and current:
                chunks.append(sep.join(current))
                # Keep overlap
                overlap_text = sep.join(current)[-self.chunk_overlap:]
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text) if overlap_text else 0
            
            current.append(split)
            current_len += len(split) + len(sep)
        
        if current:
            chunks.append(sep.join(current))
        
        # Recurse on oversized chunks
        result = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                result.extend(self._recursive_helper(chunk, separators[1:]))
            elif chunk.strip():
                result.append(chunk.strip())
        
        return result
    
    def _token_split(self, text: str) -> list[str]:
        """Token-based splitting (using word approximation)."""
        words = text.split()
        token_size = self.chunk_size // 4  # Approximate chars per token
        overlap_tokens = self.chunk_overlap // 4
        
        chunks = []
        start = 0
        while start < len(words):
            end = start + token_size
            chunks.append(' '.join(words[start:end]))
            start = end - overlap_tokens
        
        return chunks
    
    def _semantic_split(self, text: str) -> list[str]:
        """Sentence-based semantic splitting."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current = []
        current_len = 0
        
        for sentence in sentences:
            if current_len + len(sentence) > self.chunk_size and current:
                chunks.append(' '.join(current))
                # Keep last 2 sentences for overlap
                current = current[-2:]
                current_len = sum(len(s) for s in current)
            
            current.append(sentence)
            current_len += len(sentence)
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks
    
    def _add_headers(self, chunks: list[str], doc_name: str) -> list[str]:
        """Add context headers to chunks."""
        return [
            f"[Document: {doc_name} | Chunk {i+1}/{len(chunks)}]\n\n{chunk}"
            for i, chunk in enumerate(chunks)
        ]


# Example usage
splitter = UniversalTextSplitter(
    split_type=SplitType.RECURSIVE,
    chunk_size=500,
    chunk_overlap=50,
    add_context_headers=True
)

document = """Your long document text here..."""
chunks = splitter.split("machine_learning_guide.md")
```

## Key Takeaways

1. **Character splitting is simple but breaks semantic units**
2. **Recursive splitting adapts to document structure**
3. **Token splitting aligns with LLM limits accurately**
4. **Overlap preserves context across chunk boundaries**
5. **Semantic overlap (by sentences) is often better than fixed overlap**
6. **Context headers improve retrieval by adding document-level information**

## Additional Resources

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) - Production implementations
- [tiktoken Library](https://github.com/openai/tiktoken) - Accurate token counting for OpenAI models
- [Sentence Tokenization (NLTK)](https://www.nltk.org/api/nltk.tokenize.html) - Better sentence splitting
