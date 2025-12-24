"""
Demo 01: Production Text Splitters

This demo shows trainees how to:
1. Build LangChain-style recursive text splitters
2. Use language-specific separators
3. Manage chunk overlap effectively
4. Choose the right splitter for each document type

Learning Objectives:
- Implement production-quality recursive splitters
- Configure splitters for different content types
- Understand overlap trade-offs

References:
- Written Content: 01-text-splitting-techniques.md
"""

import re
from typing import List, Optional
from dataclasses import dataclass

# ============================================================================
# PART 1: Why Production Splitters Matter
# ============================================================================

print("=" * 70)
print("PART 1: Why Production Splitters Matter")
print("=" * 70)

print("""
Yesterday we built simple splitters. Today: PRODUCTION-QUALITY splitters.

THE KEY INSIGHT:
───────────────
Not all text should split the same way!

Prose:      Split at paragraphs first, then sentences
Markdown:   Split at headers first, then paragraphs  
Code:       Split at functions/classes first, then blocks

Production splitters TRY MULTIPLE SEPARATORS in order of preference.
""")

# ============================================================================
# PART 2: The Recursive Character Text Splitter
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Recursive Character Text Splitter")
print("=" * 70)

class RecursiveCharacterTextSplitter:
    """
    Production-quality text splitter that recursively tries separators.
    
    This is modeled after LangChain's RecursiveCharacterTextSplitter.
    It tries to split by the most meaningful separators first.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.keep_separator = keep_separator
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text recursively using a hierarchy of separators.
        """
        return self._split_text(text, self.separators)
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Internal recursive splitting logic."""
        final_chunks = []
        
        # Get current separator
        separator = separators[0] if separators else ""
        new_separators = separators[1:] if len(separators) > 1 else []
        
        # Split by current separator
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator means split by character
            splits = list(text)
        
        # Process splits
        good_splits = []
        for split in splits:
            # Re-add separator if keeping it
            if self.keep_separator and separator and split:
                split = separator + split
            
            if len(split) <= self.chunk_size:
                good_splits.append(split)
            elif new_separators:
                # Recursively process with next separator
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                
                other_chunks = self._split_text(split, new_separators)
                final_chunks.extend(other_chunks)
            else:
                # No more separators, force split
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                
                forced = self._force_split(split)
                final_chunks.extend(forced)
        
        # Don't forget remaining good splits
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)
        
        return [c for c in final_chunks if c.strip()]
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits up to chunk_size, with overlap."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split)
            
            if current_length + split_len > self.chunk_size:
                if current_chunk:
                    chunk_text = "".join(current_chunk)
                    chunks.append(chunk_text.strip())
                    
                    # Handle overlap
                    overlap_text = self._get_overlap_text(chunk_text)
                    current_chunk = [overlap_text] if overlap_text else []
                    current_length = len(overlap_text) if overlap_text else 0
            
            current_chunk.append(split)
            current_length += split_len
        
        if current_chunk:
            chunks.append("".join(current_chunk).strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the overlap portion from previous chunk."""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]
    
    def _force_split(self, text: str) -> List[str]:
        """Force split text that's too large."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        return chunks

print("""
The RecursiveCharacterTextSplitter:
1. Tries the first separator (e.g., \\n\\n for paragraphs)
2. If chunks are still too large, tries the next (\\n for lines)
3. Continues until chunks fit, or force-splits as last resort
4. Maintains overlap for context continuity
""")

# ============================================================================
# PART 3: Demonstrating the Splitter
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Basic Demonstration")
print("=" * 70)

sample_text = """
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables computers to learn from data. This technology has transformed numerous industries by automating complex decision-making processes.

Supervised Learning

In supervised learning, algorithms learn from labeled training data. The model makes predictions based on input features and compares them against known correct outputs.

Common applications include:
- Email spam detection
- Image classification
- Price prediction

Unsupervised Learning

Unsupervised learning works with unlabeled data. The algorithm discovers hidden patterns or structures without explicit guidance. Key techniques include clustering and dimensionality reduction.
"""

print("[Step 1] Creating a basic recursive splitter...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = splitter.split_text(sample_text)

print(f"\nInput: {len(sample_text)} characters")
print(f"Output: {len(chunks)} chunks")
print(f"\nChunks created:")

for i, chunk in enumerate(chunks):
    preview = chunk[:60].replace("\n", "\\n") + "..."
    print(f"  Chunk {i+1} ({len(chunk)} chars): {preview}")

# ============================================================================
# PART 4: Language-Specific Separators
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Language-Specific Separators")
print("=" * 70)

# Define separator configurations for different content types
SEPARATOR_CONFIGS = {
    "default": ["\n\n", "\n", ". ", " ", ""],
    
    "markdown": [
        "\n## ",      # H2 headers
        "\n### ",     # H3 headers
        "\n#### ",    # H4 headers
        "\n\n",       # Paragraphs
        "\n",         # Lines
        ". ",         # Sentences
        " "           # Words
    ],
    
    "python": [
        "\nclass ",   # Class definitions
        "\ndef ",     # Function definitions
        "\n\n",       # Double newlines
        "\n",         # Single newlines
        ". ",         # Sentences (in docstrings)
        " "           # Words
    ],
    
    "html": [
        "</div>",     # Block divs
        "</p>",       # Paragraphs
        "<br>",       # Line breaks
        "\n\n",
        "\n",
        ". ",
        " "
    ]
}

print("Available separator configurations:")
for config_name, seps in SEPARATOR_CONFIGS.items():
    sep_preview = ", ".join([repr(s) for s in seps[:4]]) + "..."
    print(f"  {config_name}: {sep_preview}")

# ============================================================================
# PART 5: Markdown Splitting
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Markdown-Aware Splitting")
print("=" * 70)

markdown_doc = """
# Machine Learning Guide

This guide covers the fundamentals of machine learning.

## Supervised Learning

Supervised learning uses labeled data to train models.

### Classification

Classification predicts discrete categories like spam/not-spam.

### Regression

Regression predicts continuous values like house prices.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.

### Clustering

Clustering groups similar data points together, like customer segmentation.
"""

print("Markdown document preview:")
print("-" * 50)
print(markdown_doc[:300] + "...")
print("-" * 50)

print("\n[Step 2] Splitting with markdown-aware separators...")

md_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=SEPARATOR_CONFIGS["markdown"]
)

md_chunks = md_splitter.split_text(markdown_doc)

print(f"\nGenerated {len(md_chunks)} chunks:")
for i, chunk in enumerate(md_chunks):
    # Check if chunk starts with a header
    is_header = chunk.strip().startswith("#")
    prefix = "[HEADER] " if is_header else ""
    preview = chunk[:50].replace("\n", "\\n").strip()
    print(f"  Chunk {i+1}: {prefix}{preview}...")

print("""
OBSERVATION: 
- Headers stay with their content
- Sections are split at meaningful boundaries
- No header is orphaned from its content
""")

# ============================================================================
# PART 6: Python Code Splitting
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Python Code Splitting")
print("=" * 70)

python_code = '''
"""
Module for machine learning utilities.
"""

class DataProcessor:
    """Process and clean data for ML models."""
    
    def __init__(self, config):
        self.config = config
        self.data = None
    
    def load_data(self, path):
        """Load data from file path."""
        with open(path) as f:
            self.data = f.read()
        return self.data
    
    def clean_data(self):
        """Clean the loaded data."""
        if self.data is None:
            raise ValueError("No data loaded")
        return self.data.strip()

def train_model(data, epochs=10):
    """Train a machine learning model."""
    for epoch in range(epochs):
        # Training logic here
        pass
    return "trained_model"

def evaluate_model(model, test_data):
    """Evaluate model performance."""
    # Evaluation logic
    return {"accuracy": 0.95}
'''

print("Python code preview:")
print("-" * 50)
print(python_code[:300] + "...")
print("-" * 50)

print("\n[Step 3] Splitting with Python-aware separators...")

py_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=SEPARATOR_CONFIGS["python"]
)

py_chunks = py_splitter.split_text(python_code)

print(f"\nGenerated {len(py_chunks)} chunks:")
for i, chunk in enumerate(py_chunks):
    # Detect what this chunk contains
    has_class = "class " in chunk
    has_def = "def " in chunk
    chunk_type = "CLASS" if has_class else ("FUNCTION" if has_def else "OTHER")
    preview = chunk[:50].replace("\n", "\\n").strip()
    print(f"  Chunk {i+1} [{chunk_type:8}]: {preview}...")

print("""
OBSERVATION:
- Classes can be kept together (if small enough)
- Functions stay with their docstrings
- Code structure is preserved
""")

# ============================================================================
# PART 7: Overlap Management
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: Overlap Management")
print("=" * 70)

print("""
OVERLAP preserves context across chunk boundaries.

Example without overlap:
  Chunk 1: "...the algorithm uses gradient descent."
  Chunk 2: "This optimization technique is powerful..."
  
  Problem: Reader doesn't know what "This" refers to!

Example WITH overlap:
  Chunk 1: "...the algorithm uses gradient descent."
  Chunk 2: "...gradient descent. This optimization technique is powerful..."
  
  Better: Context is preserved!
""")

# Demonstrate overlap impact
test_text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."

print("[Step 4] Comparing different overlap settings...")

for overlap in [0, 20, 50]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=40,
        chunk_overlap=overlap,
        separators=[". ", " "]
    )
    chunks = splitter.split_text(test_text)
    print(f"\nOverlap={overlap}: {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3]):
        print(f"  Chunk {i+1}: '{c}'")

print("""
TRADE-OFFS:
- More overlap = better context continuity
- More overlap = more chunks = more storage
- More overlap = more search results to process
- Typical: 10-20% of chunk_size
""")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO 1 COMPLETE: Production Text Splitters")
print("=" * 70)

print("""
Key Takeaways:

1. RECURSIVE SPLITTING tries meaningful boundaries first
   - Paragraphs, then sentences, then words
   - Falls back gracefully when needed

2. LANGUAGE-SPECIFIC separators improve quality
   - Markdown: Split at headers
   - Python: Split at classes/functions
   - HTML: Split at block elements

3. OVERLAP preserves context
   - 10-20% of chunk size is typical
   - Trade-off: context vs. storage/speed

4. PRODUCTION SYSTEMS use these patterns
   - LangChain's default splitter works this way
   - Most RAG frameworks follow similar logic

Coming Next: Demo 2 covers multi-format document loading!
""")

# ============================================================================
# INSTRUCTOR NOTES
# ============================================================================

print("\n" + "=" * 70)
print("INSTRUCTOR NOTES")
print("=" * 70)

print("""
Discussion Questions:
1. "What separators would you use for JSON documents?"
2. "How would you handle a document mixing prose and code?"
3. "When might you want to disable overlap?"

Interactive Exercise:
- Have trainees design separators for their own document types

Common Confusions:
- "Why keep the separator?" → Helps with context, especially headers
- "What's the perfect chunk size?" → Depends on embedding model limits
- "Is more overlap always better?" → No, diminishing returns

If Running Short on Time:
- Skip HTML splitting
- Focus on markdown and overlap discussion

If Trainees Are Advanced:
- Discuss token-based splitting for precise LLM limits
- Mention semantic chunking based on embedding similarity
""")

print("\n" + "=" * 70)
