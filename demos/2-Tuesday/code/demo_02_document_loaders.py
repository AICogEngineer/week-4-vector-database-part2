"""
Demo 02: Multi-Format Document Loading

This demo shows trainees how to:
1. Extract text from HTML documents
2. Process Markdown files while preserving structure
3. Handle code files and extract documentation
4. Build a unified document loader

Learning Objectives:
- Handle diverse document formats in RAG pipelines
- Extract meaningful text while removing noise
- Build extensible document loading systems

References:
- Written Content: 02-handling-document-types.md
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from html import unescape

# ============================================================================
# PART 1: The Multi-Format Challenge
# ============================================================================

print("=" * 70)
print("PART 1: The Multi-Format Challenge")
print("=" * 70)

print("""
REAL RAG SYSTEMS face diverse document types:

PDF      → Binary format, tables, images, scanned pages
HTML     → Tags, scripts, styles mixed with content  
Markdown → Headers, code blocks, links, formatting
Code     → Functions, classes, docstrings, comments
JSON     → Nested structures, key-value pairs

Each format needs SPECIALIZED EXTRACTION to get clean text!
""")

# ============================================================================
# PART 2: HTML Processing
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: HTML Processing")
print("=" * 70)

def extract_html_text(html: str) -> str:
    """
    Extract clean text from HTML content.
    
    Steps:
    1. Remove script and style elements entirely
    2. Decode HTML entities (&amp; → &)
    3. Replace block elements with newlines
    4. Strip all remaining tags
    5. Normalize whitespace
    """
    # First decode HTML entities
    text = unescape(html)
    
    # Remove script and style elements completely
    text = re.sub(
        r'<script[^>]*>.*?</script>', 
        '', 
        text, 
        flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(
        r'<style[^>]*>.*?</style>', 
        '', 
        text, 
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Replace block elements with newlines
    block_tags = r'<(p|div|br|h[1-6]|li|tr|article|section|header|footer)[^>]*>'
    text = re.sub(block_tags, '\n', text, flags=re.IGNORECASE)
    
    # Remove all remaining tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


# Sample HTML document
sample_html = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Guide</title>
    <script>console.log("This should be removed");</script>
    <style>body { color: black; }</style>
</head>
<body>
    <h1>Machine Learning Basics</h1>
    
    <p>Machine learning is a <b>powerful</b> subset of AI.</p>
    
    <div class="section">
        <h2>Supervised Learning</h2>
        <p>Uses labeled data for training.</p>
        <ul>
            <li>Classification</li>
            <li>Regression</li>
        </ul>
    </div>
    
    <footer>Copyright &copy; 2024</footer>
</body>
</html>
"""

print("[Step 1] Processing HTML document...")
print("\nOriginal HTML (truncated):")
print("-" * 50)
print(sample_html[:300] + "...")
print("-" * 50)

html_text = extract_html_text(sample_html)

print("\nExtracted text:")
print("-" * 50)
print(html_text)
print("-" * 50)

print("""
OBSERVATIONS:
- Script and style content removed completely
- HTML entities decoded (© shown correctly)
- Block elements converted to newlines
- Bold tags removed but text preserved
""")

# ============================================================================
# PART 3: Markdown Processing
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Markdown Processing")
print("=" * 70)

def extract_markdown_text(markdown: str, keep_structure: bool = True) -> str:
    """
    Extract text from Markdown while optionally preserving structure.
    
    Options:
    - keep_structure=True: Headers become plain text with newlines
    - keep_structure=False: All formatting removed
    """
    text = markdown
    
    # Remove code blocks (replace with placeholder or remove)
    text = re.sub(r'```[\s\S]*?```', '\n[CODE BLOCK]\n', text)
    
    # Remove inline code backticks but keep content
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    if keep_structure:
        # Convert headers to plain text with extra spacing
        text = re.sub(r'^#{1,6}\s+(.+)$', r'\n\1\n', text, flags=re.MULTILINE)
    else:
        # Remove header markers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)       # Italic
    text = re.sub(r'__([^_]+)__', r'\1', text)       # Bold underscore
    text = re.sub(r'_([^_]+)_', r'\1', text)         # Italic underscore
    
    # Convert links to just text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove images
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[Image: \1]', text)
    
    # Convert blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


sample_markdown = """
# Machine Learning Guide

This guide covers the **fundamentals** of ML.

## Getting Started

Before diving in, make sure you have:
- Python 3.8+
- Basic *statistics* knowledge

Check out [our tutorial](https://example.com/tutorial) for more.

## Code Example

```python
def train_model(data):
    model = create_model()
    model.fit(data)
    return model
```

### Key Concepts

The main concept is `gradient descent`, which optimizes the model.

> This is an important quote about ML.

![Diagram](https://example.com/diagram.png)
"""

print("[Step 2] Processing Markdown document...")
print("\nOriginal Markdown:")
print("-" * 50)
print(sample_markdown[:400] + "...")
print("-" * 50)

md_text = extract_markdown_text(sample_markdown)

print("\nExtracted text:")
print("-" * 50)
print(md_text)
print("-" * 50)

print("""
OBSERVATIONS:
- Headers preserved as plain text
- Bold/italic markers removed
- Link text kept, URLs removed
- Code blocks replaced with placeholder
- Images noted but not included
""")

# ============================================================================
# PART 4: Code File Processing
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Code File Processing")
print("=" * 70)

def extract_python_docs(code: str) -> List[Dict]:
    """
    Extract documentation from Python code.
    
    Returns:
    - Module docstrings
    - Class names and docstrings
    - Function signatures and docstrings
    """
    results = []
    
    # Check for module docstring (triple quotes at start)
    module_match = re.match(r'^[\s\n]*"""([\s\S]*?)"""', code)
    if module_match:
        results.append({
            "type": "module",
            "content": module_match.group(1).strip()
        })
    
    # Find class definitions
    class_pattern = r'class\s+(\w+)[^:]*:\s*(?:"""([\s\S]*?)""")?'
    for match in re.finditer(class_pattern, code):
        results.append({
            "type": "class",
            "name": match.group(1),
            "docstring": match.group(2).strip() if match.group(2) else ""
        })
    
    # Find function definitions
    func_pattern = r'def\s+(\w+)\s*\(([^)]*)\)[^:]*:\s*(?:"""([\s\S]*?)""")?'
    for match in re.finditer(func_pattern, code):
        results.append({
            "type": "function",
            "name": match.group(1),
            "signature": f"{match.group(1)}({match.group(2)})",
            "docstring": match.group(3).strip() if match.group(3) else ""
        })
    
    return results


sample_python = '''
"""
Machine Learning Utilities Module

This module provides utilities for training and evaluating ML models.
"""

class MLModel:
    """A simple machine learning model wrapper."""
    
    def __init__(self, config):
        """Initialize the model with configuration."""
        self.config = config
    
    def train(self, data, epochs=10):
        """
        Train the model on provided data.
        
        Args:
            data: Training data
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        pass
    
    def predict(self, inputs):
        """Generate predictions for inputs."""
        pass

def load_data(path):
    """Load and preprocess data from file path."""
    pass
'''

print("[Step 3] Extracting Python documentation...")
print("\nOriginal Python code:")
print("-" * 50)
print(sample_python[:400] + "...")
print("-" * 50)

python_docs = extract_python_docs(sample_python)

print("\nExtracted documentation:")
print("-" * 50)
for doc in python_docs:
    if doc["type"] == "module":
        print(f"MODULE: {doc['content'][:80]}...")
    elif doc["type"] == "class":
        print(f"CLASS: {doc['name']} - {doc['docstring']}")
    elif doc["type"] == "function":
        print(f"FUNCTION: {doc['signature']} - {doc['docstring'][:50] if doc['docstring'] else 'No docstring'}...")
print("-" * 50)

print("""
OBSERVATIONS:
- Module documentation extracted
- Class names and docstrings captured
- Function signatures and docs preserved
- Implementation details ignored (often desired)
""")

# ============================================================================
# PART 5: Unified Document Loader
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Unified Document Loader")
print("=" * 70)

@dataclass
class LoadedDocument:
    """Represents a loaded and processed document."""
    source: str
    doc_type: str
    content: str
    metadata: Dict


class DocumentLoader:
    """
    Unified interface for loading various document formats.
    
    Usage:
        loader = DocumentLoader()
        doc = loader.load_text(content, "markdown")
        # or
        doc = loader.load_file("path/to/file.md")
    """
    
    EXTENSION_MAP = {
        '.html': 'html',
        '.htm': 'html',
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.txt': 'text',
        '.py': 'python',
        '.js': 'javascript'
    }
    
    def load_text(self, content: str, doc_type: str, source: str = "unknown") -> LoadedDocument:
        """Load from text content with specified type."""
        if doc_type == "html":
            processed = extract_html_text(content)
        elif doc_type == "markdown":
            processed = extract_markdown_text(content)
        elif doc_type == "python":
            docs = extract_python_docs(content)
            processed = "\n\n".join([
                f"{d['type'].upper()}: {d.get('name', '')} {d.get('docstring', d.get('content', ''))}"
                for d in docs
            ])
        else:
            processed = content  # Plain text
        
        return LoadedDocument(
            source=source,
            doc_type=doc_type,
            content=processed,
            metadata={
                "original_length": len(content),
                "processed_length": len(processed)
            }
        )
    
    def detect_type(self, path: str) -> str:
        """Detect document type from file extension."""
        ext = Path(path).suffix.lower()
        return self.EXTENSION_MAP.get(ext, "text")


print("DocumentLoader provides a unified interface:")
print("-" * 50)

loader = DocumentLoader()

# Process different formats
formats_demo = [
    ("html", sample_html),
    ("markdown", sample_markdown),
    ("python", sample_python)
]

for doc_type, content in formats_demo:
    doc = loader.load_text(content, doc_type, source=f"demo.{doc_type}")
    print(f"\n{doc_type.upper()}:")
    print(f"  Source: {doc.source}")
    print(f"  Original: {doc.metadata['original_length']} chars")
    print(f"  Processed: {doc.metadata['processed_length']} chars")
    print(f"  Content preview: {doc.content[:60].replace(chr(10), ' ')}...")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO 2 COMPLETE: Multi-Format Document Loading")
print("=" * 70)

print("""
Key Takeaways:

1. HTML PROCESSING
   - Remove scripts, styles, comments
   - Decode entities, strip tags
   - Preserve text structure

2. MARKDOWN PROCESSING
   - Keep headers as plain text
   - Remove formatting markers
   - Handle code blocks and links

3. CODE FILE PROCESSING
   - Extract docstrings and signatures
   - Focus on documentation, not implementation
   - Preserve function/class structure

4. UNIFIED LOADING
   - Single interface for all formats
   - Auto-detect by extension
   - Consistent output structure

Coming Next: Demo 3 covers metadata storage and filtering!
""")

# ============================================================================
# INSTRUCTOR NOTES
# ============================================================================

print("\n" + "=" * 70)
print("INSTRUCTOR NOTES")
print("=" * 70)

print("""
Discussion Questions:
1. "What would you do with a PDF containing images?"
2. "How would you handle a document with code snippets in prose?"
3. "What metadata would you extract from each format?"

Interactive Exercise:
- Have trainees bring their own documents to test

Common Confusions:
- "What about PDFs?" → Need libraries like PyMuPDF, out of scope today
- "What about images in HTML?" → Extract alt text, or ignore
- "Should we keep code implementation?" → Usually no for RAG

If Running Short on Time:
- Skip code file processing
- Focus on HTML and Markdown

If Trainees Are Advanced:
- Discuss OCR for scanned documents
- Mention langchain document loaders
""")

print("\n" + "=" * 70)
