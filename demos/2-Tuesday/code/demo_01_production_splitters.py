"""
Demo 01: Production Text Splitters with LangChain

This demo shows trainees how to:
1. Use LangChain's production-ready text splitters
2. Split Markdown with header metadata preservation
3. Split code (Python) with language-aware boundaries
4. Split JSON data while preserving structure
5. Split HTML with header tracking

Learning Objectives:
- Use RecursiveCharacterTextSplitter for general text
- Use MarkdownHeaderTextSplitter for structured documents
- Use RecursiveCharacterTextSplitter.from_language() for code
- Use RecursiveJsonSplitter for API responses and configs
- Use HTMLHeaderTextSplitter for web content

Prerequisites:
- pip install langchain-text-splitters

References:
- https://docs.langchain.com/oss/python/integrations/splitters
- https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter
- https://docs.langchain.com/oss/python/integrations/splitters/code_splitter
- https://docs.langchain.com/oss/python/integrations/splitters/recursive_json_splitter
- https://docs.langchain.com/oss/python/integrations/splitters/split_html
"""

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveJsonSplitter,
    HTMLHeaderTextSplitter,
    Language,
)

# ============================================================================
# PART 1: Why LangChain Splitters?
# ============================================================================

print("=" * 70)
print("PART 1: Why LangChain Splitters?")
print("=" * 70)

print("""
Don't reinvent the wheel! LangChain provides battle-tested splitters.

AVAILABLE SPLITTERS:
───────────────────
• RecursiveCharacterTextSplitter - The go-to for most documents
• MarkdownHeaderTextSplitter     - Preserves Markdown structure + metadata
• RecursiveJsonSplitter          - Splits JSON while preserving structure
• HTMLHeaderTextSplitter         - Tracks HTML header hierarchy
• Language.PYTHON, .JS, etc.     - Code-aware splitting

These splitters TRY MULTIPLE SEPARATORS in order of preference.
""")

# ============================================================================
# PART 2: Markdown Header Text Splitter
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: MarkdownHeaderTextSplitter")
print("=" * 70)

print("""
From: https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter

The MarkdownHeaderTextSplitter:
• Splits at headers (#, ##, ###)
• Adds header hierarchy to metadata
• Perfect for documentation and README files
""")

markdown_document = """# Machine Learning Guide

This guide covers ML fundamentals.

## Supervised Learning

Supervised learning uses labeled data to train models.

### Classification

Classification predicts discrete categories like spam/not-spam.

### Regression

Regression predicts continuous values like house prices.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.

### Clustering

Clustering groups similar data points together.
"""

print("[Step 1] Splitting Markdown by headers...")

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)

print(f"\nGenerated {len(md_header_splits)} documents with metadata:")
for i, doc in enumerate(md_header_splits):
    content_preview = doc.page_content[:50].replace("\n", "\\n")
    print(f"\n  Document {i+1}:")
    print(f"    Content: {content_preview}...")
    print(f"    Metadata: {doc.metadata}")

print("""
OBSERVATION: 
• Each section becomes a Document with metadata tracking the header hierarchy
• "Classification" knows it's under "Supervised Learning" under "ML Guide"
• This metadata is GOLD for RAG - you can filter by section!
""")

# With strip_headers=False to keep headers in content
print("\n[Step 2] Keeping headers in content (strip_headers=False)...")

markdown_splitter_keep = MarkdownHeaderTextSplitter(
    headers_to_split_on, 
    strip_headers=False
)
md_splits_with_headers = markdown_splitter_keep.split_text(markdown_document)

print(f"  First doc content: {md_splits_with_headers[0].page_content[:60]}...")

# ============================================================================
# PART 3: Code Splitter (Python)
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Code Splitter (Language.PYTHON)")
print("=" * 70)

print("""
From: https://docs.langchain.com/oss/python/integrations/splitters/code_splitter

RecursiveCharacterTextSplitter.from_language():
• Knows Python syntax (class, def, etc.)
• Splits at function/class boundaries
• Also supports: JS, TS, HTML, Markdown, Go, Rust, Java, and more!
""")

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

class DataProcessor:
    \"\"\"Process data for ML models.\"\"\"
    
    def __init__(self, config):
        self.config = config
        self.data = None
    
    def load_data(self, path):
        \"\"\"Load data from file.\"\"\"
        with open(path) as f:
            self.data = f.read()
        return self.data
    
    def clean_data(self):
        \"\"\"Clean the loaded data.\"\"\"
        if self.data is None:
            raise ValueError("No data loaded")
        return self.data.strip()

def train_model(data, epochs=10):
    \"\"\"Train a machine learning model.\"\"\"
    for epoch in range(epochs):
        pass
    return "trained_model"

# Call the function
hello_world()
"""

print("[Step 3] Splitting Python code...")

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0
)

python_docs = python_splitter.create_documents([PYTHON_CODE])

print(f"\nGenerated {len(python_docs)} code chunks:")
for i, doc in enumerate(python_docs):
    # Detect what this chunk contains
    has_class = "class " in doc.page_content
    has_def = "def " in doc.page_content
    chunk_type = "CLASS" if has_class else ("FUNCTION" if has_def else "OTHER")
    preview = doc.page_content[:45].replace("\n", "\\n")
    print(f"  Chunk {i+1} [{chunk_type:8}]: {preview}...")

# Show the separators LangChain uses for Python
print("\n[LangChain's Python separators]:")
py_separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
for i, sep in enumerate(py_separators[:6]):
    print(f"  {i+1}. {repr(sep)}")

print("""
OBSERVATION:
• LangChain knows Python syntax!
• Functions stay with their docstrings
• Class methods can stay grouped
""")

# ============================================================================
# PART 4: JSON Splitter
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: RecursiveJsonSplitter")
print("=" * 70)

print("""
From: https://docs.langchain.com/oss/python/integrations/splitters/recursive_json_splitter

RecursiveJsonSplitter:
• Splits JSON while preserving valid structure
• Each chunk is still valid JSON
• Perfect for API responses, config files, etc.
""")

json_data = {
    "company": "TechCorp",
    "products": [
        {
            "name": "Widget A",
            "description": "A fantastic widget for all your needs. It comes with premium features.",
            "price": 99.99,
            "features": ["feature1", "feature2", "feature3"]
        },
        {
            "name": "Widget B", 
            "description": "An economy widget with basic functionality. Great for beginners.",
            "price": 49.99,
            "features": ["basic1", "basic2"]
        },
        {
            "name": "Widget C",
            "description": "Professional grade widget for enterprise use. Maximum performance.",
            "price": 299.99,
            "features": ["pro1", "pro2", "pro3", "pro4"]
        }
    ],
    "contact": {
        "email": "info@techcorp.com",
        "phone": "1-800-TECH"
    }
}

print("[Step 4] Splitting JSON data...")

json_splitter = RecursiveJsonSplitter(max_chunk_size=200)

# Method 1: Get JSON chunks (still as dicts)
json_chunks = json_splitter.split_json(json_data=json_data)

print(f"\nGenerated {len(json_chunks)} JSON chunks:")
for i, chunk in enumerate(json_chunks[:4]):
    print(f"  Chunk {i+1}: {str(chunk)[:60]}...")

# Method 2: Get as Documents
json_docs = json_splitter.create_documents(texts=[json_data])

print(f"\nAs Documents ({len(json_docs)} docs):")
for i, doc in enumerate(json_docs[:3]):
    print(f"  Doc {i+1}: {doc.page_content[:60]}...")

print("""
OBSERVATION:
• Each chunk is still valid JSON!
• Nested structures are preserved where possible
• Great for embedding structured data
""")

# ============================================================================
# PART 5: HTML Header Text Splitter
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: HTMLHeaderTextSplitter")
print("=" * 70)

print("""
From: https://docs.langchain.com/oss/python/integrations/splitters/split_html

HTMLHeaderTextSplitter:
• Splits at HTML headers (h1, h2, h3)
• Tracks header hierarchy in metadata
• Similar to MarkdownHeaderTextSplitter but for HTML
""")

html_string = """
<html>
<body>
    <h1>Main Title</h1>
    <p>This is an introductory paragraph with some basic content.</p>
    
    <h2>Section 1: Introduction</h2>
    <p>This section introduces the topic.</p>
    <ul>
        <li>First item</li>
        <li>Second item</li>
        <li>Third item</li>
    </ul>
    
    <h3>Subsection 1.1: Details</h3>
    <p>This subsection provides additional details.</p>
    
    <h2>Section 2: Implementation</h2>
    <p>This section covers implementation details.</p>
    
    <h2>Conclusion</h2>
    <p>This is the conclusion of the document.</p>
</body>
</html>
"""

print("[Step 5] Splitting HTML by headers...")

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)

print(f"\nGenerated {len(html_header_splits)} documents with metadata:")
for i, doc in enumerate(html_header_splits):
    content_preview = doc.page_content[:40].replace("\n", "\\n")
    print(f"\n  Document {i+1}:")
    print(f"    Content: {content_preview}...")
    print(f"    Metadata: {doc.metadata}")

print("""
OBSERVATION:
• HTML tags are stripped, text is extracted
• Header hierarchy is preserved in metadata
• Great for scraping documentation sites!
""")

# ============================================================================
# PART 6: Quick Reference
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Quick Reference - Which Splitter to Use")
print("=" * 70)

print("""
┌─────────────────────────┬──────────────────────────────────────────────┐
│ Document Type           │ Recommended Splitter                         │
├─────────────────────────┼──────────────────────────────────────────────┤
│ Plain text / prose      │ RecursiveCharacterTextSplitter (default)     │
│ Markdown documentation  │ MarkdownHeaderTextSplitter                   │
│ Python code             │ Recursive.from_language(Language.PYTHON)     │
│ JavaScript/TypeScript   │ Recursive.from_language(Language.JS/TS)      │
│ JSON / API responses    │ RecursiveJsonSplitter                        │
│ HTML / web pages        │ HTMLHeaderTextSplitter                       │
└─────────────────────────┴──────────────────────────────────────────────┘

IMPORTS CHEAT SHEET:
───────────────────
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveJsonSplitter,
    HTMLHeaderTextSplitter,
    Language,
)

CHUNK SIZE RECOMMENDATIONS:
• For embeddings: 500-1000 characters (100-200 tokens)
• For LLM context: Match model's token limit minus prompt
• Overlap: 10-20% of chunk size
""")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO 1 COMPLETE: Production Text Splitters with LangChain")
print("=" * 70)

print("""
Key Takeaways:

1. USE LANGCHAIN - Don't reinvent the wheel!
   pip install langchain-text-splitters

2. MARKDOWNHEADERTEXTSPLITTER
   • Tracks header hierarchy in metadata
   • Perfect for documentation

3. LANGUAGE.PYTHON (and others)
   • Code-aware splitting at function/class boundaries
   • Supports 15+ programming languages

4. RECURSIVEJSONSPLITTER
   • Keeps chunks as valid JSON
   • Great for structured data

5. HTMLHEADERTEXTSPLITTER
   • Tracks HTML header hierarchy
   • Strips tags, extracts text

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
1. "Why is metadata from MarkdownHeaderTextSplitter useful for RAG?"
2. "When would you use RecursiveJsonSplitter vs just stringify JSON?"
3. "What Language.X would you use for SQL files?"

Interactive Exercise:
- Have trainees try from_language(Language.MARKDOWN) vs MarkdownHeaderTextSplitter
- Compare the outputs and discuss trade-offs

LangChain Import Note:
- Use: from langchain_text_splitters import ...
- Install: pip install langchain-text-splitters
- Docs: https://docs.langchain.com/oss/python/integrations/splitters

Common Confusions:
- "MarkdownHeaderTextSplitter vs from_language(Language.MARKDOWN)"
  → Header splitter tracks metadata, from_language just uses MD separators
  
- "Why use JSONSplitter vs just splitting stringified JSON?"
  → JSONSplitter guarantees each chunk is valid JSON

If Running Short on Time:
- Focus on Markdown and Python examples
- Skip HTML splitting

If Trainees Are Advanced:
- Show ExperimentalMarkdownSyntaxTextSplitter
- Discuss HTMLSemanticPreservingSplitter for tables
- Explore creating custom separators
""")

print("\n" + "=" * 70)
