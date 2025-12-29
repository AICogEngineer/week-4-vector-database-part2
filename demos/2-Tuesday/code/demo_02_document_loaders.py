"""
Demo 02: Multi-Format Document Loading with LangChain

This demo shows trainees how to use LangChain's document loaders for:
1. Web pages (WebBaseLoader)
2. PDF files (PyPDFLoader)
3. CSV files (CSVLoader)
4. JSON files (JSONLoader)

Learning Objectives:
- Use LangChain's production document loaders
- Understand the common Document interface
- Handle diverse document formats in RAG pipelines

Prerequisites:
- pip install langchain langchain-community beautifulsoup4 pypdf

References:
- https://docs.langchain.com/oss/python/integrations/document_loaders
"""

import tempfile
import os
from pathlib import Path

# LangChain Document Loaders
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    CSVLoader,
    JSONLoader,
)

# ============================================================================
# PART 1: The Document Loader Interface
# ============================================================================

print("=" * 70)
print("PART 1: LangChain Document Loader Interface")
print("=" * 70)

print("""
LangChain provides 100+ document loaders with a COMMON INTERFACE:

    loader = SomeLoader(...)
    documents = loader.load()      # Load all at once
    # or
    for doc in loader.lazy_load(): # Stream lazily
        process(doc)

Each Document has:
    - page_content: str   (the text)
    - metadata: dict      (source, page number, etc.)

This means YOUR CODE stays the same regardless of document format!
""")

# ============================================================================
# PART 2: WebBaseLoader - Loading Web Pages
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: WebBaseLoader - Loading Web Pages")
print("=" * 70)

print("""
WebBaseLoader uses BeautifulSoup to extract text from web pages.
Great for documentation sites, blogs, and articles.

From: https://docs.langchain.com/oss/python/integrations/document_loaders/web_base
""")

print("[Step 1] Loading a web page...")

try:
    # Load a simple, stable documentation page
    web_loader = WebBaseLoader(
        web_paths=["https://python.langchain.com/docs/introduction/"],
    )
    
    web_docs = web_loader.load()
    
    print(f"\n  Loaded {len(web_docs)} document(s)")
    
    if web_docs:
        doc = web_docs[0]
        print(f"  Content length: {len(doc.page_content)} chars")
        print(f"  Metadata: {list(doc.metadata.keys())}")
        print(f"  Source: {doc.metadata.get('source', 'N/A')}")
        print(f"\n  Content preview:")
        print(f"  {doc.page_content[:200].strip()}...")
        
except Exception as e:
    print(f"  [Note] Web loading requires internet: {e}")
    print("  Simulating web content for demo...")

print("""
OBSERVATIONS:
- WebBaseLoader extracts text from HTML automatically
- Metadata includes the source URL
- Multiple URLs can be loaded at once with web_paths=["url1", "url2"]
""")

# ============================================================================
# PART 3: PyPDFLoader - Loading PDF Files
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: PyPDFLoader - Loading PDF Files")
print("=" * 70)

print("""
PyPDFLoader extracts text from PDF files, page by page.
Each page becomes a separate Document with page number metadata.

From: https://docs.langchain.com/oss/python/integrations/document_loaders/pypdfloader
""")

# Create a sample PDF for demo (using reportlab would need extra dep)
# Instead, we'll demonstrate the API and show what the output looks like

print("[Step 2] Demonstrating PDF loading...")

# Create a temporary PDF-like file for API demonstration
pdf_demo_path = Path(tempfile.mkdtemp()) / "sample.pdf"

print(f"""
  PyPDFLoader usage:
  
  from langchain_community.document_loaders import PyPDFLoader
  
  loader = PyPDFLoader("path/to/document.pdf")
  pages = loader.load()
  
  # Each page is a separate Document
  for page in pages:
      print(f"Page {{page.metadata['page']}}: {{page.page_content[:100]}}...")

  Output structure:
  - Document 1: Page 0 content, metadata={{'source': 'doc.pdf', 'page': 0}}
  - Document 2: Page 1 content, metadata={{'source': 'doc.pdf', 'page': 1}}
  - etc.
""")

print("""
OBSERVATIONS:
- Each PDF page becomes a separate Document
- Page numbers are in metadata (0-indexed)
- Works with most PDF formats
- For scanned PDFs, consider OCR-based loaders like Amazon Textract
""")

# ============================================================================
# PART 4: CSVLoader - Loading CSV Files
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: CSVLoader - Loading CSV Files")
print("=" * 70)

print("""
CSVLoader creates one Document per row, with columns accessible in content.
Perfect for FAQs, product catalogs, or any tabular data.

From: https://docs.langchain.com/oss/python/integrations/document_loaders/csv
""")

# Create a sample CSV
csv_path = Path(tempfile.mkdtemp()) / "faq.csv"
csv_content = """question,answer,category
What is machine learning?,ML is a subset of AI that learns from data,basics
How does gradient descent work?,It minimizes loss by following the gradient,optimization
What are neural networks?,Computational models inspired by the brain,deep learning
What is overfitting?,When a model memorizes training data too well,problems
"""

with open(csv_path, "w") as f:
    f.write(csv_content)

print("[Step 3] Loading CSV file...")

csv_loader = CSVLoader(
    file_path=str(csv_path),
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
    }
)

csv_docs = csv_loader.load()

print(f"\n  Loaded {len(csv_docs)} documents (one per row)")
print("\n  Sample documents:")
for i, doc in enumerate(csv_docs[:3]):
    print(f"\n  Document {i+1}:")
    print(f"    Content: {doc.page_content[:80]}...")
    print(f"    Source: {doc.metadata.get('source', 'N/A')}")
    print(f"    Row: {doc.metadata.get('row', 'N/A')}")

print("""
OBSERVATIONS:
- Each CSV row becomes a Document
- Column names and values are in page_content
- Row number is in metadata
- Great for FAQ datasets!
""")

# ============================================================================
# PART 5: JSONLoader - Loading JSON Files
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: JSONLoader - Loading JSON Files")
print("=" * 70)

print("""
JSONLoader extracts content from JSON using jq-style queries.
Perfect for API responses, config files, or nested data.

From: https://docs.langchain.com/oss/python/integrations/document_loaders/json
""")

# Create a sample JSON file
json_path = Path(tempfile.mkdtemp()) / "docs.json"
json_content = """[
    {
        "title": "Introduction to ML",
        "content": "Machine learning enables computers to learn from data without explicit programming.",
        "category": "basics"
    },
    {
        "title": "Neural Networks",
        "content": "Neural networks are composed of layers of interconnected nodes that process information.",
        "category": "deep-learning"
    },
    {
        "title": "Training Models",
        "content": "Model training involves feeding data through the network and adjusting weights.",
        "category": "training"
    }
]"""

with open(json_path, "w") as f:
    f.write(json_content)

print("[Step 4] Loading JSON file...")

json_loader = JSONLoader(
    file_path=str(json_path),
    jq_schema=".[].content",  # Extract the 'content' field from each object
    text_content=False
)

json_docs = json_loader.load()

print(f"\n  Loaded {len(json_docs)} documents")
print("\n  Sample documents:")
for i, doc in enumerate(json_docs[:3]):
    print(f"\n  Document {i+1}:")
    print(f"    Content: {doc.page_content[:80]}...")
    print(f"    Metadata: {doc.metadata}")

# Also show how to extract with metadata
print("\n  [Advanced] Loading with metadata extraction:")

json_loader_meta = JSONLoader(
    file_path=str(json_path),
    jq_schema=".[]",  # Get full objects
    content_key="content",  # Use 'content' field as page_content
    metadata_func=lambda record, metadata: {
        **metadata,
        "title": record.get("title", ""),
        "category": record.get("category", "")
    }
)

json_docs_meta = json_loader_meta.load()

for i, doc in enumerate(json_docs_meta[:2]):
    print(f"\n  Document {i+1}:")
    print(f"    Content: {doc.page_content[:50]}...")
    print(f"    Title: {doc.metadata.get('title', 'N/A')}")
    print(f"    Category: {doc.metadata.get('category', 'N/A')}")

print("""
OBSERVATIONS:
- jq_schema selects which parts of JSON to extract
- content_key specifies which field becomes page_content
- metadata_func allows custom metadata extraction
- Great for API responses and structured data!
""")

# ============================================================================
# PART 6: Quick Reference
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Quick Reference - Common Document Loaders")
print("=" * 70)

print("""
┌─────────────────────────┬──────────────────────────────────────────────┐
│ Document Type           │ Loader                                       │
├─────────────────────────┼──────────────────────────────────────────────┤
│ Web pages               │ WebBaseLoader                                │
│ PDFs                    │ PyPDFLoader, PyMuPDFLoader                   │
│ CSV files               │ CSVLoader                                    │
│ JSON files              │ JSONLoader                                   │
│ HTML files              │ BSHTMLLoader (BeautifulSoup)                 │
│ Word docs               │ Docx2txtLoader                               │
│ Markdown                │ UnstructuredMarkdownLoader                   │
│ YouTube                 │ YoutubeLoader                                │
│ GitHub                  │ GitHubLoader                                 │
│ Notion                  │ NotionLoader                                 │
└─────────────────────────┴──────────────────────────────────────────────┘

INSTALLATION:
    pip install langchain langchain-community
    pip install pypdf           # For PyPDFLoader
    pip install beautifulsoup4  # For WebBaseLoader, BSHTMLLoader

ALL LOADERS RETURN:
    List[Document] where Document has:
    - page_content: str
    - metadata: dict
""")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO 2 COMPLETE: Multi-Format Document Loading with LangChain")
print("=" * 70)

print("""
Key Takeaways:

1. WEBBASELOADER
   - Extracts text from web pages
   - Uses BeautifulSoup under the hood
   - Great for documentation and articles

2. PYPDFLOADER
   - Extracts text from PDFs, page by page
   - Each page becomes a Document
   - Page number in metadata

3. CSVLOADER
   - One Document per row
   - Row number in metadata
   - Perfect for FAQs and catalogs

4. JSONLOADER
   - Uses jq-style queries to extract data
   - Can extract metadata from JSON fields
   - Great for API responses

5. COMMON INTERFACE
   - loader.load() returns List[Document]
   - loader.lazy_load() for streaming
   - Your code stays the same for any format!

Coming Next: Demo 3 covers metadata storage and filtering!
""")

# Cleanup temp files
import shutil
shutil.rmtree(csv_path.parent, ignore_errors=True)
shutil.rmtree(json_path.parent, ignore_errors=True)

# ============================================================================
# INSTRUCTOR NOTES
# ============================================================================

print("\n" + "=" * 70)
print("INSTRUCTOR NOTES")
print("=" * 70)

print("""
Discussion Questions:
1. "When would you use WebBaseLoader vs. scraping yourself?"
2. "What are the limitations of PDF text extraction?"
3. "How would you handle a 1000-page PDF efficiently?"

Setup Notes:
- WebBaseLoader needs internet access
- Have sample PDFs ready in case of connection issues

Common Confusions:
- "Why does PDF have multiple documents?" → One per page
- "Can I load multiple URLs?" → Yes, pass a list to web_paths
- "What about authentication?" → Most loaders support auth params

If Running Short on Time:
- Focus on CSVLoader and JSONLoader (work offline)
- Skip WebBaseLoader live demo

If Trainees Are Advanced:
- Show DirectoryLoader for batch loading
- Discuss RecursiveUrlLoader for sitemaps
- Mention UnstructuredLoader for 50+ file types
""")

print("\n" + "=" * 70)
