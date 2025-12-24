"""
Exercise 01: Multi-Format Document Ingestion - Solution

Complete implementation of the multi-format document ingestion pipeline.
"""

import re
from html import unescape
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================================
# SAMPLE DOCUMENTS
# ============================================================================

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Support Guide</title>
    <style>body { font-family: Arial; }</style>
</head>
<body>
    <h1>Customer Support Guide</h1>
    <p>Welcome to our <strong>support</strong> documentation.</p>
    
    <h2>Getting Help</h2>
    <p>Contact us at support@example.com or call 1-800-HELP.</p>
    <p>Our team responds within &lt;24 hours&gt;.</p>
    
    <script>console.log("tracking");</script>
    
    <h2>FAQ</h2>
    <p>Common questions and answers are listed below.</p>
</body>
</html>
"""

SAMPLE_MARKDOWN = """
# Getting Started Guide

Welcome to our product documentation.

## Installation

To install the software, follow these steps:

1. Download the installer from our website
2. Run the setup wizard
3. Configure your settings

```python
# Example configuration
config = {"api_key": "your-key"}
```

## Configuration

The system can be configured through the **settings panel**.

For more details, see [advanced configuration](./advanced.md).
"""

SAMPLE_TEXT = """
Configuration Reference Guide

This document describes all available configuration options.

Database Settings
-----------------
HOST: The database server hostname
PORT: The database server port (default: 5432)
USER: Database username
PASS: Database password

Application Settings
-------------------
DEBUG: Enable debug mode (true/false)
LOG_LEVEL: Logging verbosity (info, warn, error)

For support, contact admin@example.com
"""


@dataclass
class Document:
    """Represents a loaded and processed document."""
    content: str
    metadata: Dict
    source: str
    format: str


# ============================================================================
# SOLUTION IMPLEMENTATIONS
# ============================================================================

def load_html(html_content: str, source: str = "unknown.html") -> Document:
    """
    Load and clean an HTML document.
    """
    text = html_content
    
    # Extract title first (before removing tags)
    title_match = re.search(r'<title[^>]*>([^<]+)</title>', text, re.IGNORECASE)
    if not title_match:
        title_match = re.search(r'<h1[^>]*>([^<]+)</h1>', text, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else source
    
    # Remove script blocks
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove style blocks
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove all tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Decode HTML entities
    text = unescape(text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return Document(
        content=text,
        metadata={
            "title": title,
            "word_count": len(text.split()),
            "loaded_at": datetime.now().isoformat()
        },
        source=source,
        format="html"
    )


def load_markdown(md_content: str, source: str = "unknown.md") -> Document:
    """
    Load and clean a Markdown document.
    """
    text = md_content
    
    # Extract first header as title
    header_match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
    title = header_match.group(1).strip() if header_match else source
    
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # Remove inline code but keep content
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove header markers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove formatting - bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    
    # Remove formatting - italic
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # Convert links to text only
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove images
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return Document(
        content=text,
        metadata={
            "title": title,
            "word_count": len(text.split()),
            "loaded_at": datetime.now().isoformat()
        },
        source=source,
        format="markdown"
    )


def load_text(text_content: str, source: str = "unknown.txt") -> Document:
    """
    Load a plain text document.
    """
    text = text_content.strip()
    
    # Extract first line as title
    first_line = text.split('\n')[0].strip()
    title = first_line if first_line else source
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return Document(
        content=text,
        metadata={
            "title": title,
            "word_count": len(text.split()),
            "loaded_at": datetime.now().isoformat()
        },
        source=source,
        format="text"
    )


class DocumentLoader:
    """
    Unified document loader that handles multiple formats.
    """
    
    def __init__(self):
        self.loaders = {
            '.html': load_html,
            '.htm': load_html,
            '.md': load_markdown,
            '.markdown': load_markdown,
            '.txt': load_text,
        }
    
    def load(self, content: str, source: str) -> Document:
        """
        Load a document by detecting format from source filename.
        """
        # Extract extension
        ext = ''
        if '.' in source:
            ext = '.' + source.rsplit('.', 1)[-1].lower()
        
        # Find loader
        loader = self.loaders.get(ext, load_text)
        
        # Load and return
        return loader(content, source)


def chunk_document(doc: Document, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    """
    Chunk a document and prepare for vector store ingestion.
    """
    content = doc.content
    chunks = []
    
    # Simple fixed-size chunking
    step = chunk_size - overlap
    
    start = 0
    chunk_index = 0
    
    while start < len(content):
        end = start + chunk_size
        chunk_text = content[start:end].strip()
        
        if chunk_text:
            # Combine document metadata with chunk metadata
            chunk_meta = dict(doc.metadata)
            chunk_meta.update({
                "source": doc.source,
                "format": doc.format,
                "chunk_index": chunk_index,
            })
            
            chunks.append({
                "content": chunk_text,
                "metadata": chunk_meta
            })
            
            chunk_index += 1
        
        start += step
        if end >= len(content):
            break
    
    # Add total_chunks to all chunks
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)
    
    return chunks


def ingest_documents(documents: List[Document], collection) -> Dict:
    """
    Ingest documents into a Chroma collection.
    """
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_chunks = []
    
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        return {"total_docs": len(documents), "total_chunks": 0}
    
    # Extract content and metadata
    contents = [c["content"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    
    # Embed
    embeddings = embedder.encode(contents).tolist()
    
    # Add to collection
    collection.add(
        ids=ids,
        documents=contents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    return {
        "total_docs": len(documents),
        "total_chunks": len(all_chunks)
    }


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_tests():
    """Test the document ingestion pipeline."""
    print("=" * 60)
    print("Exercise 01: Multi-Format Document Ingestion - SOLUTION")
    print("=" * 60)
    
    loader = DocumentLoader()
    
    test_cases = [
        ("support.html", SAMPLE_HTML),
        ("guide.md", SAMPLE_MARKDOWN),
        ("config.txt", SAMPLE_TEXT),
    ]
    
    documents = []
    
    print("\n=== Loading Sample Documents ===\n")
    
    for source, content in test_cases:
        fmt = source.split('.')[-1].upper()
        print(f"[{fmt}] {source}")
        
        doc = loader.load(content, source)
        
        print(f"  Title: {doc.metadata.get('title', 'Unknown')}")
        print(f"  Length: {len(doc.content)} chars")
        print(f"  Words: {doc.metadata.get('word_count', 0)}")
        print(f"  [OK] Loaded successfully")
        
        documents.append(doc)
        print()
    
    # Test ingestion
    print("=== Ingesting to Vector Store ===")
    
    client = chromadb.Client()
    
    try:
        client.delete_collection("exercise_01")
    except:
        pass
    
    collection = client.create_collection("exercise_01")
    
    stats = ingest_documents(documents, collection)
    
    print(f"  Ingested {stats['total_chunks']} chunks from {stats['total_docs']} documents")
    
    # Test query
    print("\n=== Test Query ===")
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    query = "How do I configure the system?"
    query_emb = embedder.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_emb,
        n_results=2
    )
    
    print(f'Query: "{query}"')
    print("Results:")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
        print(f"  {i}. [{meta['source']}] \"{doc[:50]}...\"")
    
    print("\n" + "=" * 60)
    print("[OK] Multi-format ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
