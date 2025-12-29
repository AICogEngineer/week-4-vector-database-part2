"""
Exercise 01: Multi-Format Document Ingestion with LangChain - Solution

Complete implementation using LangChain text splitters for multi-format document ingestion.

Prerequisites:
- pip install langchain-text-splitters chromadb sentence-transformers
"""

from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

# LangChain splitters
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
)

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
    <p>Welcome to our support documentation.</p>
    
    <h2>Getting Help</h2>
    <p>Contact us at support@example.com or call 1-800-HELP.</p>
    <p>Our team responds within 24 hours.</p>
    
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
# SOLUTION: LANGCHAIN-BASED SPLITTERS
# ============================================================================

def split_html(html_content: str, source: str = "unknown.html") -> List[Document]:
    """
    Split HTML using LangChain HTMLHeaderTextSplitter.
    
    Returns multiple Documents, each with header hierarchy in metadata.
    """
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
    html_docs = html_splitter.split_text(html_content)
    
    # Convert LangChain Documents to our Document format
    documents = []
    for i, doc in enumerate(html_docs):
        metadata = dict(doc.metadata)
        metadata.update({
            "source": source,
            "chunk_index": i,
            "total_chunks": len(html_docs),
            "loaded_at": datetime.now().isoformat()
        })
        
        documents.append(Document(
            content=doc.page_content,
            metadata=metadata,
            source=source,
            format="html"
        ))
    
    return documents


def split_markdown(md_content: str, source: str = "unknown.md") -> List[Document]:
    """
    Split Markdown using LangChain MarkdownHeaderTextSplitter.
    
    Returns multiple Documents, each with header hierarchy in metadata.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on,
        strip_headers=False  # Keep headers in content for context
    )
    md_docs = markdown_splitter.split_text(md_content)
    
    # Convert LangChain Documents to our Document format
    documents = []
    for i, doc in enumerate(md_docs):
        metadata = dict(doc.metadata)
        metadata.update({
            "source": source,
            "chunk_index": i,
            "total_chunks": len(md_docs),
            "loaded_at": datetime.now().isoformat()
        })
        
        documents.append(Document(
            content=doc.page_content,
            metadata=metadata,
            source=source,
            format="markdown"
        ))
    
    return documents


def split_text(text_content: str, source: str = "unknown.txt") -> List[Document]:
    """
    Split plain text using LangChain RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = text_splitter.split_text(text_content)
    
    # Extract title from first line
    title = text_content.strip().split('\n')[0].strip()
    
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append(Document(
            content=chunk,
            metadata={
                "title": title,
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "loaded_at": datetime.now().isoformat()
            },
            source=source,
            format="text"
        ))
    
    return documents


class DocumentLoader:
    """
    Unified document loader using LangChain splitters.
    """
    
    def __init__(self):
        self.splitters = {
            '.html': split_html,
            '.htm': split_html,
            '.md': split_markdown,
            '.markdown': split_markdown,
            '.txt': split_text,
        }
    
    def load(self, content: str, source: str) -> List[Document]:
        """
        Load and split a document by detecting format from source filename.
        
        Returns a list of Document chunks with metadata.
        """
        # Extract extension
        ext = ''
        if '.' in source:
            ext = '.' + source.rsplit('.', 1)[-1].lower()
        
        # Find splitter
        splitter = self.splitters.get(ext, split_text)
        
        # Split and return
        return splitter(content, source)


def ingest_documents(doc_chunks: List[Document], collection) -> Dict:
    """
    Ingest document chunks into a Chroma collection.
    """
    if not doc_chunks:
        return {"total_docs": 0, "total_chunks": 0}
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract content and metadata
    contents = [d.content for d in doc_chunks]
    metadatas = [d.metadata for d in doc_chunks]
    ids = [f"chunk_{i}" for i in range(len(doc_chunks))]
    
    # Embed
    embeddings = embedder.encode(contents).tolist()
    
    # Add to collection
    collection.add(
        ids=ids,
        documents=contents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    # Count unique sources
    sources = set(d.source for d in doc_chunks)
    
    return {
        "total_docs": len(sources),
        "total_chunks": len(doc_chunks)
    }


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_tests():
    """Test the LangChain-based document ingestion pipeline."""
    print("=" * 60)
    print("Exercise 01: Multi-Format Ingestion with LangChain - SOLUTION")
    print("=" * 60)
    
    loader = DocumentLoader()
    
    test_cases = [
        ("support.html", SAMPLE_HTML),
        ("guide.md", SAMPLE_MARKDOWN),
        ("config.txt", SAMPLE_TEXT),
    ]
    
    all_chunks = []
    
    print("\n=== Loading and Splitting Documents ===\n")
    
    for source, content in test_cases:
        fmt = source.split('.')[-1].upper()
        print(f"[{fmt}] {source}")
        
        chunks = loader.load(content, source)
        
        print(f"  Chunks created: {len(chunks)}")
        
        # Show metadata from first chunk
        if chunks:
            first_meta = chunks[0].metadata
            if 'Header 1' in first_meta:
                print(f"  Header 1: {first_meta.get('Header 1', 'N/A')}")
            if 'Header 2' in first_meta:
                print(f"  Header 2: {first_meta.get('Header 2', 'N/A')}")
            print(f"  First chunk preview: {chunks[0].content[:50]}...")
        
        all_chunks.extend(chunks)
        print()
    
    # Test ingestion
    print("=== Ingesting to Vector Store ===")
    
    client = chromadb.Client()
    
    try:
        client.delete_collection("exercise_01_langchain")
    except:
        pass
    
    collection = client.create_collection("exercise_01_langchain")
    
    stats = ingest_documents(all_chunks, collection)
    
    print(f"  Ingested {stats['total_chunks']} chunks from {stats['total_docs']} documents")
    
    # Test query
    print("\n=== Test Query ===")
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    query = "How do I configure the system?"
    query_emb = embedder.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_emb,
        n_results=3
    )
    
    print(f'Query: "{query}"')
    print("Results:")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
        source = meta.get('source', 'unknown')
        header = meta.get('Header 2', meta.get('Header 1', ''))
        print(f"  {i}. [{source}] {header}")
        print(f"     \"{doc[:50]}...\"")
    
    print("\n" + "=" * 60)
    print("[OK] LangChain-based multi-format ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
