"""
Exercise 01: Multi-Format Document Ingestion - Starter Code

Build a unified document ingestion pipeline for HTML, Markdown, and text files.

Instructions:
1. Implement each TODO function
2. Run this file to test your implementations
3. Check the expected output in the exercise guide
"""

import re
from html import unescape
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================================
# SAMPLE DOCUMENTS (DO NOT MODIFY)
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
# TODO: IMPLEMENT THESE FUNCTIONS
# ============================================================================

def load_html(html_content: str, source: str = "unknown.html") -> Document:
    """
    Load and clean an HTML document.
    
    Tasks:
    1. Remove <script> and <style> elements entirely
    2. Remove all HTML tags
    3. Decode HTML entities (&amp; -> &, &lt; -> <, etc.)
    4. Extract title from <title> or first <h1> for metadata
    5. Normalize whitespace
    
    Args:
        html_content: Raw HTML string
        source: Source filename
        
    Returns:
        Document with cleaned content and metadata
    """
    # TODO: Implement this function
    # Hints:
    # - Use re.sub() to remove script/style blocks
    # - Use re.search() to find title
    # - Use html.unescape() for entity decoding
    
    pass  # Remove this and add your implementation


def load_markdown(md_content: str, source: str = "unknown.md") -> Document:
    """
    Load and clean a Markdown document.
    
    Tasks:
    1. Remove code blocks (``` ... ```)
    2. Remove inline code backticks but keep content
    3. Remove formatting markers (*, _, #)
    4. Convert links [text](url) to just text
    5. Extract first header as title
    
    Args:
        md_content: Raw Markdown string
        source: Source filename
        
    Returns:
        Document with cleaned content and metadata
    """
    # TODO: Implement this function
    # Hints:
    # - Use re.sub(r'```[\s\S]*?```', '', text) for code blocks
    # - Use re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text) for links
    
    pass  # Remove this and add your implementation


def load_text(text_content: str, source: str = "unknown.txt") -> Document:
    """
    Load a plain text document.
    
    Tasks:
    1. Normalize whitespace
    2. Extract first line or section header as title
    3. Create standard metadata
    
    Args:
        text_content: Raw text string
        source: Source filename
        
    Returns:
        Document with content and metadata
    """
    # TODO: Implement this function
    # This one is simpler - mostly whitespace cleanup
    
    pass  # Remove this and add your implementation


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
        
        Args:
            content: Document content
            source: Source filename (used for format detection)
            
        Returns:
            Loaded Document
        """
        # TODO: Implement this method
        # 1. Extract extension from source
        # 2. Find appropriate loader
        # 3. Call loader with content and source
        # 4. Return the result
        
        pass  # Remove this and add your implementation


def chunk_document(doc: Document, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    """
    Chunk a document and prepare for vector store ingestion.
    
    Args:
        doc: Document to chunk
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        
    Returns:
        List of dicts with 'content' and 'metadata' keys
    """
    # TODO: Implement this function
    # 1. Split content into chunks
    # 2. For each chunk, create metadata including:
    #    - All original document metadata
    #    - chunk_index
    #    - total_chunks
    # 3. Return list of {content, metadata} dicts
    
    pass  # Remove this and add your implementation


def ingest_documents(documents: List[Document], collection) -> Dict:
    """
    Ingest documents into a Chroma collection.
    
    Args:
        documents: List of Document objects
        collection: Chroma collection
        
    Returns:
        Statistics dict with counts
    """
    # TODO: Implement this function
    # 1. Chunk each document
    # 2. Embed all chunks
    # 3. Add to collection with IDs and metadata
    # 4. Return stats
    
    pass  # Remove this and add your implementation


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_tests():
    """Test the document ingestion pipeline."""
    print("=" * 60)
    print("Exercise 01: Multi-Format Document Ingestion")
    print("=" * 60)
    
    loader = DocumentLoader()
    
    # Test documents
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
        
        try:
            doc = loader.load(content, source)
            
            if doc is None:
                print(f"  [ERROR] Loader returned None - not implemented yet")
                continue
            
            print(f"  Title: {doc.metadata.get('title', 'Unknown')}")
            print(f"  Length: {len(doc.content)} chars")
            print(f"  [OK] Loaded successfully")
            documents.append(doc)
            
        except Exception as e:
            print(f"  [ERROR] {e}")
        
        print()
    
    if not documents:
        print("[INFO] No documents loaded. Implement the loader functions first.")
        return
    
    # Test ingestion
    print("=== Ingesting to Vector Store ===")
    
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        client = chromadb.Client()
        
        try:
            client.delete_collection("exercise_01")
        except:
            pass
        
        collection = client.create_collection("exercise_01")
        
        stats = ingest_documents(documents, collection)
        
        if stats:
            print(f"  Ingested {stats.get('total_chunks', 0)} chunks from {stats.get('total_docs', 0)} documents")
        else:
            print("  [INFO] ingest_documents not implemented yet")
        
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    print("\n" + "=" * 60)
    print("[OK] Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
