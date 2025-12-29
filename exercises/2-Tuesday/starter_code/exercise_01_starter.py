"""
Exercise 01: Multi-Format Document Ingestion with LangChain - Starter Code

Build a unified document ingestion pipeline using LangChain text splitters.

Instructions:
1. Implement each TODO function using the appropriate LangChain splitter
2. Run this file to test your implementations
3. Check the expected output in the exercise guide

Prerequisites:
- pip install langchain-text-splitters chromadb sentence-transformers

References:
- https://docs.langchain.com/oss/python/integrations/splitters/split_html
- https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter
"""

from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

# LangChain splitters - these are your tools!
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
)

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
# TODO: IMPLEMENT THESE FUNCTIONS USING LANGCHAIN SPLITTERS
# ============================================================================

def split_html(html_content: str, source: str = "unknown.html") -> List[Document]:
    """
    Split HTML using LangChain HTMLHeaderTextSplitter.
    
    Tasks:
    1. Define headers_to_split_on for h1, h2, h3
    2. Create HTMLHeaderTextSplitter with those headers
    3. Call split_text() on the HTML content
    4. Convert LangChain Documents to our Document format
    5. Add source, chunk_index, total_chunks to metadata
    
    Example from docs:
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
        ]
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
        docs = html_splitter.split_text(html_string)
    
    Returns:
        List of Document objects with header hierarchy in metadata
    """
    # TODO: Implement using HTMLHeaderTextSplitter
    # Hint: Each LangChain doc has .page_content and .metadata
    
    pass  # Remove this and add your implementation


def split_markdown(md_content: str, source: str = "unknown.md") -> List[Document]:
    """
    Split Markdown using LangChain MarkdownHeaderTextSplitter.
    
    Tasks:
    1. Define headers_to_split_on for #, ##, ###
    2. Create MarkdownHeaderTextSplitter with strip_headers=False
    3. Call split_text() on the Markdown content
    4. Convert to our Document format with proper metadata
    
    Example from docs:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        docs = md_splitter.split_text(markdown_document)
    
    Returns:
        List of Document objects with header hierarchy in metadata
    """
    # TODO: Implement using MarkdownHeaderTextSplitter
    
    pass  # Remove this and add your implementation


def split_text(text_content: str, source: str = "unknown.txt") -> List[Document]:
    """
    Split plain text using LangChain RecursiveCharacterTextSplitter.
    
    Tasks:
    1. Create RecursiveCharacterTextSplitter with:
       - chunk_size=400
       - chunk_overlap=50
       - separators=["\n\n", "\n", ". ", " "]
    2. Call split_text() on the content
    3. Convert to our Document format
    
    Example from docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
    
    Returns:
        List of Document objects
    """
    # TODO: Implement using RecursiveCharacterTextSplitter
    
    pass  # Remove this and add your implementation


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
        
        TODO:
        1. Extract extension from source
        2. Find appropriate splitter
        3. Call splitter with content and source
        4. Return the result
        
        Returns:
            List of Document chunks with metadata
        """
        # TODO: Implement this method
        
        pass  # Remove this and add your implementation


def ingest_documents(doc_chunks: List[Document], collection) -> Dict:
    """
    Ingest document chunks into a Chroma collection.
    
    TODO:
    1. Extract content and metadata from all chunks
    2. Generate embeddings using SentenceTransformer
    3. Add to collection with unique IDs
    4. Return stats
    """
    # TODO: Implement this function
    
    pass  # Remove this and add your implementation


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_tests():
    """Test the LangChain-based document ingestion pipeline."""
    print("=" * 60)
    print("Exercise 01: Multi-Format Ingestion with LangChain")
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
        
        try:
            chunks = loader.load(content, source)
            
            if chunks is None:
                print(f"  [ERROR] Loader returned None - not implemented yet")
                continue
            
            print(f"  Chunks created: {len(chunks)}")
            
            # Show metadata from first chunk
            if chunks:
                first_meta = chunks[0].metadata
                if 'Header 1' in first_meta:
                    print(f"  Header 1: {first_meta.get('Header 1', 'N/A')}")
                print(f"  First chunk preview: {chunks[0].content[:50]}...")
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"  [ERROR] {e}")
        
        print()
    
    if not all_chunks:
        print("[INFO] No documents loaded. Implement the splitter functions first.")
        return
    
    # Test ingestion
    print("=== Ingesting to Vector Store ===")
    
    try:
        client = chromadb.Client()
        
        try:
            client.delete_collection("exercise_01_langchain")
        except:
            pass
        
        collection = client.create_collection("exercise_01_langchain")
        
        stats = ingest_documents(all_chunks, collection)
        
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
