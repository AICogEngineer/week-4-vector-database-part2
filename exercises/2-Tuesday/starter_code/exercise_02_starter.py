"""
Exercise 02: Metadata Search System - Starter Code

Build a search system combining semantic similarity with metadata filtering.

Instructions:
1. Implement each TODO function
2. Run this file to test your implementations
3. Check the expected output in the exercise guide
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================================
# SAMPLE DATA (DO NOT MODIFY)
# ============================================================================

SAMPLE_DOCUMENTS = [
    {
        "content": "Installing the software on Windows requires downloading the installer and running setup.exe. Follow the prompts to complete installation.",
        "metadata": {"category": "tutorial", "version": "2.0", "language": "en", "author": "Alice", "created_date": "2024-01-15"}
    },
    {
        "content": "Installing the software on Linux involves using apt-get or downloading the tarball. Make sure to set execute permissions.",
        "metadata": {"category": "tutorial", "version": "2.0", "language": "en", "author": "Bob", "created_date": "2024-02-20"}
    },
    {
        "content": "Installation troubleshooting: If you see error code 1001, check your system permissions and try running as administrator.",
        "metadata": {"category": "troubleshooting", "version": "2.0", "language": "en", "author": "Alice", "created_date": "2024-03-10"}
    },
    {
        "content": "The API reference provides detailed documentation of all available endpoints and their parameters.",
        "metadata": {"category": "reference", "version": "2.0", "language": "en", "author": "Charlie", "created_date": "2024-01-20"}
    },
    {
        "content": "Configuration tutorial: Learn how to set up your development environment with the correct settings.",
        "metadata": {"category": "tutorial", "version": "1.0", "language": "en", "author": "Alice", "created_date": "2023-06-15"}
    },
    {
        "content": "Error handling best practices: Always wrap API calls in try-catch blocks and log errors appropriately.",
        "metadata": {"category": "tutorial", "version": "2.0", "language": "en", "author": "Bob", "created_date": "2024-04-01"}
    },
    {
        "content": "Troubleshooting connection errors: Verify your network settings and check firewall configurations.",
        "metadata": {"category": "troubleshooting", "version": "1.0", "language": "en", "author": "Charlie", "created_date": "2023-08-20"}
    },
    {
        "content": "Version 3.0 changelog: New features include improved performance, better error messages, and async support.",
        "metadata": {"category": "changelog", "version": "3.0", "language": "en", "author": "Alice", "created_date": "2024-06-01"}
    },
    {
        "content": "Guia de instalacion en espanol: Descarga el instalador y sigue las instrucciones en pantalla.",
        "metadata": {"category": "tutorial", "version": "2.0", "language": "es", "author": "Maria", "created_date": "2024-03-15"}
    },
    {
        "content": "Performance optimization reference: Use batch processing for large datasets to improve throughput.",
        "metadata": {"category": "reference", "version": "2.0", "language": "en", "author": "Bob", "created_date": "2024-05-10"}
    },
]


# ============================================================================
# TODO: IMPLEMENT THESE FUNCTIONS
# ============================================================================

def create_metadata(
    source: str,
    category: str,
    version: str,
    language: str = "en",
    author: str = "unknown",
    created_date: str = None
) -> Dict:
    """
    Create a standardized metadata dictionary.
    
    Args:
        source: Document source/filename
        category: One of [tutorial, reference, troubleshooting, changelog]
        version: Product version (e.g., "2.0")
        language: Language code (e.g., "en", "es")
        author: Author name
        created_date: ISO date string (defaults to today)
        
    Returns:
        Metadata dictionary
    """
    # TODO: Implement this function
    # 1. Create dictionary with all fields
    # 2. Add word_count (placeholder for now)
    # 3. Default created_date to today if not provided
    
    pass  # Remove this and add your implementation


class FilteredSearch:
    """
    A search class that supports metadata filtering.
    
    Usage:
        search = FilteredSearch(collection, embedder)
        results = (search
            .by_category("tutorial")
            .by_version("2.0")
            .search("how to install"))
    """
    
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder
        self._filters = []
    
    def by_category(self, category: str) -> 'FilteredSearch':
        """Add a category filter."""
        # TODO: Implement this method
        # Add category equality filter to self._filters
        
        pass  # Remove this and add your implementation
        return self
    
    def by_version(self, version: str) -> 'FilteredSearch':
        """Add a version filter."""
        # TODO: Implement this method
        
        pass  # Remove this and add your implementation
        return self
    
    def by_language(self, language: str) -> 'FilteredSearch':
        """Add a language filter."""
        # TODO: Implement this method
        
        pass  # Remove this and add your implementation
        return self
    
    def by_categories(self, categories: List[str]) -> 'FilteredSearch':
        """Add a filter for multiple categories (OR)."""
        # TODO: Implement this method
        # Use $in operator for Chroma
        
        pass  # Remove this and add your implementation
        return self
    
    def by_date_after(self, date: str) -> 'FilteredSearch':
        """Add a filter for documents after a date."""
        # TODO: Implement this method
        # Use $gt operator for Chroma
        
        pass  # Remove this and add your implementation
        return self
    
    def _build_where(self) -> Optional[Dict]:
        """Build the Chroma where clause from accumulated filters."""
        # TODO: Implement this method
        # Combine all filters with $and if multiple
        
        pass  # Remove this and add your implementation
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Execute the search with accumulated filters."""
        # TODO: Implement this method
        # 1. Embed the query
        # 2. Build where clause
        # 3. Query collection
        # 4. Reset filters for next query
        # 5. Return results
        
        pass  # Remove this and add your implementation
    
    def reset(self) -> 'FilteredSearch':
        """Reset filters for a new query."""
        self._filters = []
        return self


class SearchInterface:
    """
    User-friendly search interface that parses filter syntax.
    
    Supports queries like:
        "how to install version:2.0 category:tutorial"
    """
    
    def __init__(self, collection, embedder):
        self.searcher = FilteredSearch(collection, embedder)
    
    def parse_query(self, user_input: str) -> tuple:
        """
        Parse user input into query text and filters.
        
        Args:
            user_input: Query string like "search text key:value key2:value2"
            
        Returns:
            (query_text, filters_dict)
        """
        # TODO: Implement this method
        # 1. Find all key:value patterns
        # 2. Extract them as filters
        # 3. Return remaining text as query
        
        pass  # Remove this and add your implementation
    
    def search(self, user_input: str, n_results: int = 5) -> Dict:
        """Search with parsed user input."""
        # TODO: Implement this method
        # 1. Parse input
        # 2. Apply filters
        # 3. Execute search
        
        pass  # Remove this and add your implementation


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_tests():
    """Test the metadata search system."""
    print("=" * 60)
    print("Exercise 02: Metadata Search System")
    print("=" * 60)
    
    print("\n[INFO] Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("[INFO] Setting up vector store...")
    client = chromadb.Client()
    
    try:
        client.delete_collection("exercise_02")
    except:
        pass
    
    collection = client.create_collection("exercise_02")
    
    # Populate with sample data
    print("[INFO] Populating sample documents...")
    
    embeddings = embedder.encode([d["content"] for d in SAMPLE_DOCUMENTS]).tolist()
    
    collection.add(
        ids=[f"doc_{i}" for i in range(len(SAMPLE_DOCUMENTS))],
        documents=[d["content"] for d in SAMPLE_DOCUMENTS],
        embeddings=embeddings,
        metadatas=[d["metadata"] for d in SAMPLE_DOCUMENTS]
    )
    
    print(f"[INFO] Populated {len(SAMPLE_DOCUMENTS)} sample documents\n")
    
    # Test FilteredSearch
    print("=== Basic Filter Tests ===\n")
    
    search = FilteredSearch(collection, embedder)
    
    # Test 1: No filter
    print('Query: "installation" (no filter)')
    try:
        results = search.search("installation")
        if results and results.get("documents"):
            print(f"  Results: {len(results['documents'][0])} matches")
        else:
            print("  [INFO] search() not implemented yet")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test 2: Category filter
    print('\nQuery: "installation" category=tutorial')
    try:
        results = search.by_category("tutorial").search("installation")
        if results and results.get("documents"):
            print(f"  Results: {len(results['documents'][0])} matches")
        else:
            print("  [INFO] by_category() not implemented yet")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test 3: Version filter
    print('\nQuery: "installation" version=2.0')
    try:
        results = search.by_version("2.0").search("installation")
        if results and results.get("documents"):
            print(f"  Results: {len(results['documents'][0])} matches")
        else:
            print("  [INFO] by_version() not implemented yet")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test SearchInterface
    print("\n=== User Interface Test ===\n")
    
    interface = SearchInterface(collection, embedder)
    
    print('Input: "how to install version:2.0 category:tutorial"')
    try:
        query, filters = interface.parse_query("how to install version:2.0 category:tutorial")
        if query is not None:
            print(f"  Query text: \"{query}\"")
            print(f"  Filters: {filters}")
        else:
            print("  [INFO] parse_query() not implemented yet")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    print("\n" + "=" * 60)
    print("[OK] Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
