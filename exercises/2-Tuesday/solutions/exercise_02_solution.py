"""
Exercise 02: Metadata Search System - Solution

Complete implementation of the metadata search system.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================================
# SAMPLE DATA
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
# SOLUTION IMPLEMENTATIONS
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
    """
    if created_date is None:
        created_date = datetime.now().strftime("%Y-%m-%d")
    
    return {
        "source": source,
        "category": category,
        "version": version,
        "language": language,
        "author": author,
        "created_date": created_date,
        "word_count": 0  # Placeholder, would be set when content is known
    }


class FilteredSearch:
    """
    A search class that supports metadata filtering with a fluent interface.
    """
    
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder
        self._filters = []
    
    def by_category(self, category: str) -> 'FilteredSearch':
        """Add a category filter."""
        self._filters.append({"category": category})
        return self
    
    def by_version(self, version: str) -> 'FilteredSearch':
        """Add a version filter."""
        self._filters.append({"version": version})
        return self
    
    def by_language(self, language: str) -> 'FilteredSearch':
        """Add a language filter."""
        self._filters.append({"language": language})
        return self
    
    def by_author(self, author: str) -> 'FilteredSearch':
        """Add an author filter."""
        self._filters.append({"author": author})
        return self
    
    def by_categories(self, categories: List[str]) -> 'FilteredSearch':
        """Add a filter for multiple categories (OR)."""
        self._filters.append({"category": {"$in": categories}})
        return self
    
    def by_date_after(self, date: str) -> 'FilteredSearch':
        """Add a filter for documents after a date."""
        self._filters.append({"created_date": {"$gt": date}})
        return self
    
    def by_date_before(self, date: str) -> 'FilteredSearch':
        """Add a filter for documents before a date."""
        self._filters.append({"created_date": {"$lt": date}})
        return self
    
    def _build_where(self) -> Optional[Dict]:
        """Build the Chroma where clause from accumulated filters."""
        if not self._filters:
            return None
        
        if len(self._filters) == 1:
            return self._filters[0]
        
        # Multiple filters - combine with $and
        return {"$and": self._filters}
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Execute the search with accumulated filters."""
        # Embed the query
        query_embedding = self.embedder.encode([query]).tolist()
        
        # Build where clause
        where = self._build_where()
        
        # Query collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Reset filters for next query
        self._filters = []
        
        return results
    
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
    
    # Map of supported filters
    FILTER_MAP = {
        "category": "by_category",
        "version": "by_version",
        "language": "by_language",
        "author": "by_author",
    }
    
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder
        self.searcher = FilteredSearch(collection, embedder)
    
    def parse_query(self, user_input: str) -> tuple:
        """
        Parse user input into query text and filters.
        """
        filters = {}
        
        # Find all key:value patterns
        pattern = r'(\w+):(\S+)'
        matches = re.findall(pattern, user_input)
        
        for key, value in matches:
            if key in self.FILTER_MAP:
                filters[key] = value
        
        # Remove filters from query text
        query_text = re.sub(pattern, '', user_input).strip()
        # Clean up extra whitespace
        query_text = re.sub(r'\s+', ' ', query_text)
        
        return query_text, filters
    
    def search(self, user_input: str, n_results: int = 5) -> Dict:
        """Search with parsed user input."""
        query_text, filters = self.parse_query(user_input)
        
        # Reset searcher
        self.searcher.reset()
        
        # Apply filters
        for key, value in filters.items():
            method_name = self.FILTER_MAP.get(key)
            if method_name and hasattr(self.searcher, method_name):
                getattr(self.searcher, method_name)(value)
        
        # Execute search
        return self.searcher.search(query_text, n_results)


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_tests():
    """Test the metadata search system."""
    print("=" * 60)
    print("Exercise 02: Metadata Search System - SOLUTION")
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
    results = search.search("installation")
    print(f"  Results: {len(results['documents'][0])} matches")
    for doc, meta in zip(results['documents'][0][:2], results['metadatas'][0][:2]):
        print(f"    [{meta['category']}] {doc[:50]}...")
    
    # Test 2: Category filter
    print('\nQuery: "installation" category=tutorial')
    results = search.by_category("tutorial").search("installation")
    print(f"  Results: {len(results['documents'][0])} matches")
    for doc, meta in zip(results['documents'][0][:2], results['metadatas'][0][:2]):
        print(f"    [{meta['category']}] {doc[:50]}...")
    
    # Test 3: Version filter
    print('\nQuery: "installation" version=2.0')
    results = search.by_version("2.0").search("installation")
    print(f"  Results: {len(results['documents'][0])} matches")
    
    # Test 4: Combined filters
    print('\nQuery: "error" category=troubleshooting AND version=2.0')
    results = search.by_category("troubleshooting").by_version("2.0").search("error")
    print(f"  Results: {len(results['documents'][0])} matches")
    
    # Test 5: Multiple categories
    print('\nQuery: "configuration" category IN [tutorial, reference]')
    results = search.by_categories(["tutorial", "reference"]).search("configuration")
    print(f"  Results: {len(results['documents'][0])} matches")
    
    # Test SearchInterface
    print("\n=== User Interface Test ===\n")
    
    interface = SearchInterface(collection, embedder)
    
    user_input = "how to install version:2.0 category:tutorial"
    print(f'Input: "{user_input}"')
    
    query, filters = interface.parse_query(user_input)
    print(f"  Query text: \"{query}\"")
    print(f"  Filters: {filters}")
    
    results = interface.search(user_input)
    print("  Results:")
    for i, (doc, meta) in enumerate(zip(results['documents'][0][:2], results['metadatas'][0][:2]), 1):
        print(f"    {i}. [v{meta['version']}, {meta['category']}] {doc[:50]}...")
    
    print("\n" + "=" * 60)
    print("[OK] Metadata search system complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
