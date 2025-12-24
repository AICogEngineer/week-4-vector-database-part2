# Storing Metadata with Vectors

## Learning Objectives
- Understand the role of metadata in vector databases
- Design effective metadata schemas for enhanced retrieval
- Store and organize metadata alongside embeddings
- Use metadata to improve search relevance and filtering

## Why This Matters

Vector similarity finds semantically related content. But what if you want:
- Only results from a specific document?
- Content from the last month?
- Results filtered by author or category?

**Metadata** provides this capability. It transforms vector databases from pure semantic search into powerful hybrid systems that combine meaning with structured attributes.

## The Concept

### What is Metadata?

Metadata is structured information stored alongside vectors that describes the content.

```
VECTOR ALONE:
[0.23, -0.45, 0.67, ...] → "Some text about machine learning"

VECTOR + METADATA:
[0.23, -0.45, 0.67, ...] → "Some text about machine learning"
                          + source: "ml_guide.pdf"
                          + page: 42
                          + author: "Dr. Smith"
                          + date: "2024-01-15"
                          + category: "tutorial"
```

### Why Store Metadata?

1. **Filtering**: Narrow searches to specific subsets
2. **Context**: Provide additional information about results
3. **Citations**: Track sources for RAG responses
4. **Organization**: Manage large document collections
5. **Analytics**: Track usage patterns and content statistics

### Metadata Types

Different types of metadata serve different purposes:

| Type | Examples | Use Case |
|------|----------|----------|
| **Source tracking** | filename, URL, page number | Citations, provenance |
| **Temporal** | created_date, modified_date | Time-based filtering |
| **Categorical** | category, type, department | Faceted search |
| **Structural** | chunk_index, parent_id, section | Navigation, context |
| **Custom** | priority, confidence, language | Application-specific |

### Designing Metadata Schemas

A good metadata schema balances comprehensiveness with simplicity.

```python
# Minimal schema - source tracking only
minimal_metadata = {
    "source": "document.pdf",
    "chunk_id": "doc_chunk_0"
}

# Standard schema - common use cases
standard_metadata = {
    "source": "document.pdf",
    "source_type": "pdf",
    "chunk_index": 0,
    "total_chunks": 10,
    "created_at": "2024-01-15T10:30:00Z",
    "category": "documentation"
}

# Rich schema - comprehensive tracking
rich_metadata = {
    # Source tracking
    "source": "document.pdf",
    "source_type": "pdf",
    "source_url": "https://example.com/docs/document.pdf",
    
    # Chunk information
    "chunk_id": "doc_chunk_0",
    "chunk_index": 0,
    "total_chunks": 10,
    "parent_chunk_id": None,
    
    # Content information
    "content_type": "text",
    "section_heading": "Introduction",
    "page_number": 1,
    "word_count": 245,
    
    # Temporal
    "created_at": "2024-01-15T10:30:00Z",
    "indexed_at": "2024-01-20T08:00:00Z",
    
    # Categorical
    "category": "documentation",
    "subcategory": "api-reference",
    "language": "en",
    "audience": "developers",
    
    # Custom
    "version": "2.1.0",
    "confidence_score": 0.95,
    "reviewed": True
}
```

### Metadata Guidelines

**Do:**
- Keep keys consistent (use snake_case or camelCase, not both)
- Use appropriate data types (strings, numbers, booleans)
- Include source information for citations
- Add chunk position for context reconstruction
- Store timestamps in ISO 8601 format

**Don't:**
- Store large text in metadata (that's what documents are for)
- Use deeply nested structures (keep it flat)
- Include sensitive information
- Create too many unique values for filterable fields

## Code Example

Implementing metadata storage with Chroma:

```python
import chromadb
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class DocumentMetadata:
    """Structured metadata for documents."""
    source: str
    source_type: str
    chunk_index: int
    total_chunks: int
    created_at: str
    category: Optional[str] = None
    section_heading: Optional[str] = None
    page_number: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Remove None values (Chroma doesn't like them)
        return {k: v for k, v in data.items() if v is not None}


class MetadataAwareVectorStore:
    """Vector store with comprehensive metadata support."""
    
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Collection-level metadata
        )
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: DocumentMetadata
    ):
        """Add a document with metadata."""
        self.collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata.to_dict()]
        )
    
    def add_documents_batch(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[DocumentMetadata]
    ):
        """Add multiple documents with metadata."""
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[m.to_dict() for m in metadatas]
        )
    
    def query_with_metadata(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        include_metadata: bool = True
    ) -> dict:
        """Query and return results with metadata."""
        include = ["documents", "distances"]
        if include_metadata:
            include.append("metadatas")
        
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=include
        )
    
    def get_by_metadata(self, where: dict, limit: int = 10) -> dict:
        """Retrieve documents by metadata filters."""
        return self.collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"]
        )
    
    def update_metadata(self, doc_id: str, metadata_updates: dict):
        """Update metadata for an existing document."""
        # Get existing
        existing = self.collection.get(ids=[doc_id], include=["metadatas"])
        
        if not existing["ids"]:
            raise ValueError(f"Document {doc_id} not found")
        
        # Merge updates
        current_metadata = existing["metadatas"][0]
        updated_metadata = {**current_metadata, **metadata_updates}
        
        # Update
        self.collection.update(
            ids=[doc_id],
            metadatas=[updated_metadata]
        )


# Usage Example
store = MetadataAwareVectorStore("my_docs")

# Create metadata
metadata = DocumentMetadata(
    source="ml_guide.pdf",
    source_type="pdf",
    chunk_index=0,
    total_chunks=50,
    created_at=datetime.now().isoformat(),
    category="machine_learning",
    section_heading="Introduction to Neural Networks",
    page_number=5
)

# Add document (assuming you have an embedding)
embedding = [0.1, 0.2, 0.3]  # Placeholder
store.add_document(
    doc_id="ml_guide_chunk_0",
    text="Neural networks are computational models...",
    embedding=embedding,
    metadata=metadata
)

# Query with metadata retrieval
results = store.query_with_metadata(
    query_embedding=[0.1, 0.2, 0.35],
    n_results=3
)

# Access results with metadata
for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"Source: {meta['source']}, Page: {meta.get('page_number', 'N/A')}")
    print(f"Content: {doc[:100]}...")
    print(f"Distance: {dist}")
    print()
```

### Building a Document Pipeline with Metadata

```python
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer


class DocumentPipeline:
    """Complete pipeline for ingesting documents with metadata."""
    
    def __init__(self, collection_name: str = "documents"):
        self.store = MetadataAwareVectorStore(collection_name)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def ingest_file(
        self,
        file_path: str,
        category: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> int:
        """Ingest a file with automatic metadata extraction."""
        path = Path(file_path)
        
        # Read file
        content = path.read_text(encoding='utf-8', errors='replace')
        
        # Chunk content
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        
        # Prepare batch
        ids = []
        texts = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{path.stem}_chunk_{i}"
            
            metadata = DocumentMetadata(
                source=path.name,
                source_type=path.suffix.lstrip('.'),
                chunk_index=i,
                total_chunks=len(chunks),
                created_at=datetime.now().isoformat(),
                category=category
            )
            
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append(metadata)
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts).tolist()
        
        # Store
        self.store.add_documents_batch(ids, texts, embeddings, metadatas)
        
        return len(chunks)
    
    def ingest_directory(
        self,
        dir_path: str,
        category: str = None,
        extensions: list[str] = None
    ) -> dict:
        """Ingest all matching files from a directory."""
        extensions = extensions or ['.txt', '.md', '.py']
        path = Path(dir_path)
        
        results = {"files": 0, "chunks": 0, "errors": []}
        
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix in extensions:
                try:
                    chunks = self.ingest_file(str(file_path), category)
                    results["files"] += 1
                    results["chunks"] += chunks
                except Exception as e:
                    results["errors"].append(f"{file_path}: {e}")
        
        return results
    
    def _chunk_text(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> list[str]:
        """Simple chunking with overlap."""
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk = ' '.join(words[start:end])
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks


# Usage
pipeline = DocumentPipeline("knowledge_base")

# Ingest a single file
chunks = pipeline.ingest_file(
    "docs/ml_guide.md",
    category="machine_learning"
)
print(f"Ingested {chunks} chunks")

# Ingest a directory
results = pipeline.ingest_directory(
    "docs/",
    category="documentation",
    extensions=[".md", ".txt"]
)
print(f"Ingested {results['files']} files, {results['chunks']} total chunks")
```

## Key Takeaways

1. **Metadata enables hybrid search** - combine semantic similarity with structured filters
2. **Design schemas thoughtfully** - balance comprehensiveness with simplicity
3. **Always include source tracking** - essential for citations and debugging
4. **Keep metadata flat** - avoid nested structures
5. **Use consistent naming** - stick to one convention throughout
6. **Remove null values** - most vector databases don't handle them well

## Additional Resources

- [Chroma Metadata Documentation](https://docs.trychroma.com/usage-guide#adding-data-to-a-collection) - Official Chroma metadata guide
- [Pinecone Metadata Filtering](https://docs.pinecone.io/docs/metadata-filtering) - Comprehensive metadata filtering examples
- [Vector Database Schema Design](https://www.pinecone.io/learn/vector-database-schema/) - Best practices for schema design
