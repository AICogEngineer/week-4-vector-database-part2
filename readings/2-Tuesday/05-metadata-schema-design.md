# Metadata Schema Design

## Learning Objectives
- Design scalable metadata schemas for production vector databases
- Understand indexing considerations for metadata fields
- Plan for schema evolution and migration
- Implement best practices for multi-tenant and large-scale systems

## Why This Matters

A poorly designed metadata schema can cripple your RAG system as it scales. Issues include:
- Slow queries due to unindexed fields
- Storage bloat from redundant data
- Inability to add new fields without migrations
- Inconsistent data that breaks filters

Good schema design is invisible when it works - and painful when it doesn't. Learning these patterns now saves significant refactoring later.

## The Concept

### Schema Design Principles

#### 1. Start with Use Cases

Before designing fields, list your query patterns:

```
USE CASES                          REQUIRED FIELDS
----------                          ---------------
"Find docs from this source"     → source, source_type
"Get recent content"             → created_at, updated_at
"Filter by category"             → category, subcategory
"Show related chunks"            → chunk_index, parent_id
"Attribute sources"              → author, version
```

#### 2. Choose the Right Data Types

| Type | Best For | Example |
|------|----------|---------|
| **String** | Categories, identifiers | `"category": "tutorial"` |
| **Number** | Counts, scores, positions | `"page_number": 42` |
| **Boolean** | Flags, binary states | `"is_verified": true` |
| **ISO Datetime** | Temporal data (as string) | `"created_at": "2024-01-15T10:30:00Z"` |

**Avoid:**
- Arrays (not supported in most filter operations)
- Nested objects (keep schema flat)
- Null values (use empty strings or sentinels instead)

#### 3. Plan for Filtering vs. Display

```python
# FILTERING FIELDS: Keep limited, well-typed values
# These are indexed and used in WHERE clauses
filtering_metadata = {
    "category": "documentation",        # Enum-like string
    "language": "en",                   # ISO code
    "year": 2024,                       # Integer for range queries
    "is_public": True                   # Boolean
}

# DISPLAY FIELDS: Rich information for UI/context
# These are retrieved but not filtered on
display_metadata = {
    "title": "Introduction to Machine Learning",
    "author_name": "Dr. Jane Smith",
    "summary": "A comprehensive overview...",
    "source_url": "https://example.com/doc.pdf"
}

# COMBINED: Practical schema
full_metadata = {
    # Filterable
    "source": "ml_guide.pdf",
    "source_type": "pdf",
    "category": "tutorial",
    "language": "en",
    "year": 2024,
    
    # Display
    "title": "ML Guide Chapter 1",
    "author": "AI Team"
}
```

### Schema Patterns

#### Pattern 1: Basic Document Schema
```python
BASIC_SCHEMA = {
    # Identity
    "chunk_id": str,         # Unique identifier
    "source": str,           # Source file/URL
    "source_type": str,      # pdf, md, html, etc.
    
    # Position
    "chunk_index": int,      # Position in document
    "total_chunks": int,     # Total chunks in document
    
    # Temporal
    "created_at": str,       # ISO datetime
    
    # Content
    "category": str          # Optional categorization
}
```

#### Pattern 2: Multi-Document RAG Schema
```python
RAG_SCHEMA = {
    # Source tracking
    "source_id": str,        # Document unique ID
    "source_name": str,      # Human-readable name
    "source_type": str,      # pdf, webpage, api, etc.
    "source_url": str,       # Original URL if applicable
    
    # Chunk hierarchy
    "chunk_id": str,         # Unique chunk ID
    "chunk_index": int,      # Position in document
    "parent_chunk_id": str,  # For hierarchical chunking
    
    # Content classification
    "content_type": str,     # text, code, table, image_caption
    "section": str,          # Document section/chapter
    "heading": str,          # Nearest heading
    
    # Temporal
    "document_date": str,    # Document creation date
    "indexed_at": str,       # When added to index
    
    # Quality
    "confidence": float,     # Extraction confidence
    "word_count": int        # Chunk size metric
}
```

#### Pattern 3: Multi-Tenant Schema
```python
MULTI_TENANT_SCHEMA = {
    # Tenant isolation
    "tenant_id": str,        # Organization/workspace ID
    "user_id": str,          # Optional user-level isolation
    
    # Access control
    "visibility": str,       # public, private, team
    "team_ids": str,         # Comma-separated team IDs (workaround for no arrays)
    
    # Document info
    "source": str,
    "source_type": str,
    "chunk_index": int,
    
    # Temporal
    "created_at": str,
    "created_by": str        # User who added the document
}
```

#### Pattern 4: Versioned Content Schema
```python
VERSIONED_SCHEMA = {
    # Identity
    "document_id": str,      # Stable ID across versions
    "chunk_id": str,         # This specific chunk
    
    # Versioning
    "version": str,          # "1.0.0", "2.1.3"
    "version_major": int,    # For version range queries
    "is_latest": bool,       # Quick filter for current version
    "superseded_by": str,    # Next version's document_id
    
    # Source
    "source": str,
    "chunk_index": int,
    
    # Temporal
    "published_at": str,
    "deprecated_at": str     # Empty string if not deprecated
}
```

### Schema Evolution

Schemas change as requirements evolve. Plan for this:

```python
class SchemaVersionManager:
    """Manage metadata schema versions."""
    
    CURRENT_VERSION = 2
    
    MIGRATIONS = {
        1: {
            # Version 1 -> 2: Added 'confidence' field
            "added_fields": {"confidence": 1.0},
            "renamed_fields": {},
            "removed_fields": []
        }
    }
    
    @classmethod
    def migrate_metadata(cls, metadata: dict, from_version: int) -> dict:
        """Migrate metadata to current version."""
        if from_version >= cls.CURRENT_VERSION:
            return metadata
        
        migrated = metadata.copy()
        
        for v in range(from_version, cls.CURRENT_VERSION):
            migration = cls.MIGRATIONS.get(v, {})
            
            # Add new fields with defaults
            for field, default in migration.get("added_fields", {}).items():
                if field not in migrated:
                    migrated[field] = default
            
            # Rename fields
            for old_name, new_name in migration.get("renamed_fields", {}).items():
                if old_name in migrated:
                    migrated[new_name] = migrated.pop(old_name)
            
            # Remove deprecated fields
            for field in migration.get("removed_fields", []):
                migrated.pop(field, None)
        
        # Update version marker
        migrated["_schema_version"] = cls.CURRENT_VERSION
        
        return migrated


# Usage
old_metadata = {"source": "doc.pdf", "chunk_index": 0, "_schema_version": 1}
new_metadata = SchemaVersionManager.migrate_metadata(old_metadata, 1)
# Result: {"source": "doc.pdf", "chunk_index": 0, "confidence": 1.0, "_schema_version": 2}
```

### Validation

Validate metadata before storage:

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class ValidatedMetadata:
    """Metadata with validation."""
    
    # Required fields
    source: str
    source_type: str
    chunk_index: int
    
    # Optional fields with defaults
    category: str = "uncategorized"
    language: str = "en"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate fields after initialization."""
        # Validate source_type
        valid_types = ["pdf", "md", "html", "txt", "py", "json"]
        if self.source_type not in valid_types:
            raise ValueError(f"source_type must be one of {valid_types}")
        
        # Validate chunk_index
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        
        # Validate confidence
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        
        # Validate language code (basic check)
        if len(self.language) != 2:
            raise ValueError("language must be 2-letter ISO code")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "source": self.source,
            "source_type": self.source_type,
            "chunk_index": self.chunk_index,
            "category": self.category,
            "language": self.language,
            "created_at": self.created_at,
            "confidence": self.confidence
        }


# Usage
try:
    metadata = ValidatedMetadata(
        source="doc.pdf",
        source_type="pdf",
        chunk_index=0,
        category="tutorial"
    )
    print(metadata.to_dict())
except ValueError as e:
    print(f"Validation error: {e}")
```

## Code Example

Complete schema design and management system:

```python
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import json


class SourceType(Enum):
    PDF = "pdf"
    MARKDOWN = "md"
    HTML = "html"
    TEXT = "txt"
    CODE = "code"
    JSON = "json"


class ContentCategory(Enum):
    DOCUMENTATION = "documentation"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    GUIDE = "guide"
    FAQ = "faq"
    UNCATEGORIZED = "uncategorized"


@dataclass
class MetadataSchema:
    """
    Production-ready metadata schema with validation.
    
    This schema is designed for a multi-document RAG system with:
    - Source tracking for citations
    - Chunk positioning for context reconstruction
    - Temporal data for freshness queries
    - Categories for filtered search
    """
    
    # Required: Source identification
    source: str
    source_type: SourceType
    
    # Required: Chunk positioning
    chunk_id: str
    chunk_index: int
    total_chunks: int
    
    # Optional: Content classification
    category: ContentCategory = ContentCategory.UNCATEGORIZED
    section_heading: Optional[str] = None
    language: str = "en"
    
    # Optional: Temporal
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    document_date: Optional[str] = None
    
    # Optional: Quality metrics
    confidence: float = 1.0
    word_count: Optional[int] = None
    
    # Internal: Schema versioning
    _schema_version: int = 1
    
    def __post_init__(self):
        """Validate and normalize fields."""
        # Convert enums to strings if needed
        if isinstance(self.source_type, SourceType):
            self.source_type = self.source_type.value
        if isinstance(self.category, ContentCategory):
            self.category = self.category.value
        
        # Validation
        if not self.source:
            raise ValueError("source is required")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.total_chunks < 1:
            raise ValueError("total_chunks must be at least 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to storage-ready dictionary."""
        data = asdict(self)
        
        # Remove None values (Chroma doesn't like them)
        return {k: v for k, v in data.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetadataSchema':
        """Create from dictionary."""
        # Handle enum conversion
        if "source_type" in data:
            data["source_type"] = SourceType(data["source_type"])
        if "category" in data:
            data["category"] = ContentCategory(data["category"])
        
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        
        return cls(**filtered)


class SchemaManager:
    """Manage metadata schemas across a vector database."""
    
    def __init__(self, collection):
        self.collection = collection
        self._schema_version = 1
    
    def create_metadata(
        self,
        source: str,
        source_type: str,
        chunk_id: str,
        chunk_index: int,
        total_chunks: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Create validated metadata."""
        schema = MetadataSchema(
            source=source,
            source_type=SourceType(source_type),
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            **kwargs
        )
        return schema.to_dict()
    
    def add_with_metadata(
        self,
        chunk_id: str,
        document: str,
        embedding: List[float],
        source: str,
        source_type: str,
        chunk_index: int,
        total_chunks: int,
        **extra_metadata
    ):
        """Add document with validated metadata."""
        metadata = self.create_metadata(
            source=source,
            source_type=source_type,
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            word_count=len(document.split()),
            **extra_metadata
        )
        
        self.collection.add(
            ids=[chunk_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    
    def get_schema_stats(self) -> Dict[str, Any]:
        """Get statistics about metadata in the collection."""
        all_docs = self.collection.get(include=["metadatas"])
        
        if not all_docs["metadatas"]:
            return {"total_documents": 0}
        
        stats = {
            "total_documents": len(all_docs["metadatas"]),
            "sources": set(),
            "source_types": set(),
            "categories": set(),
            "schema_versions": set()
        }
        
        for meta in all_docs["metadatas"]:
            stats["sources"].add(meta.get("source", "unknown"))
            stats["source_types"].add(meta.get("source_type", "unknown"))
            stats["categories"].add(meta.get("category", "unknown"))
            stats["schema_versions"].add(meta.get("_schema_version", 0))
        
        # Convert sets to lists for JSON serialization
        for key in ["sources", "source_types", "categories", "schema_versions"]:
            stats[key] = list(stats[key])
        
        return stats


# Example usage
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")
manager = SchemaManager(collection)

# Add document with validated metadata
manager.add_with_metadata(
    chunk_id="doc1_chunk_0",
    document="Machine learning is a subset of AI...",
    embedding=[0.1, 0.2, 0.3],
    source="ml_guide.pdf",
    source_type="pdf",
    chunk_index=0,
    total_chunks=10,
    category=ContentCategory.TUTORIAL,
    section_heading="Introduction"
)

# Get schema statistics
stats = manager.get_schema_stats()
print(json.dumps(stats, indent=2))
```

## Key Takeaways

1. **Design for your query patterns** - schema should serve your use cases
2. **Keep schemas flat** - avoid nesting and arrays
3. **Use appropriate types** - strings for categories, numbers for ranges
4. **Plan for evolution** - include version markers and migration strategies
5. **Validate early** - catch errors before they reach the database
6. **Separate filtering from display fields** - not all metadata needs indexing

## Additional Resources

- [Chroma Data Types](https://docs.trychroma.com/usage-guide#adding-data-to-a-collection) - Supported metadata types
- [Database Schema Design Patterns](https://www.mongodb.com/docs/manual/core/data-modeling-introduction/) - General schema best practices
- [Event Sourcing for Schema Evolution](https://martinfowler.com/eaaDev/EventSourcing.html) - Advanced evolution strategies
