# Exercise 02: Metadata Search System

## Overview

Vector similarity alone often isn't enough for precise retrieval. In this exercise, you'll build a search system that combines semantic similarity with metadata filtering for production-ready RAG.

## Learning Objectives

- Design a metadata schema for document retrieval
- Implement filtered queries using Chroma's where clause
- Combine semantic search with structured filters
- Build a query interface that supports multiple filter types

## The Scenario

You're building a knowledge base for a software product with:
- Multiple product versions (v1.0, v2.0, v3.0)
- Different document categories (tutorial, reference, troubleshooting)
- Various authors and update dates
- Different languages (English, Spanish)

Users need to search within specific versions, categories, or time ranges.

## Your Tasks

### Task 1: Schema Design (15 min)

Design a metadata schema in `create_metadata()`:
- document_id: Unique identifier
- source: Original filename
- category: One of [tutorial, reference, troubleshooting, changelog]
- version: Product version (e.g., "2.0")
- language: Document language
- author: Author name
- created_date: ISO date string
- word_count: Number of words

### Task 2: Filtered Query Builder (25 min)

Implement `FilteredSearch`:
- `by_category(category)`: Filter by single category
- `by_version(version)`: Filter by version
- `by_date_range(start, end)`: Filter by date range
- `search(query, n_results)`: Execute the combined search

### Task 3: Complex Queries (25 min)

Implement these query patterns:
- AND queries: `category = 'tutorial' AND version = '2.0'`
- OR queries: `category IN ['tutorial', 'reference']`
- Comparison: `created_date > '2024-01-01'`

### Task 4: Search Interface (25 min)

Build `SearchInterface` class:
- Parse user queries with optional filters
- Support syntax like: `"error handling" category:troubleshooting version:2.0`
- Return formatted results with metadata

## Definition of Done

- [_] Metadata schema implemented correctly
- [_] FilteredSearch supports all filter types
- [_] Complex queries (AND/OR) working
- [_] Search interface parses user input
- [_] Results show both content and metadata

## Testing Your Solution

```bash
cd exercises/2-Tuesday/starter_code
python exercise_02_starter.py
```

Expected output:
```
=== Metadata Search System ===

[INFO] Populated 20 sample documents

=== Basic Filter Tests ===

Query: "installation" (no filter)
  Results: 5 matches

Query: "installation" category=tutorial
  Results: 2 matches (both tutorials)

Query: "installation" version=2.0
  Results: 3 matches (all v2.0)

=== Complex Filter Tests ===

Query: "configuration" category IN [tutorial, reference]
  Results: 4 matches

Query: "error" version=2.0 AND category=troubleshooting
  Results: 2 matches

=== User Interface Test ===

Input: "how to install version:2.0 category:tutorial"
Parsed:
  - Query text: "how to install"
  - Filters: version=2.0, category=tutorial
Results:
  1. [v2.0, tutorial] "Installing the software requires..."

[OK] Metadata search system complete!
```

## Stretch Goals (Optional)

1. Add full-text search on metadata fields
2. Implement faceted search (show filter counts)
3. Add query auto-complete based on metadata values
