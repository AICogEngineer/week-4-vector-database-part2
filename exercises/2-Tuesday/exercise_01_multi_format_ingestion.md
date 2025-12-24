# Exercise 01: Multi-Format Document Ingestion

## Overview

Real-world RAG systems must handle multiple document formats. In this exercise, you'll build a unified ingestion pipeline that processes HTML pages, Markdown files, and plain text documents into a single vector store.

## Learning Objectives

- Extract clean text from HTML documents
- Process Markdown while preserving structure
- Create a unified document loader interface
- Store documents with format-specific metadata

## The Scenario

Your company has documentation in three formats:
1. **HTML**: Legacy web pages from the old documentation site
2. **Markdown**: New documentation written by developers
3. **Plain Text**: Log files and configuration guides

Your task: Build a pipeline that ingests all formats into one searchable system.

## Your Tasks

### Task 1: HTML Loader (20 min)

Implement `load_html()`:
- Remove script and style elements completely
- Strip all HTML tags
- Decode HTML entities
- Extract title from `<title>` or `<h1>` for metadata

### Task 2: Markdown Loader (20 min)

Implement `load_markdown()`:
- Remove code blocks (or optionally keep them)
- Convert headers to plain text
- Remove formatting markers (* _ ` #)
- Extract first header as title for metadata

### Task 3: Document Loader Class (20 min)

Implement `DocumentLoader`:
- Auto-detect format from file extension
- Route to appropriate loader function
- Add standard metadata (source, format, length, timestamp)
- Return consistent document structure

### Task 4: Batch Ingestion (15 min)

Implement `ingest_documents()`:
- Load multiple documents from a directory
- Chunk each document appropriately
- Store chunks with metadata in Chroma
- Return statistics on ingested documents

## Definition of Done

- [_] HTML loader extracts clean text
- [_] Markdown loader preserves structure info
- [_] DocumentLoader handles all three formats
- [_] Documents stored in Chroma with correct metadata
- [_] Can query across all document types

## Testing Your Solution

```bash
cd exercises/2-Tuesday/starter_code
python exercise_01_starter.py
```

Expected output:
```
=== Loading Sample Documents ===

[HTML] support.html
  Title: Customer Support
  Length: 342 chars
  [OK] Loaded successfully

[Markdown] guide.md
  Title: Getting Started
  Length: 567 chars
  [OK] Loaded successfully

[Text] config.txt
  Title: Configuration Guide
  Length: 234 chars
  [OK] Loaded successfully

=== Ingesting to Vector Store ===
  Ingested 12 chunks from 3 documents

=== Test Query ===
Query: "How do I configure the system?"
Results:
  1. [config.txt] "To configure the system..."
  2. [guide.md] "Configuration is done through..."

[OK] Multi-format ingestion complete!
```

## Stretch Goals (Optional)

1. Add PDF support using PyMuPDF
2. Extract images/tables as separate metadata
3. Implement format detection without extensions
