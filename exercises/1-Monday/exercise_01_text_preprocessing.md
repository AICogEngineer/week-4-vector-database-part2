# Exercise 01: Text Preprocessing Pipeline

## Overview

You've joined a team building a RAG system for customer support. The document corpus includes HTML pages, markdown docs, and plain text files. Unfortunately, the data is messy - inconsistent encoding, HTML artifacts, excessive whitespace, and more.

Your task: Build a preprocessing pipeline that cleans this data before embedding.

## Learning Objectives

- Implement text normalization (encoding, case, whitespace)
- Remove HTML tags and entities
- Handle special characters and unicode
- Chain preprocessing steps into a pipeline

## The Scenario

The support team has provided you with sample documents that exhibit these problems:

1. **Encoding issues**: Mixed UTF-8 and Latin-1, special quote characters
2. **HTML remnants**: Tags, entities like `&amp;` and `&nbsp;`
3. **Formatting noise**: Excessive newlines, tabs, inconsistent spacing
4. **Special characters**: Non-breaking spaces, zero-width characters

## Your Tasks

### Task 1: Encoding Normalization (15 min)

Implement `normalize_encoding()` in the starter code:
- Convert to UTF-8
- Replace smart quotes with standard quotes
- Handle common problematic characters

> **Hint**: The `unicodedata` module has functions for normalization.

### Task 2: HTML Cleaning (20 min)

Implement `remove_html()`:
- Remove all HTML tags
- Decode HTML entities (&amp; -> &)
- Remove script and style content entirely

> **Hint**: Be careful about nested tags. Consider using `html.unescape()`.

### Task 3: Whitespace Normalization (15 min)

Implement `normalize_whitespace()`:
- Collapse multiple spaces into one
- Normalize all newlines (no more than 2 consecutive)
- Strip leading/trailing whitespace
- Handle tabs appropriately

### Task 4: Complete Pipeline (20 min)

Combine your functions into `preprocess_document()`:
- Chain all cleaning steps
- Return cleaned text ready for embedding
- Test on the provided sample documents

## Definition of Done

- [_] All four functions implemented and passing tests
- [_] Sample documents produce clean output
- [_] Pipeline handles edge cases gracefully
- [_] Console output shows before/after comparison

## Testing Your Solution

```bash
cd exercises/1-Monday/starter_code
python exercise_01_starter.py
```

Expected output format:
```
=== Document 1: HTML Page ===
BEFORE (first 100 chars): <html><body>Our support <b>team</b> is here...
AFTER: Our support team is here to help...

[OK] Preprocessing complete for 3 documents
```

## Stretch Goals (Optional)

If you finish early:
1. Add language detection (skip non-English docs)
2. Implement sentence boundary detection
3. Add logging to track what was removed
