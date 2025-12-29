# Exercise 01: Local to Cloud Deployment with Pinecone

## Overview

Build a **base class pattern** for vector stores, then implement both local (Chroma) and cloud (Pinecone) backends. This teaches DRY principles and production deployment patterns.

## Learning Objectives

- Design a base class with shared logic
- Implement abstract methods in subclasses
- Test locally, deploy to cloud
- Build environment-agnostic code

## Prerequisites

1. **Pinecone Account** - https://www.pinecone.io (free tier)
2. **Environment**: `export PINECONE_API_KEY="your-key"`
3. **Install**: `pip install pinecone-client chromadb sentence-transformers`

---

## Part 1: Implement VectorStoreBase (20 min)

The abstract methods are defined. Implement the **shared methods**:

### `__init__(self)`
```python
self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
self._initialize()
```

### `_embed(self, texts)`
```python
return self.embedder.encode(texts).tolist()
```

### `_generate_ids(self, count, prefix="doc")`
```python
return [f"{prefix}_{i}" for i in range(count)]
```

### `add_documents(self, texts, metadatas)`
```python
embeddings = self._embed(texts)
ids = self._generate_ids(len(texts))
self._store(ids, embeddings, texts, metadatas)
return len(texts)
```

### `search(self, query, n_results=5)`
```python
embedding = self._embed([query])[0]
return self._query(embedding, n_results)
```

---

## Part 2: Implement LocalVectorStore (15 min)

### `_initialize(self)`
- Create directory, PersistentClient, get/create collection

### `_store(self, ids, embeddings, texts, metadatas)`
- `collection.add(...)`

### `_query(self, embedding, n_results)`
- Query and return `[{text, metadata, score}]`

### `_get_all_data(self)`
- Export for migration

---

## Part 3: Implement CloudVectorStore (20 min)

### `_initialize(self)`
- Get API key, create client, create index if needed

### `_store(self, ids, embeddings, texts, metadatas)`
- Build vectors (include text in metadata), batch upsert

### `_query(self, embedding, n_results)`
- Query and return `[{text, metadata, score}]`

---

## Part 4: Test Migration (10 min)

Migration helpers are provided. Just run:
```bash
export PINECONE_API_KEY="your-key"
python exercise_01_starter.py
```

---

## Definition of Done

- [ ] `VectorStoreBase` shared methods work
- [ ] `LocalVectorStore` adds and searches
- [ ] `CloudVectorStore` connects and queries
- [ ] Migration successful

## Expected Output

```
=== Phase 1: Local ===
[OK] Added 8 docs
[OK] Search: Python is the most popular...

=== Phase 2: Cloud ===
[OK] Pinecone connected
[OK] Migrated 8 docs
[OK] Verified: PASS

[DONE]
```
