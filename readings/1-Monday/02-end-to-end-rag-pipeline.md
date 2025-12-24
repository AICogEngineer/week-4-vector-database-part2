# End-to-End RAG Pipeline

## Learning Objectives
- Understand the complete architecture of a production RAG pipeline
- Learn the role of each component from ingestion to response generation
- Build mental models for data flow through RAG systems
- Identify integration points and potential bottlenecks

## Why This Matters

In Week 3, you were introduced to the concept of RAG (Retrieval Augmented Generation) and its basic architecture. Now we move from understanding *what* RAG is to building *complete* RAG systems.

A production RAG pipeline isn't just "embed documents and query them." It's an orchestrated system with multiple stages, each requiring careful design. Companies like Notion, Replit, and countless startups are building RAG pipelines to add AI capabilities to their products. Understanding the end-to-end flow is essential for building systems that work reliably.

## The Concept

### RAG Pipeline Overview

A complete RAG pipeline consists of two major phases:

1. **Ingestion Phase** (Offline): Prepare and store documents
2. **Query Phase** (Online): Retrieve and generate answers

```
INGESTION PHASE (Offline)
==========================
Documents → Load → Clean → Chunk → Embed → Store
                                              ↓
                                        Vector Database

QUERY PHASE (Online)
====================
User Query → Embed → Search → Retrieve → Context → LLM → Response
                        ↑
                  Vector Database
```

### Component 1: Document Loading

The first step is getting your documents into the pipeline.

**Common Document Sources**:
- Files: PDF, Word, Markdown, Text, HTML
- Databases: PostgreSQL, MongoDB exports
- APIs: Notion, Confluence, Google Drive
- Web: Crawled pages, scraped content

```python
from pathlib import Path

def load_documents(source_path: str) -> list[dict]:
    """Load documents from various file types."""
    documents = []
    path = Path(source_path)
    
    for file in path.glob("**/*"):
        if file.suffix == ".txt":
            content = file.read_text(encoding="utf-8")
        elif file.suffix == ".md":
            content = file.read_text(encoding="utf-8")
        elif file.suffix == ".pdf":
            content = extract_pdf_text(file)  # Custom function
        else:
            continue
        
        documents.append({
            "content": content,
            "source": str(file),
            "type": file.suffix
        })
    
    return documents
```

### Component 2: Text Preprocessing

Raw documents need cleaning before embedding.

**Preprocessing Tasks**:
- Remove excessive whitespace
- Handle special characters and encodings
- Strip headers/footers/boilerplate
- Normalize formatting

```python
import re

def preprocess_text(text: str) -> str:
    """Clean and normalize text for embedding."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that don't add meaning
    text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    
    return text.strip()
```

### Component 3: Chunking

Breaking documents into appropriately sized pieces for embedding.

**Why Chunking Matters**:
- Embedding models have max input lengths (typically 512 tokens)
- Smaller chunks = more precise retrieval
- Larger chunks = more context per retrieval
- Balance is key

```python
def chunk_document(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split document into overlapping chunks."""
    words = text.split()
    chunks = []
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # Overlap for context continuity
    
    return chunks
```

### Component 4: Embedding Generation

Convert text chunks into vector representations.

**Embedding Options**:
- **Local Models**: sentence-transformers, all-MiniLM-L6-v2
- **API Services**: OpenAI embeddings, Cohere embed
- **Chroma Default**: Built-in embedding function

```python
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.model.encode([text])[0].tolist()
```

### Component 5: Vector Storage

Store embeddings with their metadata for later retrieval.

```python
import chromadb

def store_documents(
    collection,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata: list[dict]
):
    """Store document chunks in the vector database."""
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata
    )
```

### Component 6: Query Processing

When a user asks a question, process and embed the query.

```python
def process_query(query: str, embedding_service) -> dict:
    """Process user query for retrieval."""
    # Clean the query
    cleaned_query = preprocess_text(query)
    
    # Generate embedding
    query_embedding = embedding_service.embed_single(cleaned_query)
    
    return {
        "original": query,
        "cleaned": cleaned_query,
        "embedding": query_embedding
    }
```

### Component 7: Retrieval

Search the vector database for relevant chunks.

```python
def retrieve_context(
    collection,
    query_embedding: list[float],
    n_results: int = 5
) -> list[dict]:
    """Retrieve relevant context from vector database."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    context_chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        context_chunks.append({
            "content": doc,
            "metadata": meta,
            "relevance_score": 1 / (1 + dist)
        })
    
    return context_chunks
```

### Component 8: Context Assembly

Combine retrieved chunks into a coherent context for the LLM.

```python
def assemble_context(
    chunks: list[dict],
    max_tokens: int = 3000
) -> str:
    """Assemble retrieved chunks into context string."""
    context_parts = []
    current_tokens = 0
    
    # Sort by relevance
    sorted_chunks = sorted(chunks, key=lambda x: x["relevance_score"], reverse=True)
    
    for chunk in sorted_chunks:
        chunk_tokens = len(chunk["content"].split()) * 1.3
        if current_tokens + chunk_tokens <= max_tokens:
            source = chunk["metadata"].get("source", "Unknown")
            context_parts.append(f"[Source: {source}]\n{chunk['content']}")
            current_tokens += chunk_tokens
    
    return "\n\n---\n\n".join(context_parts)
```

### Component 9: LLM Generation

Send the context and query to an LLM for response generation.

```python
def generate_response(query: str, context: str, llm_client) -> str:
    """Generate response using LLM with retrieved context."""
    system_prompt = """You are a helpful assistant. Answer the user's question 
    based on the provided context. If the context doesn't contain enough 
    information, say so. Always cite your sources."""
    
    user_prompt = f"""Context:
{context}

Question: {query}

Answer based on the context provided:"""
    
    response = llm_client.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.content
```

## Code Example

Here's a complete, minimal RAG pipeline implementation:

```python
import chromadb
from sentence_transformers import SentenceTransformer

class SimpleRAGPipeline:
    """A complete end-to-end RAG pipeline."""
    
    def __init__(self, collection_name: str = "documents"):
        # Initialize components
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # ============ INGESTION PHASE ============
    
    def ingest(self, documents: list[dict]):
        """
        Ingest documents into the RAG system.
        
        Args:
            documents: List of {"content": str, "source": str}
        """
        all_chunks = []
        all_metadata = []
        all_ids = []
        
        chunk_id = 0
        for doc in documents:
            # Preprocess
            cleaned = self._preprocess(doc["content"])
            
            # Chunk
            chunks = self._chunk(cleaned)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": doc["source"],
                    "chunk_index": i
                })
                all_ids.append(f"doc_{chunk_id}")
                chunk_id += 1
        
        # Embed all chunks
        embeddings = self.embedder.encode(all_chunks).tolist()
        
        # Store
        self.collection.add(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadata
        )
        
        print(f"Ingested {len(all_chunks)} chunks from {len(documents)} documents")
    
    def _preprocess(self, text: str) -> str:
        """Clean text for processing."""
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _chunk(self, text: str, size: int = 300, overlap: int = 50) -> list[str]:
        """Split text into chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            chunk = ' '.join(words[start:start + size])
            if chunk:
                chunks.append(chunk)
            start += size - overlap
        return chunks
    
    # ============ QUERY PHASE ============
    
    def query(self, question: str, n_results: int = 3) -> dict:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            n_results: Number of chunks to retrieve
        
        Returns:
            Dict with context and retrieved chunks
        """
        # Embed query
        query_embedding = self.embedder.encode([question]).tolist()
        
        # Retrieve
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Assemble context
        context = self._assemble_context(results)
        
        return {
            "question": question,
            "context": context,
            "sources": [m["source"] for m in results["metadatas"][0]],
            "chunks": results["documents"][0]
        }
    
    def _assemble_context(self, results: dict) -> str:
        """Combine chunks into context string."""
        parts = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            parts.append(f"[{meta['source']}]: {doc}")
        return "\n\n".join(parts)


# Usage Example
if __name__ == "__main__":
    # Initialize pipeline
    rag = SimpleRAGPipeline()
    
    # Ingest some documents
    documents = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            "source": "python_intro.txt"
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. It uses algorithms to identify patterns and make decisions.",
            "source": "ml_basics.txt"
        }
    ]
    rag.ingest(documents)
    
    # Query the system
    result = rag.query("What is Python used for?")
    print(f"Context:\n{result['context']}")
    print(f"\nSources: {result['sources']}")
```

## Key Takeaways

1. **RAG has two phases**: Ingestion (offline) and Query (online)
2. **Each component has a specific role**: Loading, preprocessing, chunking, embedding, storage, retrieval, context assembly, generation
3. **Data flows through the pipeline**: Documents become chunks become embeddings become context
4. **Component quality compounds**: Poor chunking leads to poor retrieval leads to poor answers
5. **The vector database is central**: It bridges ingestion and query phases

## Additional Resources

- [LangChain RAG Documentation](https://python.langchain.com/docs/tutorials/rag/) - Popular framework for building RAG
- [LlamaIndex RAG Guide](https://docs.llamaindex.ai/en/stable/understanding/rag/) - Another approach to RAG pipelines
- [Building RAG Applications (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) - Free course on advanced RAG
