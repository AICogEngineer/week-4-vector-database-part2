"""
Demo 03: Complete End-to-End RAG Pipeline

This demo shows trainees how to:
1. Build a complete RAG pipeline from scratch
2. Combine preprocessing, chunking, embedding, and storage
3. Implement the query pipeline with retrieval
4. See the full flow from document ingestion to answer

Learning Objectives:
- Understand RAG pipeline architecture
- Build working ingestion and query pipelines  
- See how components integrate together

References:
- Written Content: 01-document-retrieval-patterns.md
- Written Content: 02-end-to-end-rag-pipeline.md
"""

import re
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================================
# OPENAI CONFIGURATION
# ============================================================================
# Set to True if you have an OpenAI API key and want to see actual LLM responses
# Set to False to run the demo without OpenAI (shows prompts without generation)
USE_OPENAI = True

# If USE_OPENAI is True, the API key can be set via:
# 1. Environment variable: OPENAI_API_KEY
# 2. Or uncomment and set directly below (not recommended for production)
# os.environ["OPENAI_API_KEY"] = "sk-your-key-here"

if USE_OPENAI:
    try:
        from openai import OpenAI
        client = OpenAI()  # Uses OPENAI_API_KEY environment variable
        print("[OpenAI] Successfully initialized OpenAI client")
    except ImportError:
        print("[OpenAI] openai package not installed. Run: pip install openai")
        print("[OpenAI] Falling back to mock responses")
        USE_OPENAI = False
    except Exception as e:
        print(f"[OpenAI] Error initializing client: {e}")
        print("[OpenAI] Falling back to mock responses")
        USE_OPENAI = False

# ============================================================================
# PART 1: RAG Architecture Overview
# ============================================================================

print("=" * 70)
print("PART 1: RAG Pipeline Architecture")
print("=" * 70)

print("""
RAG (Retrieval-Augmented Generation) has TWO main phases:

INGESTION PHASE (Offline - run once)
════════════════════════════════════
Documents → [Preprocess] → [Chunk] → [Embed] → [Store in Vector DB]

QUERY PHASE (Online - run for each query)
═════════════════════════════════════════
User Query → [Embed] → [Search Vector DB] → [Retrieve Top K]
                                                   │
                                                   ▼
                                    [Assemble Context] → [Send to LLM] → Answer

Today we'll build BOTH phases from scratch!
""")

# ============================================================================
# PART 2: Sample Documents
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Our Document Corpus")
print("=" * 70)

# Sample documents representing a knowledge base
DOCUMENTS = [
    {
        "id": "ml_basics",
        "title": "Machine Learning Fundamentals",
        "content": """
Machine learning is a subset of artificial intelligence that enables computers 
to learn from data without being explicitly programmed. The core idea is that 
systems can identify patterns in data and make decisions with minimal human 
intervention.

There are three main types of machine learning: supervised learning, 
unsupervised learning, and reinforcement learning. Each approach has different 
use cases and requirements for training data.

Supervised learning uses labeled data to train models. The algorithm learns 
the relationship between inputs and known outputs, then applies this knowledge 
to new, unseen data.
        """
    },
    {
        "id": "neural_networks",
        "title": "Neural Networks Explained",
        "content": """
Neural networks are computing systems inspired by biological neural networks 
in the human brain. They consist of interconnected nodes (neurons) organized 
in layers that process information.

A typical neural network has an input layer, one or more hidden layers, and 
an output layer. Data flows forward through the network, with each layer 
transforming the information.

Deep learning uses neural networks with many hidden layers (hence "deep"). 
These deep networks can learn complex patterns and representations that 
shallow networks cannot capture.

Backpropagation is the algorithm used to train neural networks. It calculates 
gradients of the loss function with respect to each weight, allowing the 
network to improve through gradient descent.
        """
    },
    {
        "id": "nlp_intro",
        "title": "Natural Language Processing",
        "content": """
Natural Language Processing (NLP) is a field of AI focused on enabling 
computers to understand, interpret, and generate human language.

Modern NLP relies heavily on deep learning, particularly transformer 
architectures. Models like BERT and GPT have revolutionized how machines 
process text.

Key NLP tasks include text classification, named entity recognition, 
sentiment analysis, machine translation, and question answering. Each 
task requires different approaches and training data.

Embeddings are fundamental to NLP. They convert words or sentences into 
dense vector representations that capture semantic meaning. Similar 
concepts have similar embeddings.
        """
    },
    {
        "id": "vector_dbs",
        "title": "Vector Databases",
        "content": """
Vector databases are specialized systems designed to store and query 
high-dimensional vectors efficiently. They are essential for modern AI 
applications like semantic search and recommendation systems.

Unlike traditional databases that use exact matching, vector databases 
use similarity search. They find vectors that are "close" to a query 
vector in high-dimensional space.

Popular vector databases include Pinecone, Weaviate, Milvus, and Chroma. 
Each has different strengths for various use cases and scale requirements.

HNSW (Hierarchical Navigable Small World) is a common indexing algorithm 
used by vector databases. It enables fast approximate nearest neighbor 
search in high-dimensional spaces.
        """
    },
    {
        "id": "rag_systems",
        "title": "RAG Systems",
        "content": """
Retrieval-Augmented Generation (RAG) combines information retrieval with 
text generation. When a user asks a question, the system first retrieves 
relevant documents, then uses those documents as context for generation.

RAG solves the knowledge limitation problem of language models. Instead of 
relying solely on training data, RAG systems can access external knowledge 
bases for up-to-date and domain-specific information.

Building a RAG system requires several components: a document processor, 
chunking strategy, embedding model, vector database, retrieval logic, 
and a language model for generation.

The quality of RAG answers depends heavily on retrieval quality. If the 
wrong documents are retrieved, the generated answer will be wrong too.
        """
    }
]

print(f"Loaded {len(DOCUMENTS)} documents:")
for doc in DOCUMENTS:
    print(f"  - {doc['title']} ({len(doc['content'].split())} words)")

# ============================================================================
# PART 3: Building the RAG Pipeline Class
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Building the RAG Pipeline")
print("=" * 70)

class SimpleRAGPipeline:
    """
    A complete, simple RAG pipeline implementation.
    
    This class demonstrates:
    1. Document preprocessing
    2. Chunking
    3. Embedding generation
    4. Vector storage with Chroma
    5. Query processing
    6. Context retrieval
    """
    
    def __init__(self, collection_name: str = "rag_demo"):
        """Initialize the RAG pipeline components."""
        print("\n[Initializing RAG Pipeline...]")
        
        # Embedding configuration
        self.use_openai_embeddings = USE_OPENAI
        
        if self.use_openai_embeddings:
            # Use OpenAI's text-embedding-3-small model
            print("  Using OpenAI text-embedding-3-small for embeddings...")
            self.embedding_model = "text-embedding-3-small"
            self.embedding_dimension = 1536
            print(f"  ✓ OpenAI embeddings configured (dimension: {self.embedding_dimension})")
        else:
            # Fallback to local sentence-transformers model
            print("  Loading local embedding model (all-MiniLM-L6-v2)...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dimension = self.embedder.get_sentence_embedding_dimension()
            print(f"  ✓ Model loaded (dimension: {self.embedding_dimension})")
        
        # Vector database
        print("  Initializing Chroma DB...")
        self.client = chromadb.Client()
        
        # Delete existing collection if it exists (for clean demo)
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
            
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"  ✓ Collection '{collection_name}' created")
        
        # Configuration
        self.chunk_size = 300
        self.chunk_overlap = 50
        
        print("  ✓ RAG Pipeline initialized!")
    
    # ==================== EMBEDDING ====================
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Uses OpenAI API if enabled, otherwise uses local sentence-transformers.
        """
        if self.use_openai_embeddings:
            # Use OpenAI embeddings API
            response = client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            # Extract embeddings from response
            return [item.embedding for item in response.data]
        else:
            # Use local sentence-transformers
            embeddings = self.embedder.encode(texts)
            return embeddings.tolist()
    
    # ==================== PREPROCESSING ====================
    
    def preprocess(self, text: str) -> str:
        """Clean and normalize text for embedding."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep alphanumeric and basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
        
        return text.strip()
    
    # ==================== CHUNKING ====================
    
    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        # Convert word-based chunking for simplicity
        words_per_chunk = self.chunk_size // 5  # Rough estimate: 5 chars per word
        overlap_words = self.chunk_overlap // 5
        
        start = 0
        while start < len(words):
            end = start + words_per_chunk
            chunk = ' '.join(words[start:end])
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - overlap_words
            
            # Safety check to prevent infinite loop
            if start >= len(words) - overlap_words:
                break
        
        return chunks
    
    # ==================== INGESTION ====================
    
    def ingest(self, documents: List[Dict]) -> Dict:
        """
        Ingest documents into the vector database.
        
        Pipeline: Documents → Preprocess → Chunk → Embed → Store
        """
        print("\n[INGESTION PHASE]")
        print("-" * 50)
        
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        for doc in documents:
            doc_id = doc["id"]
            title = doc["title"]
            content = doc["content"]
            
            print(f"\nProcessing: {title}")
            
            # Step 1: Preprocess
            clean_content = self.preprocess(content)
            print(f"  Preprocessed: {len(content)} → {len(clean_content)} chars")
            
            # Step 2: Chunk
            chunks = self.chunk(clean_content)
            print(f"  Chunked into {len(chunks)} pieces")
            
            # Step 3: Prepare for storage
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadatas.append({
                    "source": doc_id,
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
        
        # Step 4: Generate embeddings (batch)
        print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
        if self.use_openai_embeddings:
            print("  Using OpenAI text-embedding-3-small...")
        embeddings = self.embed(all_chunks)
        print(f"  Generated {len(embeddings)} embeddings")
        
        # Step 5: Store in vector database
        print("\nStoring in Chroma DB...")
        self.collection.add(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas
        )
        print(f"  ✓ Stored {len(all_chunks)} chunks")
        
        return {
            "documents_processed": len(documents),
            "chunks_created": len(all_chunks),
            "collection_size": self.collection.count()
        }
    
    # ==================== QUERY ====================
    
    def query(self, question: str, n_results: int = 3) -> Dict:
        """
        Query the RAG system.
        
        Pipeline: Question → Embed → Search → Retrieve → Return Context
        """
        print(f"\n[QUERY PHASE]")
        print(f"Question: {question}")
        print("-" * 50)
        
        # Step 1: Embed the question
        print("  Embedding question...")
        question_embedding = self.embed([question])
        
        # Step 2: Search vector database
        print(f"  Searching for top {n_results} results...")
        results = self.collection.query(
            query_embeddings=question_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Step 3: Process results
        retrieved_chunks = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            similarity = 1 / (1 + dist)  # Convert distance to similarity
            retrieved_chunks.append({
                "rank": i + 1,
                "content": doc,
                "source": meta["title"],
                "similarity": similarity
            })
            print(f"  Result {i+1}: {meta['title']} (similarity: {similarity:.3f})")
        
        return {
            "question": question,
            "retrieved": retrieved_chunks,
            "context": self._build_context(retrieved_chunks)
        }
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Assemble retrieved chunks into context for LLM."""
        context_parts = []
        
        for chunk in chunks:
            context_parts.append(
                f"[Source: {chunk['source']}]\n{chunk['content']}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    # ==================== GENERATE PROMPT ====================
    
    def create_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the LLM with retrieved context."""
        return f"""Based on the following context, answer the question.
If the context doesn't contain enough information, say "I don't have enough information."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    # ==================== GENERATE RESPONSE ====================
    
    def generate_response(self, question: str, context: str) -> str:
        """
        Generate a response using OpenAI (if enabled) or return a mock response.
        
        This is the final step of RAG: taking retrieved context and generating
        a natural language answer.
        """
        prompt = self.create_prompt(question, context)
        
        if USE_OPENAI:
            try:
                print("  Calling OpenAI GPT-4o-mini...")
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Cost-effective model for demos
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant. Answer questions based only on the provided context. Be concise but accurate."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,  # Lower temperature for more focused answers
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"  [OpenAI Error] {e}")
                return f"[Error calling OpenAI: {e}]"
        else:
            # Mock response for those without API key
            return "[Mock Response - Set USE_OPENAI=True with valid API key for real generation]\n" \
                   f"Based on the context, I would answer the question: '{question}'"

# ============================================================================
# PART 4: Running the Pipeline
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Running the Complete Pipeline")
print("=" * 70)

# Create the pipeline
rag = SimpleRAGPipeline()

# Ingest documents
print("\n" + "=" * 50)
print("STEP 1: INGESTING DOCUMENTS")
print("=" * 50)

ingestion_result = rag.ingest(DOCUMENTS)
print(f"\nIngestion complete!")
print(f"  Documents: {ingestion_result['documents_processed']}")
print(f"  Chunks: {ingestion_result['chunks_created']}")
print(f"  Collection size: {ingestion_result['collection_size']}")

# ============================================================================
# PART 5: Testing Queries
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Testing Queries")
print("=" * 70)

test_questions = [
    "What is machine learning?",
    "How does backpropagation work?",
    "What are embeddings used for in NLP?",
    "What is the advantage of RAG systems?"
]

for question in test_questions:
    print("\n" + "=" * 50)
    result = rag.query(question, n_results=2)
    
    print(f"\n[Retrieved Context]")
    print("-" * 30)
    print(result["context"][:500] + "..." if len(result["context"]) > 500 else result["context"])
    
    # Generate response using OpenAI (or mock)
    print(f"\n[LLM Response]")
    print("-" * 30)
    answer = rag.generate_response(question, result["context"])
    print(answer)
    print()  # Extra spacing between questions

# ============================================================================
# PART 6: Understanding the Flow
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Understanding the Complete Flow")
print("=" * 70)

print("""
WHAT HAPPENED:

INGESTION (One-time):
┌─────────────────────────────────────────────────────────────────┐
│ 5 Documents                                                      │
│      ↓ preprocess()                                             │
│ Clean text (normalized whitespace, removed special chars)        │
│      ↓ chunk()                                                  │
│ ~15+ chunks (300 chars each, 50 overlap)                        │
│      ↓ embedder.encode()                                        │
│ 15+ vectors (384 dimensions each)                                │
│      ↓ collection.add()                                         │
│ Stored in Chroma with metadata                                   │
└─────────────────────────────────────────────────────────────────┘

QUERY (Every request):
┌─────────────────────────────────────────────────────────────────┐
│ "What is machine learning?"                                      │
│      ↓ embedder.encode()                                        │
│ Query vector (384 dimensions)                                    │
│      ↓ collection.query()                                       │
│ Top 2 most similar chunks                                        │
│      ↓ _build_context()                                         │
│ Assembled context with sources                                   │
│      ↓ create_prompt()                                          │
│ Ready to send to LLM (GPT-4, Claude, etc.)                       │
└─────────────────────────────────────────────────────────────────┘

KEY INSIGHT:
The quality of the RETRIEVAL determines the quality of the ANSWER.
If we retrieve wrong chunks, the LLM will generate a wrong answer!
""")

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO 3 COMPLETE: End-to-End RAG Pipeline")
print("=" * 70)

print("""
Key Takeaways:

1. RAG HAS TWO PHASES
   - Ingestion: Process documents once, store in vector DB
   - Query: Search, retrieve, and generate answers in real-time

2. EACH STEP MATTERS
   - Preprocessing → Clean data produces better embeddings
   - Chunking → Affects what can be retrieved
   - Embedding → Captures semantic meaning
   - Retrieval → Finds relevant context
   - Generation → Produces the final answer

3. QUALITY COMPOUNDS
   - Bad preprocessing → Bad embeddings → Bad retrieval → Bad answers
   - Good preprocessing → Good embeddings → Good retrieval → Good answers

4. THE VECTOR DATABASE IS CENTRAL
   - Bridges ingestion and query phases
   - Enables fast similarity search
   - Stores metadata for filtering

Coming Tomorrow: We'll add proper text splitters and metadata filtering!
""")

# ============================================================================
# INSTRUCTOR NOTES
# ============================================================================

print("\n" + "=" * 70)
print("INSTRUCTOR NOTES")
print("=" * 70)

print("""
Discussion Questions:
1. "What would happen if we used larger or smaller chunks?"
2. "How would you handle documents in different languages?"
3. "What if two chunks have the same content from different sources?"

Interactive Ideas:
- Let trainees suggest new questions to test
- Ask them to predict which documents will be retrieved
- Show what happens with a query that has no good match

Common Confusions:
- "Where's the LLM?" → We prepare the context; LLM integration is separate
- "Why not embed the whole document?" → Token limits, precision loss
- "Is similarity score accuracy?" → No, it's closeness in vector space

If Running Short on Time:
- Skip Part 6 (flow explanation)
- Focus on the working demo

If Trainees Are Advanced:
- Discuss reranking retrieved results
- Mention hybrid search (vector + keyword)
- Talk about evaluation metrics (recall, precision)
""")

print("\n" + "=" * 70)
