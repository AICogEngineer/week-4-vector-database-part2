# Weekly Knowledge Check: Vector Databases (Part 2)

This quiz covers Week 4 topics: Document Preprocessing, Chunking Strategies, Metadata Filtering, Deployment, and Performance Optimization.

---

## Part 1: Multiple Choice

### 1. What is the primary purpose of text preprocessing before embedding?

- [ ] A) To remove noise and normalize text for better embedding quality
- [ ] B) To make embeddings faster to compute
- [ ] C) To reduce storage costs
- [ ] D) To compress the document size

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) To remove noise and normalize text for better embedding quality

**Explanation:** Preprocessing removes HTML tags, normalizes whitespace, fixes encoding issues, etc. This ensures embeddings capture the actual semantic content rather than artifacts.
- **Why others are wrong:**
  - B) Speed impact is minimal
  - C) Storage isn't the main concern
  - D) Compression is a side effect, not the goal

</details>

---

### 2. Which chunking strategy respects document structure like headers and paragraphs?

- [ ] A) Fixed-size chunking
- [ ] B) Recursive chunking
- [ ] C) Sentence-based chunking
- [ ] D) Token-based chunking

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Recursive chunking

**Explanation:** Recursive chunking tries separators in priority order (paragraphs → sentences → characters), respecting document structure before falling back to simpler splits.
- **Why others are wrong:**
  - A) Fixed-size ignores structure entirely
  - C) Sentence-based only considers sentences
  - D) Token-based focuses on token count, not structure

</details>

---

### 3. What does "chunk overlap" help prevent in RAG systems?

- [ ] A) Memory overflow during embedding
- [ ] B) Duplicate documents in the database
- [ ] C) Loss of context when relevant information spans chunk boundaries
- [ ] D) Slow query performance

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Loss of context when relevant information spans chunk boundaries

**Explanation:** Overlap ensures that if important context is split between chunks, it appears in both, so retrieval can still find the complete information.
- **Why others are wrong:**
  - A) Memory usage is unrelated
  - B) Overlap may actually add some redundancy
  - D) Overlap can slightly increase storage, not improve query speed

</details>

---

### 4. What is the typical recommended chunk size for text embeddings?

- [ ] A) 50-100 characters
- [ ] B) Exactly 1000 words
- [ ] C) The entire document as one chunk
- [ ] D) 100-500 tokens (roughly 400-2000 characters)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) 100-500 tokens (roughly 400-2000 characters)

**Explanation:** This range balances semantic coherence with embedding model token limits. Too small loses context; too large dilutes meaning.
- **Why others are wrong:**
  - A) Too small for meaningful context
  - B) Too rigid; optimal size varies
  - C) Large documents lose retrieval precision

</details>

---

### 5. Which Python function is commonly used to decode HTML entities like `&amp;`?

- [ ] A) `html.unescape()`
- [ ] B) `html.parse()`
- [ ] C) `html.decode()`
- [ ] D) `html.strip()`

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) `html.unescape()`

**Explanation:** `html.unescape()` converts HTML entities to their corresponding characters: `&amp;` → `&`, `&lt;` → `<`, etc.
- **Why others are wrong:**
  - B) Not a valid function
  - C) Not a valid function
  - D) Not a valid function

</details>

---

### 6. In Chroma, which parameter filters results by metadata during a query?

- [ ] A) `filter`
- [ ] B) `where`
- [ ] C) `metadata`
- [ ] D) `match`

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) `where`

**Explanation:** Chroma uses `where={"field": "value"}` to filter query results by metadata.
- **Why others are wrong:**
  - A) Filter is not the parameter name
  - C) Metadata is used when adding, not querying
  - D) Match is not a Chroma parameter

</details>

---

### 7. Which Chroma operator allows filtering for values in a list?

- [ ] A) `$or`
- [ ] B) `$contains`
- [ ] C) `$in`
- [ ] D) `$list`

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) `$in`

**Explanation:** Use `{"field": {"$in": ["value1", "value2"]}}` to match any value in the list.
- **Why others are wrong:**
  - A) `$or` combines multiple conditions, not values for one field
  - B) Not a valid Chroma operator
  - D) Not a valid Chroma operator

</details>

---

### 8. What is the main advantage of using `chromadb.PersistentClient()` over `chromadb.Client()`?

- [ ] A) Faster query performance
- [ ] B) Supports more embedding models
- [ ] C) Uses less memory
- [ ] D) Data survives program restarts

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Data survives program restarts

**Explanation:** PersistentClient saves data to disk, so it's available the next time you run your program. Client() stores data in memory only.
- **Why others are wrong:**
  - A) Performance is similar
  - B) Both support the same models
  - C) Persistence actually uses disk storage

</details>

---

### 9. Which optimization has the LARGEST impact on document ingestion speed?

- [ ] A) Batching documents instead of single inserts
- [ ] B) Using a faster SSD
- [ ] C) Increasing RAM
- [ ] D) Using a different distance metric

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Batching documents instead of single inserts

**Explanation:** Batching can improve insertion speed by 10-50x because it reduces per-operation overhead and allows batch embedding.
- **Why others are wrong:**
  - B) Hardware helps but not as dramatically
  - C) RAM helps with large datasets, not ingestion speed
  - D) Distance metric doesn't affect insertion

</details>

---

### 10. What is the purpose of caching embeddings?

- [ ] A) To reduce storage space
- [ ] B) To avoid recomputing embeddings for repeated queries
- [ ] C) To improve embedding quality
- [ ] D) To enable offline mode

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) To avoid recomputing embeddings for repeated queries

**Explanation:** Embedding computation is expensive. Caching stores results so identical inputs don't need to be re-embedded.
- **Why others are wrong:**
  - A) Caching may use more storage
  - C) Quality isn't improved, just efficiency
  - D) Caching doesn't enable offline mode

</details>

---

## Part 2: True or False

### 11. Fixed-size chunking always produces the highest quality retrieval results.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** FALSE

**Explanation:** Fixed-size chunking can break sentences mid-thought and ignore document structure. Semantic or recursive chunking often produces better results.

</details>

---

### 12. Metadata filters are applied BEFORE the vector similarity search in most implementations.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** TRUE

**Explanation:** Pre-filtering by metadata reduces the search space before computing vector similarities, which is more efficient than post-filtering.

</details>

---

### 13. Increasing the HNSW M parameter always improves query speed.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** FALSE

**Explanation:** Higher M improves recall (quality) but increases memory usage and can slow down queries due to more connections to traverse.

</details>

---

### 14. HTML `<script>` tags should be removed during preprocessing because they contain non-content text.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** TRUE

**Explanation:** Script tags contain JavaScript code, not document content. Including them would pollute embeddings with irrelevant programming code.

</details>

---

### 15. In Chroma, you can combine multiple metadata filters using the `$and` operator.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** TRUE

**Explanation:** Use `{"$and": [{"field1": "value1"}, {"field2": "value2"}]}` to require multiple conditions to be true.

</details>

---

### 16. Chunk overlap should typically be 0% to avoid storing duplicate information.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** FALSE

**Explanation:** 10-20% overlap is recommended to prevent context loss at boundaries. A small amount of duplication is acceptable for better retrieval.

</details>

---

### 17. Cloud-managed vector databases always cost more than self-hosted solutions.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** FALSE

**Explanation:** TCO depends on scale, ops overhead, and usage patterns. Small-scale cloud can be cheaper than maintaining infrastructure.

</details>

---

### 18. The `include` parameter in Chroma queries can reduce response size and improve performance.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** TRUE

**Explanation:** Only requesting needed fields (e.g., `include=["documents"]` instead of all fields) reduces data transfer and processing.

</details>

---

## Part 3: Code Prediction

### 19. What does this code output?

```python
text = "Hello   World\n\n\nTest"
import re
result = re.sub(r'\s+', ' ', text)
print(result)
```

- [ ] A) `"Hello World Test"`
- [ ] B) `"Hello   World\n\n\nTest"`
- [ ] C) `"HelloWorldTest"`
- [ ] D) Error

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) `"Hello World Test"`

**Explanation:** `\s+` matches one or more whitespace characters (spaces, newlines, tabs). Replacing with a single space normalizes all whitespace.
- **Why others are wrong:**
  - B) The original string with no changes
  - C) Words would be joined without any spaces
  - D) This is valid regex code

</details>

---

### 20. What is the length of `chunks` after running this code?

```python
text = "ABCDEFGHIJ"  # 10 characters
chunk_size = 4
overlap = 2
chunks = []
step = chunk_size - overlap  # step = 2

for i in range(0, len(text), step):
    chunks.append(text[i:i+chunk_size])

print(len(chunks))
```

- [ ] A) 3
- [ ] B) 4
- [ ] C) 5
- [ ] D) 6

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) 5

**Explanation:** With step=2 and text length 10: chunks at 0, 2, 4, 6, 8 = 5 chunks. ("ABCD", "CDEF", "EFGH", "GHIJ", "IJ")
- **Why others are wrong:**
  - A) Too few iterations
  - B) Too few iterations
  - D) Would require step=1 or longer text

</details>

---

### 21. What does this Chroma query return?

```python
collection.query(
    query_embeddings=[[0.1, 0.2, 0.3]],
    n_results=3,
    where={"category": {"$in": ["tutorial", "guide"]}}
)
```

- [ ] A) Only documents where category equals "tutorial"
- [ ] B) Documents with both categories
- [ ] C) An error because $in is invalid
- [ ] D) All documents where category equals "tutorial" OR "guide"

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) All documents where category equals "tutorial" OR "guide"

**Explanation:** `$in` matches any value in the list, equivalent to an OR condition for that field.
- **Why others are wrong:**
  - A) Only matches one category
  - B) That would require $all operator
  - C) $in is valid Chroma syntax

</details>

---

### 22. What does `html.unescape("&lt;div&gt;")` return?

- [ ] A) `"&lt;div&gt;"`
- [ ] B) `"<div>"`
- [ ] C) `"div"`
- [ ] D) Error

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) `"<div>"`

**Explanation:** `html.unescape()` converts HTML entities to their character equivalents: `&lt;` → `<`, `&gt;` → `>`.
- **Why others are wrong:**
  - A) That's the input, not the output
  - C) The tags are preserved, just decoded
  - D) This is valid Python code

</details>

---

### 23. What happens when you run this?

```python
import chromadb
client = chromadb.PersistentClient(path="./mydata")
collection = client.get_or_create_collection("test")
```

- [ ] A) Creates an in-memory database
- [ ] B) Raises an error if ./mydata doesn't exist
- [ ] C) Creates or opens a database that persists to ./mydata
- [ ] D) Connects to a remote server

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Creates or opens a database that persists to ./mydata

**Explanation:** PersistentClient creates the directory if needed and stores data there. The data survives program restarts.
- **Why others are wrong:**
  - A) That's Client(), not PersistentClient()
  - B) Chroma creates the directory automatically
  - D) That's HttpClient()

</details>

---

### 24. What is the output?

```python
cache = {}
def get_embedding(text):
    if text in cache:
        return "HIT"
    cache[text] = [0.1, 0.2]
    return "MISS"

print(get_embedding("hello"))
print(get_embedding("world"))
print(get_embedding("hello"))
```

- [ ] A) MISS, MISS, MISS
- [ ] B) HIT, HIT, HIT
- [ ] C) Error
- [ ] D) MISS, MISS, HIT

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) MISS, MISS, HIT

**Explanation:** First "hello" is a miss (not in cache). "world" is a miss. Second "hello" is a hit (was cached on first call).
- **Why others are wrong:**
  - A) The cache works, so third call hits
  - B) First calls can't be hits
  - C) This is valid Python

</details>

---

## Part 4: Fill in the Blank

### 25. To normalize Unicode text in Python, use `unicodedata.normalize('_____', text)`.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** NFKC (or NFC, NFD, NFKD are also valid forms, but NFKC is most common for text preprocessing)

**Explanation:** Unicode normalization converts various representations of the same character to a canonical form. NFKC handles compatibility decomposition and canonical composition.

</details>

---

### 26. In Chroma, to compare a numeric metadata value, use `{"field": {"$_____": 100}}`.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** gt (or gte, lt, lte, eq, ne)

**Explanation:** Chroma supports comparison operators: $gt (greater than), $gte (greater or equal), $lt (less than), $lte (less or equal).

</details>

---

### 27. The optimal batch size for document ingestion typically ranges from _____ to 200 documents.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** 50

**Explanation:** Batch sizes of 50-200 typically provide the best throughput. Too small adds overhead; too large can cause memory issues.

</details>

---

### 28. Recursive chunking tries splitting by _____ first, then sentences, then fixed size.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** paragraphs (or double newlines)

**Explanation:** Recursive chunking respects document structure by trying the largest semantic units first (paragraphs), then falling back to smaller units.

</details>

---

## Part 5: Concept Application

### 29. You're preprocessing HTML documents and notice embeddings are capturing JavaScript variable names. What step did you miss?

- [ ] A) Removing `<script>` tags before stripping other HTML
- [ ] B) Lowercasing the text
- [ ] C) Normalizing whitespace
- [ ] D) Decoding HTML entities

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Removing `<script>` tags before stripping other HTML

**Explanation:** Script content needs to be removed entirely (not just the tags) before processing the document, or the JavaScript code becomes part of the text.
- **Why others are wrong:**
  - B) Case doesn't affect script removal
  - C) Whitespace handling comes after content extraction
  - D) Entity decoding is separate from script removal

</details>

---

### 30. Your RAG system retrieves chunks but often misses the answer because relevant info spans two chunks. What should you adjust?

- [ ] A) Decrease chunk size
- [ ] B) Increase chunk overlap
- [ ] C) Use a different embedding model
- [ ] D) Add more documents

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Increase chunk overlap

**Explanation:** More overlap means information near chunk boundaries appears in multiple chunks, improving retrieval of split content.
- **Why others are wrong:**
  - A) Smaller chunks worsen the boundary problem
  - C) Model change doesn't fix boundary issues
  - D) More documents don't help with chunking

</details>

---

### 31. You need to filter documents by date in Chroma. Which metadata schema is correct?

- [ ] A) Store dates as datetime objects
- [ ] B) Store dates as ISO strings ("2024-01-15")
- [ ] C) Both ISO strings and Unix timestamps work for comparison operators
- [ ] D) Store dates as Unix timestamps (1705276800)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Both ISO strings and Unix timestamps work for comparison operators

**Explanation:** Chroma supports string comparison (works with ISO dates) and numeric comparison (works with timestamps). Both allow $gt, $lt filtering.
- **Why others are wrong:**
  - A) Chroma doesn't support datetime objects directly
  - B) True but incomplete
  - D) True but incomplete

</details>

---

### 32. Your vector database ingestion is slow. Profiling shows embedding computation takes 80% of the time. What optimization helps most?

- [ ] A) Use a faster SSD
- [ ] B) Increase batch size
- [ ] C) Use a persistent client
- [ ] D) Cache embeddings for repeated documents

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Cache embeddings for repeated documents

**Explanation:** If embedding is the bottleneck, caching prevents re-embedding identical content. This is especially effective with updates and re-indexing.
- **Why others are wrong:**
  - A) SSD doesn't speed up CPU-bound embedding
  - B) Batch size helps DB operations, not embedding time per doc
  - C) Persistence doesn't affect computation speed

</details>

---

### 33. A user wants to search for "Python tutorials" but also wants results for "programming guides". What approach works best?

- [ ] A) Single query for "Python tutorials"
- [ ] B) Run two separate queries
- [ ] C) Combine semantic search with metadata filter `category IN [tutorials, guides]`
- [ ] D) Use exact keyword matching

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Combine semantic search with metadata filter `category IN [tutorials, guides]`

**Explanation:** Semantic search handles "Python", while metadata filtering ensures results come from tutorial or guide categories.
- **Why others are wrong:**
  - A) Misses "guides" category
  - B) Less efficient than combined query
  - D) Misses semantic variations

</details>

---

### 34. You're deploying a RAG system for a startup. Which deployment is most appropriate?

- [ ] A) Local Chroma with PersistentClient
- [ ] B) Self-managed Kubernetes cluster
- [ ] C) Enterprise cloud solution with SLAs
- [ ] D) Distributed multi-region setup

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Local Chroma with PersistentClient

**Explanation:** For a startup, local persistent Chroma offers low cost, simplicity, and sufficient performance for moderate scale.
- **Why others are wrong:**
  - B) Overkill for a startup
  - C) Expensive for early stage
  - D) Unnecessary complexity initially

</details>

---

### 35. What is the relationship between HNSW parameter M and memory usage?

- [ ] A) Higher M uses less memory
- [ ] B) Higher M uses more memory due to more connections per node
- [ ] C) M has no effect on memory
- [ ] D) M only affects query time

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Higher M uses more memory due to more connections per node

**Explanation:** M controls the maximum number of connections per node in the HNSW graph. More connections = more memory to store them.
- **Why others are wrong:**
  - A) Opposite is true
  - C) M significantly affects memory
  - D) M affects both memory and index quality

</details>

---

## Part 6: Scenario-Based Questions

### 36. You notice query latency is 50ms on average, but occasionally spikes to 500ms. What's the likely cause?

- [ ] A) Wrong distance metric
- [ ] B) Too few documents
- [ ] C) Cold cache or garbage collection pauses
- [ ] D) Incorrect metadata schema

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Cold cache or garbage collection pauses

**Explanation:** Inconsistent latency with occasional spikes typically indicates cache misses (first query for a pattern) or runtime pauses like GC.
- **Why others are wrong:**
  - A) Wrong metric would consistently affect results
  - B) Fewer docs usually means faster queries
  - D) Schema issues cause errors, not latency spikes

</details>

---

### 37. You're building a multi-tenant SaaS. Each tenant's documents must be isolated. What's the best approach?

- [ ] A) Separate Chroma instance per tenant
- [ ] B) Store all documents together without isolation
- [ ] C) Use different embedding models per tenant
- [ ] D) Single collection with tenant_id in metadata and filter on every query

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Single collection with tenant_id in metadata and filter on every query

**Explanation:** Metadata filtering provides logical isolation with simpler operations. Separate instances add operational complexity.
- **Why others are wrong:**
  - A) Operationally complex to manage many instances
  - B) No isolation violates multi-tenancy requirements
  - C) Inconsistent embeddings would prevent shared learning

</details>

---

### 38. Your Markdown parser is including code block contents in embeddings. Users complain that searching for "machine learning" returns programming examples. What should you do?

- [ ] A) Increase chunk size
- [ ] B) Use a different embedding model
- [ ] C) Decrease chunk overlap
- [ ] D) Remove code blocks during preprocessing or store them separately with different metadata

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Remove code blocks during preprocessing or store them separately with different metadata

**Explanation:** Code blocks often contain implementation details, not conceptual content. Either remove them or tag them as "code" for filtered searches.
- **Why others are wrong:**
  - A) Size doesn't fix content type issues
  - B) Model won't ignore code just because
  - C) Overlap is unrelated to content filtering

</details>

---

### 39. You want to backup your Chroma database. What information must you preserve?

- [ ] A) Only the embedding vectors
- [ ] B) Only the metadata
- [ ] C) Documents, IDs, and metadata (embeddings can be regenerated)
- [ ] D) Only the collection configuration

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Documents, IDs, and metadata (embeddings can be regenerated)

**Explanation:** Embeddings can be recomputed from documents, but documents and metadata cannot be recovered from embeddings.
- **Why others are wrong:**
  - A) Vectors alone can't recover text
  - B) Metadata without documents is useless
  - D) Config alone has no data

</details>

---

### 40. You're optimizing query performance. Which change has the LEAST impact?

- [ ] A) Adding embedding cache
- [ ] B) Changing collection name from "docs" to "documents"
- [ ] C) Using `include` to request only needed fields
- [ ] D) Adding metadata filters to reduce search space

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Changing collection name from "docs" to "documents"

**Explanation:** Collection name is just an identifier with no performance impact. The other options all improve performance.
- **Why others are wrong:**
  - A) Caching significantly reduces embedding time
  - C) Include reduces data transfer
  - D) Filters reduce vectors to search

</details>

---

## Quiz Complete

**Answer Distribution:**
- A: 7 (25%)
- B: 7 (25%)
- C: 7 (25%)
- D: 7 (25%)
- TRUE: 4
- FALSE: 4

**Scoring Guide:**
- 36-40 correct: Excellent - ready for advanced RAG patterns
- 30-35 correct: Good - review weak areas
- 24-29 correct: Satisfactory - more practice needed
- Below 24: Review readings and try again
