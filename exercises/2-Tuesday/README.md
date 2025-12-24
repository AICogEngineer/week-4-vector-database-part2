# Tuesday: Metadata & Advanced Querying

## Exercise Schedule

| Exercise | Type | Duration | Prerequisites |
|----------|------|----------|---------------|
| 01: Multi-Format Ingestion | Implementation | 60-75 min | Reading 01-02, Demo 01-02 |
| 02: Metadata Search System | Implementation | 75-90 min | Reading 03-05, Demo 03 |

## Learning Objectives

By completing these exercises, you will:
- Build a document ingestion pipeline for multiple file formats (HTML, Markdown, text)
- Design and implement a metadata schema for RAG systems
- Combine semantic search with metadata filtering for precise results
- Apply production patterns for document management

## Before You Begin

1. **Complete the readings** in `readings/2-Tuesday/`
2. **Watch/run demos** in `demos/2-Tuesday/code/`
3. Ensure you have Python 3.8+ with:
   ```bash
   pip install -r requirements.txt
   ```

## Exercises

### Exercise 01: Multi-Format Ingestion (Implementation)
See [exercise_01_multi_format_ingestion.md](exercise_01_multi_format_ingestion.md)
Starter code: `starter_code/exercise_01_starter.py`

Build a pipeline that ingests HTML, Markdown, and plain text documents into a unified vector store with proper preprocessing.

### Exercise 02: Metadata Search System (Implementation)
See [exercise_02_metadata_search_system.md](exercise_02_metadata_search_system.md)
Starter code: `starter_code/exercise_02_starter.py`

Build a search system that combines semantic similarity with metadata filtering for precise retrieval.

## Estimated Time
**Total: 2.5-3 hours**
