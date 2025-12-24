"""
Demo 01: Document Preprocessing for Vector Embeddings

This demo shows trainees how to:
1. Clean raw documents for optimal embedding quality
2. Handle encoding issues and special characters
3. Remove HTML/Markdown formatting while preserving content
4. Build a complete preprocessing pipeline

Learning Objectives:
- Understand why preprocessing matters for embedding quality
- Apply text normalization techniques
- Build reusable preprocessing utilities

References:
- Written Content: 03-data-preprocessing-vectors.md
- Weekly Epic: From Search to Systems - Building Production-Ready RAG Pipelines
"""

import re
from html import unescape
import unicodedata

# ============================================================================
# PART 1: Why Preprocessing Matters
# ============================================================================

print("=" * 70)
print("PART 1: Why Preprocessing Matters")
print("=" * 70)

print("""
GARBAGE IN, GARBAGE OUT

Embedding models are trained on clean text. When you feed them:
- HTML tags: <div class="content">Hello</div>
- Weird encoding: "smart quotes" and em—dashes
- Extra whitespace:  spaces     everywhere

They waste embedding dimensions on noise instead of meaning!

BEFORE preprocessing → embedding captures formatting noise
AFTER preprocessing  → embedding captures semantic meaning
""")

# Example of messy document
messy_document = """
<html>
<head><script>alert("bad!");</script></head>
<body>
<h1>Machine Learning Overview</h1>

<p>Machine learning is a subset of <b>artificial intelligence</b> that 
enables computers to learn from data.</p>

<p>It's becoming increasingly important in today's "connected" world—
with applications ranging from   recommendation systems   to autonomous vehicles.</p>

</body>
</html>
"""

print("\n[Example] A messy HTML document:")
print("-" * 50)
print(messy_document[:300] + "...")
print("-" * 50)

# ============================================================================
# PART 2: Encoding Normalization
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Encoding Normalization")
print("=" * 70)

print("""
COMMON ENCODING ISSUES:
- "Smart quotes" → Regular quotes
- Em dashes — → Regular dashes -
- Non-breaking spaces → Regular spaces
- Ellipsis characters … → Three dots ...
""")

def normalize_encoding(text: str) -> str:
    """
    Normalize Unicode and fix common encoding issues.
    
    This is the FIRST step in preprocessing - ensure consistent encoding.
    """
    # Normalize to NFC form (composed characters)
    text = unicodedata.normalize('NFC', text)
    
    # Replace common problematic characters
    replacements = {
        '\u2018': "'",   # Left single quote
        '\u2019': "'",   # Right single quote  
        '\u201c': '"',   # Left double quote
        '\u201d': '"',   # Right double quote
        '\u2013': '-',   # En dash
        '\u2014': '-',   # Em dash
        '\u2026': '...', # Ellipsis
        '\u00a0': ' ',   # Non-breaking space
        '\u200b': '',    # Zero-width space
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

print("\n[Step 1] Demonstrating encoding normalization...")

# Example with smart quotes and dashes
text_with_encoding_issues = "He said \"Hello\" — that's nice… isn't it?"

print(f"\nBefore: {repr(text_with_encoding_issues)}")
clean_text = normalize_encoding(text_with_encoding_issues)
print(f"After:  {repr(clean_text)}")

print("""
OBSERVATION: The smart quotes and em-dash are now standard ASCII.
             This helps embedding models process text consistently.
""")

# ============================================================================
# PART 3: HTML Removal
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: HTML Tag Removal")
print("=" * 70)

print("""
HTML CLEANING STRATEGY:
1. Remove <script> and <style> blocks entirely (dangerous/useless)
2. Replace block elements (<p>, <div>, <br>) with newlines
3. Strip all remaining tags
4. Decode HTML entities (&amp; → &, &lt; → <)
""")

def remove_html(text: str) -> str:
    """
    Remove HTML tags while preserving text content.
    """
    # Decode HTML entities first (&amp; → &, etc.)
    text = unescape(text)
    
    # Remove script and style elements ENTIRELY
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Replace block elements with newlines (preserve structure)
    block_elements = r'<(p|div|br|h[1-6]|li|tr|article|section)[^>]*>'
    text = re.sub(block_elements, '\n', text, flags=re.IGNORECASE)
    
    # Remove all remaining tags
    text = re.sub(r'<[^>]+>', '', text)
    
    return text

print("\n[Step 2] Removing HTML from our messy document...")

html_removed = remove_html(messy_document)
print("\nAfter HTML removal:")
print("-" * 50)
print(html_removed[:400])
print("-" * 50)

print("""
OBSERVATION: 
- Script tags are completely gone (security and noise)
- Paragraph tags became newlines (structure preserved)
- Bold tags removed but text kept ("artificial intelligence")
""")

# ============================================================================
# PART 4: Whitespace Normalization
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Whitespace Normalization")
print("=" * 70)

print("""
WHITESPACE ISSUES:
- Multiple spaces    between    words
- Multiple newlines


  like this

- Tabs		and mixed whitespace
- Leading/trailing spaces around lines
""")

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace to single spaces and proper line breaks.
    """
    # Replace tabs and multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove leading/trailing whitespace from entire text
    return text.strip()

print("\n[Step 3] Normalizing whitespace...")

whitespace_normalized = normalize_whitespace(html_removed)
print("\nAfter whitespace normalization:")
print("-" * 50)
print(whitespace_normalized)
print("-" * 50)

# ============================================================================
# PART 5: Markdown Removal (Bonus)
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Markdown Removal")
print("=" * 70)

def remove_markdown(text: str) -> str:
    """
    Remove Markdown formatting while preserving content.
    """
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold **text**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)       # Italic *text*
    text = re.sub(r'__([^_]+)__', r'\1', text)       # Bold __text__
    text = re.sub(r'_([^_]+)_', r'\1', text)         # Italic _text_
    
    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '[CODE]', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove images
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[Image: \1]', text)
    
    return text

markdown_example = """
# Machine Learning Guide

This is **important** information about *machine learning*.

Check out [this tutorial](https://example.com) for more details.

```python
def train_model():
    pass
```
"""

print("\n[Step 4] Processing Markdown document...")
print("\nOriginal Markdown:")
print("-" * 50)
print(markdown_example)
print("-" * 50)

markdown_cleaned = remove_markdown(markdown_example)
print("\nAfter Markdown removal:")
print("-" * 50)
print(markdown_cleaned)
print("-" * 50)

# ============================================================================
# PART 6: Complete Preprocessing Pipeline
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Complete Preprocessing Pipeline")
print("=" * 70)

class TextPreprocessor:
    """
    Complete text preprocessing pipeline for vector embeddings.
    
    Use this class to clean documents before chunking and embedding.
    """
    
    def __init__(
        self,
        remove_html: bool = True,
        remove_markdown: bool = True,
        normalize_case: bool = False  # Usually keep original case
    ):
        self.do_remove_html = remove_html
        self.do_remove_markdown = remove_markdown
        self.normalize_case = normalize_case
    
    def process(self, text: str) -> str:
        """
        Run the complete preprocessing pipeline.
        
        Steps (in order):
        1. Encoding normalization
        2. HTML removal (optional)
        3. Markdown removal (optional)
        4. Whitespace normalization
        5. Case normalization (optional)
        """
        # Step 1: Fix encoding
        text = self._normalize_encoding(text)
        
        # Step 2: Remove HTML
        if self.do_remove_html:
            text = self._remove_html(text)
        
        # Step 3: Remove Markdown
        if self.do_remove_markdown:
            text = self._remove_markdown(text)
        
        # Step 4: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Step 5: Case normalization (optional)
        if self.normalize_case:
            text = text.lower()
        
        return text
    
    def _normalize_encoding(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        replacements = {
            '\u2018': "'", '\u2019': "'",
            '\u201c': '"', '\u201d': '"',
            '\u2013': '-', '\u2014': '-',
            '\u2026': '...', '\u00a0': ' ',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def _remove_html(self, text: str) -> str:
        text = unescape(text)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<(p|div|br|h[1-6]|li)[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    def _remove_markdown(self, text: str) -> str:
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines).strip()


print("\n[Step 5] Using the complete preprocessor...")

preprocessor = TextPreprocessor(
    remove_html=True,
    remove_markdown=True,
    normalize_case=False  # Preserve case for embeddings
)

final_result = preprocessor.process(messy_document)

print("\nFinal cleaned document:")
print("-" * 50)
print(final_result)
print("-" * 50)

# ============================================================================
# DEMO SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO 1 COMPLETE: Document Preprocessing")
print("=" * 70)

print("""
Key Takeaways:

1. ENCODING NORMALIZATION is the first step
   - Fixes smart quotes, dashes, and special characters
   - Ensures consistent byte representation

2. HTML REMOVAL strips markup but keeps text
   - Remove script/style completely
   - Convert block elements to newlines
   - Strip remaining tags

3. WHITESPACE NORMALIZATION cleans up spacing
   - Single spaces between words
   - Double newlines for paragraphs
   - No leading/trailing spaces

4. USE A PIPELINE for consistent processing
   - Apply steps in the right order
   - Configure based on document type
   - Test on real data!

Coming Next: Demo 2 will show how to CHUNK preprocessed text!
""")

# ============================================================================
# INSTRUCTOR NOTES
# ============================================================================

print("\n" + "=" * 70)
print("INSTRUCTOR NOTES")
print("=" * 70)

print("""
Discussion Questions:
1. "What would happen if we embedded the raw HTML document?"
2. "When might you want to preserve some formatting (like code blocks)?"
3. "How would you handle documents in multiple languages?"

Common Confusions:
- "Should I remove all punctuation?" 
  → No! Punctuation carries meaning. Only remove formatting.
  
- "What about tables?" 
  → Special handling needed. Tomorrow's topic.
  
- "How do I know if preprocessing is working?"
  → Compare embedding similarity before/after cleaning.

If Running Short on Time:
- Skip Markdown section (Part 5)
- Focus on HTML and the complete pipeline

If Trainees Are Advanced:
- Discuss OCR preprocessing for scanned documents
- Mention language detection and handling
""")

print("\n" + "=" * 70)
