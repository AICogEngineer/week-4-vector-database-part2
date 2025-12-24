# Data Preprocessing for Vectors

## Learning Objectives
- Understand why preprocessing is critical for embedding quality
- Learn text cleaning and normalization techniques
- Handle special characters, formatting, and document structure
- Prepare documents from various sources for vector embedding

## Why This Matters

"Garbage in, garbage out" applies directly to RAG systems. The quality of your embeddings depends entirely on the quality of the text you feed into them. A document filled with HTML tags, excessive whitespace, or encoding errors will produce poor embeddings that fail to capture semantic meaning.

In our journey to build production-ready RAG pipelines, preprocessing is the often-overlooked foundation. Spending time here saves debugging hours later when retrieval mysteriously fails.

## The Concept

### Why Preprocess?

Embedding models are trained on clean, well-formatted text. When you feed them:
- HTML: `<div class="content">Hello World</div>`
- Excessive spaces: `This    has    weird    spacing`
- Encoding issues: `Hello World` (smart quotes)

The model wastes embedding dimensions on noise instead of meaning.

**Before Preprocessing**:
```
<p>Machine learning is <b>amazing</b>!!!</p>\n\n\n\nIt can do many things...
```

**After Preprocessing**:
```
Machine learning is amazing! It can do many things.
```

Both sentences mean the same thing, but the clean version embeds more effectively.

### Preprocessing Pipeline Steps

A typical preprocessing pipeline follows this order:

```
Raw Text
    ↓
1. Encoding Normalization
    ↓
2. HTML/Markup Removal
    ↓
3. Whitespace Normalization
    ↓
4. Special Character Handling
    ↓
5. Case Normalization (optional)
    ↓
Clean Text for Embedding
```

### Step 1: Encoding Normalization

Ensure consistent text encoding (UTF-8) and handle common encoding issues.

```python
import unicodedata

def normalize_encoding(text: str) -> str:
    """Normalize Unicode and fix common encoding issues."""
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
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text
```

### Step 2: HTML and Markup Removal

Strip HTML tags while preserving the text content.

```python
import re
from html import unescape

def remove_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    # Decode HTML entities first
    text = unescape(text)
    
    # Remove script and style elements entirely
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    
    # Replace block elements with newlines
    text = re.sub(r'<(p|div|br|h[1-6]|li)[^>]*>', '\n', text, flags=re.IGNORECASE)
    
    # Remove remaining tags
    text = re.sub(r'<[^>]+>', '', text)
    
    return text
```

**Handling Markdown**:
```python
def remove_markdown(text: str) -> str:
    """Remove common Markdown formatting."""
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)       # Italic
    text = re.sub(r'__([^_]+)__', r'\1', text)       # Bold
    text = re.sub(r'_([^_]+)_', r'\1', text)         # Italic
    
    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove code blocks
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    return text
```

### Step 3: Whitespace Normalization

Clean up excessive whitespace while preserving meaningful structure.

```python
def normalize_whitespace(text: str) -> str:
    """Normalize whitespace to single spaces and proper line breaks."""
    # Replace tabs and multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove leading/trailing whitespace from entire text
    return text.strip()
```

### Step 4: Special Character Handling

Decide what to do with special characters based on your use case.

```python
def handle_special_characters(text: str, mode: str = 'preserve') -> str:
    """
    Handle special characters based on mode.
    
    Modes:
        'preserve': Keep punctuation, remove other special chars
        'minimal': Remove most special characters
        'code': Preserve characters common in code
    """
    if mode == 'preserve':
        # Keep standard punctuation, remove unusual symbols
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()\[\]@#$%&]', '', text)
    
    elif mode == 'minimal':
        # Keep only alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
    
    elif mode == 'code':
        # Preserve characters common in programming
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()\[\]{}=+<>/*@#$%&|\\]', '', text)
    
    return text
```

### Step 5: Case Normalization (Optional)

Lowercasing can help with consistency but may lose information.

```python
def normalize_case(text: str, mode: str = 'none') -> str:
    """
    Normalize text case.
    
    Modes:
        'none': No change
        'lower': All lowercase
        'preserve_acronyms': Lowercase but keep ALL-CAPS words
    """
    if mode == 'none':
        return text
    
    elif mode == 'lower':
        return text.lower()
    
    elif mode == 'preserve_acronyms':
        words = text.split()
        normalized = []
        for word in words:
            # Keep if all uppercase (likely acronym)
            if word.isupper() and len(word) > 1:
                normalized.append(word)
            else:
                normalized.append(word.lower())
        return ' '.join(normalized)
    
    return text
```

### Complete Preprocessing Pipeline

Combine all steps into a cohesive pipeline:

```python
class TextPreprocessor:
    """Complete text preprocessing pipeline for embeddings."""
    
    def __init__(
        self,
        remove_html: bool = True,
        remove_markdown: bool = True,
        special_char_mode: str = 'preserve',
        case_mode: str = 'none'
    ):
        self.remove_html = remove_html
        self.remove_markdown = remove_markdown
        self.special_char_mode = special_char_mode
        self.case_mode = case_mode
    
    def process(self, text: str) -> str:
        """Run the complete preprocessing pipeline."""
        # Step 1: Encoding
        text = self._normalize_encoding(text)
        
        # Step 2: Markup removal
        if self.remove_html:
            text = self._remove_html(text)
        if self.remove_markdown:
            text = self._remove_markdown(text)
        
        # Step 3: Whitespace
        text = self._normalize_whitespace(text)
        
        # Step 4: Special characters
        text = self._handle_special_chars(text)
        
        # Step 5: Case
        text = self._normalize_case(text)
        
        return text
    
    # ... (include all the helper methods from above)
```

## Code Example

Here's a practical example processing different document types:

```python
import re
from pathlib import Path

class DocumentPreprocessor:
    """Preprocess documents for RAG pipeline ingestion."""
    
    def __init__(self):
        self.text_processor = TextPreprocessor()
    
    def process_file(self, file_path: str) -> str:
        """Process a file based on its type."""
        path = Path(file_path)
        
        # Read file
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Process based on file type
        if path.suffix == '.html':
            return self._process_html(content)
        elif path.suffix == '.md':
            return self._process_markdown(content)
        elif path.suffix == '.txt':
            return self._process_text(content)
        elif path.suffix == '.json':
            return self._process_json(content)
        else:
            return self._process_text(content)
    
    def _process_html(self, content: str) -> str:
        """Process HTML content."""
        processor = TextPreprocessor(remove_html=True, remove_markdown=False)
        return processor.process(content)
    
    def _process_markdown(self, content: str) -> str:
        """Process Markdown content."""
        processor = TextPreprocessor(remove_html=False, remove_markdown=True)
        return processor.process(content)
    
    def _process_text(self, content: str) -> str:
        """Process plain text content."""
        processor = TextPreprocessor(remove_html=False, remove_markdown=False)
        return processor.process(content)
    
    def _process_json(self, content: str) -> str:
        """Extract and process text from JSON."""
        import json
        try:
            data = json.loads(content)
            # Extract text fields (customize based on your JSON structure)
            texts = self._extract_text_from_json(data)
            combined = ' '.join(texts)
            return self.text_processor.process(combined)
        except json.JSONDecodeError:
            return self._process_text(content)
    
    def _extract_text_from_json(self, data, texts=None) -> list:
        """Recursively extract text from JSON structure."""
        if texts is None:
            texts = []
        
        if isinstance(data, dict):
            for value in data.values():
                self._extract_text_from_json(value, texts)
        elif isinstance(data, list):
            for item in data:
                self._extract_text_from_json(item, texts)
        elif isinstance(data, str):
            texts.append(data)
        
        return texts


# Usage
preprocessor = DocumentPreprocessor()

# Process different file types
html_text = preprocessor.process_file("docs/page.html")
markdown_text = preprocessor.process_file("docs/readme.md")
plain_text = preprocessor.process_file("docs/notes.txt")

print("Processed HTML:", html_text[:200])
```

## Key Takeaways

1. **Preprocessing directly impacts embedding quality** - clean text produces meaningful embeddings
2. **Follow a consistent pipeline**: encoding, markup removal, whitespace, special chars, case
3. **Tailor preprocessing to your content type** - HTML needs different handling than Markdown
4. **Preserve semantic information** - don't over-clean and lose meaning
5. **Test your preprocessing** - compare embeddings before and after to verify improvement

## Additional Resources

- [Text Preprocessing Techniques (Towards Data Science)](https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8) - Comprehensive preprocessing overview
- [Unicode and Character Encoding (Python Docs)](https://docs.python.org/3/howto/unicode.html) - Understanding text encoding
- [Beautiful Soup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - Library for HTML/XML parsing
