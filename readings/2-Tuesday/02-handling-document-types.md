# Handling Different Document Types

## Learning Objectives
- Process PDFs, HTML, Markdown, and code files for embedding
- Extract text while preserving meaningful structure
- Handle format-specific challenges and edge cases
- Build multi-format ingestion pipelines

## Why This Matters

Real-world RAG systems rarely work with clean `.txt` files. You'll encounter:
- Technical documentation in Markdown
- Research papers in PDF
- Web content in HTML
- Code repositories with mixed file types

Each format has unique challenges. Understanding how to extract and preserve meaningful content from each is essential for building robust RAG pipelines.

## The Concept

### The Multi-Format Challenge

```
RAW INPUT                    EXTRACTED TEXT
----------                   --------------
<html><body>Hello</body>  →  "Hello"
                             (remove tags, keep text)

## Header                  →  "Header\nContent here"  
Content here                 (preserve structure)

%PDF-1.4...binary...      →  "Document content..."
                             (extract text layer)

def foo():                →  "Function foo: docstring..."
    """docstring"""           (preserve code context)
```

### PDF Processing

PDFs are containers that can hold text, images, and complex layouts.

**Challenges**:
- Text extraction order may not match reading order
- Tables and columns can scramble content
- Scanned PDFs contain images, not text (need OCR)

```python
from pathlib import Path


def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF using pymupdf."""
    import fitz  # PyMuPDF
    
    doc = fitz.open(file_path)
    text_parts = []
    
    for page_num, page in enumerate(doc):
        # Extract text with layout preservation
        text = page.get_text("text")
        
        # Add page marker for reference
        text_parts.append(f"\n--- Page {page_num + 1} ---\n")
        text_parts.append(text)
    
    doc.close()
    return ''.join(text_parts)


def extract_pdf_with_structure(file_path: str) -> list[dict]:
    """Extract PDF with structural information."""
    import fitz
    
    doc = fitz.open(file_path)
    sections = []
    
    for page_num, page in enumerate(doc):
        # Get text blocks with position info
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block.get("lines", []):
                    text = ""
                    font_size = 0
                    
                    for span in line.get("spans", []):
                        text += span["text"]
                        font_size = max(font_size, span["size"])
                    
                    if text.strip():
                        sections.append({
                            "text": text.strip(),
                            "page": page_num + 1,
                            "font_size": font_size,
                            "is_heading": font_size > 14  # Heuristic
                        })
    
    doc.close()
    return sections


# Alternative: Using pypdf
def extract_pdf_pypdf(file_path: str) -> str:
    """Extract text using pypdf library."""
    from pypdf import PdfReader
    
    reader = PdfReader(file_path)
    text_parts = []
    
    for page in reader.pages:
        text_parts.append(page.extract_text())
    
    return '\n'.join(text_parts)
```

### HTML Processing

HTML mixes content with markup. The goal is to extract readable text.

```python
from html import unescape
import re


def extract_html_text(html_content: str) -> str:
    """Extract clean text from HTML."""
    # Remove script and style elements
    html_content = re.sub(
        r'<script[^>]*>.*?</script>', 
        '', 
        html_content, 
        flags=re.DOTALL | re.IGNORECASE
    )
    html_content = re.sub(
        r'<style[^>]*>.*?</style>', 
        '', 
        html_content, 
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Replace block elements with newlines
    block_elements = ['p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                      'li', 'tr', 'article', 'section']
    for elem in block_elements:
        html_content = re.sub(
            f'<{elem}[^>]*>', 
            '\n', 
            html_content, 
            flags=re.IGNORECASE
        )
    
    # Remove all remaining tags
    text = re.sub(r'<[^>]+>', '', html_content)
    
    # Decode HTML entities
    text = unescape(text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


def extract_html_with_structure(html_content: str) -> list[dict]:
    """Extract HTML preserving heading structure."""
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove scripts and styles
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    sections = []
    current_section = {"heading": "", "content": [], "level": 0}
    
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
        tag_name = element.name
        text = element.get_text(strip=True)
        
        if not text:
            continue
        
        if tag_name.startswith('h'):
            # Save previous section
            if current_section["content"]:
                sections.append({
                    "heading": current_section["heading"],
                    "content": ' '.join(current_section["content"]),
                    "level": current_section["level"]
                })
            
            # Start new section
            level = int(tag_name[1])
            current_section = {"heading": text, "content": [], "level": level}
        else:
            current_section["content"].append(text)
    
    # Don't forget last section
    if current_section["content"]:
        sections.append({
            "heading": current_section["heading"],
            "content": ' '.join(current_section["content"]),
            "level": current_section["level"]
        })
    
    return sections
```

### Markdown Processing

Markdown is semi-structured. Preserve meaningful structure while removing formatting.

```python
import re


def extract_markdown_text(md_content: str) -> str:
    """Extract text from Markdown, preserving structure."""
    text = md_content
    
    # Remove code blocks but keep inline code
    text = re.sub(r'```[\s\S]*?```', '[CODE BLOCK]', text)
    
    # Convert headers to plain text with newlines
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\n\1\n', text, flags=re.MULTILINE)
    
    # Remove bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)       # Italic
    text = re.sub(r'__([^_]+)__', r'\1', text)       # Bold
    text = re.sub(r'_([^_]+)_', r'\1', text)         # Italic
    
    # Keep link text, remove URL
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove image syntax
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[Image: \1]', text)
    
    # Convert bullet points
    text = re.sub(r'^[-*+]\s+', '- ', text, flags=re.MULTILINE)
    
    return text.strip()


def extract_markdown_sections(md_content: str) -> list[dict]:
    """Extract Markdown into sections by headers."""
    lines = md_content.split('\n')
    sections = []
    current_section = {"heading": "Introduction", "content": [], "level": 0}
    
    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        
        if header_match:
            # Save current section
            if current_section["content"]:
                sections.append({
                    "heading": current_section["heading"],
                    "content": '\n'.join(current_section["content"]),
                    "level": current_section["level"]
                })
            
            # Start new section
            level = len(header_match.group(1))
            heading = header_match.group(2)
            current_section = {"heading": heading, "content": [], "level": level}
        else:
            if line.strip():
                current_section["content"].append(line)
    
    # Add final section
    if current_section["content"]:
        sections.append({
            "heading": current_section["heading"],
            "content": '\n'.join(current_section["content"]),
            "level": current_section["level"]
        })
    
    return sections
```

### Code File Processing

Code requires special handling to preserve semantic context.

```python
import re
from pathlib import Path


def extract_code_documentation(code: str, language: str = "python") -> list[dict]:
    """Extract documentation and signatures from code."""
    if language == "python":
        return extract_python_docs(code)
    elif language in ["javascript", "typescript"]:
        return extract_js_docs(code)
    else:
        return [{"type": "raw", "content": code}]


def extract_python_docs(code: str) -> list[dict]:
    """Extract docstrings and function signatures from Python."""
    import ast
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [{"type": "raw", "content": code}]
    
    docs = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Module):
            if ast.get_docstring(node):
                docs.append({
                    "type": "module_docstring",
                    "content": ast.get_docstring(node)
                })
        
        elif isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node) or ""
            docs.append({
                "type": "class",
                "name": node.name,
                "docstring": docstring,
                "content": f"class {node.name}: {docstring}"
            })
        
        elif isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node) or ""
            # Build signature
            args = [arg.arg for arg in node.args.args]
            signature = f"{node.name}({', '.join(args)})"
            
            docs.append({
                "type": "function",
                "name": node.name,
                "signature": signature,
                "docstring": docstring,
                "content": f"def {signature}: {docstring}"
            })
    
    return docs


def extract_js_docs(code: str) -> list[dict]:
    """Extract JSDoc comments and function signatures."""
    docs = []
    
    # Find JSDoc comments
    jsdoc_pattern = r'/\*\*\s*([\s\S]*?)\*/'
    jsdocs = re.findall(jsdoc_pattern, code)
    
    for jsdoc in jsdocs:
        # Clean up the comment
        lines = jsdoc.split('\n')
        cleaned = ' '.join(
            line.strip().lstrip('* ')
            for line in lines
        )
        docs.append({"type": "jsdoc", "content": cleaned})
    
    # Find function declarations
    func_pattern = r'(?:function\s+(\w+)|(\w+)\s*=\s*(?:async\s+)?function|\(\s*\)|const\s+(\w+)\s*=\s*\([^)]*\)\s*=>)'
    functions = re.findall(func_pattern, code)
    
    for match in functions:
        name = next((m for m in match if m), "anonymous")
        docs.append({"type": "function", "name": name})
    
    return docs
```

## Code Example

Complete multi-format document loader:

```python
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class DocumentType(Enum):
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class LoadedDocument:
    """Represents a loaded and processed document."""
    source: str
    doc_type: DocumentType
    content: str
    metadata: dict
    sections: list[dict]


class MultiFormatLoader:
    """Load and process documents from various formats."""
    
    EXTENSION_MAP = {
        '.pdf': DocumentType.PDF,
        '.html': DocumentType.HTML,
        '.htm': DocumentType.HTML,
        '.md': DocumentType.MARKDOWN,
        '.markdown': DocumentType.MARKDOWN,
        '.txt': DocumentType.TEXT,
        '.py': DocumentType.CODE,
        '.js': DocumentType.CODE,
        '.ts': DocumentType.CODE,
    }
    
    def __init__(self):
        self.processors = {
            DocumentType.PDF: self._process_pdf,
            DocumentType.HTML: self._process_html,
            DocumentType.MARKDOWN: self._process_markdown,
            DocumentType.TEXT: self._process_text,
            DocumentType.CODE: self._process_code,
        }
    
    def load(self, file_path: str) -> LoadedDocument:
        """Load and process a document from file path."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine document type
        doc_type = self.EXTENSION_MAP.get(path.suffix.lower(), DocumentType.UNKNOWN)
        
        if doc_type == DocumentType.UNKNOWN:
            doc_type = DocumentType.TEXT  # Fall back to text
        
        # Read file content
        if doc_type == DocumentType.PDF:
            content = ""  # PDF needs special handling
        else:
            content = path.read_text(encoding='utf-8', errors='replace')
        
        # Process based on type
        processor = self.processors.get(doc_type, self._process_text)
        processed_content, sections = processor(content if content else file_path, path)
        
        return LoadedDocument(
            source=str(path),
            doc_type=doc_type,
            content=processed_content,
            metadata={
                "filename": path.name,
                "extension": path.suffix,
                "size_bytes": path.stat().st_size
            },
            sections=sections
        )
    
    def load_directory(self, dir_path: str, recursive: bool = True) -> list[LoadedDocument]:
        """Load all supported documents from a directory."""
        path = Path(dir_path)
        documents = []
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.EXTENSION_MAP:
                try:
                    doc = self.load(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _process_pdf(self, file_path: str, path: Path) -> tuple[str, list]:
        """Process PDF file."""
        try:
            import fitz
            doc = fitz.open(str(path))
            
            text_parts = []
            sections = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                text_parts.append(text)
                sections.append({
                    "type": "page",
                    "number": page_num + 1,
                    "content": text
                })
            
            doc.close()
            return '\n\n'.join(text_parts), sections
        except ImportError:
            return "PDF processing requires PyMuPDF (fitz)", []
    
    def _process_html(self, content: str, path: Path) -> tuple[str, list]:
        """Process HTML content."""
        # Simple extraction without BeautifulSoup dependency
        text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text, [{"type": "html", "content": text}]
    
    def _process_markdown(self, content: str, path: Path) -> tuple[str, list]:
        """Process Markdown content."""
        sections = []
        lines = content.split('\n')
        current_section = {"heading": "Document", "content": [], "level": 0}
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                if current_section["content"]:
                    sections.append({
                        "type": "section",
                        "heading": current_section["heading"],
                        "content": '\n'.join(current_section["content"]),
                        "level": current_section["level"]
                    })
                level = len(header_match.group(1))
                heading = header_match.group(2)
                current_section = {"heading": heading, "content": [], "level": level}
            else:
                current_section["content"].append(line)
        
        if current_section["content"]:
            sections.append({
                "type": "section",
                "heading": current_section["heading"],
                "content": '\n'.join(current_section["content"]),
                "level": current_section["level"]
            })
        
        # Clean content for embedding
        clean_content = re.sub(r'```[\s\S]*?```', '', content)
        clean_content = re.sub(r'[#*_`\[\]]', '', clean_content)
        
        return clean_content.strip(), sections
    
    def _process_text(self, content: str, path: Path) -> tuple[str, list]:
        """Process plain text."""
        paragraphs = content.split('\n\n')
        sections = [
            {"type": "paragraph", "content": p.strip()}
            for p in paragraphs if p.strip()
        ]
        return content.strip(), sections
    
    def _process_code(self, content: str, path: Path) -> tuple[str, list]:
        """Process code file."""
        sections = []
        
        # Extract docstrings for Python
        if path.suffix == '.py':
            docstring_pattern = r'"""([\s\S]*?)"""'
            docstrings = re.findall(docstring_pattern, content)
            for i, doc in enumerate(docstrings):
                sections.append({
                    "type": "docstring",
                    "index": i,
                    "content": doc.strip()
                })
        
        # Extract function names
        func_pattern = r'def\s+(\w+)\s*\('
        functions = re.findall(func_pattern, content)
        for func in functions:
            sections.append({"type": "function", "name": func})
        
        return content, sections


# Usage
loader = MultiFormatLoader()

# Load single file
doc = loader.load("documentation.md")
print(f"Loaded {doc.doc_type.value}: {len(doc.content)} chars, {len(doc.sections)} sections")

# Load directory
docs = loader.load_directory("./docs", recursive=True)
for doc in docs:
    print(f"  {doc.source}: {doc.doc_type.value}")
```

## Key Takeaways

1. **Each format requires specialized extraction** - one size does not fit all
2. **PDFs are complex** - text order may not match visual order
3. **HTML needs tag removal** - preserve text, remove markup
4. **Markdown is semi-structured** - leverage headers for sectioning
5. **Code benefits from AST parsing** - extract docstrings and signatures
6. **Build unified loaders** - consistent interface for all formats

## Additional Resources

- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/) - Powerful PDF processing
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - HTML parsing library
- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/) - Production loaders for 100+ formats
