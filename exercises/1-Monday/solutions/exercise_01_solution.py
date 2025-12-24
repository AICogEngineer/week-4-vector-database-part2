"""
Exercise 01: Text Preprocessing Pipeline - Solution

Complete implementation of the text preprocessing pipeline.
"""

import re
import unicodedata
from html import unescape
from typing import List

# ============================================================================
# SAMPLE DOCUMENTS
# ============================================================================

SAMPLE_DOCUMENTS = [
    """
    <html><body>
    <h1>Customer Support FAQ</h1>
    <p>Our support <b>team</b> is here to help you 24/7.</p>
    <p>Contact us at support@example.com&nbsp;or call 1-800-HELP.</p>
    <script>console.log('tracking');</script>
    <p>We&apos;ll respond within &lt;24 hours&gt;.</p>
    </body></html>
    """,
    """
    Welcome to our "premium" service!
    
    We're excited to announce our new features:
    • Real-time analytics
    • 24/7 monitoring
    
    Don't miss out – sign up today!
    
    Note: Prices start at $99/month…
    """,
    """
    
    
    Product    Description
    
    
    Widget A        Our best-selling product.
    
        It features   advanced   technology.
    
    
    Widget B        Economy option.
    
    
    
    Contact sales   for bulk pricing.
    
    
    """,
]

# ============================================================================
# SOLUTION IMPLEMENTATIONS
# ============================================================================

def normalize_encoding(text: str) -> str:
    """
    Normalize text encoding and replace problematic characters.
    
    Uses NFKC normalization which:
    - Decomposes characters into base + combining characters
    - Recomposes using canonical forms
    - Applies compatibility mappings (e.g., ligatures -> separate chars)
    """
    # Apply Unicode NFKC normalization
    # This handles many special characters automatically
    text = unicodedata.normalize('NFKC', text)
    
    # Replace smart quotes with straight quotes
    # Left/right double quotes
    text = text.replace('"', '"').replace('"', '"')
    # Left/right single quotes and apostrophes
    text = text.replace(''', "'").replace(''', "'")
    
    # Replace dashes
    text = text.replace('—', '-')  # Em dash
    text = text.replace('–', '-')  # En dash
    
    # Replace ellipsis
    text = text.replace('…', '...')
    
    # Replace bullet points
    text = text.replace('•', '-')
    
    return text


def remove_html(text: str) -> str:
    """
    Remove HTML tags and decode entities.
    
    Order of operations matters:
    1. Remove script/style blocks entirely (including content)
    2. Remove remaining tags
    3. Decode HTML entities
    """
    # Remove script blocks entirely (content and tags)
    text = re.sub(
        r'<script[^>]*>.*?</script>',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Remove style blocks entirely
    text = re.sub(
        r'<style[^>]*>.*?</style>',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    # This converts &amp; -> &, &nbsp; -> space, &apos; -> ', etc.
    text = unescape(text)
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace to clean, consistent formatting.
    """
    # Replace tabs with single spaces
    text = text.replace('\t', ' ')
    
    # Collapse multiple spaces into one
    text = re.sub(r' +', ' ', text)
    
    # Normalize newlines: replace 3+ consecutive newlines with 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up spaces around newlines
    text = re.sub(r' *\n *', '\n', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text


def preprocess_document(text: str) -> str:
    """
    Complete preprocessing pipeline combining all steps.
    
    Order:
    1. Remove HTML first (tags might interfere with other processing)
    2. Normalize encoding (fix character issues in clean text)
    3. Normalize whitespace (final cleanup)
    """
    # Step 1: Remove HTML tags and decode entities
    text = remove_html(text)
    
    # Step 2: Normalize encoding (smart quotes, dashes, etc.)
    text = normalize_encoding(text)
    
    # Step 3: Normalize whitespace
    text = normalize_whitespace(text)
    
    return text


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_tests():
    """Run preprocessing on sample documents and display results."""
    print("=" * 60)
    print("Exercise 01: Text Preprocessing Pipeline - SOLUTION")
    print("=" * 60)
    
    for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        print(f"\n=== Document {i} ===")
        print(f"BEFORE (first 100 chars):")
        print(f"  {repr(doc[:100])}")
        print()
        
        result = preprocess_document(doc)
        
        print(f"AFTER:")
        print(f"  {result[:200]}..." if len(result) > 200 else f"  {result}")
        print()
        
        # Validation
        issues = []
        if '<' in result and '>' in result:
            issues.append("HTML tags may still be present")
        if '&nbsp;' in result or '&amp;' in result:
            issues.append("HTML entities may still be present")
        if '  ' in result:
            issues.append("Multiple consecutive spaces detected")
        
        if issues:
            for issue in issues:
                print(f"  [WARNING] {issue}")
        else:
            print("  [OK] All checks passed")
    
    print("\n" + "=" * 60)
    print("[OK] All documents preprocessed successfully!")
    print("=" * 60)
    
    # Demonstrate individual functions
    print("\n=== Function Breakdown ===\n")
    
    test_text = '<p>Hello &amp; "welcome"!</p>'
    print(f"Original: {test_text}")
    
    step1 = remove_html(test_text)
    print(f"After remove_html: {step1}")
    
    step2 = normalize_encoding(step1)
    print(f"After normalize_encoding: {step2}")
    
    step3 = normalize_whitespace(step2)
    print(f"After normalize_whitespace: {step3}")


if __name__ == "__main__":
    run_tests()
