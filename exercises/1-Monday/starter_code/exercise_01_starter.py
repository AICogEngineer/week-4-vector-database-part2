"""
Exercise 01: Text Preprocessing Pipeline - Starter Code

Build a preprocessing pipeline to clean messy documents before embedding.

Instructions:
1. Implement each TODO function
2. Run this file to test your implementations
3. Check the expected output in the exercise guide
"""

import re
import unicodedata
from html import unescape
from typing import List

# ============================================================================
# SAMPLE DOCUMENTS (DO NOT MODIFY)
# ============================================================================

SAMPLE_DOCUMENTS = [
    # Document 1: HTML with entities and tags
    """
    <html><body>
    <h1>Customer Support FAQ</h1>
    <p>Our support <b>team</b> is here to help you 24/7.</p>
    <p>Contact us at support@example.com&nbsp;or call 1-800-HELP.</p>
    <script>console.log('tracking');</script>
    <p>We&apos;ll respond within &lt;24 hours&gt;.</p>
    </body></html>
    """,
    
    # Document 2: Encoding issues and smart quotes
    """
    Welcome to our "premium" service!
    
    We're excited to announce our new features:
    • Real-time analytics
    • 24/7 monitoring
    
    Don't miss out – sign up today!
    
    Note: Prices start at $99/month…
    """,
    
    # Document 3: Excessive whitespace and formatting
    """
    
    
    Product    Description
    
    
    Widget A        Our best-selling product.
    
        It features   advanced   technology.
    
    
    Widget B        Economy option.
    
    
    
    Contact sales   for bulk pricing.
    
    
    """,
]

# ============================================================================
# TODO: IMPLEMENT THESE FUNCTIONS
# ============================================================================

def normalize_encoding(text: str) -> str:
    """
    Normalize text encoding and replace problematic characters.
    
    Tasks:
    - Normalize unicode (use NFKC normalization)
    - Replace smart quotes ("" '') with straight quotes
    - Replace em-dashes (—) and en-dashes (–) with regular dashes
    - Replace ellipsis (…) with three dots
    - Replace bullet points (•) with dashes
    
    Args:
        text: Input text with potential encoding issues
        
    Returns:
        Normalized text
    """
    # TODO: Implement this function
    # Hint: unicodedata.normalize('NFKC', text) handles most normalization
    # Then use str.replace() for specific character substitutions
    
    pass  # Remove this and add your implementation


def remove_html(text: str) -> str:
    """
    Remove HTML tags and decode entities.
    
    Tasks:
    - Remove <script>...</script> and <style>...</style> entirely
    - Remove all HTML tags
    - Decode HTML entities (&amp; -> &, &nbsp; -> space, etc.)
    
    Args:
        text: Input text potentially containing HTML
        
    Returns:
        Clean text without HTML
    """
    # TODO: Implement this function
    # Hint: Use re.sub() for removing tags
    # Hint: Use html.unescape() for entities
    
    pass  # Remove this and add your implementation


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace to clean, consistent formatting.
    
    Tasks:
    - Replace tabs with single spaces
    - Collapse multiple spaces into one
    - Limit consecutive newlines to maximum of 2
    - Strip leading and trailing whitespace
    
    Args:
        text: Input text with inconsistent whitespace
        
    Returns:
        Text with normalized whitespace
    """
    # TODO: Implement this function
    # Hint: Use re.sub() with regex patterns
    # Pattern for multiple spaces: r' +'
    # Pattern for multiple newlines: r'\n{3,}'
    
    pass  # Remove this and add your implementation


def preprocess_document(text: str) -> str:
    """
    Complete preprocessing pipeline combining all steps.
    
    Order of operations:
    1. Remove HTML (get rid of tags first)
    2. Normalize encoding (fix character issues)
    3. Normalize whitespace (clean up formatting)
    
    Args:
        text: Raw document text
        
    Returns:
        Cleaned text ready for embedding
    """
    # TODO: Implement this function
    # Chain the three functions in the correct order
    
    pass  # Remove this and add your implementation


# ============================================================================
# TEST HARNESS (DO NOT MODIFY)
# ============================================================================

def run_tests():
    """Run preprocessing on sample documents and display results."""
    print("=" * 60)
    print("Exercise 01: Text Preprocessing Pipeline")
    print("=" * 60)
    
    all_passed = True
    
    for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        print(f"\n=== Document {i} ===")
        print(f"BEFORE (first 100 chars):")
        print(f"  {repr(doc[:100])}")
        print()
        
        try:
            result = preprocess_document(doc)
            
            if result is None:
                print("[ERROR] Function returned None - not implemented yet")
                all_passed = False
                continue
            
            print(f"AFTER:")
            print(f"  {result[:200]}..." if len(result) > 200 else f"  {result}")
            print()
            
            # Basic validation
            if '<' in result and '>' in result:
                print("[WARNING] HTML tags may still be present")
            if '&nbsp;' in result or '&amp;' in result:
                print("[WARNING] HTML entities may still be present")
            if '  ' in result:
                print("[WARNING] Multiple consecutive spaces detected")
                
        except Exception as e:
            print(f"[ERROR] Exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[OK] Preprocessing complete for all documents!")
    else:
        print("[INFO] Some functions need implementation. Check TODOs above.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
