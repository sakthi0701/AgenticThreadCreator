"""
Text cleaning utilities for social media content.
Removes markdown artifacts and ensures platform-specific formatting.
"""
import re


def clean_linkedin_text(text):
    """
    Remove markdown artifacts from LinkedIn posts.
    
    LinkedIn doesn't support markdown, so we need to strip:
    - Bold (**text**)
    - Italics (*text*)
    - Headers (##)
    - Em dashes (—)
    
    Args:
        text: String to clean
        
    Returns:
        Cleaned string with markdown removed
    """
    if not isinstance(text, str):
        return text
    
    # 1. Normalize bullets: "* item" -> "• item" (Aesthetic fix)
    text = re.sub(r'^\s*\*\s+', '• ', text, flags=re.MULTILINE)
    
    # 2. Remove markdown artifacts
    text = text.replace('**', '')  # Remove bolding
    text = text.replace('##', '')  # Remove headers
    
    # 3. Replace em dashes with regular dashes
    text = text.replace('—', ' - ').replace('–', ' - ')
    
    # 4. Cleanup remaining * (likely italics) but keep text
    # e.g. *italic* -> italic
    text = text.replace('*', '')
    
    return text.strip()


def clean_text_universal(text):
    """
    Universal text cleaning for all platforms.
    Removes common markdown artifacts and normalizes formatting.
    
    Args:
        text: String to clean
        
    Returns:
        Cleaned string
    """
    if not isinstance(text, str):
        return text
    
    # 1. Normalize bullets
    text = re.sub(r'^\s*\*\s+', '• ', text, flags=re.MULTILINE)
    
    # 2. Remove Bold (**) and Em/En Dashes
    text = text.replace('**', '').replace('—', ' - ').replace('–', ' - ')
    
    # 3. Remove Header marks (#)
    text = text.replace('#', '')
    
    # 4. Cleanup remaining * (italics)
    text = text.replace('*', '')
    
    return text.strip()
