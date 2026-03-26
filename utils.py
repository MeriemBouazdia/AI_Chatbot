"""
Utility functions for the Smart Greenhouse Chatbot.
This module contains shared functions to avoid circular imports.
"""


def detect_language(text):
    """Detect the language of input text.
    
    Uses langdetect library if available, otherwise falls back to
    detecting Arabic characters.
    
    Args:
        text: The input text to analyze.
        
    Returns:
        str: Language code ('ar' for Arabic, 'en' for English).
    """
    try:
        from langdetect import detect
        return detect(text)
    except:
        # Fallback: check for Arabic characters
        if any('\u0600' <= c <= '\u06FF' for c in text):
            return 'ar'
        return 'en'