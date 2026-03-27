def detect_language(text):
    try:
        from langdetect import detect
        lang = detect(text)
        if lang not in ['en', 'fr', 'ar']:
            return 'en'

        return lang

    except:
        if any('\u0600' <= c <= '\u06FF' for c in text):
            return 'ar'
        elif any(c in text for c in "éèêàçùôî"):
            return 'fr'
        return 'en'