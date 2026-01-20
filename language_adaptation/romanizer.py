"""
Language-aware romanizer.

Romanization is OPTIONAL.
Pipeline may generate it, UI decides whether to display it.
No language-specific library is imported unless required.
"""

def romanize_text(text: str, language: str) -> str:
    if not text or not language:
        return text

    lang = language.lower()

    # --------------------------------------------------
    # Latin-script languages → return as-is
    # --------------------------------------------------
    if lang in {"en", "fr", "de", "es", "it", "pt", "nl"}:
        return text

    # --------------------------------------------------
    # Indic languages
    # --------------------------------------------------
    if lang in {"hi", "ta", "te", "kn", "ml", "mr"}:
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
            return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        except Exception:
            return text

    # --------------------------------------------------
    # Chinese
    # --------------------------------------------------
    if lang in {"zh", "zh-cn", "zh-tw"}:
        try:
            from pypinyin import lazy_pinyin
            return " ".join(lazy_pinyin(text))
        except Exception:
            return text

    # --------------------------------------------------
    # Universal fallback (ASCII-ish)
    # --------------------------------------------------
    try:
        from unidecode import unidecode
        return unidecode(text)
    except Exception:
        return text
