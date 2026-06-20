"""
Multilingual processing module for TruthShield Fake News Detector.

Provides language detection (via ``langdetect``), Unicode-based script
detection, translation to English (via ``deep_translator``), basic
stopword lists for Indian languages, and a convenience pipeline that
combines detection and translation in a single call.

All optional dependencies are imported with graceful fallbacks so the
module is always importable.
"""

import re
import unicodedata
from typing import Dict, List, Optional

try:
    from langdetect import detect as _ld_detect, detect_langs as _ld_detect_langs
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False


class MultilingualProcessor:
    """Detect language, detect script, translate, and preprocess multilingual text."""

    SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
        "en": {"name": "English", "script": "Latin"},
        "hi": {"name": "Hindi", "script": "Devanagari"},
        "bn": {"name": "Bengali", "script": "Bengali"},
        "ta": {"name": "Tamil", "script": "Tamil"},
        "te": {"name": "Telugu", "script": "Telugu"},
        "mr": {"name": "Marathi", "script": "Devanagari"},
        "ur": {"name": "Urdu", "script": "Arabic"},
        "gu": {"name": "Gujarati", "script": "Gujarati"},
        "kn": {"name": "Kannada", "script": "Kannada"},
        "ml": {"name": "Malayalam", "script": "Malayalam"},
        "pa": {"name": "Punjabi", "script": "Gurmukhi"},
        "es": {"name": "Spanish", "script": "Latin"},
        "fr": {"name": "French", "script": "Latin"},
        "de": {"name": "German", "script": "Latin"},
        "ar": {"name": "Arabic", "script": "Arabic"},
        "zh": {"name": "Chinese", "script": "Han"},
    }

    # Unicode block ranges → script names
    _SCRIPT_RANGES: List[tuple] = [
        (0x0900, 0x097F, "Devanagari"),
        (0x0980, 0x09FF, "Bengali"),
        (0x0B80, 0x0BFF, "Tamil"),
        (0x0C00, 0x0C7F, "Telugu"),
        (0x0C80, 0x0CFF, "Kannada"),
        (0x0D00, 0x0D7F, "Malayalam"),
        (0x0A80, 0x0AFF, "Gujarati"),
        (0x0A00, 0x0A7F, "Gurmukhi"),
        (0x0600, 0x06FF, "Arabic"),
        (0x4E00, 0x9FFF, "Han"),
        (0x0041, 0x024F, "Latin"),
    ]

    _CHUNK_SIZE: int = 4500

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> Dict:
        """
        Detect the language of *text* with confidence.

        Returns:
            Dict with keys: language_code, language_name, confidence,
            script, is_supported.  Falls back to ``'en'`` on failure.
        """
        if not text or not text.strip():
            return self._fallback_detection()

        if HAS_LANGDETECT:
            try:
                langs = _ld_detect_langs(text)
                if langs:
                    top = langs[0]
                    code = str(top.lang)
                    confidence = round(top.prob, 3)
                    lang_info = self.SUPPORTED_LANGUAGES.get(code, {})
                    return {
                        "language_code": code,
                        "language_name": lang_info.get("name", code),
                        "confidence": confidence,
                        "script": lang_info.get("script", self.detect_script(text)),
                        "is_supported": code in self.SUPPORTED_LANGUAGES,
                    }
            except Exception:
                pass

        # Fallback: script-based heuristic
        script = self.detect_script(text)
        code = self._script_to_code(script)
        lang_info = self.SUPPORTED_LANGUAGES.get(code, {})
        return {
            "language_code": code,
            "language_name": lang_info.get("name", code),
            "confidence": 0.5,
            "script": script,
            "is_supported": code in self.SUPPORTED_LANGUAGES,
        }

    # ------------------------------------------------------------------
    # Script detection
    # ------------------------------------------------------------------

    def detect_script(self, text: str) -> str:
        """
        Detect the dominant writing script of *text* using Unicode ranges.

        Returns:
            The script name (e.g. ``'Devanagari'``, ``'Latin'``).
        """
        counts: Dict[str, int] = {}
        for ch in text:
            cp = ord(ch)
            for low, high, name in self._SCRIPT_RANGES:
                if low <= cp <= high:
                    counts[name] = counts.get(name, 0) + 1
                    break
        if not counts:
            return "Unknown"
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def translate_to_english(self, text: str, source_lang: Optional[str] = None) -> Dict:
        """
        Translate *text* to English, chunking long inputs at sentence
        boundaries to respect API limits.

        Args:
            text: Input text in any supported language.
            source_lang: Optional ISO-639-1 code; auto-detected if *None*.

        Returns:
            Dict with keys: translated_text, source_language, was_translated,
            chunks_count.
        """
        if not text or not text.strip():
            return {
                "translated_text": "",
                "source_language": source_lang or "en",
                "was_translated": False,
                "chunks_count": 0,
            }

        if source_lang is None:
            detection = self.detect_language(text)
            source_lang = detection["language_code"]

        if source_lang == "en":
            return {
                "translated_text": text,
                "source_language": "en",
                "was_translated": False,
                "chunks_count": 0,
            }

        if not HAS_TRANSLATOR:
            return {
                "translated_text": text,
                "source_language": source_lang,
                "was_translated": False,
                "chunks_count": 0,
            }

        chunks = self._chunk_text(text)
        translated_parts: List[str] = []

        for chunk in chunks:
            try:
                result = GoogleTranslator(
                    source=source_lang, target="en"
                ).translate(chunk)
                translated_parts.append(result or chunk)
            except Exception:
                translated_parts.append(chunk)

        return {
            "translated_text": " ".join(translated_parts),
            "source_language": source_lang,
            "was_translated": True,
            "chunks_count": len(chunks),
        }

    # ------------------------------------------------------------------
    # Stopwords
    # ------------------------------------------------------------------

    def get_stopwords(self, language_code: str) -> List[str]:
        """
        Return language-specific stopwords.

        Includes basic Hindi and Bengali lists; falls back to an empty
        list for unsupported languages.
        """
        _STOPWORDS: Dict[str, List[str]] = {
            "en": [
                "the", "a", "an", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "shall", "can",
                "of", "in", "to", "for", "with", "on", "at", "by", "from",
                "that", "this", "it", "and", "or", "but", "not", "as", "if",
                "about", "into", "through", "during", "before", "after",
            ],
            "hi": [
                "का", "के", "की", "में", "है", "हैं", "को", "पर", "इस",
                "से", "ने", "एक", "और", "यह", "भी", "कि", "था", "थे",
                "पर", "हो", "जो", "तो", "कर", "वह", "अपने", "नहीं",
                "सब", "कुछ", "या", "तक", "जब", "बहुत", "लेकिन", "अब",
                "उस", "इसे", "साथ", "अगर", "बाद", "दो", "बस", "सकता",
                "रहा", "गया", "होता", "रहे", "वाले", "ऐसे", "कई",
            ],
            "bn": [
                "এ", "এই", "এর", "একটি", "ও", "তা", "তার", "না", "হয়",
                "করে", "থেকে", "যে", "আর", "কিন্তু", "জন্য", "সে",
                "হয়েছে", "এবং", "হবে", "তবে", "করা", "নয়", "পর",
                "মধ্যে", "দিয়ে", "কোনো", "আমি", "তুমি", "আপনি", "যা",
                "নিয়ে", "আমার", "তোমার", "সব", "কারণ", "অনেক",
                "বলে", "হলে", "আগে", "পারে", "হতে", "গিয়ে", "ছিল",
            ],
        }
        return _STOPWORDS.get(language_code, [])

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_for_analysis(self, text: str) -> Dict:
        """
        Detect language → translate if needed → return processed text.

        Returns:
            Dict with keys: original_text, processed_text, language_code,
            language_name, was_translated, confidence, script.
        """
        detection = self.detect_language(text)
        translated = self.translate_to_english(text, source_lang=detection["language_code"])

        return {
            "original_text": text,
            "processed_text": translated["translated_text"],
            "language_code": detection["language_code"],
            "language_name": detection["language_name"],
            "was_translated": translated["was_translated"],
            "confidence": detection["confidence"],
            "script": detection["script"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fallback_detection(self) -> Dict:
        return {
            "language_code": "en",
            "language_name": "English",
            "confidence": 0.0,
            "script": "Latin",
            "is_supported": True,
        }

    def _script_to_code(self, script: str) -> str:
        mapping: Dict[str, str] = {
            "Devanagari": "hi",
            "Bengali": "bn",
            "Tamil": "ta",
            "Telugu": "te",
            "Kannada": "kn",
            "Malayalam": "ml",
            "Gujarati": "gu",
            "Gurmukhi": "pa",
            "Arabic": "ar",
            "Han": "zh",
            "Latin": "en",
        }
        return mapping.get(script, "en")

    def _chunk_text(self, text: str) -> List[str]:
        """Split *text* into chunks ≤ ``_CHUNK_SIZE`` chars at sentence boundaries."""
        sentences = re.split(r'(?<=[.!?।])\s+', text)
        chunks: List[str] = []
        current: List[str] = []
        length = 0

        for sent in sentences:
            if length + len(sent) > self._CHUNK_SIZE and current:
                chunks.append(" ".join(current))
                current = []
                length = 0
            current.append(sent)
            length += len(sent) + 1

        if current:
            chunks.append(" ".join(current))

        return chunks
