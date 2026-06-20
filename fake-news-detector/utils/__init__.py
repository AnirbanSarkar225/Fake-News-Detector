from .preprocess import TextPreprocessor

def __getattr__(name):
    """Lazy-load heavy modules to avoid import errors when their deps are missing."""
    if name == "ArticleScraper":
        from .scraper import ArticleScraper
        return ArticleScraper
    elif name == "ClickbaitDetector":
        from .clickbait_detector import ClickbaitDetector
        return ClickbaitDetector
    elif name == "AIContentDetector":
        from .ai_detector import AIContentDetector
        return AIContentDetector
    elif name == "ClaimVerifier":
        from .claim_verifier import ClaimVerifier
        return ClaimVerifier
    elif name == "MultilingualProcessor":
        from .multilingual import MultilingualProcessor
        return MultilingualProcessor
    elif name == "BertPredictor":
        from .bert_predictor import BertPredictor
        return BertPredictor
    elif name == "SourceEngine":
        from .source_engine import SourceEngine
        return SourceEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "TextPreprocessor",
    "ArticleScraper",
    "ClickbaitDetector",
    "AIContentDetector",
    "ClaimVerifier",
    "MultilingualProcessor",
    "BertPredictor",
    "SourceEngine"
]

