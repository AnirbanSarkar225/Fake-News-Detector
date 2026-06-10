from .preprocess import TextPreprocessor

def __getattr__(name):
    """Lazy-load heavy modules to avoid import errors when their deps are missing."""
    if name == "ArticleScraper":
        from .scraper import ArticleScraper
        return ArticleScraper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["TextPreprocessor", "ArticleScraper"]
