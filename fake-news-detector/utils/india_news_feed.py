"""
India News RSS Feed Module.

Fetches live headlines from major Indian news outlets via their public RSS feeds.
No API key required — these are free, public XML feeds.

Sources:
    - NDTV (Top Stories, India)
    - Times of India (Top Stories, India)
    - The Hindu (Top Stories)
"""

import time
import re
from datetime import datetime

try:
    import feedparser
except ImportError:
    feedparser = None


# ── RSS Feed URLs by category ──────────────────────────────────────────────
INDIA_RSS_FEEDS = {
    "🔥 Top Stories": [
        ("NDTV", "https://feeds.feedburner.com/ndtvnews-top-stories"),
        ("Times of India", "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
        ("The Hindu", "https://www.thehindu.com/feeder/default.rss"),
    ],
    "🇮🇳 India": [
        ("NDTV", "https://feeds.feedburner.com/ndtvnews-india-news"),
        ("Times of India", "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms"),
        ("The Hindu", "https://www.thehindu.com/news/national/feeder/default.rss"),
    ],
    "💼 Business": [
        ("NDTV", "https://feeds.feedburner.com/ndtvprofit-latest"),
        ("Times of India", "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms"),
        ("The Hindu", "https://www.thehindu.com/business/feeder/default.rss"),
    ],
    "💻 Technology": [
        ("NDTV", "https://feeds.feedburner.com/gadgets360-latest"),
        ("Times of India", "https://timesofindia.indiatimes.com/rssfeeds/66949542.cms"),
        ("The Hindu", "https://www.thehindu.com/sci-tech/technology/feeder/default.rss"),
    ],
    "🏏 Sports": [
        ("NDTV", "https://feeds.feedburner.com/ndtvsports-latest"),
        ("Times of India", "https://timesofindia.indiatimes.com/rssfeeds/4719148.cms"),
        ("The Hindu", "https://www.thehindu.com/sport/feeder/default.rss"),
    ],
}

# Simple in-memory cache
_cache = {}
_CACHE_TTL = 300  # 5 minutes


def _strip_html(text):
    """Remove HTML tags from RSS descriptions."""
    if not text:
        return ""
    return re.sub(r'<[^>]+>', '', text).strip()


def _parse_date(entry):
    """Try to extract a datetime from a feed entry."""
    for attr in ('published_parsed', 'updated_parsed'):
        parsed = getattr(entry, attr, None)
        if parsed:
            try:
                return datetime(*parsed[:6])
            except Exception:
                pass
    return None


class IndiaNewsFeed:
    """Fetches and caches live Indian news headlines from public RSS feeds."""

    def __init__(self):
        if feedparser is None:
            raise ImportError(
                "feedparser is required for the India News Feed. "
                "Install it with: pip install feedparser>=6.0"
            )

    def fetch_category(self, category="🔥 Top Stories", max_per_source=5):
        """
        Fetch headlines for a single category.

        Args:
            category: One of the keys in INDIA_RSS_FEEDS
            max_per_source: Max articles to return per source

        Returns:
            List of dicts with keys: title, description, link, source, published
        """
        feeds = INDIA_RSS_FEEDS.get(category, INDIA_RSS_FEEDS["🔥 Top Stories"])
        cache_key = f"{category}_{max_per_source}"

        # Check cache
        if cache_key in _cache:
            cached_time, cached_data = _cache[cache_key]
            if time.time() - cached_time < _CACHE_TTL:
                return cached_data

        articles = []
        for source_name, feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:max_per_source]:
                    title = _strip_html(getattr(entry, 'title', ''))
                    description = _strip_html(
                        getattr(entry, 'description', '') or
                        getattr(entry, 'summary', '')
                    )
                    link = getattr(entry, 'link', '')
                    pub_date = _parse_date(entry)

                    if title:
                        articles.append({
                            "title": title,
                            "description": description,
                            "link": link,
                            "source": source_name,
                            "published": pub_date,
                        })
            except Exception:
                # If one feed fails, continue with others
                continue

        # Sort by published date (newest first), putting None dates last
        articles.sort(
            key=lambda a: a["published"] or datetime.min,
            reverse=True,
        )

        # Cache the result
        _cache[cache_key] = (time.time(), articles)
        return articles

    def fetch_all(self, max_per_source=3):
        """Fetch headlines across all categories."""
        all_articles = []
        seen_titles = set()
        for category in INDIA_RSS_FEEDS:
            for article in self.fetch_category(category, max_per_source):
                if article["title"] not in seen_titles:
                    seen_titles.add(article["title"])
                    article["category"] = category
                    all_articles.append(article)
        return all_articles

    @staticmethod
    def get_categories():
        """Return available feed categories."""
        return list(INDIA_RSS_FEEDS.keys())

    @staticmethod
    def get_source_color(source):
        """Return a CSS color for each news source."""
        colors = {
            "NDTV": "#e63946",
            "Times of India": "#457b9d",
            "The Hindu": "#2a9d8f",
        }
        return colors.get(source, "#6c757d")
