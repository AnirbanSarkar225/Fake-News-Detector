"""
Article Scraper Module for Fake News Detector.

Extracts article text, metadata, and content from URLs.
Uses a multi-fallback strategy:
  1. Newspaper3k with browser-like config
  2. Manual requests download → Newspaper3k parse
  3. Google Webcache mirror
  4. Wayback Machine (Internet Archive) snapshot
"""

import re
import requests
from newspaper import Article, ArticleException, Config
from urllib.parse import quote_plus


class ArticleScraper:
    """
    Web article scraper that extracts full text and metadata from news URLs.
    Multiple fallback strategies to handle sites that block automated access.
    """

    USER_AGENT = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/125.0.0.0 Safari/537.36'
    )

    def __init__(self):
        # Full browser-like headers
        self.headers = {
            'User-Agent': self.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'DNT': '1',
            'Cache-Control': 'max-age=0',
        }

        # Newspaper3k config
        self.config = Config()
        self.config.browser_user_agent = self.USER_AGENT
        self.config.request_timeout = 20
        self.config.fetch_images = False

        # Persistent session for cookies
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _download_with_requests(self, url: str) -> str:
        """Download HTML using requests with full browser headers + session cookies."""
        resp = self.session.get(url, timeout=20, allow_redirects=True)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or 'utf-8'
        return resp.text

    def _try_google_cache(self, url: str) -> str:
        """Try fetching the page from Google's webcache mirror."""
        cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{quote_plus(url)}"
        resp = self.session.get(cache_url, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or 'utf-8'
        return resp.text

    def _try_wayback_machine(self, url: str) -> str:
        """Try fetching the latest snapshot from the Wayback Machine."""
        api_url = f"https://archive.org/wayback/available?url={quote_plus(url)}"
        api_resp = self.session.get(api_url, timeout=10)
        api_resp.raise_for_status()
        data = api_resp.json()

        snapshots = data.get("archived_snapshots", {})
        closest = snapshots.get("closest", {})
        if closest.get("available") and closest.get("url"):
            snap_url = closest["url"]
            resp = self.session.get(snap_url, timeout=20, allow_redirects=True)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or 'utf-8'
            return resp.text

        raise ValueError("No Wayback Machine snapshot available")

    def _parse_html_with_newspaper(self, url: str, html: str) -> dict:
        """Parse downloaded HTML using Newspaper3k and return extracted fields."""
        article = Article(url, config=self.config)
        article.set_html(html)
        article.parse()

        try:
            article.nlp()
            summary = article.summary or ''
        except Exception:
            summary = ''

        return {
            'title': article.title or '',
            'text': article.text or '',
            'authors': article.authors or [],
            'publish_date': article.publish_date.strftime('%Y-%m-%d') if article.publish_date else None,
            'top_image': article.top_image or '',
            'summary': summary,
        }

    def extract_from_url(self, url: str) -> dict:
        """
        Extract article content from a URL with multiple fallback strategies.

        Returns:
            Dictionary with success, title, text, authors, publish_date,
            top_image, summary, source_url, error, and fetch_method.
        """
        result = {
            'success': False,
            'title': '',
            'text': '',
            'authors': [],
            'publish_date': None,
            'top_image': '',
            'summary': '',
            'source_url': url,
            'error': '',
            'fetch_method': '',
        }

        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        errors_log = []

        # ── Strategy 1: Newspaper3k direct download ──
        try:
            article = Article(url, config=self.config)
            article.download()
            article.parse()
            try:
                article.nlp()
                result['summary'] = article.summary or ''
            except Exception:
                pass

            result['title'] = article.title or ''
            result['text'] = article.text or ''
            result['authors'] = article.authors or []
            result['top_image'] = article.top_image or ''
            if article.publish_date:
                result['publish_date'] = article.publish_date.strftime('%Y-%m-%d')

            if result['text'] and len(result['text'].strip()) > 50:
                result['success'] = True
                result['fetch_method'] = 'Direct download'
                return result
            else:
                errors_log.append("Newspaper3k: extracted text too short")
        except Exception as e:
            errors_log.append(f"Newspaper3k: {e}")

        # ── Strategy 2: Manual requests download → Newspaper3k parse ──
        try:
            html = self._download_with_requests(url)
            parsed = self._parse_html_with_newspaper(url, html)
            if parsed['text'] and len(parsed['text'].strip()) > 50:
                result.update(parsed)
                result['source_url'] = url
                result['success'] = True
                result['fetch_method'] = 'Requests fallback'
                return result
            else:
                errors_log.append("Requests fallback: extracted text too short")
        except Exception as e:
            errors_log.append(f"Requests fallback: {e}")

        # ── Strategy 3: Google Webcache ──
        try:
            html = self._try_google_cache(url)
            parsed = self._parse_html_with_newspaper(url, html)
            if parsed['text'] and len(parsed['text'].strip()) > 50:
                result.update(parsed)
                result['source_url'] = url
                result['success'] = True
                result['fetch_method'] = 'Google Cache'
                return result
            else:
                errors_log.append("Google Cache: extracted text too short")
        except Exception as e:
            errors_log.append(f"Google Cache: {e}")

        # ── Strategy 4: Wayback Machine ──
        try:
            html = self._try_wayback_machine(url)
            parsed = self._parse_html_with_newspaper(url, html)
            if parsed['text'] and len(parsed['text'].strip()) > 50:
                result.update(parsed)
                result['source_url'] = url
                result['success'] = True
                result['fetch_method'] = 'Wayback Machine'
                return result
            else:
                errors_log.append("Wayback Machine: extracted text too short")
        except Exception as e:
            errors_log.append(f"Wayback Machine: {e}")

        # ── All strategies failed ──
        is_auth_error = any(
            term in str(errors_log).lower()
            for term in ['401', '403', 'forbidden', 'unauthorized', 'captcha']
        )

        if is_auth_error:
            result['error'] = (
                'This website blocks automated access (common for paywalled '
                'sites like Reuters, NYT, WSJ). Please copy-paste the article '
                'text directly using the "📝 Paste Article Text" mode in the sidebar.'
            )
        else:
            result['error'] = (
                'Could not extract article text after trying multiple methods. '
                'The site may require JavaScript or login. '
                'Please copy-paste the article text directly instead.'
            )

        return result

    def validate_url(self, url: str) -> bool:
        """Quick validation to check if a URL is accessible."""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            response = self.session.head(url, timeout=10, allow_redirects=True)
            return response.status_code < 400
        except Exception:
            return False
