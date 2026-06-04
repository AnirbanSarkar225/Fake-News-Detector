"""
Article Scraper Module for Fake News Detector.

Extracts article text, metadata, and content from URLs
using Newspaper3k library with enhanced headers to avoid blocks.
Falls back to requests + BeautifulSoup when Newspaper3k fails.
"""

import requests
from newspaper import Article, ArticleException, Config


class ArticleScraper:
    """
    Web article scraper that extracts full text and metadata from news URLs.
    
    Uses Newspaper3k for intelligent article parsing and extraction.
    Falls back to requests download + Newspaper3k parse if direct download fails.
    """

    # Realistic browser User-Agent to avoid 401/403 blocks from news sites
    USER_AGENT = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/125.0.0.0 Safari/537.36'
    )

    def __init__(self):
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
        }

        # Configure Newspaper3k to use our custom User-Agent
        self.config = Config()
        self.config.browser_user_agent = self.USER_AGENT
        self.config.request_timeout = 20
        self.config.fetch_images = False          # speed up; we don't need images

    def _download_html_manually(self, url: str) -> str:
        """
        Download the raw HTML of a page using requests with full browser headers.
        This avoids the 401/403 errors that newspaper3k's default downloader gets.
        """
        resp = requests.get(url, headers=self.headers, timeout=20, allow_redirects=True)
        resp.raise_for_status()
        return resp.text

    def extract_from_url(self, url: str) -> dict:
        """
        Extract article content and metadata from a given URL.

        Strategy:
          1. Try Newspaper3k with custom config (browser UA).
          2. If that fails (401/403), manually download HTML with requests,
             then feed it to Newspaper3k for parsing.

        Args:
            url: Full URL of the news article

        Returns:
            Dictionary containing:
                - success (bool): Whether extraction succeeded
                - title (str): Article title
                - text (str): Full article text
                - authors (list): List of author names
                - publish_date (str): Publication date
                - top_image (str): URL of the article's main image
                - summary (str): Auto-generated summary
                - source_url (str): The source domain
                - error (str): Error message if failed
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
            'error': ''
        }

        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            article = Article(url, config=self.config)

            # --- Attempt 1: let Newspaper3k download with our custom config ---
            try:
                article.download()
            except Exception:
                # --- Attempt 2: download HTML ourselves, feed to Newspaper3k ---
                html = self._download_html_manually(url)
                article.set_html(html)

            article.parse()

            try:
                article.nlp()
                result['summary'] = article.summary or ''
            except Exception:
                result['summary'] = ''

            result['title'] = article.title or ''
            result['text'] = article.text or ''
            result['authors'] = article.authors or []
            result['top_image'] = article.top_image or ''

            if article.publish_date:
                result['publish_date'] = article.publish_date.strftime('%Y-%m-%d')

            if result['text'] and len(result['text']) > 50:
                result['success'] = True
            else:
                # One more fallback: try manual download if newspaper got no text
                if not result['text'] or len(result['text']) <= 50:
                    try:
                        html = self._download_html_manually(url)
                        article2 = Article(url, config=self.config)
                        article2.set_html(html)
                        article2.parse()
                        if article2.text and len(article2.text) > 50:
                            result['text'] = article2.text
                            result['title'] = article2.title or result['title']
                            result['authors'] = article2.authors or result['authors']
                            if article2.publish_date:
                                result['publish_date'] = article2.publish_date.strftime('%Y-%m-%d')
                            result['success'] = True
                        else:
                            result['error'] = (
                                'Could not extract enough article text. '
                                'The site may use heavy JavaScript rendering. '
                                'Try pasting the article text directly instead.'
                            )
                    except Exception:
                        result['error'] = (
                            'Could not extract enough article text. '
                            'Try pasting the article text directly instead.'
                        )

        except ArticleException as e:
            result['error'] = f'Failed to parse article: {str(e)}'
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 'unknown'
            if status in (401, 403):
                result['error'] = (
                    f'The website blocked automated access (HTTP {status}). '
                    'This is common for paywalled or protected sites like Reuters. '
                    'Please copy-paste the article text directly instead.'
                )
            else:
                result['error'] = f'HTTP error {status} when fetching the article.'
        except requests.exceptions.ConnectionError:
            result['error'] = 'Could not connect to the website. Check the URL and your internet connection.'
        except requests.exceptions.Timeout:
            result['error'] = 'The request timed out. The website may be slow or unresponsive.'
        except requests.exceptions.RequestException as e:
            result['error'] = f'Network error: {str(e)}'
        except Exception as e:
            result['error'] = f'Unexpected error: {str(e)}'

        return result

    def validate_url(self, url: str) -> bool:
        """
        Quick validation to check if a URL is accessible.

        Args:
            url: URL to validate

        Returns:
            True if URL is accessible, False otherwise
        """
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            response = requests.head(
                url,
                headers=self.headers,
                timeout=10,
                allow_redirects=True
            )
            return response.status_code < 400

        except Exception:
            return False
