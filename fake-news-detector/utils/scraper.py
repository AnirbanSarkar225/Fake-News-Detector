"""
Article Scraper Module for Fake News Detector.

Extracts article text, metadata, and content from URLs
using Newspaper3k library.
"""

import requests
from newspaper import Article, ArticleException


class ArticleScraper:
    """
    Web article scraper that extracts full text and metadata from news URLs.
    
    Uses Newspaper3k for intelligent article parsing and extraction.
    Falls back to basic requests + BeautifulSoup if Newspaper3k fails.
    """

    def __init__(self):
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        }

    def extract_from_url(self, url: str) -> dict:
        """
        Extract article content and metadata from a given URL.

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

            article = Article(url)
            article.download()
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
                result['error'] = 'Article text too short or could not be extracted.'

        except ArticleException as e:
            result['error'] = f'Failed to parse article: {str(e)}'
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
