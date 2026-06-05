import re
from urllib.parse import urlparse
from langdetect import detect
from deep_translator import GoogleTranslator

class SourceEngine:
    def __init__(self):
        # Database of domain trust profiles
        self.reputation_db = {
            # High Trust (Mainstream / Verified News Agencies)
            "reuters.com": {"score": 98, "category": "High Trust", "notes": "Independent international news agency with strict editorial standards."},
            "apnews.com": {"score": 97, "category": "High Trust", "notes": "Associated Press: global news network widely trusted for objective reporting."},
            "bbc.com": {"score": 95, "category": "High Trust", "notes": "British Broadcasting Corporation: highly regarded public broadcaster."},
            "bbc.co.uk": {"score": 95, "category": "High Trust", "notes": "British Broadcasting Corporation: highly regarded public broadcaster."},
            "nytimes.com": {"score": 92, "category": "High Trust", "notes": "New York Times: publication of record with strong factual reporting history."},
            "wsj.com": {"score": 93, "category": "High Trust", "notes": "Wall Street Journal: highly factual business and political reporting."},
            "bloomberg.com": {"score": 94, "category": "High Trust", "notes": "Global financial news network with strict data-driven reporting standards."},
            "npr.org": {"score": 92, "category": "High Trust", "notes": "National Public Radio: nonprofit public media organization with high factuality."},
            "theguardian.com": {"score": 90, "category": "High Trust", "notes": "The Guardian: reputable journalism, leaning slightly left-liberal in commentary."},
            "politico.com": {"score": 91, "category": "High Trust", "notes": "Highly factual political journalism and policy news."},
            "pbs.org": {"score": 95, "category": "High Trust", "notes": "Public Broadcasting Service: educational and objective journalism."},
            
            # Medium Trust / Mixed Factuality / Tabloids / Leaning Opinions
            "foxnews.com": {"score": 65, "category": "Mixed Trust", "notes": "Highly popular network; factual reporting is mixed, with strong right-wing opinion leaning."},
            "msnbc.com": {"score": 68, "category": "Mixed Trust", "notes": "Left-leaning political coverage with high opinion content; factual news is generally reliable."},
            "dailymail.co.uk": {"score": 55, "category": "Mixed / Tabloid", "notes": "Sensational headlines and tabloid style; often criticized for factual inaccuracies and clickbait."},
            "huffpost.com": {"score": 70, "category": "Mixed Trust", "notes": "Left-leaning opinion and aggregator hub; factual news coverage is generally reliable."},
            "buzzfeed.com": {"score": 72, "category": "Mixed Trust", "notes": "BuzzFeed News (separate from entertainment) is generally reliable, but the main domain is tabloid-heavy."},
            "ndtv.com": {"score": 78, "category": "Factual / Mixed", "notes": "Major Indian news site; generally factual, but occasionally subjective in opinions."},
            "indiatimes.com": {"score": 70, "category": "Mixed / Tabloid", "notes": "Times of India network: large coverage with significant clickbait and tabloid features."},
            
            # Low Trust / Satire / Known Unreliable / Conspiracy
            "theonion.com": {"score": 10, "category": "Satire / Parody", "notes": "100% Satirical publication. Not intended to represent actual news."},
            "clickhole.com": {"score": 10, "category": "Satire / Parody", "notes": "Satirical website mimicking internet clickbait articles."},
            "babylonbee.com": {"score": 15, "category": "Satire / Parody", "notes": "Conservative Christian satirical news site; articles are humorous fiction."},
            "infowars.com": {"score": 5, "category": "Conspiracy / Fake News", "notes": "Promotes unsubstantiated conspiracy theories and severe misinformation."},
            "nationalenquirer.com": {"score": 25, "category": "Tabloid / Low Trust", "notes": "Sensational tabloid with a history of paid story suppression and low factuality."},
            "rt.com": {"score": 30, "category": "State-Controlled / Biased", "notes": "Russia Today: Russian state-funded media promoting political propaganda."}
        }

    def clean_domain(self, url_or_domain):
        """
        Extract the core domain name from a URL or raw domain string.
        """
        if not url_or_domain:
            return ""
        
        # Remove whitespace
        url_or_domain = url_or_domain.strip().lower()
        
        # Add scheme if missing to allow urlparse to work correctly
        if not url_or_domain.startswith(('http://', 'https://')):
            # Check if it looks like a URL
            if '/' in url_or_domain or url_or_domain.endswith(('.com', '.org', '.net', '.edu', '.gov', '.in', '.co.uk')):
                url_str = 'https://' + url_or_domain
            else:
                url_str = url_or_domain
        else:
            url_str = url_or_domain
            
        try:
            parsed = urlparse(url_str)
            domain = parsed.netloc or parsed.path
            # Remove www.
            if domain.startswith('www.'):
                domain = domain[4:]
            # Remove port if present
            domain = domain.split(':')[0]
            return domain
        except Exception:
            return url_or_domain

    def check_domain_reputation(self, url_or_domain):
        """
        Check the trustworthiness rating of a domain against our reputation database.
        """
        domain = self.clean_domain(url_or_domain)
        if not domain:
            return None
            
        # Try exact match
        if domain in self.reputation_db:
            profile = self.reputation_db[domain]
            return {
                "domain": domain,
                "score": profile["score"],
                "category": profile["category"],
                "description": profile["notes"]
            }
            
        # Try parent domain match (e.g. news.bbc.co.uk -> bbc.co.uk)
        parts = domain.split('.')
        for i in range(len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in self.reputation_db:
                profile = self.reputation_db[parent]
                return {
                    "domain": domain,
                    "score": profile["score"],
                    "category": profile["category"],
                    "description": f"Inherited from {parent}: {profile['notes']}"
                }
                
        # Default unverified profile
        return {
            "domain": domain,
            "score": 50,
            "category": "Unverified Source",
            "description": "This source domain is not listed in our reputation database. Proceed with careful cross-referencing."
        }

    def detect_and_translate(self, text):
        """
        Detect language using langdetect. If non-English, translate to English.
        Returns a dict: {"is_translated": bool, "detected_lang": str, "translated_text": str, "warning": str}
        """
        if not text or len(text.strip()) < 10:
            return {
                "is_translated": False,
                "detected_lang": "en",
                "translated_text": text,
                "warning": ""
            }
            
        try:
            # We take a sample of the text for faster language detection
            sample = text[:1000]
            lang = detect(sample)
        except Exception as e:
            # Fallback to English on error
            return {
                "is_translated": False,
                "detected_lang": "unknown",
                "translated_text": text,
                "warning": f"Language detection error: {str(e)}"
            }
            
        # Supported translation mapping (specifically supporting Hindi 'hi' and Bengali 'bn')
        lang_names = {
            "en": "English",
            "hi": "Hindi",
            "bn": "Bengali",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "zh-cn": "Chinese (Simplified)",
            "ar": "Arabic"
        }
        
        detected_name = lang_names.get(lang, lang.upper())
        
        if lang != 'en':
            try:
                # Use GoogleTranslator (deep-translator) to translate to English
                # It handles long texts in chunks automatically
                translator = GoogleTranslator(source='auto', target='en')
                # For very long articles, deep-translator might have issues, so we limit size
                text_to_translate = text[:5000] # safe limit
                translated_text = translator.translate(text_to_translate)
                
                warning_msg = ""
                if len(text) > 5000:
                    warning_msg = "⚠️ Article is very long; only the first 5,000 characters were translated for credibility analysis."
                    
                return {
                    "is_translated": True,
                    "detected_lang": lang,
                    "detected_lang_name": detected_name,
                    "translated_text": translated_text,
                    "warning": warning_msg
                }
            except Exception as e:
                return {
                    "is_translated": False,
                    "detected_lang": lang,
                    "detected_lang_name": detected_name,
                    "translated_text": text,
                    "warning": f"⚠️ Failed to translate from {detected_name} to English: {str(e)}. Analyzing original text."
                }
                
        return {
            "is_translated": False,
            "detected_lang": "en",
            "detected_lang_name": "English",
            "translated_text": text,
            "warning": ""
        }
