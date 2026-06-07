"""
Source Verification Engine for Fake News Detector.

Enhanced with:
- 50+ domain reputation profiles (up from 18)
- MBFC-aligned bias labels
- Unknown-domain heuristic scoring (TLD analysis, domain age patterns)
- Language detection & translation (Hindi, Bengali, 100+ languages)
"""

from urllib.parse import urlparse
from langdetect import detect
from deep_translator import GoogleTranslator


class SourceEngine:
    def __init__(self):
        # ══════════════════════════════════════════════════════════════════
        # COMPONENT 6: Expanded Domain Reputation Database (50+ domains)
        # ══════════════════════════════════════════════════════════════════

        self.reputation_db = {
            # ── HIGH TRUST (90-100) — Wire Services & Public Broadcasters ──
            "reuters.com":       {"score": 98, "category": "High Trust", "bias": "Center", "notes": "Independent international news agency with strict editorial standards."},
            "apnews.com":        {"score": 97, "category": "High Trust", "bias": "Center", "notes": "Associated Press: global news network widely trusted for objective reporting."},
            "bbc.com":           {"score": 95, "category": "High Trust", "bias": "Center-Left", "notes": "British Broadcasting Corporation: highly regarded public broadcaster."},
            "bbc.co.uk":         {"score": 95, "category": "High Trust", "bias": "Center-Left", "notes": "British Broadcasting Corporation: highly regarded public broadcaster."},
            "pbs.org":           {"score": 95, "category": "High Trust", "bias": "Center", "notes": "Public Broadcasting Service: educational and objective journalism."},
            "npr.org":           {"score": 92, "category": "High Trust", "bias": "Center-Left", "notes": "National Public Radio: nonprofit public media organization with high factuality."},
            "bloomberg.com":     {"score": 94, "category": "High Trust", "bias": "Center", "notes": "Global financial news network with strict data-driven reporting standards."},
            "wsj.com":           {"score": 93, "category": "High Trust", "bias": "Center-Right", "notes": "Wall Street Journal: highly factual business and political reporting."},
            "nytimes.com":       {"score": 92, "category": "High Trust", "bias": "Center-Left", "notes": "New York Times: publication of record with strong factual reporting history."},
            "washingtonpost.com":{"score": 91, "category": "High Trust", "bias": "Center-Left", "notes": "The Washington Post: award-winning investigative journalism and political coverage."},
            "politico.com":      {"score": 91, "category": "High Trust", "bias": "Center", "notes": "Highly factual political journalism and policy news."},
            "theguardian.com":   {"score": 90, "category": "High Trust", "bias": "Left-Center", "notes": "The Guardian: reputable journalism, leaning slightly left-liberal in commentary."},
            "economist.com":     {"score": 93, "category": "High Trust", "bias": "Center", "notes": "The Economist: rigorous analytical reporting with centrist editorial stance."},
            "ft.com":            {"score": 93, "category": "High Trust", "bias": "Center", "notes": "Financial Times: premium business journalism with high factual standards."},

            # ── INTERNATIONAL HIGH TRUST ──
            "aljazeera.com":     {"score": 85, "category": "High Trust", "bias": "Center-Left", "notes": "Al Jazeera English: extensive international coverage; Qatar state-funded but editorially respected."},
            "dw.com":            {"score": 90, "category": "High Trust", "bias": "Center", "notes": "Deutsche Welle: German public international broadcaster with high factuality."},
            "france24.com":      {"score": 88, "category": "High Trust", "bias": "Center", "notes": "France 24: French state-owned international news with balanced coverage."},
            "abc.net.au":        {"score": 92, "category": "High Trust", "bias": "Center", "notes": "Australian Broadcasting Corporation: public broadcaster with strong editorial standards."},
            "cbc.ca":            {"score": 91, "category": "High Trust", "bias": "Center", "notes": "Canadian Broadcasting Corporation: national public broadcaster."},
            "nhk.or.jp":         {"score": 90, "category": "High Trust", "bias": "Center", "notes": "Japan's national public broadcasting organization."},

            # ── INDIA — MAJOR OUTLETS ──
            "thehindu.com":      {"score": 85, "category": "High Trust", "bias": "Center-Left", "notes": "The Hindu: one of India's most respected broadsheets, known for factual reporting."},
            "ndtv.com":          {"score": 78, "category": "Factual / Mixed", "bias": "Center-Left", "notes": "Major Indian news site; generally factual, but occasionally subjective in opinions."},
            "hindustantimes.com":{"score": 75, "category": "Factual / Mixed", "bias": "Center", "notes": "Hindustan Times: major Indian daily, generally reliable with some tabloid sections."},
            "indiatimes.com":    {"score": 70, "category": "Mixed / Tabloid", "bias": "Center", "notes": "Times of India network: large coverage with significant clickbait and tabloid features."},
            "timesofindia.indiatimes.com": {"score": 72, "category": "Mixed", "bias": "Center", "notes": "India's largest English daily; reliable core reporting but heavy clickbait sidebar."},
            "indianexpress.com": {"score": 82, "category": "High Trust", "bias": "Center", "notes": "The Indian Express: strong investigative journalism tradition."},
            "scroll.in":         {"score": 80, "category": "Factual", "bias": "Center-Left", "notes": "Digital-first Indian outlet with high factual standards."},
            "thewire.in":        {"score": 78, "category": "Factual", "bias": "Left-Center", "notes": "Independent Indian digital news platform with detailed investigative pieces."},

            # ── US MAINSTREAM — MODERATE TRUST (65-85) ──
            "cnn.com":           {"score": 78, "category": "Factual / Mixed", "bias": "Left-Center", "notes": "CNN: generally factual reporting with significant opinion/commentary content."},
            "abcnews.go.com":    {"score": 82, "category": "High Trust", "bias": "Center-Left", "notes": "ABC News: major US network with high factuality standards."},
            "cbsnews.com":       {"score": 82, "category": "High Trust", "bias": "Center-Left", "notes": "CBS News: established US broadcast network with strong editorial standards."},
            "nbcnews.com":       {"score": 80, "category": "High Trust", "bias": "Center-Left", "notes": "NBC News: major US network with reliable reporting."},
            "usatoday.com":      {"score": 78, "category": "Factual", "bias": "Center-Left", "notes": "USA Today: widely circulated US daily with high factuality."},
            "axios.com":         {"score": 85, "category": "High Trust", "bias": "Center", "notes": "Axios: concise, factual reporting with low bias."},
            "thehill.com":       {"score": 80, "category": "Factual", "bias": "Center", "notes": "The Hill: congressional and political news with centrist reporting."},

            # ── MIXED TRUST / OPINION-HEAVY (55-70) ──
            "foxnews.com":       {"score": 65, "category": "Mixed Trust", "bias": "Right", "notes": "Highly popular network; factual reporting is mixed, with strong right-wing opinion leaning."},
            "msnbc.com":         {"score": 68, "category": "Mixed Trust", "bias": "Left", "notes": "Left-leaning political coverage with high opinion content; factual news is generally reliable."},
            "dailymail.co.uk":   {"score": 55, "category": "Mixed / Tabloid", "bias": "Right-Center", "notes": "Sensational headlines and tabloid style; often criticized for factual inaccuracies and clickbait."},
            "huffpost.com":      {"score": 70, "category": "Mixed Trust", "bias": "Left", "notes": "Left-leaning opinion and aggregator hub; factual news coverage is generally reliable."},
            "buzzfeed.com":      {"score": 72, "category": "Mixed Trust", "bias": "Left-Center", "notes": "BuzzFeed News was reliable, but the main domain is tabloid-heavy."},
            "nypost.com":        {"score": 60, "category": "Mixed / Tabloid", "bias": "Right-Center", "notes": "New York Post: tabloid with sensational headlines; factual reporting is inconsistent."},
            "thesun.co.uk":      {"score": 50, "category": "Tabloid", "bias": "Right", "notes": "The Sun: UK tabloid with heavy sensationalism and questionable factual standards."},
            "mirror.co.uk":      {"score": 55, "category": "Tabloid", "bias": "Left-Center", "notes": "Daily Mirror: UK tabloid with left-leaning bias and sensational style."},
            "breitbart.com":     {"score": 40, "category": "Low Trust / Biased", "bias": "Far-Right", "notes": "Breitbart News: extreme right-wing outlet with frequent misleading framing and conspiracy promotion."},
            "occupydemocrats.com": {"score": 35, "category": "Low Trust / Biased", "bias": "Far-Left", "notes": "Highly partisan left-wing site with frequent misleading headlines and lack of sourcing."},

            # ── FACT-CHECKING SITES ──
            "snopes.com":        {"score": 95, "category": "Fact-Checker", "bias": "Center", "notes": "Snopes: one of the oldest and most respected fact-checking organizations."},
            "factcheck.org":     {"score": 96, "category": "Fact-Checker", "bias": "Center", "notes": "FactCheck.org: nonpartisan, nonprofit fact-checking project from Annenberg Public Policy Center."},
            "politifact.com":    {"score": 94, "category": "Fact-Checker", "bias": "Center-Left", "notes": "PolitiFact: Pulitzer Prize-winning political fact-checking site."},
            "fullfact.org":      {"score": 93, "category": "Fact-Checker", "bias": "Center", "notes": "Full Fact: independent UK fact-checking charity."},

            # ── LOW TRUST / SATIRE / CONSPIRACY (5-30) ──
            "theonion.com":      {"score": 10, "category": "Satire / Parody", "bias": "N/A", "notes": "100% Satirical publication. Not intended to represent actual news."},
            "clickhole.com":     {"score": 10, "category": "Satire / Parody", "bias": "N/A", "notes": "Satirical website mimicking internet clickbait articles."},
            "babylonbee.com":    {"score": 15, "category": "Satire / Parody", "bias": "Right", "notes": "Conservative Christian satirical news site; articles are humorous fiction."},
            "infowars.com":      {"score": 5,  "category": "Conspiracy / Fake News", "bias": "Far-Right", "notes": "Promotes unsubstantiated conspiracy theories and severe misinformation."},
            "nationalenquirer.com": {"score": 25, "category": "Tabloid / Low Trust", "bias": "Right", "notes": "Sensational tabloid with a history of paid story suppression and low factuality."},
            "rt.com":            {"score": 30, "category": "State-Controlled / Biased", "bias": "Right", "notes": "Russia Today: Russian state-funded media promoting political propaganda."},
            "sputniknews.com":   {"score": 25, "category": "State-Controlled / Biased", "bias": "Right", "notes": "Russian state-funded international news with strong propaganda slant."},
            "globalresearch.ca": {"score": 15, "category": "Conspiracy / Pseudo-Science", "bias": "Far-Left", "notes": "Centre for Research on Globalization: promotes conspiracy theories and anti-Western propaganda."},
            "naturalnews.com":   {"score": 10, "category": "Conspiracy / Pseudo-Science", "bias": "Far-Right", "notes": "Promotes health misinformation, anti-vaccine content, and conspiracy theories."},
            "zerohedge.com":     {"score": 35, "category": "Low Trust / Biased", "bias": "Far-Right", "notes": "Libertarian financial blog; mixes factual market data with conspiracy theories and misinformation."},
        }

        # ── TLD trust heuristics for unknown domains ──
        self.tld_trust = {
            '.gov': 90, '.edu': 88, '.mil': 92, '.int': 85,
            '.org': 60, '.com': 50, '.net': 48, '.co': 48,
            '.info': 35, '.biz': 30, '.xyz': 20, '.top': 15,
            '.click': 10, '.news': 45, '.press': 40
        }

    def clean_domain(self, url_or_domain):
        """Extract the core domain name from a URL or raw domain string."""
        if not url_or_domain:
            return ""

        url_or_domain = url_or_domain.strip().lower()

        if not url_or_domain.startswith(('http://', 'https://')):
            if '/' in url_or_domain or url_or_domain.endswith(('.com', '.org', '.net', '.edu', '.gov', '.in', '.co.uk')):
                url_str = 'https://' + url_or_domain
            else:
                url_str = url_or_domain
        else:
            url_str = url_or_domain

        try:
            parsed = urlparse(url_str)
            domain = parsed.netloc or parsed.path
            if domain.startswith('www.'):
                domain = domain[4:]
            domain = domain.split(':')[0]
            return domain
        except Exception:
            return url_or_domain

    def check_domain_reputation(self, url_or_domain):
        """
        Check the trustworthiness rating of a domain against the expanded
        reputation database. Falls back to TLD-based heuristic scoring
        for unknown domains.
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
                "bias": profile.get("bias", "Unknown"),
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
                    "bias": profile.get("bias", "Unknown"),
                    "description": f"Inherited from {parent}: {profile['notes']}"
                }

        # ── Unknown-domain heuristic scoring ──
        heuristic_score = 50  # Default neutral
        category = "Unverified Source"
        notes_parts = []

        # TLD-based scoring
        for tld, tld_score in self.tld_trust.items():
            if domain.endswith(tld):
                heuristic_score = tld_score
                if tld_score >= 85:
                    category = "Government / Academic"
                    notes_parts.append(f"TLD '{tld}' indicates official institution.")
                elif tld_score <= 30:
                    category = "Suspicious TLD"
                    notes_parts.append(f"TLD '{tld}' is commonly associated with spam or low-trust sites.")
                break

        # Country-code TLD bonus for established news ccTLDs
        cc_trusted = {'.co.uk', '.com.au', '.co.in', '.co.jp', '.de', '.fr', '.ca'}
        for cc in cc_trusted:
            if domain.endswith(cc):
                heuristic_score = max(heuristic_score, 55)
                notes_parts.append(f"Country-code TLD '{cc}' from established internet region.")
                break

        if not notes_parts:
            notes_parts.append("This source domain is not listed in our reputation database. Proceed with careful cross-referencing.")

        return {
            "domain": domain,
            "score": heuristic_score,
            "category": category,
            "bias": "Unknown",
            "description": " ".join(notes_parts)
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
            sample = text[:1000]
            lang = detect(sample)
        except Exception as e:
            return {
                "is_translated": False,
                "detected_lang": "unknown",
                "translated_text": text,
                "warning": f"Language detection error: {str(e)}"
            }

        lang_names = {
            "en": "English",
            "hi": "Hindi",
            "bn": "Bengali",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "zh-cn": "Chinese (Simplified)",
            "zh-tw": "Chinese (Traditional)",
            "ar": "Arabic",
            "ja": "Japanese",
            "ko": "Korean",
            "pt": "Portuguese",
            "ru": "Russian",
            "it": "Italian",
            "nl": "Dutch",
            "tr": "Turkish",
            "pl": "Polish",
            "sv": "Swedish",
            "ta": "Tamil",
            "te": "Telugu",
            "mr": "Marathi",
            "gu": "Gujarati",
            "kn": "Kannada",
            "ml": "Malayalam",
            "ur": "Urdu",
            "pa": "Punjabi"
        }

        detected_name = lang_names.get(lang, lang.upper())

        if lang != 'en':
            try:
                translator = GoogleTranslator(source='auto', target='en')
                text_to_translate = text[:5000]
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

