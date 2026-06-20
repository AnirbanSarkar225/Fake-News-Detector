"""
Text Preprocessing Module for Fake News Detector.

Handles all text cleaning, normalization, and feature extraction
required for the classification pipeline.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

def ensure_nltk_data():
    """Download required NLTK datasets if not already available."""
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


class TextPreprocessor:
    """
    Comprehensive text preprocessor for news article classification.
    
    Handles:
    - HTML tag removal
    - URL removal
    - Special character cleaning
    - Stopword removal
    - Lemmatization
    - Sentence-level tokenization for explainability
    """

    def __init__(self):
        ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

        self.sensational_patterns = [
            re.compile(r'\b(?:breaking|exclusive|shocking|urgent|alert)\b', re.IGNORECASE),
            re.compile(r'\b(?:exposed|revealed|leaked|secret|coverup|cover-up)\b', re.IGNORECASE),
            re.compile(r'\b(?:you won\'t believe|they don\'t want you to know)\b', re.IGNORECASE),
            re.compile(r'\b(?:mainstream media|msm|fake news|deep state)\b', re.IGNORECASE),
            re.compile(r'\b(?:miracle|cure-all|conspiracy|hoax)\b', re.IGNORECASE),
            re.compile(r'!{2,}'),
            re.compile(r'\?{2,}'),
            re.compile(r'\b[A-Z]{5,}\b'), # Case-sensitive: only matches actual ALL-CAPS words!
        ]

        self.credibility_patterns = [
            # ── Attribution & sourcing (multi-word phrases only) ──
            re.compile(r'\b(?:according to|study finds|research shows|data suggests)\b', re.IGNORECASE),
            re.compile(r'\b(?:reported by|as reported|sources say|sources said)\b', re.IGNORECASE),
            re.compile(r'\b(?:stated that|said that|noted that|added that|explained that)\b', re.IGNORECASE),
            re.compile(r'\b(?:told reporters|in a statement|press conference|press release)\b', re.IGNORECASE),
            re.compile(r'\b(?:expected to|is expected|are expected|was expected)\b', re.IGNORECASE),

            # ── Academic & research (specific terms) ──
            re.compile(r'\b(?:peer-reviewed|published in|university|institute|laboratory)\b', re.IGNORECASE),
            re.compile(r'\b(?:spokesperson|confirmed|acknowledged)\b', re.IGNORECASE),

            # ── Specific named institutions ──
            re.compile(r'\b(?:United Nations|NATO|EU|ASEAN|G7|G20)\b', re.IGNORECASE),
            re.compile(r'\b(?:FDA|WHO|CDC|NIH)\b', re.IGNORECASE),
            re.compile(r'\b(?:Reuters|Associated Press|AFP)\b', re.IGNORECASE),

            # ── Quantitative signals (specific formats) ──
            re.compile(r'\b(?:percent|percentage|\d+%)\b', re.IGNORECASE),
            re.compile(r'\b(?:billion|million)\b', re.IGNORECASE),

            # ── India-specific named institutions ──
            re.compile(r'\b(?:Lok Sabha|Rajya Sabha|Vidhan Sabha|Parliament of India)\b', re.IGNORECASE),
            re.compile(r'\b(?:Supreme Court|High Court|NCLAT)\b', re.IGNORECASE),
            re.compile(r'\b(?:Prime Minister|Chief Minister|President of India)\b', re.IGNORECASE),
            re.compile(r'\b(?:ISRO|DRDO|BARC|ICAR|CSIR|IIT|IIM|AIIMS)\b', re.IGNORECASE),
            re.compile(r'\b(?:RBI|SEBI|NITI Aayog|CAG|CBI|NIA)\b', re.IGNORECASE),
            re.compile(r'\b(?:BCCI|IPL)\b', re.IGNORECASE),
            re.compile(r'\b(?:PTI|ANI|PIB|Doordarshan|All India Radio|Prasar Bharati)\b', re.IGNORECASE),
            re.compile(r'\b(?:crore|lakh)\b', re.IGNORECASE),
            re.compile(r'\b(?:Aadhaar|UPI|GST|NEET|JEE|UPSC)\b', re.IGNORECASE),
        ]

    def clean_text(self, text: str) -> str:
        """
        Clean raw text by removing noise while preserving meaningful content.

        Args:
            text: Raw article text

        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove publisher/location headers at start (e.g. "WASHINGTON (Reuters) - ", "New Delhi (PTI) : ", "MUMBAI - ")
        # Limits location names to 2-30 characters and parenthesized agency names to 2-15 characters to prevent false positives.
        text = re.sub(r'^\s*(?:[A-Za-z\s,/]{2,30}(?:\s*\([A-Za-z\s\./]{2,15}\))?|[A-Za-z\s,/]{2,30}\((?:Reuters|AFP|AP|UPI|Bloomberg|PTI|ANI|IANS|Reuters\.com)\))\s*(?:-+|:|–|—)\s*', '', text)
        text = re.sub(r'^\s*(?:Reuters|AFP|AP|Bloomberg|Associated Press|Press Trust of India|PTI|ANI|IANS)\b\s*(?:-+|:|–|—)?\s*', '', text, flags=re.IGNORECASE)

        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def preprocess_for_model(self, text: str) -> str:
        """
        Full preprocessing pipeline for model input.
        
        Applies cleaning, lowercasing, and stopword removal.

        Args:
            text: Raw article text

        Returns:
            Preprocessed text ready for TF-IDF vectorization
        """
        text = self.clean_text(text)
        text = text.lower()
        # Fast regex tokenization: matches words of length 3 or more
        tokens = re.findall(r'\b[a-z]{3,}\b', text)
        # Fast stopword filtering
        tokens = [t for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)

    def get_sentences(self, text: str) -> list:
        """
        Split text into individual sentences for explainability analysis.

        Args:
            text: Article text

        Returns:
            List of sentence strings
        """
        text = self.clean_text(text)
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            sentences = sent_tokenize(text)

        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def analyze_suspicious_indicators(self, text: str) -> dict:
        """
        Analyze text for patterns commonly associated with misinformation.

        Returns a dictionary with scores and matched patterns for explainability.

        Args:
            text: Article text to analyze

        Returns:
            Dictionary with sensationalism score, credibility indicators, and details
        """
        results = {
            'sensationalism_score': 0.0,
            'credibility_score': 0.0,
            'suspicious_patterns': [],
            'credibility_indicators': [],
            'caps_ratio': 0.0,
            'exclamation_count': 0,
            'question_count': 0,
        }

        if not text:
            return results

        for pattern in self.sensational_patterns:
            matches = pattern.findall(text)
            if matches:
                results['suspicious_patterns'].extend([m for m in matches if m])

        for pattern in self.credibility_patterns:
            matches = pattern.findall(text)
            if matches:
                results['credibility_indicators'].extend([m for m in matches if m])

        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            results['caps_ratio'] = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)

        results['exclamation_count'] = text.count('!')
        results['question_count'] = text.count('?')

        word_count = max(len(text.split()), 1)

        sensational_density = len(results['suspicious_patterns']) / word_count
        results['sensationalism_score'] = min(sensational_density * 100, 1.0)

        credibility_density = len(results['credibility_indicators']) / word_count
        results['credibility_score'] = min(credibility_density * 50, 1.0)

        # ── Sensationalism penalties ──
        if results['caps_ratio'] > 0.3:
            results['sensationalism_score'] = min(results['sensationalism_score'] + 0.2, 1.0)
        if results['exclamation_count'] > 3:
            results['sensationalism_score'] = min(results['sensationalism_score'] + 0.1, 1.0)

        # ── Formal writing style bonus ──
        # Neutral tone: low exclamation usage + no ALL-CAPS shouting = credible writing
        if results['exclamation_count'] == 0 and results['caps_ratio'] < 0.12:
            results['credibility_score'] = min(results['credibility_score'] + 0.15, 1.0)
        # Proper sentence structure (avg sentence length > 12 words)
        sentences = text.split('.')
        real_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]
        if real_sentences:
            avg_sent_len = sum(len(s.split()) for s in real_sentences) / len(real_sentences)
            if avg_sent_len > 12:
                results['credibility_score'] = min(results['credibility_score'] + 0.1, 1.0)

        return results

    def score_sentence_suspicion(self, sentence: str) -> float:
        """
        Score an individual sentence for how suspicious it appears.

        Args:
            sentence: Single sentence text

        Returns:
            Suspicion score between 0.0 (credible) and 1.0 (suspicious)
        """
        score = 0.0

        if not sentence:
            return score

        for pattern in self.sensational_patterns:
            if pattern.search(sentence):
                score += 0.15

        alpha_chars = [c for c in sentence if c.isalpha()]
        if alpha_chars:
            caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if caps_ratio > 0.4:
                score += 0.2

        for pattern in self.credibility_patterns:
            if pattern.search(sentence):
                score -= 0.1

        if sentence.count('!') > 1:
            score += 0.1
        if sentence.count('?') > 2:
            score += 0.05

        vague_patterns = [re.compile(r'\b(?:some people say|many believe|sources say|experts claim)\b', re.IGNORECASE)]
        for pattern in vague_patterns:
            if pattern.search(sentence):
                score += 0.15

        return max(0.0, min(score, 1.0))
