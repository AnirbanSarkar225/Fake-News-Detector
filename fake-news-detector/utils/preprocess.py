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
            # ── Attribution & sourcing ──
            re.compile(r'\b(?:according to|study finds|research shows|data suggests)\b', re.IGNORECASE),
            re.compile(r'\b(?:reported by|as reported|sources say|sources said)\b', re.IGNORECASE),
            re.compile(r'\b(?:stated that|said that|noted that|added that|explained that)\b', re.IGNORECASE),
            re.compile(r'\b(?:told reporters|in a statement|press conference|press release)\b', re.IGNORECASE),

            # ── Institutional & academic ──
            re.compile(r'\b(?:university|institute|journal|peer-reviewed|laboratory)\b', re.IGNORECASE),
            re.compile(r'\b(?:official|spokesperson|statement|confirmed|acknowledged)\b', re.IGNORECASE),
            re.compile(r'\b(?:evidence|analysis|statistics|survey|findings|concluded)\b', re.IGNORECASE),

            # ── Government & policy ──
            re.compile(r'\b(?:ministry|department|government|federal|parliament|legislature)\b', re.IGNORECASE),
            re.compile(r'\b(?:announced|initiative|program|programme|policy|regulation)\b', re.IGNORECASE),
            re.compile(r'\b(?:scheme|subsidy|budget|allocation|funding|grant)\b', re.IGNORECASE),
            re.compile(r'\b(?:legislation|amendment|bill|act|ordinance|directive)\b', re.IGNORECASE),
            re.compile(r'\b(?:commission|committee|council|authority|agency|bureau)\b', re.IGNORECASE),
            re.compile(r'\b(?:election|ballot|vote|referendum|constituency|polling)\b', re.IGNORECASE),

            # ── Formal reporting verbs ──
            re.compile(r'\b(?:announced|reported|disclosed|released|published|issued)\b', re.IGNORECASE),
            re.compile(r'\b(?:implemented|proposed|approved|authorized|ratified)\b', re.IGNORECASE),
            re.compile(r'\b(?:expected to|is expected|are expected|was expected)\b', re.IGNORECASE),

            # ── Quantitative & factual ──
            re.compile(r'\b(?:percent|percentage|\d+%)\b', re.IGNORECASE),
            re.compile(r'\b(?:billion|million|thousand|quarter|fiscal)\b', re.IGNORECASE),
            re.compile(r'\b(?:increase|decrease|growth|decline|rose|fell)\b', re.IGNORECASE),

            # ── Geographic & organizational ──
            re.compile(r'\b(?:city|state|district|region|country|nation|province)\b', re.IGNORECASE),
            re.compile(r'\b(?:organization|organisation|corporation|company|firm)\b', re.IGNORECASE),

            # ── Science & health ──
            re.compile(r'\b(?:clinical|trial|vaccine|treatment|therapy|diagnosis)\b', re.IGNORECASE),
            re.compile(r'\b(?:patients|symptoms|disease|infection|outbreak|pandemic)\b', re.IGNORECASE),
            re.compile(r'\b(?:researcher|scientist|physician|doctor|surgeon|nurse)\b', re.IGNORECASE),
            re.compile(r'\b(?:hospital|clinic|medical|pharmaceutical|FDA|WHO)\b', re.IGNORECASE),
            re.compile(r'\b(?:study|experiment|published in|lancet|nature|JAMA)\b', re.IGNORECASE),

            # ── Business & finance ──
            re.compile(r'\b(?:revenue|profit|earnings|shares|stock|market)\b', re.IGNORECASE),
            re.compile(r'\b(?:CEO|CFO|chairman|director|executive|management)\b', re.IGNORECASE),
            re.compile(r'\b(?:quarterly|annual|fiscal year|dividend|valuation)\b', re.IGNORECASE),
            re.compile(r'\b(?:acquisition|merger|IPO|investment|venture|startup)\b', re.IGNORECASE),
            re.compile(r'\b(?:inflation|GDP|economy|recession|interest rate|central bank)\b', re.IGNORECASE),

            # ── Technology ──
            re.compile(r'\b(?:launched|unveiled|released|update|version|upgrade)\b', re.IGNORECASE),
            re.compile(r'\b(?:software|hardware|platform|application|device|processor)\b', re.IGNORECASE),
            re.compile(r'\b(?:artificial intelligence|machine learning|cybersecurity|cloud)\b', re.IGNORECASE),
            re.compile(r'\b(?:patent|innovation|prototype|beta|rollout)\b', re.IGNORECASE),

            # ── Sports ──
            re.compile(r'\b(?:scored|defeated|championship|tournament|league|season)\b', re.IGNORECASE),
            re.compile(r'\b(?:coach|manager|captain|player|athlete|team)\b', re.IGNORECASE),
            re.compile(r'\b(?:match|game|final|semifinal|qualifier|fixture)\b', re.IGNORECASE),
            re.compile(r'\b(?:medal|record|Olympic|World Cup|FIFA|UEFA|ICC)\b', re.IGNORECASE),
            re.compile(r'\b(?:innings|wicket|goal|touchdown|set|round)\b', re.IGNORECASE),

            # ── Crime & legal ──
            re.compile(r'\b(?:court|judge|verdict|trial|prosecution|defendant)\b', re.IGNORECASE),
            re.compile(r'\b(?:arrested|charged|convicted|sentenced|investigation)\b', re.IGNORECASE),
            re.compile(r'\b(?:police|detective|officer|sheriff|FBI|enforcement)\b', re.IGNORECASE),
            re.compile(r'\b(?:suspect|witness|testimony|evidence|forensic)\b', re.IGNORECASE),
            re.compile(r'\b(?:lawsuit|hearing|ruling|appeal|bail|parole)\b', re.IGNORECASE),

            # ── International & diplomatic ──
            re.compile(r'\b(?:treaty|summit|bilateral|diplomatic|embassy|consul)\b', re.IGNORECASE),
            re.compile(r'\b(?:United Nations|NATO|EU|ASEAN|G7|G20)\b', re.IGNORECASE),
            re.compile(r'\b(?:sanctions|tariff|trade agreement|ceasefire|peacekeeping)\b', re.IGNORECASE),
            re.compile(r'\b(?:ambassador|diplomat|foreign minister|secretary of state)\b', re.IGNORECASE),

            # ── Weather & environment ──
            re.compile(r'\b(?:forecast|temperature|rainfall|hurricane|cyclone|tornado)\b', re.IGNORECASE),
            re.compile(r'\b(?:flood|drought|wildfire|earthquake|tsunami|eruption)\b', re.IGNORECASE),
            re.compile(r'\b(?:evacuation|advisory|warning|alert issued|emergency)\b', re.IGNORECASE),
            re.compile(r'\b(?:climate|carbon|emissions|renewable|sustainability)\b', re.IGNORECASE),
            re.compile(r'\b(?:meteorological|seismological|conservation|endangered)\b', re.IGNORECASE),
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

        # Remove publisher/location headers like "WASHINGTON (Reuters) - " or "LONDON (AP) - " or "SEOUL - " at start
        text = re.sub(r'^\s*(?:[A-Z\s,/]+(?:\s*\([A-Za-z\s]+\))?|[A-Za-z\s,/]+\((?:Reuters|AFP|AP|UPI|Bloomberg|Reuters\.com)\))\s*-\s*', '', text)
        text = re.sub(r'^\s*(?:Reuters|AFP|AP|Bloomberg|Associated Press)\b\s*(?:-\s*)?', '', text, flags=re.IGNORECASE)

        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def preprocess_for_model(self, text: str) -> str:
        """
        Full preprocessing pipeline for model input.
        
        Applies cleaning, lowercasing, stopword removal, and lemmatization.

        Args:
            text: Raw article text

        Returns:
            Preprocessed text ready for TF-IDF vectorization
        """
        text = self.clean_text(text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))

        try:
            tokens = word_tokenize(text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            tokens = word_tokenize(text)

        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

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
