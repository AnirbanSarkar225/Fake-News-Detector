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
            r'\b(breaking|exclusive|shocking|urgent|alert)\b',
            r'\b(exposed|revealed|leaked|secret|coverup|cover-up)\b',
            r'\b(you won\'t believe|they don\'t want you to know)\b',
            r'\b(mainstream media|msm|fake news|deep state)\b',
            r'\b(miracle|cure-all|conspiracy|hoax)\b',
            r'!{2,}',
            r'\?{2,}',
            r'[A-Z]{5,}',
        ]

        self.credibility_patterns = [
            r'\b(according to|study finds|research shows|data suggests)\b',
            r'\b(university|institute|journal|peer-reviewed)\b',
            r'\b(official|spokesperson|statement|confirmed)\b',
            r'\b(evidence|analysis|statistics|survey)\b',
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
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results['suspicious_patterns'].extend(matches)

        for pattern in self.credibility_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results['credibility_indicators'].extend(matches)

        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            results['caps_ratio'] = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)

        results['exclamation_count'] = text.count('!')
        results['question_count'] = text.count('?')

        word_count = max(len(text.split()), 1)

        sensational_density = len(results['suspicious_patterns']) / word_count
        results['sensationalism_score'] = min(sensational_density * 100, 1.0)

        credibility_density = len(results['credibility_indicators']) / word_count
        results['credibility_score'] = min(credibility_density * 100, 1.0)

        if results['caps_ratio'] > 0.3:
            results['sensationalism_score'] = min(results['sensationalism_score'] + 0.2, 1.0)
        if results['exclamation_count'] > 3:
            results['sensationalism_score'] = min(results['sensationalism_score'] + 0.1, 1.0)

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
            if re.search(pattern, sentence, re.IGNORECASE):
                score += 0.15

        alpha_chars = [c for c in sentence if c.isalpha()]
        if alpha_chars:
            caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if caps_ratio > 0.4:
                score += 0.2

        for pattern in self.credibility_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                score -= 0.1

        if sentence.count('!') > 1:
            score += 0.1
        if sentence.count('?') > 2:
            score += 0.05

        vague_patterns = [r'\b(some people say|many believe|sources say|experts claim)\b']
        for pattern in vague_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                score += 0.15

        return max(0.0, min(score, 1.0))
