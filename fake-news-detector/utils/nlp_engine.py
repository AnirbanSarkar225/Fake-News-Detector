"""
NLP Engine for Fake News Detector.

Enhanced with:
- VADER Sentiment Analysis (compound/pos/neg/neutral + fear/anger/joy granularity)
- TextRank Summarization (sentence similarity graph + PageRank scoring)
- Real SHAP Explainability (LinearExplainer for Shapley values)
- spaCy NER Integration (PERSON, ORG, GPE, DATE, MONEY entities)
"""

import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK tokenizers if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# ── Optional imports with graceful fallbacks ──────────────────────────────

# VADER
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False

# SHAP
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# spaCy
try:
    import spacy
    try:
        _spacy_nlp = spacy.load("en_core_web_sm")
        _SPACY_AVAILABLE = True
    except OSError:
        _spacy_nlp = None
        _SPACY_AVAILABLE = False
except ImportError:
    _spacy_nlp = None
    _SPACY_AVAILABLE = False


class NLPEngine:
    def __init__(self):
        # ── VADER Sentiment Analyzer ──
        if _VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None

        # ── Fallback lexicons for emotion granularity ──
        self.fear_words = {
            'panic', 'scared', 'terrified', 'fear', 'afraid', 'deadly', 'threat',
            'warning', 'danger', 'hazard', 'epidemic', 'crisis', 'fatal', 'poison',
            'kill', 'destroy', 'dread', 'terror', 'alarming', 'catastrophe',
            'devastating', 'horrifying', 'nightmare', 'plague', 'lethal', 'peril',
            'menace', 'ominous', 'sinister', 'grim', 'treacherous', 'doom'
        }
        self.anger_words = {
            'rage', 'angry', 'hate', 'furious', 'outrage', 'disgust', 'liar',
            'cheat', 'scam', 'betray', 'offensive', 'shame', 'scandal', 'corrupt',
            'propaganda', 'con', 'conspiracy', 'fake', 'infuriating', 'deplorable',
            'despicable', 'vile', 'atrocious', 'reprehensible', 'treachery',
            'deceitful', 'crooked', 'fraudulent', 'manipulate', 'exploit'
        }
        self.joy_words = {
            'happy', 'joy', 'celebrate', 'wonderful', 'great', 'excellent',
            'amazing', 'fantastic', 'brilliant', 'success', 'triumph', 'victory',
            'delighted', 'thrilled', 'grateful', 'blessed', 'optimistic',
            'hopeful', 'inspiring', 'uplifting', 'remarkable', 'proud'
        }

        # Fallback NER resources
        self.org_suffixes = {
            'Inc', 'Corp', 'Co', 'LLC', 'Group', 'Agency', 'Department',
            'Administration', 'University', 'Foundation', 'Association',
            'Institute', 'Bureau', 'Commission', 'Council', 'Ministry'
        }
        self.loc_keywords = {
            'USA', 'UK', 'China', 'Russia', 'Europe', 'America', 'London',
            'Washington', 'Beijing', 'Moscow', 'Paris', 'Berlin', 'Tokyo',
            'India', 'Canada', 'Australia', 'Brazil', 'Mexico', 'Japan',
            'Germany', 'France', 'Italy', 'Spain', 'Egypt', 'Nigeria',
            'Iran', 'Iraq', 'Syria', 'Ukraine', 'Israel', 'Palestine'
        }

    # ══════════════════════════════════════════════════════════════════════
    # COMPONENT 1: VADER Sentiment Analysis
    # ══════════════════════════════════════════════════════════════════════

    def get_sentiment_metrics(self, text):
        """
        Enhanced sentiment analyzer using VADER compound scores
        plus granular emotion detection (fear, anger, joy).

        Returns:
            dict with keys: compound, positive, negative, neutral,
                            fear, anger, joy, dominant_emotion
        """
        if not text:
            return {
                "compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0,
                "fear": 0.0, "anger": 0.0, "joy": 0.0, "dominant_emotion": "Neutral"
            }

        words = re.findall(r'\b\w+\b', text.lower())
        total_words = max(len(words), 1)

        # ── VADER scores ──
        if self.vader:
            vader_scores = self.vader.polarity_scores(text)
            compound = vader_scores['compound']
            positive = vader_scores['pos']
            negative = vader_scores['neg']
            neutral_vader = vader_scores['neu']
        else:
            # Fallback: simple positive/negative word ratio
            compound = 0.0
            positive = 0.0
            negative = 0.0
            neutral_vader = 1.0

        # ── Granular emotion scores (lexicon-based) ──
        fear_count = sum(1 for w in words if w in self.fear_words)
        anger_count = sum(1 for w in words if w in self.anger_words)
        joy_count = sum(1 for w in words if w in self.joy_words)

        fear_score = min((fear_count / total_words) * 30.0, 1.0)
        anger_score = min((anger_count / total_words) * 30.0, 1.0)
        joy_score = min((joy_count / total_words) * 30.0, 1.0)

        # Normalize emotion scores to sum ≤ 1
        emotion_total = fear_score + anger_score + joy_score
        if emotion_total > 1.0:
            fear_score /= emotion_total
            anger_score /= emotion_total
            joy_score /= emotion_total

        # Determine dominant emotion
        emotions = {"Fear": fear_score, "Anger": anger_score, "Joy": joy_score, "Neutral": neutral_vader}
        dominant = max(emotions, key=emotions.get)

        return {
            "compound": float(round(compound, 3)),
            "positive": float(round(positive, 3)),
            "negative": float(round(negative, 3)),
            "neutral": float(round(neutral_vader, 3)),
            "fear": float(round(fear_score, 3)),
            "anger": float(round(anger_score, 3)),
            "joy": float(round(joy_score, 3)),
            "dominant_emotion": dominant
        }

    # ══════════════════════════════════════════════════════════════════════
    # COMPONENT 2: TextRank Summarization
    # ══════════════════════════════════════════════════════════════════════

    def generate_summary(self, text, max_sentences=3):
        """
        TextRank extractive summarization using sentence similarity graph.

        Builds a cosine similarity matrix between sentence TF-IDF vectors,
        then applies iterative PageRank-like scoring to extract the most
        important sentences.
        """
        if not text or len(text.strip()) == 0:
            return ""

        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        # ── Build TF-IDF sentence vectors ──
        from sklearn.feature_extraction.text import TfidfVectorizer

        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except Exception:
            # Fallback to frequency-based method
            return self._frequency_summary(text, sentences, max_sentences)

        # ── Cosine similarity matrix ──
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()

        # ── PageRank-like iterative scoring ──
        n = len(sentences)
        damping = 0.85
        scores = np.ones(n) / n
        
        # Normalize similarity matrix rows
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        norm_matrix = similarity_matrix / row_sums

        for _ in range(50):  # 50 iterations is typically sufficient
            new_scores = (1 - damping) / n + damping * norm_matrix.T.dot(scores)
            if np.abs(new_scores - scores).sum() < 1e-6:
                break
            scores = new_scores

        # ── Select top sentences in original order ──
        ranked_indices = np.argsort(scores)[::-1][:max_sentences]
        selected = sorted(ranked_indices)
        summary_sentences = [sentences[i] for i in selected]

        return " ".join(summary_sentences)

    def _frequency_summary(self, text, sentences, max_sentences):
        """Fallback frequency-based summarization."""
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = {
            'the', 'a', 'an', 'and', 'but', 'or', 'for', 'in', 'on', 'at',
            'to', 'from', 'of', 'by', 'with', 'is', 'are', 'was', 'were',
            'that', 'this', 'it', 'he', 'she', 'they'
        }
        word_freq = {}
        for w in words:
            if w not in stopwords and len(w) > 3:
                word_freq[w] = word_freq.get(w, 0) + 1

        if not word_freq:
            return " ".join(sentences[:max_sentences])

        sent_scores = []
        for i, sent in enumerate(sentences):
            sent_words = re.findall(r'\b\w+\b', sent.lower())
            score = sum(word_freq.get(w, 0) for w in sent_words if w in word_freq)
            norm_score = score / (len(sent_words) + 1)
            sent_scores.append((i, norm_score))

        top_indices = [idx for idx, _ in sorted(sent_scores, key=lambda x: x[1], reverse=True)[:max_sentences]]
        return " ".join([sentences[idx] for idx in sorted(top_indices)])

    # ══════════════════════════════════════════════════════════════════════
    # COMPONENT 3: SHAP Explainability
    # ══════════════════════════════════════════════════════════════════════

    def explain_with_shap(self, text, model_pipeline):
        """
        Compute real SHAP values using LinearExplainer for the prediction.

        Returns:
            dict with keys: shap_values (list of {word, value}),
                            base_value, predicted_value, available
        """
        result = {"shap_values": [], "base_value": 0.0, "predicted_value": 0.0, "available": False}

        if not _SHAP_AVAILABLE or not text or len(text.strip()) == 0:
            return result

        try:
            tfidf = model_pipeline.named_steps['tfidf']
            classifier = model_pipeline.named_steps['classifier']

            # Transform text
            X = tfidf.transform([text])
            feature_names = tfidf.get_feature_names_out()

            # Create SHAP explainer with a zero background matrix
            import scipy.sparse as sp
            bg = sp.csr_matrix((1, X.shape[1]))
            explainer = shap.LinearExplainer(classifier, bg)
            shap_values = explainer.shap_values(X)

            if shap_values is not None:
                sv = shap_values[0]  # First (only) sample
                feature_indices = X.nonzero()[1]

                shap_list = []
                for idx in feature_indices:
                    word = feature_names[idx]
                    value = float(sv[idx])
                    if abs(value) > 1e-5:  # Filter near-zero contributions
                        shap_list.append({"word": word, "value": value})

                # Sort by absolute SHAP value
                shap_list.sort(key=lambda x: abs(x['value']), reverse=True)

                result["shap_values"] = shap_list[:20]  # Top 20
                result["base_value"] = float(explainer.expected_value)
                result["predicted_value"] = float(explainer.expected_value + sv.sum())
                result["available"] = True

        except Exception:
            pass  # Graceful fallback — explain_features() will be used instead

        return result

    def explain_features(self, text, model_pipeline, max_features=10):
        """
        Token-level feature attribution (TF-IDF weight approximation).
        Kept as fast fallback when SHAP is unavailable.
        """
        if not text or len(text.strip()) == 0:
            return [], []

        tfidf = model_pipeline.named_steps['tfidf']
        classifier = model_pipeline.named_steps['classifier']

        feature_names = tfidf.get_feature_names_out()
        weights = classifier.coef_[0]

        vec = tfidf.transform([text])
        feature_indices = vec.nonzero()[1]

        explanations = []
        for idx in feature_indices:
            word = feature_names[idx]
            tfidf_val = vec[0, idx]
            weight = weights[idx]
            contribution = tfidf_val * weight

            explanations.append({
                "word": word,
                "tfidf": tfidf_val,
                "weight": weight,
                "contribution": contribution
            })

        df_exp = pd.DataFrame(explanations)
        if df_exp.empty:
            return [], []

        fake_drivers = df_exp[df_exp['contribution'] < 0].sort_values(
            by='contribution', ascending=True
        ).head(max_features).to_dict('records')
        real_drivers = df_exp[df_exp['contribution'] > 0].sort_values(
            by='contribution', ascending=False
        ).head(max_features).to_dict('records')

        return fake_drivers, real_drivers

    # ══════════════════════════════════════════════════════════════════════
    # COMPONENT 4: spaCy NER Integration
    # ══════════════════════════════════════════════════════════════════════

    def extract_entities(self, text):
        """
        Named Entity Recognition using spaCy (en_core_web_sm) with
        graceful fallback to rule-based extraction.

        Extracts: PERSON, ORG, GPE/LOC, DATE, MONEY entities.
        """
        if not text:
            return {"people": [], "organizations": [], "locations": [], "dates": [], "money": []}

        if _SPACY_AVAILABLE and _spacy_nlp is not None:
            return self._spacy_ner(text)
        else:
            return self._regex_ner(text)

    def _spacy_ner(self, text):
        """spaCy-powered NER extraction."""
        # Limit text length for performance
        doc = _spacy_nlp(text[:10000])

        people = set()
        orgs = set()
        locs = set()
        dates = set()
        money = set()

        for ent in doc.ents:
            clean_text = ent.text.strip()
            if len(clean_text) < 2:
                continue

            if ent.label_ == "PERSON":
                people.add(clean_text)
            elif ent.label_ == "ORG":
                orgs.add(clean_text)
            elif ent.label_ in ("GPE", "LOC", "FAC"):
                locs.add(clean_text)
            elif ent.label_ == "DATE":
                dates.add(clean_text)
            elif ent.label_ == "MONEY":
                money.add(clean_text)

        return {
            "people": sorted(list(people))[:10],
            "organizations": sorted(list(orgs))[:10],
            "locations": sorted(list(locs))[:10],
            "dates": sorted(list(dates))[:8],
            "money": sorted(list(money))[:5]
        }

    def _regex_ner(self, text):
        """Fallback rule-based NER extraction."""
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        people = set()
        orgs = set()
        locs = set()

        for name in proper_nouns:
            words = name.split()
            if words[0] in {
                'Mr.', 'Mr', 'Ms.', 'Ms', 'Mrs.', 'Mrs', 'Dr.', 'Dr',
                'President', 'Senator', 'Governor', 'Representative', 'Minister'
            }:
                people.add(name)
            elif any(w in self.org_suffixes for w in words):
                orgs.add(name)
            elif any(w in self.loc_keywords for w in words) or name in self.loc_keywords:
                locs.add(name)
            else:
                if len(words) >= 3 or any(kw in words for kw in ['News', 'Party', 'Court', 'Committee']):
                    orgs.add(name)
                else:
                    people.add(name)

        return {
            "people": sorted(list(people))[:10],
            "organizations": sorted(list(orgs))[:10],
            "locations": sorted(list(locs))[:10],
            "dates": [],
            "money": []
        }

    # ══════════════════════════════════════════════════════════════════════
    # Sentence Analysis (unchanged)
    # ══════════════════════════════════════════════════════════════════════

    def analyze_sentences(self, text, model_pipeline):
        """
        Split text into sentences and score each sentence.
        Uses the decision boundary distance to compute a pseudo-probability.
        """
        if not text or len(text.strip()) == 0:
            return []

        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.!?])\s+', text)

        results = []

        tfidf = model_pipeline.named_steps['tfidf']
        classifier = model_pipeline.named_steps['classifier']

        for sent in sentences:
            sent_str = sent.strip()
            if len(sent_str) < 5:
                continue

            vec = tfidf.transform([sent_str])
            decision_score = classifier.decision_function(vec)[0]

            prob_real = 1.0 / (1.0 + np.exp(-decision_score))
            prob_fake = 1.0 - prob_real

            label = "REAL" if prob_real >= 0.5 else "FAKE"
            confidence = prob_real if label == "REAL" else prob_fake

            results.append({
                "text": sent_str,
                "score_fake": prob_fake,
                "score_real": prob_real,
                "label": label,
                "confidence": confidence
            })

        return results

