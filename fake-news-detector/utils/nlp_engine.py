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

class NLPEngine:
    def __init__(self):
        # Fallback wordlists for Sentiment & Emotions
        self.fear_words = {
            'panic', 'scared', 'terrified', 'fear', 'afraid', 'deadly', 'threat', 'warning', 'danger', 
            'hazard', 'epidemic', 'crisis', 'fatal', 'poison', 'kill', 'destroy', 'dread', 'terror'
        }
        self.anger_words = {
            'rage', 'angry', 'hate', 'furious', 'outrage', 'disgust', 'liar', 'cheat', 'scam', 'betray',
            'offensive', 'shame', 'scandal', 'corrupt', 'propaganda', 'con', 'conspiracy', 'fake'
        }
        # Fallback list of locations and organizations suffixes for entity extraction
        self.org_suffixes = {'Inc', 'Corp', 'Co', 'LLC', 'Group', 'Agency', 'Department', 'Administration', 'University', 'Foundation', 'Association'}
        self.loc_keywords = {'USA', 'UK', 'China', 'Russia', 'Europe', 'America', 'London', 'Washington', 'Beijing', 'Moscow', 'Paris', 'Berlin', 'Tokyo', 'India'}

    def analyze_sentences(self, text, model_pipeline):
        """
        Split text into sentences and score each sentence.
        Uses the decision boundary distance to compute a pseudo-probability of FAKE vs REAL.
        """
        if not text or len(text.strip()) == 0:
            return []

        # Split into sentences using nltk
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Simple regex fallback if nltk fails
            sentences = re.split(r'(?<=[.!?])\s+', text)

        results = []
        
        # Get components from model pipeline
        tfidf = model_pipeline.named_steps['tfidf']
        classifier = model_pipeline.named_steps['classifier']
        
        for sent in sentences:
            sent_str = sent.strip()
            if len(sent_str) < 5:
                continue
            
            # Predict score for this sentence
            # Transform using the global vectorizer
            vec = tfidf.transform([sent_str])
            
            # Compute distance to decision boundary
            decision_score = classifier.decision_function(vec)[0]
            
            # Convert decision score to probability of FAKE (0) and REAL (1)
            # Sigmoid: prob_real = 1 / (1 + exp(-x))
            prob_real = 1.0 / (1.0 + np.exp(-decision_score))
            prob_fake = 1.0 - prob_real
            
            # Label
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

    def explain_features(self, text, model_pipeline, max_features=10):
        """
        Token-level feature attribution (SHAP / LIME approximation).
        Calculates tfidf_value * classifier_weight for each word in the text.
        """
        if not text or len(text.strip()) == 0:
            return [], []

        tfidf = model_pipeline.named_steps['tfidf']
        classifier = model_pipeline.named_steps['classifier']
        
        # Get feature names and weights
        feature_names = tfidf.get_feature_names_out()
        weights = classifier.coef_[0]
        
        # Vectorize single text
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
            
        # Sort by contribution
        # Negative contribution => drives prediction towards FAKE (Class index 0)
        # Positive contribution => drives prediction towards REAL (Class index 1)
        fake_drivers = df_exp[df_exp['contribution'] < 0].sort_values(by='contribution', ascending=True).head(max_features).to_dict('records')
        real_drivers = df_exp[df_exp['contribution'] > 0].sort_values(by='contribution', ascending=False).head(max_features).to_dict('records')
        
        return fake_drivers, real_drivers

    def get_sentiment_metrics(self, text):
        """
        Lexicon-based sentiment analyzer estimating Fear, Anger, and Neutral scores.
        """
        if not text:
            return {"fear": 0.0, "anger": 0.0, "neutral": 1.0}
            
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return {"fear": 0.0, "anger": 0.0, "neutral": 1.0}
            
        fear_count = sum(1 for w in words if w in self.fear_words)
        anger_count = sum(1 for w in words if w in self.anger_words)
        total_words = len(words)
        
        # Normalize scores
        fear_score = min((fear_count / total_words) * 35.0, 1.0) # scaling factor to make visible
        anger_score = min((anger_count / total_words) * 35.0, 1.0)
        neutral_score = max(1.0 - (fear_score + anger_score), 0.0)
        
        # Re-normalize to sum to 1
        total = fear_score + anger_score + neutral_score
        if total > 0:
            fear_score /= total
            anger_score /= total
            neutral_score /= total
            
        return {
            "fear": float(round(fear_score, 3)),
            "anger": float(round(anger_score, 3)),
            "neutral": float(round(neutral_score, 3))
        }

    def extract_entities(self, text):
        """
        Rule-based named entity recognition (NER) extracting People, Organizations, and Locations.
        """
        if not text:
            return {"people": [], "organizations": [], "locations": []}
            
        # Extract potential proper nouns (sequences of capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        people = set()
        orgs = set()
        locs = set()
        
        for name in proper_nouns:
            words = name.split()
            # Title prefixes indicate People
            if words[0] in {'Mr.', 'Mr', 'Ms.', 'Ms', 'Mrs.', 'Mrs', 'Dr.', 'Dr', 'President', 'Senator', 'Governor', 'Representative', 'Minister'}:
                people.add(name)
            # Suffixes indicate Organizations
            elif any(w in self.org_suffixes for w in words):
                orgs.add(name)
            # Location keywords
            elif any(w in self.loc_keywords for w in words) or name in self.loc_keywords:
                locs.add(name)
            else:
                # Fallback heuristic: 3+ words or generic keywords often ORGs
                if len(words) >= 3 or 'News' in words or 'Party' in words or 'Court' in words or 'Committee' in words:
                    orgs.add(name)
                else:
                    people.add(name)
                    
        return {
            "people": sorted(list(people))[:10],
            "organizations": sorted(list(orgs))[:10],
            "locations": sorted(list(locs))[:10]
        }

    def generate_summary(self, text, max_sentences=3):
        """
        TextRank / Frequency-based summarization to extract high-importance sentences.
        """
        if not text or len(text.strip()) == 0:
            return ""
            
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
        if len(sentences) <= max_sentences:
            return text
            
        # Simple word frequency scoring
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter short words and common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'but', 'or', 'for', 'in', 'on', 'at', 'to', 'from', 'of', 'by', 'with', 'is', 'are', 'was', 'were', 'that', 'this', 'it', 'he', 'she', 'they'}
        word_freq = {}
        for w in words:
            if w not in stopwords and len(w) > 3:
                word_freq[w] = word_freq.get(w, 0) + 1
                
        if not word_freq:
            return " ".join(sentences[:max_sentences])
            
        # Score sentences
        sent_scores = []
        for i, sent in enumerate(sentences):
            sent_words = re.findall(r'\b\w+\b', sent.lower())
            score = sum(word_freq.get(w, 0) for w in sent_words if w in word_freq)
            # Normalize by length to avoid favoring extremely long sentences too much
            norm_score = score / (len(sent_words) + 1)
            sent_scores.append((i, norm_score))
            
        # Get top scoring sentences
        top_indices = [idx for idx, score in sorted(sent_scores, key=lambda x: x[1], reverse=True)[:max_sentences]]
        # Return in original chronological order
        summary_sentences = [sentences[idx] for idx in sorted(top_indices)]
        
        return " ".join(summary_sentences)
