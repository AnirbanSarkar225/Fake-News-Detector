"""
DistilBERT Predictor for Fake News Detection.

Provides a lazy-loading interface to the fine-tuned DistilBERT model.
Falls back gracefully if torch/transformers are not installed.
"""

import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model', 'distilbert_fakenews')


class BertPredictor:
    """Lazy-loading DistilBERT predictor for fake news classification."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None
        self._available = None

    @property
    def is_available(self):
        """Check if DistilBERT model and dependencies are available."""
        if self._available is None:
            self._available = (
                os.path.exists(os.path.join(MODEL_DIR, 'config.json'))
                and self._check_deps()
            )
        return self._available

    def _check_deps(self):
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is not None:
            return
        
        import torch
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
        self._model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(self._device)
        self._model.eval()

    def predict(self, text, max_length=256):
        """
        Predict FAKE/REAL for a single text.
        Returns dict: {prediction, confidence, probabilities: {FAKE, REAL}}
        """
        if not self.is_available:
            return None

        import torch
        import torch.nn.functional as F

        self._load_model()

        text = str(text)[:5000]
        encoding = self._tokenizer(
            text, max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self._device)
            attention_mask = encoding['attention_mask'].to(self._device)
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        labels = ['FAKE', 'REAL']
        
        return {
            'prediction': labels[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': {
                'FAKE': float(probs[0]),
                'REAL': float(probs[1])
            }
        }

    def predict_proba(self, text, max_length=256):
        """Return class probabilities as [P(FAKE), P(REAL)]."""
        result = self.predict(text, max_length)
        if result is None:
            return None
        return [result['probabilities']['FAKE'], result['probabilities']['REAL']]

    def get_metrics(self):
        """Load saved training metrics."""
        metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return None
