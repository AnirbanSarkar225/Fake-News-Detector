"""
Unified Transformer Predictor for TruthShield v2.

Supports DistilBERT (English) and MuRIL (Indian languages) with
ONNX Runtime optimization for fast CPU inference.  Falls back
gracefully when models or dependencies are missing.
"""

import os
import json
from typing import Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DISTILBERT_DIR = os.path.join(PROJECT_ROOT, "model", "distilbert_fakenews")
DISTILBERT_ONNX_DIR = os.path.join(PROJECT_ROOT, "model", "distilbert_fakenews_onnx")
MURIL_DIR = os.path.join(PROJECT_ROOT, "model", "muril_fakenews")
MURIL_ONNX_DIR = os.path.join(PROJECT_ROOT, "model", "muril_fakenews_onnx")

INDIAN_LANG_CODES = {"hi", "bn", "mr", "ta", "te", "gu", "kn", "ml", "pa", "ur"}


class TransformerPredictor:
    """Lazy-loading transformer predictor with ONNX acceleration.

    Automatically selects DistilBERT for English and MuRIL for Indian
    languages.  Tries ONNX Runtime first (fastest), then PyTorch,
    then returns ``None`` if neither is available.
    """

    def __init__(self) -> None:
        self._distilbert_model = None
        self._distilbert_tokenizer = None
        self._muril_model = None
        self._muril_tokenizer = None
        self._device = None
        self._distilbert_mode: Optional[str] = None  # "onnx" | "pytorch" | None
        self._muril_mode: Optional[str] = None
        self._checked_distilbert = False
        self._checked_muril = False

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if at least one transformer model is loadable."""
        return self.distilbert_available or self.muril_available

    @property
    def distilbert_available(self) -> bool:
        if not self._checked_distilbert:
            self._checked_distilbert = True
            self._distilbert_mode = self._probe_model(DISTILBERT_DIR, DISTILBERT_ONNX_DIR)
        return self._distilbert_mode is not None

    @property
    def muril_available(self) -> bool:
        if not self._checked_muril:
            self._checked_muril = True
            self._muril_mode = self._probe_model(MURIL_DIR, MURIL_ONNX_DIR)
        return self._muril_mode is not None

    @staticmethod
    def _probe_model(pytorch_dir: str, onnx_dir: str) -> Optional[str]:
        """Return 'onnx', 'pytorch', or None based on what's available."""
        if os.path.isdir(onnx_dir):
            onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]
            if onnx_files:
                try:
                    import onnxruntime  # noqa: F401
                    return "onnx"
                except ImportError:
                    pass
        if os.path.isdir(pytorch_dir) and os.path.exists(os.path.join(pytorch_dir, "config.json")):
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
                return "pytorch"
            except ImportError:
                pass
        return None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_distilbert(self) -> bool:
        if self._distilbert_model is not None:
            return True
        if not self.distilbert_available:
            return False
        return self._load_model("distilbert")

    def _ensure_muril(self) -> bool:
        if self._muril_model is not None:
            return True
        if not self.muril_available:
            return False
        return self._load_model("muril")

    def _load_model(self, which: str) -> bool:
        if which == "distilbert":
            mode = self._distilbert_mode
            pt_dir, onnx_dir = DISTILBERT_DIR, DISTILBERT_ONNX_DIR
        else:
            mode = self._muril_mode
            pt_dir, onnx_dir = MURIL_DIR, MURIL_ONNX_DIR
        try:
            if mode == "onnx":
                model, tokenizer = self._load_onnx(onnx_dir)
            else:
                model, tokenizer = self._load_pytorch(pt_dir)
        except Exception:
            return False
        if which == "distilbert":
            self._distilbert_model = model
            self._distilbert_tokenizer = tokenizer
        else:
            self._muril_model = model
            self._muril_tokenizer = tokenizer
        return True

    @staticmethod
    def _load_onnx(model_dir: str):
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = ORTModelForSequenceClassification.from_pretrained(model_dir)
        return model, tokenizer

    def _load_pytorch(self, model_dir: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self._device)
        model.eval()
        return model, tokenizer

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _select_model(self, language_code: str):
        """Return (model, tokenizer, mode, name) for the language."""
        if language_code in INDIAN_LANG_CODES and self._ensure_muril():
            return self._muril_model, self._muril_tokenizer, self._muril_mode, "muril"
        if self._ensure_distilbert():
            return self._distilbert_model, self._distilbert_tokenizer, self._distilbert_mode, "distilbert"
        if self._ensure_muril():
            return self._muril_model, self._muril_tokenizer, self._muril_mode, "muril"
        return None, None, None, None

    def predict(self, text: str, language_code: str = "en",
                max_length: int = 256) -> Optional[Dict]:
        """Run transformer inference.

        Returns:
            Dict with ``prediction``, ``confidence``, ``probabilities``,
            ``model_used`` — or ``None`` if no model is available.
        """
        model, tokenizer, mode, name = self._select_model(language_code)
        if model is None:
            return None
        text = str(text)[:5000]
        encoding = tokenizer(
            text, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        if mode == "onnx":
            probs = self._infer_onnx(model, encoding)
        else:
            probs = self._infer_pytorch(model, encoding)
        pred_idx = int(probs.argmax())
        labels = ["FAKE", "REAL"]
        return {
            "prediction": labels[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {"FAKE": float(probs[0]), "REAL": float(probs[1])},
            "model_used": name,
        }

    def _infer_onnx(self, model, encoding):
        import numpy as np
        outputs = model(**{k: v for k, v in encoding.items()})
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if hasattr(logits, "numpy"):
            logits = logits.detach().numpy()
        elif not isinstance(logits, np.ndarray):
            logits = np.array(logits)
        logits = logits[0]
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()

    def _infer_pytorch(self, model, encoding):
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            inputs = {k: v.to(self._device) for k, v in encoding.items()}
            outputs = model(**inputs)
            return F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def predict_proba(self, text: str, language_code: str = "en",
                      max_length: int = 256) -> Optional[List[float]]:
        """Return ``[P(FAKE), P(REAL)]``."""
        result = self.predict(text, language_code, max_length)
        if result is None:
            return None
        return [result["probabilities"]["FAKE"], result["probabilities"]["REAL"]]

    def get_transformer_score(self, text: str, language_code: str = "en",
                              max_length: int = 256) -> Optional[float]:
        """Return ``P(REAL)`` for weighted ensemble integration."""
        result = self.predict(text, language_code, max_length)
        if result is None:
            return None
        return result["probabilities"]["REAL"]

    def get_metrics(self, which: str = "distilbert") -> Optional[Dict]:
        """Load saved training metrics for the given model."""
        base = DISTILBERT_DIR if which == "distilbert" else MURIL_DIR
        metrics_path = os.path.join(base, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                return json.load(f)
        return None
