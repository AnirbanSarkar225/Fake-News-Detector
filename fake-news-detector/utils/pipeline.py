"""
Verification Pipeline Orchestrator for TruthShield v2.

Wraps predict_article() with explicit stage tracking, timing,
image analysis, evidence reports, and multilingual routing.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


class VerificationPipeline:
    """End-to-end verification with transparent stage tracking.

    Pipeline stages:
        Language Detection → Translation → Preprocessing →
        Claim Extraction → Fact-Check API → Source Analysis →
        ML Classification → Transformer Classification →
        NLP Analysis → Image Analysis → Ensemble Decision →
        Evidence Report
    """

    def __init__(self) -> None:
        self._model = None
        self._preprocessor = None
        self._clickbait_det = None
        self._ai_det = None
        self._claim_verifier = None
        self._source_engine = None
        self._multilingual = None
        self._transformer = None
        self._evidence_engine = None
        self._image_verifier = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import joblib
        from utils.preprocess import TextPreprocessor
        from utils.clickbait_detector import ClickbaitDetector
        from utils.ai_detector import AIContentDetector
        from utils.claim_verifier import ClaimVerifier
        from utils.source_engine import SourceEngine
        from utils.multilingual import MultilingualProcessor
        from utils.evidence_engine import EvidenceEngine

        model_path = os.path.join(PROJECT_ROOT, "model", "fake_news_model.pkl")
        if os.path.exists(model_path):
            self._model = joblib.load(model_path)
        self._preprocessor = TextPreprocessor()
        self._clickbait_det = ClickbaitDetector()
        self._ai_det = AIContentDetector()
        self._claim_verifier = ClaimVerifier()
        self._source_engine = SourceEngine()
        self._multilingual = MultilingualProcessor()
        self._evidence_engine = EvidenceEngine()

        try:
            from utils.transformer_predictor import TransformerPredictor
            self._transformer = TransformerPredictor()
        except Exception:
            pass

        try:
            from utils.image_verifier import ImageVerifier
            self._image_verifier = ImageVerifier()
        except Exception:
            pass

        self._loaded = True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        text: str,
        url: Optional[str] = None,
        image: Optional[Union[str, bytes]] = None,
    ) -> Dict:
        """Execute the full verification pipeline.

        Args:
            text: Article text to analyze.
            url: Optional source URL.
            image: Optional image file path or bytes for image analysis.

        Returns:
            Dict with ``stages``, ``prediction_result``,
            ``evidence_report``, ``image_analysis``,
            ``language_info``, ``total_duration_ms``.
        """
        self._ensure_loaded()
        t_start = time.perf_counter()
        stages: List[Dict] = []

        # Stage 1: Language Detection
        lang_info = self._run_stage(stages, "Language Detection",
                                    lambda: self._multilingual.detect_language(text))
        lang_code = lang_info.get("language_code", "en") if lang_info else "en"

        # Stage 2: Translation
        analysis_text = text
        was_translated = False
        if lang_code != "en":
            trans_result = self._run_stage(stages, "Translation",
                                           lambda: self._multilingual.translate_to_english(text, source_lang=lang_code))
            if trans_result and trans_result.get("was_translated"):
                analysis_text = trans_result.get("translated_text", text)
                was_translated = True
        else:
            stages.append({"name": "Translation", "status": "skipped",
                           "duration_ms": 0, "output_summary": "English detected"})

        # Stage 3: Text Preprocessing
        self._run_stage(stages, "Text Preprocessing",
                        lambda: {"processed": bool(self._preprocessor.preprocess_for_model(analysis_text))})

        # Stage 4-7: Core prediction (predict_article handles these internally)
        prediction_result = None
        try:
            from src.app import predict_article
            prediction_result = self._run_stage(
                stages, "Core Prediction Pipeline",
                lambda: predict_article(
                    analysis_text, self._model, self._preprocessor,
                    self._clickbait_det, self._ai_det,
                    self._claim_verifier, self._source_engine, url,
                ),
            )
        except Exception as e:
            stages.append({"name": "Core Prediction Pipeline", "status": "error",
                           "duration_ms": 0, "output_summary": str(e)})

        if prediction_result is None:
            prediction_result = self._fallback_prediction(analysis_text)

        # Stage 8: Transformer Classification (standalone if not triggered in predict_article)
        transformer_result = None
        if not prediction_result.get("bert_triggered") and self._transformer and self._transformer.is_available:
            transformer_result = self._run_stage(
                stages, "Transformer Classification",
                lambda: self._transformer.predict(analysis_text, language_code=lang_code),
            )
            if transformer_result:
                prediction_result["transformer_used"] = True
                prediction_result["transformer_result"] = transformer_result
                prediction_result["transformer_model"] = transformer_result.get("model_used", "unknown")
        else:
            stages.append({"name": "Transformer Classification", "status": "skipped",
                           "duration_ms": 0, "output_summary": "Already triggered or unavailable"})

        # Stage 9: Image Analysis
        image_analysis = None
        if image is not None and self._image_verifier:
            image_analysis = self._run_stage(
                stages, "Image Analysis",
                lambda: self._image_verifier.verify_image(image),
            )
        else:
            stages.append({"name": "Image Analysis", "status": "skipped",
                           "duration_ms": 0, "output_summary": "No image provided"})

        # Stage 10: Evidence Report
        evidence_report = None
        if self._evidence_engine:
            evidence_report = self._run_stage(
                stages, "Evidence Report Generation",
                lambda: self._evidence_engine.generate_report(prediction_result),
            )

        total_ms = (time.perf_counter() - t_start) * 1000

        return {
            "stages": stages,
            "prediction_result": prediction_result,
            "evidence_report": evidence_report,
            "image_analysis": image_analysis,
            "language_info": lang_info,
            "was_translated": was_translated,
            "total_duration_ms": round(total_ms, 1),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_stage(stages: List[Dict], name: str, fn) -> Optional[Dict]:
        """Execute a pipeline stage with timing and error handling."""
        t0 = time.perf_counter()
        try:
            result = fn()
            dt = (time.perf_counter() - t0) * 1000
            summary = ""
            if isinstance(result, dict):
                # Pick a representative key for the summary
                for key in ("prediction", "language_code", "text", "manipulation_score",
                            "display_verdict", "category", "processed"):
                    if key in result:
                        summary = f"{key}={result[key]}"
                        break
            stages.append({"name": name, "status": "complete",
                           "duration_ms": round(dt, 1), "output_summary": summary})
            return result
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000
            stages.append({"name": name, "status": "error",
                           "duration_ms": round(dt, 1), "output_summary": str(e)})
            return None

    @staticmethod
    def _fallback_prediction(text: str) -> Dict:
        """Minimal prediction if the main pipeline fails."""
        return {
            "prediction": "UNKNOWN",
            "confidence": 0.0,
            "credibility": 0.5,
            "reliability": 0.0,
            "category": "Error",
            "clickbait_score": 0.0,
            "ai_score": 0.0,
            "source_trust": 50.0,
            "source_profile": {"domain": "N/A", "score": 50, "category": "Unknown"},
            "verification_results": [],
            "verification_status": "Pipeline error",
            "sentence_analysis": [],
            "indicators": {},
            "bert_triggered": False,
            "bert_result": None,
            "is_sufficient": False,
            "is_clickbait": False,
            "is_ai_generated": False,
            "is_satire": False,
            "positive_factors": [],
            "negative_factors": [],
            "temporal_analysis": {},
            "stance": {},
            "ml_score": 0.5,
            "nlp_score": 0.5,
            "factcheck_score": 0.5,
            "article_stance": "NEUTRAL",
            "stance_confidence": 0.0,
            "evidence_count": 0,
            "matched_themes": [],
            "satire_score": 0.0,
        }
