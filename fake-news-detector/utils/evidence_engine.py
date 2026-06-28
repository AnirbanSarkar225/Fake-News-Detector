"""
Evidence Engine for TruthShield v2.

Consolidates all prediction signals into structured, human-readable
evidence reports with explainable verdicts.
"""

from typing import Dict, List, Optional


class EvidenceEngine:
    """Generates structured evidence reports from predict_article() results."""

    # Display verdict thresholds
    _VERDICT_MAP = [
        (0.85, "\U0001f7e2 Likely True", "Highly Credible"),
        (0.65, "\U0001f7e2 Likely True", "Likely Real"),
        (0.45, "\U0001f7e1 Needs Verification", "Uncertain"),
        (0.20, "\U0001f534 Likely Misleading", "Likely Fake"),
        (0.00, "\U0001f534 Likely Misleading", "High Risk Misinformation"),
    ]

    def generate_report(self, result: Dict) -> Dict:
        """Build a full evidence report from a ``predict_article()`` output.

        Args:
            result: The dict returned by ``predict_article()``.

        Returns:
            Dict with ``evidence_items``, ``explanation_text``,
            ``key_factors``, ``debunk_links``, ``trust_chain``,
            ``display_verdict``, ``verdict_detail``, ``confidence_pct``.
        """
        evidence_items = self._extract_evidence(result)
        explanation = self._generate_explanation(evidence_items, result)
        display_verdict, verdict_detail = self._map_display_verdict(
            result.get("credibility", 0.5)
        )
        trust_chain = self._build_trust_chain(result)
        debunk_links = self._extract_debunk_links(result)

        # Sort by absolute impact
        evidence_items.sort(key=lambda e: abs(e.get("impact_pct", 0)), reverse=True)
        key_factors = evidence_items[:3]

        return {
            "evidence_items": evidence_items,
            "explanation_text": explanation,
            "key_factors": key_factors,
            "debunk_links": debunk_links,
            "trust_chain": trust_chain,
            "display_verdict": display_verdict,
            "verdict_detail": verdict_detail,
            "confidence_pct": int(result.get("confidence", 0.5) * 100),
        }

    # ------------------------------------------------------------------
    # Evidence extraction
    # ------------------------------------------------------------------

    def _extract_evidence(self, r: Dict) -> List[Dict]:
        items: List[Dict] = []

        # 1. Source credibility
        source_trust = r.get("source_trust", 50.0)
        sp = r.get("source_profile") or {}
        if source_trust >= 75:
            items.append(self._item(
                "Source Credibility", "Trusted Publisher",
                f"Source {sp.get('domain', 'unknown')} has a high trust score ({source_trust:.0f}%).",
                impact_pct=20, sentiment="positive", icon="\U0001f3db\ufe0f",
                confidence=min(source_trust / 100, 1.0),
            ))
        elif source_trust <= 40:
            items.append(self._item(
                "Source Credibility", "Low-Trust Source",
                f"Source {sp.get('domain', 'unknown')} is flagged as unreliable ({source_trust:.0f}%).",
                impact_pct=-20, sentiment="negative", icon="\u26a0\ufe0f",
                confidence=1.0 - source_trust / 100,
            ))

        # 2. Fact-check results
        fc_score = r.get("factcheck_score", 0.5)
        ev_count = r.get("evidence_count", 0)
        if ev_count > 0:
            if fc_score >= 0.7:
                items.append(self._item(
                    "Fact-Check", "Claims Verified",
                    f"Fact-check sources support the claims ({ev_count} evidence items, {fc_score:.0%} agreement).",
                    impact_pct=25, sentiment="positive", icon="\u2705",
                    confidence=fc_score,
                ))
            elif fc_score <= 0.35:
                items.append(self._item(
                    "Fact-Check", "Claims Contradicted",
                    f"Fact-checkers contradict the article's core claims ({ev_count} evidence items).",
                    impact_pct=-25, sentiment="negative", icon="\u274c",
                    confidence=1.0 - fc_score,
                ))

        # 3. Language / NLP analysis
        nlp_score = r.get("nlp_score", 0.5)
        indicators = r.get("indicators", {})
        sens = indicators.get("sensationalism_score", 0.0)
        if sens > 0.5:
            items.append(self._item(
                "Language Analysis", "Emotionally Charged Language",
                f"Article uses sensationalist language (sensationalism score: {sens:.0%}).",
                impact_pct=-int(sens * 20), sentiment="negative", icon="\U0001f525",
                confidence=sens,
            ))
        elif nlp_score >= 0.65:
            items.append(self._item(
                "Language Analysis", "Neutral Reporting Tone",
                "Writing style is consistent with factual reporting.",
                impact_pct=10, sentiment="positive", icon="\U0001f4dd",
                confidence=nlp_score,
            ))

        # 4. Clickbait
        cb = r.get("clickbait_score", 0.0)
        if r.get("is_clickbait"):
            items.append(self._item(
                "Clickbait Detection", "Clickbait Headlines Detected",
                f"Headline uses sensationalized framing ({cb:.0%} clickbait score).",
                impact_pct=-5, sentiment="negative", icon="\U0001f3a3",
                confidence=cb,
            ))

        # 5. AI content
        ai = r.get("ai_score", 0.0)
        if r.get("is_ai_generated"):
            items.append(self._item(
                "AI Content", "AI-Generated Text Detected",
                f"High similarity to AI-generated text ({ai:.0%} AI score).",
                impact_pct=-5, sentiment="negative", icon="\U0001f916",
                confidence=ai,
            ))

        # 6. Stance
        stance = r.get("article_stance", "NEUTRAL")
        s_conf = r.get("stance_confidence", 0.5)
        if stance == "REFUTES" and s_conf >= 0.3:
            items.append(self._item(
                "Stance Analysis", "Fact-Check / Debunking Article",
                "Article refutes or debunks claims with evidence.",
                impact_pct=int(s_conf * 30), sentiment="positive", icon="\U0001f50d",
                confidence=s_conf,
            ))
        elif stance == "SUPPORTS" and s_conf >= 0.3:
            items.append(self._item(
                "Stance Analysis", "Promotes Unverified Claims",
                "Article endorses claims without sufficient evidence.",
                impact_pct=-int(s_conf * 20), sentiment="negative", icon="\U0001f4e2",
                confidence=s_conf,
            ))

        # 7. Temporal consistency
        ta = r.get("temporal_analysis", {})
        if not ta.get("is_consistent", True):
            risk = ta.get("risk_score", 0.0)
            items.append(self._item(
                "Temporal Consistency", "Outdated Data Reused",
                "Article presents old statistics or events as current.",
                impact_pct=-int(risk * 30), sentiment="negative", icon="\u23f3",
                confidence=risk,
            ))

        # 8. Satire
        if r.get("is_satire"):
            items.append(self._item(
                "Satire Detection", "Satire / Parody Content",
                f"Article matches satirical patterns (satire score: {r.get('satire_score', 0):.0%}).",
                impact_pct=-15, sentiment="negative", icon="\U0001f921",
                confidence=r.get("satire_score", 0.0),
            ))

        # 9. Misinformation patterns
        for theme in r.get("matched_themes", []):
            items.append(self._item(
                "Misinformation Pattern", f"Known Pattern: {theme['theme']}",
                f"Content matches known misinformation about {theme['theme'].lower()}.",
                impact_pct=-int(theme.get("multiplier", 0.1) * 100),
                sentiment="negative", icon="\U0001f6a8",
                confidence=theme.get("multiplier", 0.5),
            ))

        # 10. ML model signal
        ml = r.get("ml_score", 0.5)
        bert_used = r.get("bert_triggered", False)
        transformer_used = r.get("transformer_used", False)
        model_name = "Transformer + TF-IDF" if (bert_used or transformer_used) else "TF-IDF Ensemble"
        if ml > 0.7:
            items.append(self._item(
                "ML Classification", f"{model_name}: Credible",
                "Machine learning classifier validates the text patterns as genuine.",
                impact_pct=35, sentiment="positive", icon="\U0001f9e0",
                confidence=ml,
            ))
        elif ml < 0.35:
            items.append(self._item(
                "ML Classification", f"{model_name}: Suspicious",
                "Machine learning classifier detects misinformation patterns in the text.",
                impact_pct=-35, sentiment="negative", icon="\U0001f9e0",
                confidence=1.0 - ml,
            ))

        # Red flags
        rf_count = indicators.get("redflag_count", 0)
        if rf_count > 0 and stance != "REFUTES":
            rfs = indicators.get("redflags", [])
            examples = ", ".join(f'"{f}"' for f in rfs[:3])
            items.append(self._item(
                "Fabrication Indicators", f"{rf_count} Red Flags Detected",
                f"Detected suspicious phrases: {examples}.",
                impact_pct=-min(rf_count * 6, 30), sentiment="negative", icon="\U0001f6a9",
                confidence=min(rf_count * 0.15, 1.0),
            ))

        return items

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _item(category: str, title: str, description: str,
              impact_pct: int = 0, sentiment: str = "neutral",
              icon: str = "", confidence: float = 0.5) -> Dict:
        return {
            "category": category,
            "title": title,
            "description": description,
            "impact_pct": impact_pct,
            "sentiment": sentiment,
            "icon_emoji": icon,
            "confidence": round(confidence, 3),
        }

    def _generate_explanation(self, evidence: List[Dict], result: Dict) -> str:
        """Create a 2-3 sentence natural-language explanation."""
        pred = result.get("prediction", "UNKNOWN")
        cred = result.get("credibility", 0.5)
        cat = result.get("category", "Uncertain")

        positives = [e for e in evidence if e["sentiment"] == "positive"]
        negatives = [e for e in evidence if e["sentiment"] == "negative"]

        parts = []
        if pred == "REAL":
            parts.append(f"This article appears credible (credibility: {cred:.0%}, category: {cat}).")
            if positives:
                tops = ", ".join(p["title"].lower() for p in positives[:2])
                parts.append(f"Key supporting factors: {tops}.")
            if negatives:
                tops = ", ".join(n["title"].lower() for n in negatives[:1])
                parts.append(f"Minor concerns include {tops}.")
        elif pred == "FAKE":
            parts.append(f"This article is likely misleading (credibility: {cred:.0%}, category: {cat}).")
            if negatives:
                tops = ", ".join(n["title"].lower() for n in negatives[:2])
                parts.append(f"Primary concerns: {tops}.")
            if positives:
                tops = ", ".join(p["title"].lower() for p in positives[:1])
                parts.append(f"However, {tops} was noted positively.")
        else:
            parts.append(f"This article could not be conclusively classified ({cat}).")

        return " ".join(parts)

    def _map_display_verdict(self, credibility: float):
        for threshold, display, detail in self._VERDICT_MAP:
            if credibility >= threshold:
                return display, detail
        return "\U0001f534 Likely Misleading", "High Risk Misinformation"

    def _build_trust_chain(self, result: Dict) -> List[Dict]:
        """Build a pipeline-stage visualization list."""
        stages = []
        def _add(name, status, detail=""):
            stages.append({"name": name, "status": status, "detail": detail})

        # Language
        _add("Language Detection", "complete", "English")

        # Source
        st = result.get("source_trust", 50)
        sp = result.get("source_profile", {})
        _add("Source Analysis", "complete" if sp.get("category") != "Unknown" else "skipped",
             f"{sp.get('domain', 'N/A')} ({st:.0f}%)")

        # Fact-check
        ec = result.get("evidence_count", 0)
        _add("Fact-Check API", "complete" if ec > 0 else "no_results",
             f"{ec} evidence items")

        # ML
        _add("ML Classification", "complete",
             f"Score: {result.get('ml_score', 0.5):.2f}")

        # Transformer
        bt = result.get("bert_triggered", False) or result.get("transformer_used", False)
        _add("Transformer", "complete" if bt else "skipped",
             result.get("transformer_model", "N/A"))

        # Stance
        _add("Stance Detection", "complete",
             result.get("article_stance", "NEUTRAL"))

        # Decision
        _add("Ensemble Decision", "complete",
             f"Credibility: {result.get('credibility', 0.5):.0%}")

        return stages

    def _extract_debunk_links(self, result: Dict) -> List[str]:
        links = []
        for vr in result.get("verification_results", []):
            url = vr.get("url") or vr.get("link")
            if url and url.startswith("http"):
                links.append(url)
        return links[:10]
