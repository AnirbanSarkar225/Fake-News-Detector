"""
Clickbait detection module for TruthShield Fake News Detector.

Analyzes article text and titles for clickbait characteristics using
pattern matching, punctuation analysis, emotional language density,
title heuristics, and vague attribution scoring.
"""

import re
import math
from typing import Dict, List, Optional


class ClickbaitDetector:
    """Detects clickbait content using multi-signal heuristic analysis."""

    def __init__(self) -> None:
        self.clickbait_phrases: List[str] = [
            "you won't believe", "shocking", "mind-blowing", "jaw-dropping",
            "what happens next", "this will change", "secret", "they don't want you",
            "gone wrong", "epic fail", "insane", "unbelievable", "can't stop laughing",
            r"number \d+ will", "the truth about", "exposed", "what they found",
            "doctors hate", "one weird trick", "you need to see", "is dead",
            "breaks the internet", "goes viral", "will blow your mind",
            "changed forever", "the real reason", "finally reveals",
            "heartbreaking", "devastating",
        ]
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.clickbait_phrases
        ]

        self._emotional_words = {
            "amazing", "incredible", "terrifying", "horrifying", "adorable",
            "brilliant", "disgusting", "hilarious", "outrageous", "stunning",
            "tragic", "furious", "glorious", "pathetic", "phenomenal",
            "ridiculous", "spectacular", "terrible", "wonderful", "worst",
            "best", "crazy", "scary", "creepy", "awesome", "awful",
        }

        self._vague_attributions = [
            re.compile(p, re.IGNORECASE) for p in [
                r"some say", r"experts claim", r"sources reveal",
                r"people are saying", r"many believe", r"rumor has it",
                r"reports suggest", r"insiders say", r"according to sources",
                r"it is being said", r"they say", r"word is",
            ]
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str, title: Optional[str] = None) -> Dict:
        """
        Analyze text (and optional title) for clickbait characteristics.

        Args:
            text: Article body text.
            title: Optional headline / title string.

        Returns:
            Dict with keys: is_clickbait, clickbait_score, matched_patterns, analysis.
        """
        if not text and not title:
            return self._empty_result()

        combined = (title or "") + " " + text[:500]
        pattern_score, matched = self._pattern_score(combined)
        punctuation_score = self._punctuation_score(combined)
        emotional_score = self._emotional_score(combined)
        title_score = self._title_score(title) if title else 0.0
        vague_score = self._vague_attribution_score(text[:1000])

        weights = {
            "pattern": 0.3,
            "punctuation": 0.2,
            "emotional": 0.2,
            "title": 0.2,
            "vague_attribution": 0.1,
        }
        clickbait_score = (
            pattern_score * weights["pattern"]
            + punctuation_score * weights["punctuation"]
            + emotional_score * weights["emotional"]
            + title_score * weights["title"]
            + vague_score * weights["vague_attribution"]
        )
        clickbait_score = min(max(clickbait_score, 0.0), 1.0)

        analysis = {
            "pattern_score": round(pattern_score, 3),
            "punctuation_score": round(punctuation_score, 3),
            "emotional_score": round(emotional_score, 3),
            "title_score": round(title_score, 3),
            "vague_attribution_score": round(vague_score, 3),
        }

        return {
            "is_clickbait": clickbait_score > 0.5,
            "clickbait_score": round(clickbait_score, 3),
            "matched_patterns": matched,
            "analysis": analysis,
        }

    def get_explanation(self, result: Dict) -> str:
        """Return a human-readable explanation of the clickbait analysis."""
        if not result or "clickbait_score" not in result:
            return "No analysis result provided."

        score = result["clickbait_score"]
        analysis = result.get("analysis", {})
        matched = result.get("matched_patterns", [])

        if result.get("is_clickbait"):
            lines = [f"⚠️  This content is likely CLICKBAIT (score: {score:.0%})."]
        else:
            lines = [f"✅  This content does NOT appear to be clickbait (score: {score:.0%})."]

        if matched:
            lines.append(f"  • Matched clickbait phrases: {', '.join(matched[:5])}")
        if analysis.get("punctuation_score", 0) > 0.4:
            lines.append("  • Excessive punctuation or ALL-CAPS detected.")
        if analysis.get("emotional_score", 0) > 0.4:
            lines.append("  • High density of emotionally charged language.")
        if analysis.get("title_score", 0) > 0.4:
            lines.append("  • Title exhibits clickbait patterns (questions, listicles, etc.).")
        if analysis.get("vague_attribution_score", 0) > 0.4:
            lines.append("  • Vague or unsourced attributions found.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _empty_result(self) -> Dict:
        return {
            "is_clickbait": False,
            "clickbait_score": 0.0,
            "matched_patterns": [],
            "analysis": {},
        }

    def _pattern_score(self, text: str) -> tuple:
        matched: List[str] = []
        for pattern, raw in zip(self._compiled_patterns, self.clickbait_phrases):
            if pattern.search(text):
                matched.append(raw)
        if not matched:
            return 0.0, matched
        return min(len(matched) / 3.0, 1.0), matched

    def _punctuation_score(self, text: str) -> float:
        if not text:
            return 0.0
        excl = text.count("!")
        quest = text.count("?")
        ellipsis = text.count("...")
        caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
        total_words = max(len(text.split()), 1)

        punct_density = (excl + quest + ellipsis) / total_words
        caps_density = caps_words / total_words
        return min((punct_density * 5) + (caps_density * 3), 1.0)

    def _emotional_score(self, text: str) -> float:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        if not words:
            return 0.0
        emotional_count = sum(1 for w in words if w in self._emotional_words)
        density = emotional_count / len(words)
        return min(density * 15, 1.0)

    def _title_score(self, title: str) -> float:
        if not title:
            return 0.0
        score = 0.0
        # Question headline
        if title.strip().endswith("?"):
            score += 0.3
        # Listicle pattern ("7 things…", "Top 10…")
        if re.match(r"^\d+\s", title.strip()) or re.search(r"top \d+", title, re.I):
            score += 0.3
        # Very short or very long titles
        word_count = len(title.split())
        if word_count <= 3 or word_count >= 20:
            score += 0.2
        # ALL-CAPS title
        if title.isupper():
            score += 0.3
        return min(score, 1.0)

    def _vague_attribution_score(self, text: str) -> float:
        if not text:
            return 0.0
        hits = sum(1 for p in self._vague_attributions if p.search(text))
        return min(hits / 2.0, 1.0)
