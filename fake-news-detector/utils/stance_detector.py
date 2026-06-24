"""
Stance Detection Module for TruthShield Fake News Detector.

Determines whether an article SUPPORTS, REFUTES, or NEUTRALLY REPORTS on
the claims it contains. This is critical for distinguishing fact-check
articles (which debunk misinformation) from actual misinformation.

Uses linguistic heuristics — no ML model required.
"""

import re
from typing import Dict, List, Optional


class StanceDetector:
    """
    Lightweight stance detector that classifies an article's relationship
    to the claims it discusses.

    Stance categories:
      - REFUTES:  Article is debunking / fact-checking the claim
      - SUPPORTS: Article is promoting / endorsing the claim
      - NEUTRAL:  Article is reporting without clear editorial stance
    """

    def __init__(self) -> None:
        # ── Refutation signal patterns ──
        # These indicate the article is DEBUNKING a claim
        self._refutation_patterns = [
            # Explicit debunking language
            re.compile(r'\b(?:debunk(?:ed|ing|s)?|disproved?|refut(?:ed|ing|es?)|busted|dismantled)\b', re.I),
            re.compile(r'\b(?:fact[\s-]?check(?:ed|ing|ers?|s)?)\b', re.I),
            re.compile(r'\b(?:false claim|fake claim|misleading claim|baseless claim|unsubstantiated claim)\b', re.I),
            re.compile(r'\b(?:no evidence|no proof|no scientific basis|no credible evidence|lacks evidence)\b', re.I),
            re.compile(r'\b(?:has been debunked|was debunked|were debunked|previously debunked)\b', re.I),
            re.compile(r'\b(?:experts (?:say|warn|caution|note|explain|clarify|dismiss|reject))\b', re.I),
            re.compile(r'\b(?:scientists (?:say|warn|caution|note|explain|clarify|dismiss|reject))\b', re.I),
            re.compile(r'\b(?:researchers (?:say|warn|note|found|conclude|point out))\b', re.I),
            re.compile(r'\b(?:contrary to (?:the|popular|viral|this) (?:claim|belief|post|message))\b', re.I),
            re.compile(r'\b(?:the claim (?:is|was|has been) (?:false|untrue|incorrect|misleading|inaccurate|baseless|unfounded))\b', re.I),
            re.compile(r'\b(?:this (?:is|was) (?:false|untrue|incorrect|misleading|inaccurate|a hoax|misinformation))\b', re.I),
            re.compile(r'\b(?:viral (?:claim|post|message|video|image|hoax|misinformation|rumor|rumour))\b', re.I),
            re.compile(r'\b(?:(?:spreads?|spreading|circulating|went viral) (?:on social media|online|on whatsapp|on facebook))\b', re.I),
            re.compile(r'\b(?:misleading|misinformation|disinformation|false narrative|fabricated)\b', re.I),
            re.compile(r'\b(?:investigation (?:found|reveals?|shows?))\b', re.I),
            re.compile(r'\b(?:there is no (?:truth|evidence|basis|proof|link|connection))\b', re.I),
            re.compile(r'\b(?:rated? (?:as )?(?:false|mostly false|pants on fire|misleading|unproven))\b', re.I),
            re.compile(r'\b(?:(?:snopes|politifact|factcheck\.org|full fact|alt news|boom live) (?:rated|found|reported|confirmed|debunked|says?))\b', re.I),
            # Hedged refutation
            re.compile(r'\b(?:however|but|yet|nevertheless|in reality|in fact|actually|on the contrary)\b', re.I),
        ]

        # ── Support signal patterns ──
        # These indicate the article is ENDORSING/PROMOTING a claim
        self._support_patterns = [
            re.compile(r'\b(?:exposed|they don\'t want you to know|the truth (?:about|behind|they))\b', re.I),
            re.compile(r'\b(?:exposed|what they\'re hiding|exposed the truth|exposed the lie|big pharma)\b', re.I),
            re.compile(r'\b(?:mainstream media (?:won\'t|refuses?|hides?|ignores?))\b', re.I),
            re.compile(r'\b(?:wake up|open your eyes|share before (?:they|it) (?:delete|remove|censor))\b', re.I),
            re.compile(r'\b(?:100% (?:true|proven|confirmed|real)|exposed by insider)\b', re.I),
            re.compile(r'\b(?:exposed|whistleblow|cover[\s-]?up|suppressed|censored)\b', re.I),
            re.compile(r'\b(?:you won\'t believe|shocking truth|what they don\'t tell you)\b', re.I),
            re.compile(r'\b(?:exposed|exposed|exposed)\b', re.I),  # intentional weight
        ]

        # ── Quotation/attribution patterns ──
        # Claims inside quotes or attributions are being DISCUSSED, not endorsed
        self._attribution_patterns = [
            re.compile(r'(?:the|a|this) (?:viral )?(?:claim|post|message|forward|video|image) (?:states?|says?|alleges?|suggests?|claims?) (?:that )?', re.I),
            re.compile(r'(?:according to|as per) (?:the|a|this) (?:viral |social media )?(?:claim|post|message|forward)', re.I),
            re.compile(r'(?:it (?:is|was) (?:claimed|alleged|stated|suggested) (?:that|in))', re.I),
            re.compile(r'"[^"]{15,}"', re.I),  # Substantial quoted material
        ]

        # ── Fact-checker source references ──
        self._factchecker_references = [
            re.compile(r'\b(?:snopes|politifact|factcheck\.org|full fact|alt news|boom live|AFP fact check|reuters fact check)\b', re.I),
            re.compile(r'\b(?:fact[\s-]?check(?:ers?|ing)?|verified by|according to (?:experts|scientists|researchers|doctors|officials))\b', re.I),
            re.compile(r'\b(?:peer[\s-]?reviewed|published in (?:the )?(?:journal|lancet|nature|bmj|nejm|science))\b', re.I),
            re.compile(r'\b(?:world health organi[sz]ation|WHO|CDC|FDA|NIH|ICMR|NHS)\b', re.I),
        ]

    def detect(self, text: str, claims: Optional[List[str]] = None,
               verification_results: Optional[List[Dict]] = None) -> Dict:
        """
        Analyze the article's stance toward the claims it discusses.

        Args:
            text: Full article text
            claims: Optional list of extracted claims
            verification_results: Optional verification data from ClaimVerifier

        Returns:
            Dictionary with:
              - stance: 'REFUTES' | 'SUPPORTS' | 'NEUTRAL'
              - stance_confidence: float 0.0-1.0
              - refutation_signals: list of matched refutation patterns
              - support_signals: list of matched support patterns
              - factchecker_references: list of fact-checker mentions
              - attribution_count: number of attributed/quoted claims
              - is_factcheck_article: bool (high-confidence detection)
        """
        if not text or len(text.strip()) < 50:
            return self._neutral_result()

        text_lower = text.lower()

        # ── Count refutation signals ──
        refutation_signals = []
        refutation_score = 0.0
        for pattern in self._refutation_patterns:
            matches = pattern.findall(text)
            if matches:
                for m in matches:
                    if m.strip() and m.strip().lower() not in [x.lower() for x in refutation_signals]:
                        refutation_signals.append(m.strip())
                # Hedging words ('however', 'but') get lower weight
                if pattern.pattern.startswith(r'\b(?:however|but|yet'):
                    refutation_score += 0.05 * len(matches)
                else:
                    refutation_score += 0.12 * len(matches)

        # ── Count support signals ──
        support_signals = []
        support_score = 0.0
        for pattern in self._support_patterns:
            matches = pattern.findall(text)
            if matches:
                for m in matches:
                    if m.strip() and m.strip().lower() not in [x.lower() for x in support_signals]:
                        support_signals.append(m.strip())
                support_score += 0.15 * len(matches)

        # ── Count fact-checker references ──
        factchecker_refs = []
        for pattern in self._factchecker_references:
            matches = pattern.findall(text)
            if matches:
                for m in matches:
                    if m.strip() and m.strip().lower() not in [x.lower() for x in factchecker_refs]:
                        factchecker_refs.append(m.strip())
                refutation_score += 0.1 * len(matches)  # fact-checker refs boost refutation

        # ── Count attributions (claims in quotes or attributed) ──
        attribution_count = 0
        for pattern in self._attribution_patterns:
            matches = pattern.findall(text)
            attribution_count += len(matches)
        if attribution_count > 0:
            refutation_score += 0.08 * min(attribution_count, 5)

        # ── Leverage verification results if available ──
        if verification_results:
            contradicted = sum(1 for v in verification_results if v.get('status') == 'CONTRADICTED')
            verified = sum(1 for v in verification_results if v.get('status') == 'VERIFIED')
            # If external fact-checkers contradicted claims AND article has refutation language,
            # this strongly suggests the article is a fact-check
            if contradicted > 0 and refutation_score > 0.15:
                refutation_score += 0.2 * contradicted

        # ── Normalize scores ──
        refutation_score = min(refutation_score, 1.0)
        support_score = min(support_score, 1.0)

        # ── Determine stance ──
        is_factcheck = (
            refutation_score >= 0.35 and
            len(refutation_signals) >= 2 and
            (len(factchecker_refs) >= 1 or attribution_count >= 1)
        )

        if refutation_score > support_score and refutation_score >= 0.2:
            stance = "REFUTES"
            stance_confidence = min(refutation_score, 1.0)
        elif support_score > refutation_score and support_score >= 0.2:
            stance = "SUPPORTS"
            stance_confidence = min(support_score, 1.0)
        else:
            stance = "NEUTRAL"
            stance_confidence = max(0.0, 1.0 - refutation_score - support_score)

        return {
            "stance": stance,
            "stance_confidence": round(stance_confidence, 3),
            "refutation_signals": refutation_signals[:10],  # cap for readability
            "support_signals": support_signals[:10],
            "factchecker_references": factchecker_refs[:10],
            "attribution_count": attribution_count,
            "is_factcheck_article": is_factcheck,
            "scores": {
                "refutation": round(refutation_score, 3),
                "support": round(support_score, 3),
            }
        }

    def _neutral_result(self) -> Dict:
        """Return a default neutral stance result."""
        return {
            "stance": "NEUTRAL",
            "stance_confidence": 0.5,
            "refutation_signals": [],
            "support_signals": [],
            "factchecker_references": [],
            "attribution_count": 0,
            "is_factcheck_article": False,
            "scores": {"refutation": 0.0, "support": 0.0}
        }
