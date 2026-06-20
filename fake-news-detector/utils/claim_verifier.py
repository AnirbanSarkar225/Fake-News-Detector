"""
Claim extraction and verification module for TruthShield Fake News Detector.

Extracts key factual claims from article text and cross-references them
against the Google Fact Check Tools API and Wikipedia to produce an
overall verification score.
"""

import re
import os
from typing import Dict, List, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

API_KEY: str = "AIzaSyC27TYhEGCJaOJ6gwcM8fvaUundJBpbwls"

_REQUEST_TIMEOUT: int = 10  # seconds


class ClaimVerifier:
    """Extracts claims from text and verifies them via external APIs."""

    def __init__(self) -> None:
        self.factcheck_url: str = (
            "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        )
        self.wiki_api_url: str = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.wiki_search_url: str = "https://en.wikipedia.org/w/api.php"

    # ------------------------------------------------------------------
    # Claim extraction
    # ------------------------------------------------------------------

    def extract_claims(self, text: str) -> List[str]:
        """
        Extract key factual claims from article text.

        Uses sentence splitting and heuristics to find sentences containing
        numbers, dates, named entities, or definitive statements.  Filters
        out opinions, questions, and very short sentences.

        Returns:
            List of claim strings (max 10).
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        claims: List[str] = []

        for sent in sentences:
            sent = sent.strip()
            words = sent.split()
            if len(words) < 6 or len(words) > 60:
                continue
            if sent.endswith("?"):
                continue
            # Skip opinion markers
            if re.search(r"\b(I think|I believe|in my opinion|arguably)\b", sent, re.I):
                continue

            score = 0
            # Contains numbers or dates
            if re.search(r"\b\d[\d,.]*\b", sent):
                score += 1
            if re.search(r"\b(19|20)\d{2}\b", sent):
                score += 1
            # Named entities (capitalised words not at sentence start)
            caps = re.findall(r"(?<!^)(?<!\.\s)[A-Z][a-z]{2,}", sent)
            if len(caps) >= 1:
                score += 1
            # Definitive verbs
            if re.search(r"\b(is|was|are|were|will|has been|have been|had been)\b", sent, re.I):
                score += 1

            if score >= 2:
                claims.append(sent)

            if len(claims) >= 10:
                break

        return claims

    # ------------------------------------------------------------------
    # Google Fact Check API
    # ------------------------------------------------------------------

    def verify_with_factcheck_api(self, claim: str) -> List[Dict]:
        """
        Cross-reference a claim against the Google Fact Check Tools API.

        Returns:
            List of dicts with keys: source, rating, url, claim_text.
            Returns empty list on failure or missing ``requests`` library.
        """
        if not HAS_REQUESTS:
            return []

        params = {"query": claim, "key": API_KEY, "languageCode": "en"}
        try:
            resp = requests.get(self.factcheck_url, params=params, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        results: List[Dict] = []
        for item in data.get("claims", []):
            for review in item.get("claimReview", []):
                results.append({
                    "source": review.get("publisher", {}).get("name") or "Unknown",
                    "rating": review.get("textualRating") or "Unknown",
                    "url": review.get("url") or "",
                    "claim_text": item.get("text") or claim,
                })
        return results

    # ------------------------------------------------------------------
    # Wikipedia verification
    # ------------------------------------------------------------------

    def verify_with_wikipedia(self, claim: str) -> List[Dict]:
        """
        Search Wikipedia for articles relevant to *claim*.

        Extracts key entities, searches, fetches summaries for the top 3
        results, and computes a simple keyword-overlap relevance score.

        Returns:
            List of dicts with keys: title, summary, url, relevance_score.
        """
        if not HAS_REQUESTS:
            return []

        keywords = self._extract_keywords(claim)
        if not keywords:
            return []

        search_query = " ".join(keywords[:5])
        params = {
            "action": "query",
            "list": "search",
            "srsearch": search_query,
            "srlimit": 3,
            "format": "json",
        }

        try:
            resp = requests.get(self.wiki_search_url, params=params, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            search_results = resp.json().get("query", {}).get("search", [])
        except Exception:
            return []

        results: List[Dict] = []
        claim_words = set(claim.lower().split())

        for item in search_results:
            title = item.get("title", "")
            summary = self._fetch_wiki_summary(title)
            if not summary:
                continue
            summary_words = set(summary.lower().split())
            overlap = len(claim_words & summary_words)
            relevance = min(overlap / max(len(claim_words), 1), 1.0)
            results.append({
                "title": title,
                "summary": summary[:500],
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "relevance_score": round(relevance, 3),
            })

        return results

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def verify_article(self, text: str) -> Dict:
        """
        Run the full verification pipeline on article *text*.

        1. Extract claims.
        2. For each claim check the Fact Check API and Wikipedia.
        3. Aggregate into an overall verification score.

        Returns:
            Dict with keys: claims, verification_results,
            overall_verification_score, summary.
        """
        claims = self.extract_claims(text)
        if not claims:
            return {
                "claims": [],
                "verification_results": [],
                "overall_verification_score": 0.0,
                "summary": "No verifiable claims could be extracted from the text.",
            }

        verification_results: List[Dict] = []
        claim_scores: List[float] = []

        for claim in claims:
            fc_results = self.verify_with_factcheck_api(claim)
            wiki_results = self.verify_with_wikipedia(claim)

            claim_score = self._score_claim(fc_results, wiki_results)
            claim_scores.append(claim_score)

            verification_results.append({
                "claim": claim,
                "factcheck_results": fc_results,
                "wikipedia_results": wiki_results,
                "verification_score": round(claim_score, 3),
            })

        overall = sum(claim_scores) / len(claim_scores) if claim_scores else 0.0
        summary = self._build_summary(claims, overall)

        return {
            "claims": claims,
            "verification_results": verification_results,
            "overall_verification_score": round(overall, 3),
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract significant words (proper nouns, long words, numbers)."""
        stop = {
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "of", "in", "to", "for", "with", "on", "at", "by", "from",
            "that", "this", "it", "its", "and", "or", "but", "not", "as",
        }
        words = re.findall(r"[A-Za-z0-9]+", text)
        return [w for w in words if w.lower() not in stop and len(w) > 2]

    def _fetch_wiki_summary(self, title: str) -> Optional[str]:
        """Fetch the plain-text summary for a Wikipedia article."""
        if not HAS_REQUESTS:
            return None
        try:
            url = self.wiki_api_url + title.replace(" ", "_")
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("extract", "")
        except Exception:
            return None

    @staticmethod
    def _score_claim(fc: List[Dict], wiki: List[Dict]) -> float:
        """Compute a 0-1 verification score for a single claim."""
        score = 0.0

        if fc:
            score += 0.5  # existence of fact-check entries is informative
            positive = {"true", "correct", "accurate", "mostly true"}
            for entry in fc:
                rating_val = entry.get("rating") or ""
                if rating_val.lower() in positive:
                    score += 0.2

        if wiki:
            best_rel = max(w.get("relevance_score", 0) for w in wiki)
            score += best_rel * 0.3

        return min(score, 1.0)

    @staticmethod
    def _build_summary(claims: List[str], overall: float) -> str:
        n = len(claims)
        if overall >= 0.7:
            status = "well-supported by available fact-check and reference sources"
        elif overall >= 0.4:
            status = "partially verifiable; some claims lack supporting sources"
        else:
            status = "largely unverified; limited supporting evidence found"
        return (
            f"Extracted {n} claim(s). Overall verification score: {overall:.0%}. "
            f"The article's factual claims are {status}."
        )
