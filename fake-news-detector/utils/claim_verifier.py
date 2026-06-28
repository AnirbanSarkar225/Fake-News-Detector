"""
Claim extraction and verification module for TruthShield Fake News Detector.

Enhanced with:
- Local SQLite Claims Cache (claims_kb) with 30-day expiry and automatic re-validation.
- Fact-check source priority and freshness decay.
- Semantic contradiction heuristics (valence flips and entity-number mismatches).
- Temporal consistency analysis and temporal risk scoring.
- RAG retrieval confidence scoring (trust, recency, relevance).
- Evidence agreement analysis (supporting % vs contradicting %).
"""

import re
import os
import sqlite3
import hashlib
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

API_KEY: str = "AIzaSyC27TYhEGCJaOJ6gwcM8fvaUundJBpbwls"
_REQUEST_TIMEOUT: int = 10  # seconds

class ClaimVerifier:
    """Extracts claims from text and verifies them via external APIs with local caching and advanced XAI metrics."""

    # Publisher trust mapping for priority scoring
    _PUBLISHER_TRUST: Dict[str, float] = {
        "reuters": 1.0,
        "ap": 0.98,
        "associated press": 0.98,
        "factcheck.org": 0.95,
        "snopes": 0.95,
        "politifact": 0.95,
        "full fact": 0.95,
        "wikipedia": 0.85,
        "alt news": 0.90,
        "boom live": 0.90,
        "the quint": 0.85,
        "pib fact check": 0.90
    }

    def __init__(self) -> None:
        self.factcheck_url: str = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        self.wiki_api_url: str = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.wiki_search_url: str = "https://en.wikipedia.org/w/api.php"

    def _get_db_connection(self):
        # Locate the DB path dynamically
        db_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(db_dir, "data", "truthshield.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return sqlite3.connect(db_path)

    # ------------------------------------------------------------------
    # Claims Cache (Historical Claim Knowledge Base)
    # ------------------------------------------------------------------
    def _get_claim_hash(self, claim: str) -> str:
        # Normalize and MD5 hash the claim
        normalized = re.sub(r'[^\w\s]', '', claim.lower().strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def _get_cached_claim(self, claim_hash: str) -> Optional[Dict]:
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT claim_text, verdict, source, details, last_verified FROM claims_kb WHERE claim_hash = ?",
                (claim_hash,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                last_verified = datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S")
                # Check 30-day cache expiration
                if datetime.now() - last_verified < timedelta(days=30):
                    return {
                        "claim_text": row[0],
                        "verdict": row[1],
                        "source": row[2],
                        "details": json.loads(row[3]),
                        "last_verified": row[4],
                        "from_cache": True
                    }
        except Exception:
            pass
        return None

    def _cache_claim(self, claim: str, verdict: str, source: str, details: Dict):
        try:
            claim_hash = self._get_claim_hash(claim)
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO claims_kb (claim_hash, claim_text, verdict, source, details, last_verified)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                claim_hash,
                claim,
                verdict,
                source,
                json.dumps(details),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Claim extraction
    # ------------------------------------------------------------------
    def extract_claims(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        claims: List[str] = []

        for sent in sentences:
            sent = sent.strip()
            words = sent.split()
            if len(words) < 6 or len(words) > 60:
                continue
            if sent.endswith("?"):
                continue
            if re.search(r"\b(I think|I believe|in my opinion|arguably)\b", sent, re.I):
                continue

            score = 0
            if re.search(r"\b\d[\d,.]*\b", sent):
                score += 1
            if re.search(r"\b(19|20)\d{2}\b", sent):
                score += 1
            caps = re.findall(r"(?<!^)(?<!\.\s)[A-Z][a-z]{2,}", sent)
            if len(caps) >= 1:
                score += 1
            if re.search(r"\b(is|was|are|were|will|has been|have been|had been)\b", sent, re.I):
                score += 1

            if score >= 2:
                claims.append(sent)

            if len(claims) >= 10:
                break

        return claims

    # ------------------------------------------------------------------
    # Verification APIs
    # ------------------------------------------------------------------
    def verify_with_factcheck_api(self, claim: str) -> List[Dict]:
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
                publisher_name = review.get("publisher", {}).get("name") or "Unknown"
                rating_text = review.get("textualRating") or "Unknown"
                url = review.get("url") or ""
                review_date_str = review.get("reviewDate") or ""
                
                # Fetch publisher priority
                trust = 0.5
                for pub, score in self._PUBLISHER_TRUST.items():
                    if pub in publisher_name.lower():
                        trust = score
                        break
                
                # Recency calculation
                recency = 0.5
                if review_date_str:
                    try:
                        review_date = datetime.fromisoformat(review_date_str.replace("Z", "+00:00"))
                        days_old = (datetime.now() - review_date.replace(tzinfo=None)).days
                        recency = max(0.1, min(1.0, math.exp(-0.0002 * days_old)))
                    except Exception:
                        pass
                
                results.append({
                    "source": publisher_name,
                    "rating": rating_text,
                    "url": url,
                    "claim_text": item.get("text") or claim,
                    "trust_score": trust,
                    "recency_score": recency
                })
        return results

    def verify_with_wikipedia(self, claim: str) -> List[Dict]:
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
            
            # Wikipedia defaults
            trust = 0.85
            recency = 0.8  # Wikipedia is generally fresh
            
            results.append({
                "title": title,
                "summary": summary[:500],
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "relevance_score": round(relevance, 3),
                "trust_score": trust,
                "recency_score": recency
            })

        return results

    # ------------------------------------------------------------------
    # Intelligence Features: Negation, Temporal, and Quality Analysis
    # ------------------------------------------------------------------
    def _detect_contradiction(self, claim: str, text: str) -> bool:
        claim_lower = claim.lower()
        text_lower = text.lower()
        
        # 1. Valence flip check for known negative fact-check ratings
        negatives = {"false", "fake", "incorrect", "misleading", "debunked", "untrue", "hoax", "conspiracy", "pants on fire", "mostly false"}
        for neg in negatives:
            if neg in text_lower:
                claim_negated = any(w in claim_lower.split() for w in ["not", "no", "never", "incorrect", "untrue", "hoax", "false"])
                if not claim_negated:
                    return True

        # 2. Number/Entity mismatch check
        claim_nums = re.findall(r'\b\d+(?:[\.,]\d+)?\b', claim_lower)
        text_nums = re.findall(r'\b\d+(?:[\.,]\d+)?\b', text_lower)
        if claim_nums and text_nums:
            claim_words = set(re.findall(r'\b[a-z]{4,}\b', claim_lower))
            stop_words = {"about", "there", "their", "would", "could", "should", "other", "where", "which"}
            claim_nouns = claim_words - stop_words
            text_words = set(re.findall(r'\b[a-z]{4,}\b', text_lower))
            
            # High noun overlap indicates they talk about the same event
            overlap = claim_nouns & text_words
            if len(overlap) >= 3:
                # Mismatched numbers signal potential timeline or statistics corruption
                if not any(num in text_nums for num in claim_nums):
                    return True
        return False

    def _analyze_temporal_consistency(self, text: str, verification_results: List[Dict]) -> Dict:
        years = [int(y) for y in re.findall(r'\b(19\d{2}|20[0-2]\d)\b', text)]
        if not years:
            return {"is_consistent": True, "risk_score": 0.0, "reason": "No date markers detected.", "detected_years": []}

        min_year = min(years)
        max_year = max(years)
        current_year = 2026  # Time anchor

        mismatches = []
        risk_score = 0.0

        # Outdated event recycling check
        is_breaking = any(kw in text.lower() for kw in ["breaking", "now", "today", "yesterday", "recently", "current", "latest", "new", "record", "this week"])
        if is_breaking and max_year < current_year - 3:
            mismatches.append(f"Outdated content: Article references events from {max_year} as breaking/current.")
            risk_score += 0.4

        # Mismatched timeline check
        evidence_years = []
        for ev in verification_results:
            ev_str = ev.get("claim", "") + " " + str(ev.get("factcheck_results", "")) + " " + str(ev.get("wikipedia_results", ""))
            ev_yrs = [int(y) for y in re.findall(r'\b(19\d{2}|20[0-2]\d)\b', ev_str)]
            evidence_years.extend(ev_yrs)

        if evidence_years:
            avg_ev_year = sum(evidence_years) / len(evidence_years)
            if max_year >= current_year and avg_ev_year < current_year - 4:
                mismatches.append(f"Timeline drift: Claim focuses on timeline ({max_year}) but fact-checks were archived around {int(avg_ev_year)}.")
                risk_score += 0.3

        return {
            "is_consistent": len(mismatches) == 0,
            "risk_score": round(min(risk_score, 1.0), 2),
            "detected_years": sorted(list(set(years))),
            "mismatches": mismatches
        }

    # ------------------------------------------------------------------
    # Full verification pipeline
    # ------------------------------------------------------------------
    def verify_article(self, text: str) -> Dict:
        claims = self.extract_claims(text)
        if not claims:
            return {
                "claims": [],
                "verification_results": [],
                "overall_verification_score": 0.5,
                "evidence_count": 0,
                "evidence_quality": 0.0,
                "agreement_ratio": 0.5,
                "temporal_analysis": {"is_consistent": True, "risk_score": 0.0, "mismatches": []},
                "summary": "No verifiable factual claims extracted."
            }

        verification_results: List[Dict] = []
        total_evidence_pieces = 0
        total_supporting = 0
        total_contradicting = 0
        quality_accumulator = 0.0

        for claim in claims:
            claim_hash = self._get_claim_hash(claim)
            cached = self._get_cached_claim(claim_hash)

            if cached:
                fc_results = cached["details"].get("factcheck_results", [])
                wiki_results = cached["details"].get("wikipedia_results", [])
                claim_score = cached["details"].get("verification_score", 0.5)
                status = cached.get("verdict", "UNVERIFIED")
                from_cache = True
            else:
                fc_results = self.verify_with_factcheck_api(claim)
                wiki_results = self.verify_with_wikipedia(claim)
                from_cache = False

                # Calculate claim details
                supporting = 0
                contradicting = 0
                claim_score = 0.0
                status = "UNVERIFIED"

                # Check fact-checks
                if fc_results:
                    for fc in fc_results:
                        rating = fc.get("rating", "").lower()
                        is_contr = self._detect_contradiction(claim, rating)
                        
                        # Calculate retrieval quality score
                        relevance = 0.8  # Google Fact check queries are highly relevant
                        ev_quality = fc["trust_score"] * 0.4 + fc["recency_score"] * 0.2 + relevance * 0.4
                        quality_accumulator += ev_quality
                        total_evidence_pieces += 1

                        if is_contr:
                            contradicting += 1
                            fc["verdict_class"] = "CONTRADICTED"
                        else:
                            # Positive match keywords
                            positives = {"true", "correct", "accurate", "mostly true", "verified", "supported"}
                            if any(pos in rating for pos in positives):
                                supporting += 1
                                fc["verdict_class"] = "VERIFIED"
                            else:
                                fc["verdict_class"] = "UNVERIFIED"

                # Check Wikipedia
                if wiki_results:
                    for wk in wiki_results:
                        is_contr = self._detect_contradiction(claim, wk.get("summary", ""))
                        relevance = wk.get("relevance_score", 0.5)
                        ev_quality = wk["trust_score"] * 0.4 + wk["recency_score"] * 0.2 + relevance * 0.4
                        quality_accumulator += ev_quality
                        total_evidence_pieces += 1

                        if is_contr:
                            contradicting += 1
                            wk["verdict_class"] = "CONTRADICTED"
                        elif relevance > 0.65:
                            supporting += 1
                            wk["verdict_class"] = "VERIFIED"
                        else:
                            wk["verdict_class"] = "UNVERIFIED"

                # Compute verdict class for this claim
                if contradicting > 0:
                    status = "CONTRADICTED"
                    claim_score = 0.1
                elif supporting > 0:
                    status = "VERIFIED"
                    claim_score = 0.9
                else:
                    status = "UNVERIFIED"
                    claim_score = 0.5

                # Accumulate counts
                total_supporting += supporting
                total_contradicting += contradicting

                # Write to claims database cache
                details_payload = {
                    "factcheck_results": fc_results,
                    "wikipedia_results": wiki_results,
                    "verification_score": claim_score
                }
                self._cache_claim(claim, status, "Google Fact Check / Wikipedia", details_payload)

            verification_results.append({
                "claim": claim,
                "factcheck_results": fc_results,
                "wikipedia_results": wiki_results,
                "verification_score": claim_score,
                "status": status,
                "from_cache": from_cache
            })

        # Calculate evidence metrics
        evidence_quality = quality_accumulator / max(total_evidence_pieces, 1) if total_evidence_pieces > 0 else 0.0
        
        # Agreement analysis
        total_verified_signals = total_supporting + total_contradicting
        agreement_ratio = total_supporting / max(total_verified_signals, 1) if total_verified_signals > 0 else 0.5

        # Compute factcheck score
        factcheck_score = 0.5
        if total_supporting > 0 or total_contradicting > 0:
            factcheck_score = agreement_ratio * 0.8 + (evidence_quality * 0.2)
        
        # Contradiction penalty
        if total_contradicting > 0:
            factcheck_score = max(0.0, factcheck_score - 0.2)

        # Run temporal consistency checks
        temporal_analysis = self._analyze_temporal_consistency(text, verification_results)
        if not temporal_analysis["is_consistent"]:
            factcheck_score = max(0.0, factcheck_score - (temporal_analysis["risk_score"] * 0.3))

        # Overall summary message
        if total_contradicting > 0:
            summary = f"Factual check completed. Extracted {len(claims)} claims; detected {total_contradicting} CONTRADICTED claim(s)."
        elif total_supporting > 0:
            summary = f"Factual check completed. Extracted {len(claims)} claims; found supporting evidence for {total_supporting} claim(s)."
        else:
            summary = f"Factual check completed. Extracted {len(claims)} claims; evidence remains unverified."

        return {
            "claims": claims,
            "verification_results": verification_results,
            "overall_verification_score": round(factcheck_score, 3),
            "evidence_count": total_evidence_pieces,
            "evidence_quality": round(evidence_quality, 3),
            "agreement_ratio": round(agreement_ratio, 3),
            "temporal_analysis": temporal_analysis,
            "summary": summary
        }

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------
    def _extract_keywords(self, text: str) -> List[str]:
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
        if not HAS_REQUESTS:
            return None
        try:
            url = self.wiki_api_url + title.replace(" ", "_")
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("extract", "")
        except Exception:
            return None
