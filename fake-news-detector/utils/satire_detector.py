"""
Satire Detection Module using Regex and Keyword Pattern Matching.
Detects satirical news language (The Onion, Babylon Bee style) without heavy ML dependencies.
"""

import re
from typing import Dict, Optional


class SatireDetector:
    """Detects satirical articles using domain reputation and linguistic heuristics."""

    def __init__(self) -> None:
        # Known satire domains
        self.satire_domains = {
            "theonion.com",
            "babylonbee.com",
            "clickhole.com",
            "thebeaverton.com",
            "thedailymash.co.uk",
            "reductress.com",
            "borowitzreport.com",
            "newsthump.com",
            "theonion.com",
            "babylonbee.com",
            "worldnewsdailyreport.com",
            "der-postillon.com",
            "duffelblog.com",
            "thechaser.com.au"
        }

        # Satirical linguistic patterns
        self.satirical_patterns = [
            # Absurdist escalation/impossible scenario patterns
            re.compile(r'\b(?:sources confirm(?:ed)? that|sources close to.*confirm)\b', re.IGNORECASE),
            re.compile(r'\b(?:area man|local woman|area woman|local man|area child|local child)\b', re.IGNORECASE),
            re.compile(r'\b(?:nod(?:ded)? knowingly|shrugged indifferently|sighed in relief)\b', re.IGNORECASE),
            re.compile(r'\b(?:unnamed officials? was quoted as saying|officials? close to the situation say)\b', re.IGNORECASE),
            re.compile(r'\b(?:in a shocking turn of events|in what can only be described as|completely unrelated news)\b', re.IGNORECASE),
            
            # Deadpan impossible/exaggerated institutional language
            re.compile(r'\b(?:announced today that they will begin|issued a press release declaring that)\b', re.IGNORECASE),
            re.compile(r'\b(?:according to a report published by the department of|spokesperson confirmed that)\b', re.IGNORECASE),
            re.compile(r'\b(?:reported that a local|reports indicate that the local|officials confirm that the)\b', re.IGNORECASE),
            
            # Ironic juxtaposition / comedic phrasing
            re.compile(r'\b(?:despite having absolutely no|while completely ignoring the fact that)\b', re.IGNORECASE),
            re.compile(r'\b(?:solely for the purpose of|which is definitely a real thing)\b', re.IGNORECASE),
            re.compile(r'\b(?:in hopes of finally|in an attempt to seem|in a desperate bid to)\b', re.IGNORECASE),
            re.compile(r'\b(?:strongly denies that they have any connection to the|denied rumors that)\b', re.IGNORECASE)
        ]

    def detect(self, text: str, url: Optional[str] = None) -> Dict:
        """
        Analyze text and optional URL for satire.
        
        Returns:
            Dict containing 'satire_score' (0.0-1.0) and 'is_satire' (bool).
        """
        if not text or len(text.strip()) < 20:
            return {"satire_score": 0.0, "is_satire": False}

        score = 0.0
        
        # 1. Domain Check
        if url:
            cleaned_url = url.lower().strip()
            # Extract domain
            domain = cleaned_url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
            if domain in self.satire_domains:
                return {"satire_score": 1.0, "is_satire": True}

        # 2. Linguistic Pattern Check
        matched_patterns = []
        for pattern in self.satirical_patterns:
            matches = pattern.findall(text)
            if matches:
                matched_patterns.append(pattern.pattern)
                score += 0.20 * len(matches)

        # 3. Absurdist Capitalization/Formatting indicators
        # Headlines in ALL CAPS or excessive exclamation in a pseudo-serious text
        if text.isupper():
            score += 0.15

        # Normalize score to 0.0 - 1.0
        score = min(score, 1.0)
        
        # Consider it satire if score is above 0.4 or if domain matched
        is_satire = score > 0.45

        return {
            "satire_score": float(round(score, 3)),
            "is_satire": is_satire,
            "matched_patterns": matched_patterns
        }
