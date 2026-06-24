"""
Seed script for the source_reputation SQLite table.

Safe to re-run: uses INSERT OR REPLACE keyed on the `domain` PRIMARY KEY,
so existing rows are updated in place rather than duplicated.

Usage:
    python scripts/seed_fake_domains.py
"""

import os
import sqlite3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "truthshield.db")

# ANSI colors (matches existing project script style)
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# domain, trust_score, bias, category, description
FAKE_DOMAINS = [
    ("infowars.com", 10.0, "Conspiracy", "Conspiracy", "Conspiracy and disinformation outlet, repeatedly debunked."),
    ("yournewswire.com", 10.0, "Fake News", "Fake News", "Fabricated news stories, rebranded as newspunch.com."),
    ("newspunch.com", 10.0, "Fake News", "Fake News", "Rebrand of yournewswire.com, fabricated stories."),
    ("conspiracyclub.co", 10.0, "Conspiracy", "Conspiracy", "Conspiracy theory aggregator with no editorial standards."),
    ("naturalnews.com", 12.0, "Conspiracy", "Conspiracy", "Pseudo-medical conspiracy content, anti-vaccine misinformation."),
    ("vaccines.news", 12.0, "Conspiracy", "Conspiracy", "Anti-vaccine misinformation network."),
    ("veteranstoday.com", 12.0, "Conspiracy", "Conspiracy", "Conspiracy theories framed as military/veteran news."),
    ("beforeitsnews.com", 14.0, "Conspiracy", "Conspiracy", "User-submitted conspiracy and fringe content, no fact-checking."),
    ("globalresearch.ca", 15.0, "Conspiracy / Pseudo-Science", "Conspiracy / Pseudo-Science", "Conspiracy site mixing geopolitics with pseudo-science."),
    ("worldnewsdailyreport.com", 15.0, "None", "Satire / Parody", "Infamous satirical news site featuring hoax stories presented as real."),
    ("thegatewaypundit.com", 15.0, "Conspiracy", "Conspiracy", "Repeatedly published false and unverified claims."),
    ("realworldnews.com", 15.0, "Fake News", "Fake News", "Fabricated news stories with no editorial oversight."),
    ("gellerreport.com", 15.0, "Extreme Bias", "Extreme Bias", "Extreme-bias blog with a history of inflammatory misinformation."),
    ("dcclothesline.com", 15.0, "Conspiracy", "Conspiracy", "Conspiracy blog network, low editorial standards."),
    ("neonnettle.com", 18.0, "Fake News", "Fake News", "UK fake news outlet, fabricated and exaggerated stories."),
    ("humansarefree.com", 18.0, "Conspiracy", "Conspiracy", "New-age conspiracy content, anti-science claims."),
    ("investmentwatchblog.com", 20.0, "Conspiracy", "Conspiracy", "Conspiracy-leaning financial doom blog."),
    ("nationalenquirer.com", 20.0, "None", "Satire / Parody", "Tabloid known for fabricated celebrity and sensational stories."),
    ("disclose.tv", 22.0, "Conspiracy", "Conspiracy", "UFO/conspiracy aggregator with unverified claims."),
    ("climatechangedispatch.com", 22.0, "Conspiracy", "Conspiracy", "Climate change denial outlet, misrepresents scientific consensus."),
    ("sputniknews.com", 22.0, "Fake News", "Fake News", "Russian state-affiliated outlet with documented disinformation."),
    ("wnd.com", 24.0, "Conspiracy", "Conspiracy", "WorldNetDaily, history of conspiracy theories and false claims."),
    ("theonion.com", 25.0, "None", "Satire / Parody", "Long-running American satirical news publication."),
    ("clickhole.com", 25.0, "None", "Satire / Parody", "Satirical clickbait parody site from The Onion's network."),
    ("der-postillon.com", 25.0, "None", "Satire / Parody", "German satirical news site."),
    ("trueactivist.com", 25.0, "Conspiracy", "Conspiracy", "Activist blog mixing real and fabricated/conspiracy content."),
    ("palmerreport.com", 25.0, "Extreme Bias", "Extreme Bias", "Highly partisan blog with unverified speculative claims."),
    ("frontpagemag.com", 25.0, "Extreme Bias", "Extreme Bias", "Extreme-bias political commentary outlet."),
    ("mediatakeout.com", 25.0, "Fake News", "Fake News", "Celebrity gossip site with frequent fabricated stories."),
    ("rt.com", 26.0, "Extreme Bias", "Extreme Bias", "Russian state media with documented propaganda and bias."),
    ("thedailymash.co.uk", 28.0, "None", "Satire / Parody", "British satirical news site."),
    ("newsthump.com", 28.0, "None", "Satire / Parody", "British satirical news and comedy site."),
    ("breitbart.com", 28.0, "Extreme Bias", "Extreme Bias", "Far-right outlet with a history of misleading headlines."),
    ("thefreethoughtproject.com", 28.0, "Conspiracy", "Conspiracy", "Conspiracy-leaning alternative news site."),
    ("thebeaverton.com", 29.0, "None", "Satire / Parody", "Canadian satirical news site."),
    ("thechaser.com.au", 29.0, "None", "Satire / Parody", "Australian satirical news and comedy site."),
    ("babylonbee.com", 30.0, "None", "Satire / Parody", "Conservative-leaning satirical news site."),
    ("reductress.com", 30.0, "None", "Satire / Parody", "Satirical women's lifestyle/news site."),
    ("duffelblog.com", 30.0, "None", "Satire / Parody", "Military-themed satirical news site."),
    ("theantimedia.com", 30.0, "Extreme Bias", "Extreme Bias", "Alternative media outlet with mixed factual accuracy."),
    ("themindunleashed.com", 30.0, "Conspiracy", "Conspiracy", "New-age and conspiracy content mixed with pseudo-science."),
    ("collective-evolution.com", 30.0, "Conspiracy", "Conspiracy", "Alternative health and conspiracy content site."),
    ("mintpressnews.com", 32.0, "Extreme Bias", "Extreme Bias", "Alternative outlet with documented bias and unverified claims."),
    ("occupydemocrats.com", 35.0, "Low Trust / Biased", "Extreme Bias", "Highly partisan left-leaning outlet with exaggerated claims."),
    ("borowitzreport.com", 35.0, "None", "Satire / Parody", "Satirical political column, often mistaken for real news."),
    ("redstate.com", 35.0, "Extreme Bias", "Extreme Bias", "Partisan conservative commentary outlet."),
    ("politicususa.com", 35.0, "Extreme Bias", "Extreme Bias", "Partisan left-leaning commentary outlet."),
    ("zerohedge.com", 35.0, "Conspiracy", "Conspiracy", "Financial/political blog with a history of conspiracy content."),
    ("libertynation.com", 35.0, "Extreme Bias", "Extreme Bias", "Conservative-leaning commentary outlet."),
    ("westernjournal.com", 36.0, "Extreme Bias", "Extreme Bias", "Conservative outlet with a history of misleading framing."),
    ("dailywire.com", 38.0, "Extreme Bias", "Extreme Bias", "Conservative commentary outlet, strong editorial slant."),
    ("dailykos.com", 38.0, "Extreme Bias", "Extreme Bias", "Progressive commentary outlet, strong editorial slant."),
]

TRUSTED_DOMAINS = [
    ("bbc.co.uk", 95.0, "High Trust", "High Trust", "British public broadcaster, strong editorial standards."),
    ("reuters.com", 95.0, "News", "News", "International wire service with rigorous fact-checking."),
    ("apnews.com", 95.0, "News", "News", "Associated Press, leading international wire service."),
    ("afp.com", 95.0, "News", "News", "Agence France-Presse, international wire service."),
    ("nature.com", 95.0, "News", "News", "Peer-reviewed scientific journal."),
    ("science.org", 95.0, "News", "News", "Peer-reviewed scientific journal (AAAS)."),
    ("pnas.org", 95.0, "News", "News", "Peer-reviewed scientific journal (PNAS)."),
    ("nejm.org", 95.0, "News", "News", "New England Journal of Medicine, peer-reviewed."),
    ("thelancet.com", 94.0, "News", "News", "The Lancet, peer-reviewed medical journal."),
    ("fullfact.org", 93.0, "Fact-Checker", "Fact-Checker", "UK independent fact-checking organization."),
    ("abc.net.au", 92.0, "High Trust", "High Trust", "Australian public broadcaster."),
    ("bbc.com", 92.0, "News", "News", "British Broadcasting Corporation, international edition."),
    ("bloomberg.com", 92.0, "News", "News", "Financial and business news wire."),
    ("ft.com", 92.0, "News", "News", "Financial Times, strong editorial standards."),
    ("pbs.org", 92.0, "News", "News", "US public broadcaster."),
    ("dw.com", 92.0, "News", "News", "Deutsche Welle, German public broadcaster."),
    ("france24.com", 92.0, "News", "News", "French international news channel."),
    ("nationalgeographic.com", 92.0, "News", "News", "Science and geography focused publication."),
    ("snopes.com", 92.0, "Fact Check", "Fact-Checker", "Leading independent fact-checking site."),
    ("factcheck.org", 92.0, "Fact Check", "Fact-Checker", "Nonpartisan fact-checking organization."),
    ("politico.com", 91.0, "High Trust", "High Trust", "US political news outlet."),
    ("cbc.ca", 91.0, "High Trust", "High Trust", "Canadian public broadcaster."),
    ("nhk.or.jp", 90.0, "High Trust", "High Trust", "Japanese public broadcaster."),
    ("wsj.com", 90.0, "News", "News", "Wall Street Journal, financial and general news."),
    ("economist.com", 90.0, "News", "News", "The Economist, strong editorial standards."),
    ("npr.org", 90.0, "News", "News", "US public radio, strong editorial standards."),
    ("scientificamerican.com", 90.0, "News", "News", "Science journalism publication."),
    ("politifact.com", 90.0, "Fact Check", "Fact-Checker", "Pulitzer Prize-winning fact-checking organization."),
    ("nytimes.com", 88.0, "News", "News", "New York Times, major US newspaper."),
    ("washingtonpost.com", 88.0, "News", "News", "Washington Post, major US newspaper."),
    ("thehindu.com", 85.0, "High Trust", "High Trust", "Major Indian English-language newspaper."),
    ("axios.com", 85.0, "High Trust", "High Trust", "US news outlet known for concise factual reporting."),
    ("theguardian.com", 85.0, "News", "News", "The Guardian, major UK newspaper."),
    ("aljazeera.com", 85.0, "News", "News", "Qatar-based international news network."),
    ("indianexpress.com", 82.0, "High Trust", "High Trust", "Major Indian English-language newspaper."),
    ("abcnews.go.com", 82.0, "High Trust", "High Trust", "ABC News, major US broadcaster."),
    ("cbsnews.com", 82.0, "High Trust", "High Trust", "CBS News, major US broadcaster."),
    ("scroll.in", 80.0, "Factual", "Factual", "Indian digital news and commentary outlet."),
    ("nbcnews.com", 80.0, "High Trust", "High Trust", "NBC News, major US broadcaster."),
    ("thehill.com", 80.0, "Factual", "Factual", "US political news outlet."),
]


def seed():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS source_reputation (
            domain TEXT PRIMARY KEY,
            trust_score REAL,
            bias TEXT,
            category TEXT,
            description TEXT,
            fact_check_history TEXT,
            accuracy_rate REAL DEFAULT 1.0,
            frequency INTEGER DEFAULT 0
        )
    """)

    fake_inserted = 0
    for domain, trust_score, bias, category, description in FAKE_DOMAINS:
        cursor.execute("""
            INSERT OR REPLACE INTO source_reputation
                (domain, trust_score, bias, category, description, fact_check_history, accuracy_rate, frequency)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (domain, trust_score, bias, category, description, "", 0.2, 0))
        fake_inserted += 1

    trusted_inserted = 0
    for domain, trust_score, bias, category, description in TRUSTED_DOMAINS:
        cursor.execute("""
            INSERT OR REPLACE INTO source_reputation
                (domain, trust_score, bias, category, description, fact_check_history, accuracy_rate, frequency)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (domain, trust_score, bias, category, description, "", 1.0, 0))
        trusted_inserted += 1

    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM source_reputation")
    total_rows = cursor.fetchone()[0]
    conn.close()

    print(f"{BOLD}{CYAN}TruthShield — Domain Reputation Seeder{RESET}")
    print(f"{GREEN}  Fake/unreliable domains inserted or updated: {fake_inserted}{RESET}")
    print(f"{GREEN}  Trusted domains inserted or updated:        {trusted_inserted}{RESET}")
    print(f"{YELLOW}  Total rows now in source_reputation:        {total_rows}{RESET}")


if __name__ == "__main__":
    seed()