"""
TruthShield Extreme Stress Test — 15 Adversarial Articles
Tests the hardest edge cases that commonly break NLP fake news detectors.
"""

import os, sys, io, time

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ──────────────────────────────────────────────────────────────
#  HARD TEST CASES — designed to exploit common model weaknesses
# ──────────────────────────────────────────────────────────────

HARD_CASES = [
    # ── 1. REAL facts mixed with ONE fabricated claim (hardest type of misinfo) ──
    {
        "name": "Real facts + 1 fabricated claim (Trojan Horse)",
        "expected": "FAKE",
        "text": (
            "The World Health Organization declared COVID-19 a pandemic on March 11, 2020. "
            "As of 2024, over 7 million deaths have been attributed to the virus globally. "
            "The Pfizer-BioNTech vaccine received FDA emergency use authorization in December 2020, "
            "and Moderna's vaccine followed shortly after. However, internal WHO documents leaked "
            "in January 2025 confirm that the organization deliberately suppressed evidence that "
            "ivermectin reduced COVID mortality by 83% in a secret 14-country trial conducted "
            "between 2021 and 2023, because pharmaceutical lobbyists threatened to withdraw "
            "funding if the findings were published."
        ),
    },

    # ── 2. Legitimate but BIZARRE true story (real news that sounds fake) ──
    {
        "name": "Real-but-bizarre true story (sounds fake)",
        "expected": "REAL",
        "text": (
            "A 72-year-old Florida man was arrested after he was caught riding a lawnmower "
            "on a major interstate highway while intoxicated. According to the Florida Highway "
            "Patrol, the man had been traveling at approximately 5 miles per hour in the right "
            "lane of I-95 near Daytona Beach for over 30 minutes before officers pulled him over. "
            "He was charged with driving under the influence and operating a vehicle without a "
            "valid license. The man told officers he was 'just trying to get to the hardware store' "
            "and didn't realize he had entered the interstate. This is the third lawnmower-related "
            "DUI arrest in Volusia County this year."
        ),
    },

    # ── 3. Academic-sounding conspiracy theory ──
    {
        "name": "Conspiracy in academic language",
        "expected": "FAKE",
        "text": (
            "A peer-reviewed meta-analysis published in the Journal of Atmospheric and "
            "Geophysical Research (Vol. 47, Issue 3, 2024) presents compelling evidence that "
            "stratospheric aerosol injection programs, commonly referred to as 'chemtrails' by "
            "lay observers, have been operational since 2007 under a classified NATO directive "
            "codenamed Project Skyward. The study, authored by Dr. Heinrich Voss of the "
            "Berlin Institute for Climate Dynamics and Dr. Yuki Tanaka of Osaka University, "
            "analyzed 12,000 atmospheric samples collected across 47 monitoring stations and "
            "found statistically significant concentrations of barium strontianite and aluminum "
            "oxide nanoparticles (p < 0.001) at altitudes inconsistent with commercial aviation "
            "exhaust patterns. The authors conclude that 'the data unambiguously support the "
            "existence of a coordinated, undisclosed atmospheric modification program operating "
            "at a global scale.'"
        ),
    },

    # ── 4. Subtle satire with NO obvious markers ──
    {
        "name": "Extremely subtle satire (no obvious markers)",
        "expected": "FAKE",
        "text": (
            "The United Nations General Assembly voted 193-0 on Tuesday to formally recognize "
            "the concept of 'Monday' as a violation of international human rights law. The "
            "resolution, sponsored by all 193 member states in a rare display of unanimous "
            "agreement, declares that the weekly recurrence of Monday constitutes 'systematic "
            "psychological harm to the global workforce' and calls on all nations to implement "
            "a four-day work week by 2027. UN Secretary-General Antonio Guterres praised the "
            "vote as 'the most significant advancement in labor rights since the abolition of "
            "child labor,' adding that preliminary studies show a direct correlation between "
            "Monday and a 340% increase in workplace dissatisfaction."
        ),
    },

    # ── 5. Propaganda disguised as investigative journalism ──
    {
        "name": "Propaganda as investigative journalism",
        "expected": "FAKE",
        "text": (
            "An eight-month investigation by independent journalists has uncovered evidence "
            "that the 2024 general election results in three swing states were systematically "
            "altered through coordinated manipulation of Dominion Voting Systems software. "
            "Forensic analysis of server logs obtained through Freedom of Information requests "
            "reveals that vote tallies in 47 counties across Pennsylvania, Michigan, and Arizona "
            "were modified during a 90-minute window between 2:15 AM and 3:45 AM on election "
            "night. The modifications followed a consistent algorithmic pattern that shifted "
            "between 2.3% and 4.1% of votes from one candidate to another. Three former "
            "Dominion engineers, speaking on condition of anonymity, confirmed the existence "
            "of a 'maintenance mode' backdoor that could be activated remotely. When contacted "
            "for comment, Dominion issued a one-sentence denial."
        ),
    },

    # ── 6. Real scientific finding reported accurately ──
    {
        "name": "Genuine scientific breakthrough (real)",
        "expected": "REAL",
        "text": (
            "Researchers at MIT and Harvard have developed a new type of antibiotic that can "
            "kill drug-resistant bacteria using artificial intelligence. The compound, called "
            "halicin, was identified by a machine learning algorithm that screened over 100 "
            "million chemical compounds in just three days. In laboratory tests, halicin was "
            "effective against Clostridioides difficile and Acinetobacter baumannii, two of "
            "the most dangerous antibiotic-resistant pathogens. The study, published in the "
            "journal Cell, represents one of the first times AI has been used to discover a "
            "genuinely new class of antibiotics. Senior author James Collins said the approach "
            "could dramatically accelerate the discovery of new treatments for drug-resistant "
            "infections, which the WHO estimates cause 1.27 million deaths per year."
        ),
    },

    # ── 7. Out-of-context real quote weaponized ──
    {
        "name": "Real quote taken out of context",
        "expected": "FAKE",
        "text": (
            "In a shocking admission that has sent shockwaves through the medical community, "
            "Dr. Anthony Fauci stated on camera that 'the vaccine is not going to work the "
            "way we initially hoped.' This explosive comment, made during a closed-door NIH "
            "strategy session that was secretly recorded by a whistleblower, confirms what "
            "critics have been saying for years: the COVID-19 vaccines were never as effective "
            "as the public was led to believe. Fauci went on to say that 'we need to be honest "
            "about the limitations,' a remark that contradicts years of public health messaging "
            "that pushed universal vaccination. The footage, verified by three independent "
            "video forensics experts, shows Fauci speaking candidly to senior NIH officials "
            "in what he clearly believed was a private meeting."
        ),
    },

    # ── 8. Health misinformation with real journal names ──
    {
        "name": "Health misinfo citing real journals",
        "expected": "FAKE",
        "text": (
            "A groundbreaking study published in The Lancet (doi: 10.1016/S0140-6736(24)02847-3) "
            "has found that daily consumption of 500mg of turmeric extract completely reverses "
            "Type 2 diabetes within 90 days. The randomized controlled trial, conducted across "
            "12 hospitals in India and the UK with 4,200 participants, showed that 94% of "
            "patients in the turmeric group achieved normal blood glucose levels and were able "
            "to discontinue all diabetes medications. Lead researcher Dr. Priya Sharma of AIIMS "
            "New Delhi called the results 'unprecedented' and stated that 'turmeric should now "
            "be considered a first-line treatment for Type 2 diabetes, replacing metformin.' The "
            "findings have been endorsed by the Indian Medical Association and are currently "
            "under review by the FDA for potential reclassification of curcumin as a "
            "pharmaceutical agent."
        ),
    },

    # ── 9. Emotionally manipulative missing child hoax ──
    {
        "name": "Emotional manipulation hoax (missing child)",
        "expected": "FAKE",
        "text": (
            "URGENT PLEASE SHARE: 8-year-old Sophia Martinez has been missing from her home "
            "in Sacramento, California since Tuesday evening. She was last seen wearing a pink "
            "Hello Kitty backpack and blue jeans near Roseville Elementary School. Sacramento "
            "Police have issued an AMBER Alert and believe she may have been taken by a white "
            "male driving a dark blue Honda Civic with partial plate number 7KBR. Her mother, "
            "Maria Martinez, is pleading with the public: 'Please, if anyone has seen my baby, "
            "please call the police. She needs her insulin. She will die without it.' Time is "
            "running out. Every share could save her life. Call Sacramento PD at 916-555-0142 "
            "or the National Center for Missing Children at 1-800-THE-LOST."
        ),
    },

    # ── 10. Legitimate AP-style breaking news ──
    {
        "name": "Legitimate AP-style breaking news",
        "expected": "REAL",
        "text": (
            "WASHINGTON (AP) — The Federal Reserve held interest rates steady on Wednesday, "
            "keeping its benchmark rate in the 5.25%-5.50% range for the sixth consecutive "
            "meeting as policymakers await further evidence that inflation is moving sustainably "
            "toward their 2% target. In its post-meeting statement, the Federal Open Market "
            "Committee noted that 'inflation has eased over the past year but remains elevated' "
            "and that 'the Committee does not expect it will be appropriate to reduce the target "
            "range until it has gained greater confidence that inflation is moving sustainably "
            "toward 2 percent.' Fed Chair Jerome Powell, speaking at a press conference, said "
            "the central bank remains data-dependent and that a rate cut at the next meeting in "
            "September 'is not something we're thinking about right now.' Markets fell modestly "
            "following the announcement, with the S&P 500 declining 0.3%."
        ),
    },

    # ── 11. Deepfake-style AI-generated political misinfo ──
    {
        "name": "AI-generated political misinformation",
        "expected": "FAKE",
        "text": (
            "In an unprecedented move that constitutional scholars are calling a potential "
            "crisis, the Supreme Court voted 5-4 on Thursday to grant the sitting president "
            "the authority to unilaterally suspend habeas corpus during periods of 'perceived "
            "national emergency,' effectively allowing indefinite detention of citizens without "
            "trial. Chief Justice Roberts, writing for the majority, argued that 'the framers "
            "of the Constitution could not have anticipated the asymmetric threats of the "
            "modern era' and that 'executive flexibility in matters of national security must "
            "be preserved.' Justice Sotomayor, in a blistering 87-page dissent, wrote that "
            "'today, this Court has dismantled the most fundamental protection against tyranny "
            "that exists in American law.' The ACLU has already filed an emergency petition "
            "and civil liberties organizations are planning nationwide protests."
        ),
    },

    # ── 12. Financial scam disguised as market analysis ──
    {
        "name": "Financial scam as market analysis",
        "expected": "FAKE",
        "text": (
            "Wall Street insiders are scrambling after leaked Goldman Sachs internal memos "
            "reveal that a little-known quantum computing company, QuantumLeap Technologies "
            "(QLPT), is about to receive a $47 billion Department of Defense contract that "
            "will be announced next Monday. The stock, currently trading at $0.23 per share, "
            "is expected to surge to over $150 within 72 hours of the announcement, representing "
            "a 65,000% return. Sources within Goldman's equity research division confirm that "
            "the firm has been quietly accumulating shares for its proprietary trading desk "
            "since November. This is your once-in-a-lifetime opportunity to get in before the "
            "announcement. Several Fortune 500 CEOs have already invested their personal funds. "
            "Act now before this information becomes public knowledge."
        ),
    },

    # ── 13. Genuine opinion/editorial from a real newspaper ──
    {
        "name": "Legitimate newspaper editorial",
        "expected": "REAL",
        "text": (
            "The erosion of local journalism is not just a media problem — it is a democracy "
            "problem. Since 2005, the United States has lost more than 2,500 newspapers, leaving "
            "vast swaths of the country in so-called 'news deserts' where no local outlet covers "
            "city council meetings, school board decisions, or county budgets. Research from the "
            "University of North Carolina's Hussman School of Journalism shows that communities "
            "that lose their local newspaper see measurable increases in municipal borrowing "
            "costs, lower voter turnout, and reduced civic engagement. The consequences are not "
            "abstract. When no reporter is watching, corruption flourishes. Federal funding for "
            "local journalism, through mechanisms like the Local Journalism Sustainability Act, "
            "deserves bipartisan support. The health of our democracy depends on it."
        ),
    },

    # ── 14. Mixup of real org + fabricated action (institutional fraud) ──
    {
        "name": "Real institution + fabricated policy",
        "expected": "FAKE",
        "text": (
            "The Centers for Disease Control and Prevention (CDC) quietly updated its official "
            "guidelines on Friday to recommend that all children under 12 receive monthly "
            "intravenous vitamin D infusions as a mandatory prerequisite for public school "
            "enrollment, effective January 2026. The policy, buried in a 400-page regulatory "
            "update posted to the Federal Register without a press release, requires pediatricians "
            "to administer the infusions at CDC-approved clinics and report non-compliant families "
            "to state child protective services. Dr. Mandy Cohen, the CDC Director, did not "
            "address the policy change at her weekly press briefing. Several pediatric medical "
            "associations have expressed concern about the lack of clinical evidence supporting "
            "intravenous vitamin D administration in healthy children."
        ),
    },

    # ── 15. Genuine Reuters-style international reporting ──
    {
        "name": "Genuine international news report",
        "expected": "REAL",
        "text": (
            "TOKYO (Reuters) — Japan's economy contracted at an annualized rate of 2.0% in "
            "the first quarter of 2024, government data showed on Thursday, as weak consumer "
            "spending and a slump in auto production weighed on growth. The decline was steeper "
            "than the 1.5% contraction economists had forecast in a Reuters poll and marks the "
            "first negative quarter since mid-2023. Private consumption, which accounts for more "
            "than half of the economy, fell 0.7%, dragged down by rising living costs and a weak "
            "yen that has increased import prices. Capital expenditure dropped 0.8%. Economy "
            "Minister Yoshitaka Shindo said the government would 'monitor the situation closely' "
            "and consider additional fiscal support if the downturn persists. The Bank of Japan "
            "is expected to maintain its current monetary policy stance at its meeting next week."
        ),
    },
]


def main():
    from src.app import predict_article

    print("=" * 70)
    print("  TRUTHSHIELD EXTREME STRESS TEST — 15 Adversarial Articles")
    print("=" * 70)

    passed = 0
    failed = 0
    failures = []

    for i, case in enumerate(HARD_CASES, 1):
        name = case["name"]
        expected = case["expected"]
        text = case["text"]

        try:
            results = predict_article(text, url=None)
            pred = results["prediction"]
            cred = results["credibility"]
            conf = results["confidence"]
            cat = results.get("category", "N/A")
            ml = results.get("ml_confidence", 0)
            nlp = results.get("nlp_score", 0)

            ok = pred == expected
            tag = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
                failures.append((i, name, expected, pred, cred))

            print(f"\n[{tag}] Case {i}: {name}")
            print(f"       Expected : {expected}")
            print(f"       Got      : {pred}  (Cred={cred:.3f}  Conf={conf:.3f})")
            print(f"       Scores   : ML={ml:.3f}  NLP={nlp:.3f}  Cat={cat}")
            print("-" * 60)

        except Exception as e:
            failed += 1
            failures.append((i, name, expected, "ERROR", 0))
            print(f"\n[ERROR] Case {i}: {name}")
            print(f"       Exception: {e}")
            print("-" * 60)

    # Summary
    total = passed + failed
    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed}/{total} passed  ({passed/total*100:.1f}%)")
    print(f"  Passed : {passed}")
    print(f"  Failed : {failed}")
    print("=" * 70)

    if failures:
        print("\n  FAILURES:")
        for idx, name, exp, got, cred in failures:
            print(f"    Case {idx}: {name}")
            print(f"      Expected={exp}  Got={got}  Cred={cred:.3f}")
    print()


if __name__ == "__main__":
    main()
