"""
Training Data Augmentation for TruthShield.

Diversifies the training corpus with synonym replacement, random
deletion/swap, headline manipulation, and satire templates.

Usage:
    python scripts/augment_training_data.py
    python scripts/augment_training_data.py --multiplier 2 --max-samples 5000
"""

import os, sys, io, re, random, argparse
import pandas as pd
from tqdm import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# NLTK setup
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

from nltk.corpus import wordnet


# ── Satire templates ──────────────────────────────────────────────────
SATIRE_TEMPLATES = [
    "BREAKING: Area {person} Discovers That {absurd_claim}. Sources confirm {reaction}.",
    "Nation's {group} Announce Plans To {absurd_action}. \"It just makes sense,\" reports say.",
    "Study Finds {percent}% Of {group} Secretly {absurd_claim}. Experts nod knowingly.",
    "Local {person} Solves {problem} By Simply {absurd_action}. Neighbors stunned.",
    "Congress Passes Bill Requiring All Americans To {absurd_action}. President signs after {time} debate.",
    "New Study Reveals {absurd_claim}. Big {industry} reportedly tried to suppress the findings.",
    "Sources Confirm {person} Has Been {absurd_action} For {time}. Friends say they always suspected.",
    "Report: {percent}% Of All {thing} Actually Just {other_thing}.",
]

SATIRE_FILLS = {
    "person": ["Man", "Woman", "Dad", "Mom", "CEO", "Senator", "Teacher", "Dog Owner"],
    "group": ["Adults", "Millennials", "Parents", "Office Workers", "Scientists", "Politicians"],
    "absurd_claim": [
        "wifi signals make plants smarter", "the moon is a social construct",
        "breakfast is a conspiracy by Big Cereal", "sleep is optional for productivity",
        "rain is just the sky crying about climate change", "cats control the global economy",
    ],
    "absurd_action": [
        "eat lunch standing up", "work two full-time jobs", "apologize to the internet",
        "replace all doors with curtains", "speak only in hashtags", "live in a spreadsheet",
    ],
    "reaction": ["everyone nodded knowingly", "no one was surprised", "experts remained silent"],
    "problem": ["world hunger", "climate change", "traffic", "the economy", "Mondays"],
    "percent": ["47", "83", "91", "62", "100"],
    "time": ["15-minute", "30-second", "two-hour", "overnight"],
    "industry": ["Pharma", "Tech", "Food", "Energy", "Media"],
    "thing": ["news articles", "social media posts", "emails", "meetings"],
    "other_thing": ["elaborate jokes", "AI-generated content", "recycled press releases"],
}


def generate_satire(n=100):
    """Generate satirical articles from templates."""
    articles = []
    for _ in range(n):
        template = random.choice(SATIRE_TEMPLATES)
        text = template
        for key, values in SATIRE_FILLS.items():
            text = text.replace("{" + key + "}", random.choice(values), 1)
        articles.append({"text": text, "label": "FAKE", "augmentation_type": "satire_template"})
    return articles


# ── Text augmentation functions ───────────────────────────────────────

def get_synonyms(word):
    """Get synonyms from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            s = lemma.name().replace("_", " ")
            if s.lower() != word.lower():
                synonyms.add(s)
    return list(synonyms)


def synonym_replace(text, replace_ratio=0.15):
    """Replace a fraction of words with WordNet synonyms."""
    words = text.split()
    if len(words) < 5:
        return text
    n_replace = max(1, int(len(words) * replace_ratio))
    indices = random.sample(range(len(words)), min(n_replace, len(words)))
    new_words = words.copy()
    for idx in indices:
        word = re.sub(r'[^\w]', '', words[idx])
        if len(word) > 3:
            syns = get_synonyms(word)
            if syns:
                new_words[idx] = random.choice(syns)
    return " ".join(new_words)


def random_deletion(text, delete_ratio=0.12):
    """Randomly delete words from text."""
    words = text.split()
    if len(words) < 5:
        return text
    remaining = [w for w in words if random.random() > delete_ratio]
    return " ".join(remaining) if remaining else " ".join(words[:3])


def random_swap(text, n_swaps=5):
    """Swap adjacent words n times."""
    words = text.split()
    if len(words) < 3:
        return text
    for _ in range(min(n_swaps, len(words) - 1)):
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return " ".join(words)


def headline_manipulation(text):
    """Swap first sentence with sensationalized version."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 2:
        return text
    prefixes = ["BREAKING:", "SHOCKING:", "EXCLUSIVE:", "REVEALED:", "EXPOSED:"]
    sentences[0] = f"{random.choice(prefixes)} {sentences[0]}"
    return " ".join(sentences)


def augment_single(text, label):
    """Apply a random augmentation to a single article."""
    method = random.choice(["synonym", "deletion", "swap", "headline"])
    if method == "synonym":
        return synonym_replace(text), f"synonym_replace"
    elif method == "deletion":
        return random_deletion(text), f"random_deletion"
    elif method == "swap":
        return random_swap(text), f"random_swap"
    else:
        return headline_manipulation(text), f"headline_manipulation"


def main():
    parser = argparse.ArgumentParser(description="Augment training data")
    parser.add_argument("--input", default=os.path.join(PROJECT_ROOT, "data", "news.csv"))
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "data", "news_augmented.csv"))
    parser.add_argument("--multiplier", type=int, default=1, help="Augmented copies per original")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit input size")
    parser.add_argument("--satire-count", type=int, default=200, help="Number of satire samples")
    args = parser.parse_args()

    print("Loading dataset...")
    df = pd.read_csv(args.input)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[["text", "label"]].dropna()

    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42)

    original_count = len(df)
    print(f"Original dataset: {original_count:,} articles")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Add augmentation_type column to original
    df["augmentation_type"] = "original"

    # Augment
    augmented_rows = []
    for _ in range(args.multiplier):
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Augmenting (x{args.multiplier})"):
            aug_text, aug_type = augment_single(str(row["text"]), str(row["label"]))
            augmented_rows.append({
                "text": aug_text,
                "label": row["label"],
                "augmentation_type": aug_type,
            })

    aug_df = pd.DataFrame(augmented_rows)

    # Generate satire
    print(f"Generating {args.satire_count} satirical samples...")
    satire_rows = generate_satire(args.satire_count)
    satire_df = pd.DataFrame(satire_rows)

    # Combine
    final_df = pd.concat([df, aug_df, satire_df], ignore_index=True)
    final_df.to_csv(args.output, index=False)

    print(f"\n{'='*50}")
    print(f"Augmentation Summary")
    print(f"{'='*50}")
    print(f"  Original:   {original_count:,}")
    print(f"  Augmented:  {len(aug_df):,}")
    print(f"  Satire:     {len(satire_df):,}")
    print(f"  Total:      {len(final_df):,}")
    print(f"\n  Per-class distribution:")
    for label, count in final_df["label"].value_counts().items():
        print(f"    {label}: {count:,}")
    print(f"\n  Augmentation types:")
    for aug_type, count in final_df["augmentation_type"].value_counts().items():
        print(f"    {aug_type}: {count:,}")
    print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()
