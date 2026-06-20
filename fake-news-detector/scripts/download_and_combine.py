"""
Script to download LIAR, McIntire (zipped from joolsa fork), and COVID-19 fake news datasets from public sources,
combine them with the existing Kaggle Fake/True news dataset,
and export a unified news.csv ready for training.
"""

import os
import sys
import io
import zipfile
import pandas as pd
import requests

# Fix Unicode output on Windows consoles (cp1252 can't print box-drawing/emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

LIAR_ZIP_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
# Using joolsa's active fork of McIntire dataset
MCINTIRE_ZIP_URL = "https://github.com/joolsa/fake_real_news_dataset/raw/master/fake_or_real_news.csv.zip"
COVID_TRAIN_URL = "https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/Constraint_Train.csv"
COVID_VAL_URL = "https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/Constraint_Val.csv"

def download_liar_dataset():
    print("Downloading LIAR dataset ZIP from UCSB...")
    try:
        response = requests.get(LIAR_ZIP_URL, timeout=60)
        response.raise_for_status()
        
        z = zipfile.ZipFile(io.BytesIO(response.content))
        print("   ZIP downloaded successfully.")
        
        liar_dfs = []
        files_to_load = {
            "train": "train.tsv",
            "test": "test.tsv",
            "valid": "valid.tsv"
        }
        
        for part, filename in files_to_load.items():
            with z.open(filename) as f:
                df = pd.read_csv(f, sep='\t', header=None)
                df = df[[2, 1]].rename(columns={2: "text", 1: "label_raw"})
                liar_dfs.append(df)
                
    except Exception as e:
        print(f"   Error downloading/extracting LIAR dataset: {e}")
        return pd.DataFrame(columns=['text', 'label'])
        
    liar_combined = pd.concat(liar_dfs, ignore_index=True)
    
    label_map = {
        "pants-fire": "FAKE",
        "false": "FAKE",
        "barely-true": "FAKE",
        "mostly-true": "REAL",
        "true": "REAL"
    }
    
    liar_combined['label'] = liar_combined['label_raw'].map(label_map)
    liar_combined = liar_combined.dropna(subset=['label'])
    liar_combined = liar_combined[['text', 'label']]
    print(f"   ✓ Processed {len(liar_combined):,} valid LIAR statements")
    return liar_combined

def download_mcintire_dataset():
    print("Downloading McIntire Fake or Real News Dataset ZIP from active fork...")
    try:
        response = requests.get(MCINTIRE_ZIP_URL, timeout=60)
        response.raise_for_status()
        
        z = zipfile.ZipFile(io.BytesIO(response.content))
        print("   McIntire ZIP downloaded successfully.")
        
        with z.open("fake_or_real_news.csv") as f:
            df = pd.read_csv(f)
            
        df['text'] = df['title'].fillna('') + " " + df['text'].fillna('')
        df['label'] = df['label'].str.upper()
        df = df[['text', 'label']].dropna()
        print(f"   ✓ Loaded {len(df):,} rows from McIntire dataset")
        return df
    except Exception as e:
        print(f"   Error downloading/extracting McIntire dataset: {e}")
        return pd.DataFrame(columns=['text', 'label'])

def download_covid_dataset():
    print("Downloading COVID-19 Misinformation Dataset...")
    try:
        train_df = pd.read_csv(COVID_TRAIN_URL)
        val_df = pd.read_csv(COVID_VAL_URL)
        df = pd.concat([train_df, val_df], ignore_index=True)
        df = df.rename(columns={'tweet': 'text'})
        df['label'] = df['label'].str.upper()
        df = df[['text', 'label']].dropna()
        print(f"   ✓ Loaded {len(df):,} rows from COVID-19 dataset")
        return df
    except Exception as e:
        print(f"   Error downloading COVID-19 dataset: {e}")
        return pd.DataFrame(columns=['text', 'label'])


def load_ifnd_dataset():
    """
    Load the IFND (Indian Fake News Dataset) if present in data/.

    Expected format: CSV with columns 'Statement' and 'Label' (1=Real, 0=Fake).
    Download from: https://www.kaggle.com/datasets/sharmila-garg/ifnd-dataset
    """
    ifnd_path = os.path.join(DATA_DIR, "IFND_dataset.csv")
    if not os.path.exists(ifnd_path):
        # Try alternate names
        for alt_name in ["ifnd.csv", "IFND.csv", "ifnd_dataset.csv"]:
            alt_path = os.path.join(DATA_DIR, alt_name)
            if os.path.exists(alt_path):
                ifnd_path = alt_path
                break
        else:
            print("   ⏭️ IFND dataset not found in data/ (optional — download from Kaggle)")
            return pd.DataFrame(columns=['text', 'label'])

    print("Loading IFND (Indian Fake News Dataset)...")
    try:
        df = pd.read_csv(ifnd_path, encoding='latin1')
        df.columns = [col.strip() for col in df.columns]

        # Find the text column
        text_col = None
        for col in ['Statement', 'statement', 'text', 'Text', 'content', 'Content']:
            if col in df.columns:
                text_col = col
                break
        if text_col is None:
            print(f"   ⚠️ Could not find text column in IFND. Columns: {list(df.columns)}")
            return pd.DataFrame(columns=['text', 'label'])

        # Find the label column
        label_col = None
        for col in ['Label', 'label', 'class', 'Class', 'target']:
            if col in df.columns:
                label_col = col
                break
        if label_col is None:
            print(f"   ⚠️ Could not find label column in IFND. Columns: {list(df.columns)}")
            return pd.DataFrame(columns=['text', 'label'])

        df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
        df = df.dropna()

        # Map labels: 1 = REAL, 0 = FAKE (IFND convention)
        label_map = {}
        for val in df['label'].unique():
            val_str = str(val).strip().upper()
            if val_str in ['1', 'TRUE', 'REAL']:
                label_map[val] = 'REAL'
            elif val_str in ['0', 'FALSE', 'FAKE']:
                label_map[val] = 'FAKE'
        if label_map:
            df['label'] = df['label'].map(label_map)
            df = df.dropna(subset=['label'])

        df = df[['text', 'label']]
        print(f"   ✓ Loaded {len(df):,} articles from IFND Indian dataset")
        return df
    except Exception as e:
        print(f"   ⚠️ Error loading IFND dataset: {e}")
        return pd.DataFrame(columns=['text', 'label'])


def load_indian_fakenews_dataset():
    """
    Load the Indian Fake News dataset if present in data/.

    Expected format: CSV with 'text' and 'label' columns (FAKE/REAL).
    Download from: https://www.kaggle.com/datasets/ruchi798/indian-fake-news
    """
    for fname in ["indian_fake_news.csv", "Indian_Fake_News.csv", "indian-fake-news.csv"]:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            print(f"Loading Indian Fake News dataset ({fname})...")
            try:
                df = pd.read_csv(fpath)
                df.columns = [col.strip().lower() for col in df.columns]

                if 'text' not in df.columns:
                    for col in df.columns:
                        if col in ['content', 'article', 'body', 'news', 'statement']:
                            df.rename(columns={col: 'text'}, inplace=True)
                            break

                if 'label' not in df.columns:
                    for col in df.columns:
                        if col in ['class', 'target', 'category', 'fake']:
                            df.rename(columns={col: 'label'}, inplace=True)
                            break

                if 'text' in df.columns and 'label' in df.columns:
                    df = df[['text', 'label']].dropna()
                    # Normalize labels
                    label_map = {}
                    for val in df['label'].unique():
                        val_str = str(val).strip().upper()
                        if val_str in ['FAKE', '0', 'FALSE', 'UNRELIABLE']:
                            label_map[val] = 'FAKE'
                        else:
                            label_map[val] = 'REAL'
                    df['label'] = df['label'].map(label_map)
                    df = df.dropna(subset=['label'])
                    print(f"   ✓ Loaded {len(df):,} rows from Indian Fake News dataset")
                    return df
                else:
                    print(f"   ⚠️ Missing text/label columns. Found: {list(df.columns)}")
            except Exception as e:
                print(f"   ⚠️ Error loading {fname}: {e}")

    print("   ⏭️ Indian Fake News dataset not found in data/ (optional)")
    return pd.DataFrame(columns=['text', 'label'])


def combine_all_datasets():
    # 1. Download LIAR
    liar_df = download_liar_dataset()
    
    # 2. Download McIntire
    mcintire_df = download_mcintire_dataset()
    
    # 3. Download COVID-19
    covid_df = download_covid_dataset()

    # 4. Load Indian datasets (if present in data/)
    ifnd_df = load_ifnd_dataset()
    indian_fn_df = load_indian_fakenews_dataset()
    
    # 5. Load existing Fake.csv and True.csv if they exist
    existing_dfs = []
    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")
    
    if os.path.exists(fake_path) and os.path.exists(true_path):
        print("Loading local Kaggle Fake/True datasets...")
        try:
            fake_df = pd.read_csv(fake_path)
            true_df = pd.read_csv(true_path)
            
            fake_df['text'] = fake_df['title'].fillna('') + " " + fake_df['text'].fillna('')
            true_df['text'] = true_df['title'].fillna('') + " " + true_df['text'].fillna('')
            
            fake_df['label'] = 'FAKE'
            true_df['label'] = 'REAL'
            
            existing_dfs.extend([fake_df[['text', 'label']], true_df[['text', 'label']]])
            print(f"   ✓ Loaded {len(fake_df):,} fake and {len(true_df):,} real articles from local files")
        except Exception as e:
            print(f"   ⚠️ Could not load local Fake.csv/True.csv: {e}")
            
    # 6. Concatenate all datasets
    all_dfs = [liar_df, mcintire_df, covid_df, ifnd_df, indian_fn_df] + existing_dfs
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Clean up whitespace
    combined_df['text'] = combined_df['text'].str.strip()
    # Remove empty text rows
    combined_df = combined_df[combined_df['text'].str.len() > 10].dropna()
    
    # Shuffle the dataset to mix classes and sources
    print("   ✓ Shuffling dataset...")
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Export to news.csv
    output_path = os.path.join(DATA_DIR, "news.csv")
    print(f"\nSaving {len(combined_df):,} total samples to {output_path}...")
    combined_df.to_csv(output_path, index=False, encoding='utf-8')
    print("   ✓ Done! You can now run 'python train_model.py' to retrain your model.")

if __name__ == "__main__":
    combine_all_datasets()

