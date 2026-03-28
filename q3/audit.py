import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from text_processing import is_valid_english_transcript

DATA_ROOT = "mozilla_common_voice"
TSV_PATH = os.path.join(DATA_ROOT, "ss-corpus-en.tsv")

def load_split(split="train"):
    df = pd.read_csv(TSV_PATH, sep="\t")

    # -------------------------------
    # Filter by split
    # -------------------------------
    df = df[df["split"] == split]

    # -------------------------------
    # Basic cleanup
    # -------------------------------
    df = df.dropna(subset=["audio_file", "transcription"])
    df = df[df["audio_file"].apply(lambda x: isinstance(x, str))]
    df = df[df["transcription"].apply(is_valid_english_transcript)]

    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} samples for split: {split}")
    return df

# -------------------------------
# Documentation Debt Audit
# -------------------------------
def audit_dataset(df):
    print("\n Missing Values:")
    print(df[["age", "gender", "transcription"]].isna().mean())

    print("\n Gender Distribution:")
    print(df["gender"].value_counts())

    print("\n Age Distribution:")
    print(df["age"].value_counts())

    # Plots
    plt.figure()
    sns.countplot(x=df["gender"])
    plt.title("Gender Distribution")
    plt.show()

    plt.figure()
    sns.countplot(x=df["age"])
    plt.title("Age Distribution")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == '__main__':
    df = load_split("train")
    audit_dataset(df)
