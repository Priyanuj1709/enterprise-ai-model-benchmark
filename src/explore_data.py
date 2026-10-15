"""First look: labels, examples, and how many messages are near-duplicates."""
import re
import pandas as pd

df = pd.read_csv("data/raw/bitext_customer_support.csv")
print(df.shape)
print(df.columns.tolist())

print("\n=== Categories ===")
print(df["category"].value_counts())
print("\n=== Intents ===")
print(df["intent"].value_counts())

print("\n=== 3 example messages per intent ===")
for intent, group in df.groupby("intent"):
    print(f"\n--- {intent} ---")
    for text in group["instruction"].sample(3, random_state=42):
        print("   ", text)

def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\{\{.*?\}\}", "<slot>", text)   # {{Order Number}} -> <slot>
    text = re.sub(r"[^a-z<> ]", " ", text)          # drop punctuation and digits
    return re.sub(r"\s+", " ", text).strip()

print("\nExact duplicate messages:", df["instruction"].duplicated().sum())
print("Near-duplicates (after normalising):", df["instruction"].map(normalise).duplicated().sum())