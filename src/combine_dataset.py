"""Combine Bitext (4 classes) and generated Product Quality into one dataset."""
import pandas as pd

bitext = pd.read_csv("data/processed/bitext_mapped.csv")
pq = pd.read_csv("data/processed/product_quality_generated.csv")

df = pd.concat([bitext, pq]).sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, "id", range(len(df)))
df.to_csv("data/processed/dataset.csv", index=False)

print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True).round(3))
print(f"\nTotal: {len(df):,} rows")