"""Combine Bitext (4 classes) and generated Product Quality into one dataset.
Validates rows first, because manual editing can leave broken lines."""
import pandas as pd

VALID_LABELS = {"Delivery", "Refund", "Payment", "Product Quality", "Other"}

bitext = pd.read_csv("data/processed/bitext_mapped.csv")
pq = pd.read_csv("data/processed/product_quality_generated.csv")

# Check 1: both files have the columns we expect
for name, part in [("bitext_mapped", bitext), ("product_quality_generated", pq)]:
    assert {"text", "label"} <= set(part.columns), f"{name}.csv columns are {list(part.columns)}"

df = pd.concat([bitext, pq], ignore_index=True)

# Check 2: keep only rows with real text and a valid label
ok = df["text"].notna() & df["text"].str.strip().ne("") & df["label"].isin(VALID_LABELS)
if (~ok).any():
    print(f"Dropping {(~ok).sum()} broken rows:")
    print(df.loc[~ok, ["text", "label"]].head(10).to_string(), "\n")
df = df[ok]

# Check 3: no category has disappeared
missing = VALID_LABELS - set(df["label"])
assert not missing, f"Categories missing after validation: {missing}"

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, "id", range(len(df)))
df.to_csv("data/processed/dataset.csv", index=False)

print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True).round(3))
print(f"\nTotal: {len(df):,} rows")