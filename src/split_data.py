"""Split dataset.csv into train / validation / test (about 70/15/15).
- Stratified: every split keeps the same class mix (representative).
- Grouped: near-duplicate messages always land in the same split (no leakage).
The test set is used ONCE, at the very end (Step 6)."""
import hashlib
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
VALID_LABELS = {"Delivery", "Refund", "Payment", "Product Quality", "Other"}
# Values we inserted for placeholders - ignored when finding near-duplicates
FILLER_WORDS = {"sarah", "james", "priya", "tom", "aisha", "leeds", "bristol",
                "glasgow", "cardiff", "manchester", "uk", "inv"}


def group_key(text: str) -> str:
    """Lowercase, keep letters only, drop filler values -> near-duplicates share a key."""
    words = re.sub(r"[^a-z ]", " ", str(text).lower()).split()
    return " ".join(w for w in words if w not in FILLER_WORDS)


def carve(data: pd.DataFrame, n_splits: int):
    """Hold out one fold (1/n_splits of the data), stratified by label, grouped by key."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    keep_idx, held_idx = next(sgkf.split(data, data["label"], groups=data["group"]))
    return data.iloc[keep_idx], data.iloc[held_idx]


# --- Load and check input -------------------------------------------------
df = pd.read_csv("data/processed/dataset.csv")
if df["text"].isna().any() or df["label"].isna().any():
    raise SystemExit("dataset.csv has rows with no text/label - run combine_dataset.py first.")
assert set(df["label"]) == VALID_LABELS, f"Unexpected labels: {set(df['label'])}"

df["group"] = df["text"].map(group_key)
print(f"{len(df):,} messages in {df['group'].nunique():,} near-duplicate groups")
biggest = df["group"].value_counts().iloc[0]
print(f"Largest group: {biggest} messages\n")

# --- Split ----------------------------------------------------------------
rest, test = carve(df, n_splits=7)      # ~14% -> test
train, val = carve(rest, n_splits=6)    # ~14% of the total -> validation
splits = {"train": train, "val": val, "test": test}

# --- Check 1: no near-duplicate group appears in two splits ---------------
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    shared = set(splits[a]["group"]) & set(splits[b]["group"])
    print(f"Shared groups {a}/{b}: {len(shared)}")
    assert not shared, "Leakage! A near-duplicate group is in two splits."

# --- Check 2: every split has every class, in the same proportions ---------
print("\nSize:", {name: len(s) for name, s in splits.items()})
mix = pd.DataFrame({name: s["label"].value_counts(normalize=True) for name, s in splits.items()})
print("\nClass mix per split:")
print(mix.round(3))
assert not mix.isna().any().any(), "A class is missing from one of the splits."
assert (mix.max(axis=1) - mix.min(axis=1)).max() < 0.02, "Class mix differs between splits."

# --- Save -----------------------------------------------------------------
out = Path("data/splits")
out.mkdir(parents=True, exist_ok=True)
for name, s in splits.items():
    s.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")

# --- Lock the test set with a fingerprint ----------------------------------
digest = hashlib.sha256((out / "test.csv").read_bytes()).hexdigest()
(out / "test.sha256").write_text(digest + "\n")
print(f"\nTest set locked. SHA-256: {digest[:16]}...")