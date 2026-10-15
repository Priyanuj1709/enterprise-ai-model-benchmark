"""Download the Bitext customer-support dataset and save it as a CSV."""
from pathlib import Path
from datasets import load_dataset

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"
)
df = ds.to_pandas()

out = RAW_DIR / "bitext_customer_support.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df):,} rows to {out}")