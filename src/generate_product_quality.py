"""Generate synthetic Product Quality enquiries with Mistral via Ollama.
LLM output is not exactly reproducible, so the result CSV is committed to Git."""
import json
import random
import re
from pathlib import Path

import ollama
import pandas as pd
from tqdm import tqdm

SEED = 42
random.seed(SEED)
MODEL = "mistral"
TARGET = 1000          # rows we want in the end
BATCH = 10             # messages per request
MAX_BATCHES = 160      # safety limit

PRODUCTS = ["kettle", "wireless headphones", "running trainers", "sofa", "phone case",
            "winter coat", "blender", "laptop bag", "bedsheets", "toaster", "smartwatch",
            "children's toy", "desk lamp", "frying pan", "jeans", "hair dryer", "suitcase",
            "electric toothbrush", "office chair", "bluetooth speaker"]
PROBLEMS = ["stopped working after a few days", "broke the first time I used it",
            "has stitching coming apart", "has a strange chemical smell",
            "doesn't match the description on the website",
            "is a completely different colour from the photos",
            "is much smaller than advertised", "keeps overheating",
            "has a battery that dies within an hour", "feels cheap and flimsy",
            "was scratched when I opened the box", "arrived with a part missing",
            "leaks", "makes a loud rattling noise", "arrived cracked"]
TONES = ["polite", "angry", "frustrated but calm", "very brief", "confused", "disappointed"]
STYLES = ["all lowercase with no punctuation", "with a couple of spelling mistakes",
          "formal British English", "casual text-message style",
          "phrased as a question", "plain and simple"]
WANTS = ["nothing specific, just reporting it", "asking what to do next",
         "asking for a replacement", "asking if it is covered by the warranty",
         "asking for a refund or replacement"]

PROMPT = """You write realistic messages that customers send to a UK online shop's support inbox.
Write exactly {n} messages, one per specification below.
Rules:
- each message is one or two sentences, 8 to 30 words
- no greeting (no "Dear", "Hi") and no sign-off
- first person, written as the customer
- British English spelling; use £ for money
Return JSON only, in this form: {{"messages": ["...", "..."]}}

Specifications:
{specs}"""


def make_specs(n: int) -> str:
    lines = []
    for i in range(1, n + 1):
        lines.append(f"{i}. product: {random.choice(PRODUCTS)} | problem: {random.choice(PROBLEMS)}"
                     f" | tone: {random.choice(TONES)} | style: {random.choice(STYLES)}"
                     f" | customer wants: {random.choice(WANTS)}")
    return "\n".join(lines)


def clean(msg: str) -> str:
    msg = re.sub(r"^\s*\d+[\.\)]\s*", "", str(msg))   # remove "1. " numbering
    return msg.strip().strip('"').strip()


messages, seen = [], set()
with tqdm(total=int(TARGET * 1.15), desc="Generating") as bar:
    for batch_no in range(MAX_BATCHES):
        if len(messages) >= TARGET * 1.15:      # make 15% extra, to allow filtering
            break
        prompt = PROMPT.format(n=BATCH, specs=make_specs(BATCH))
        try:
            resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                               format="json", options={"temperature": 0.9, "seed": batch_no})
            batch = json.loads(resp["message"]["content"]).get("messages", [])
        except Exception as e:                   # bad JSON etc. -> skip this batch
            print(f"Batch {batch_no} failed: {e}")
            continue
        for m in batch:
            m = clean(m)
            key = m.lower()
            if 5 <= len(m.split()) <= 40 and key not in seen:
                seen.add(key)
                messages.append(m)
                bar.update(1)

print(f"\nUnique usable messages: {len(messages)}")
df = pd.DataFrame({"text": messages[:TARGET] if len(messages) >= TARGET else messages})
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df["label"] = "Product Quality"
df["intent"] = "generated"
df["source"] = "mistral"

out = Path("data/processed/product_quality_generated.csv")
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
print("\n20 random examples:")
for t in df["text"].sample(20, random_state=1):
    print("  -", t)