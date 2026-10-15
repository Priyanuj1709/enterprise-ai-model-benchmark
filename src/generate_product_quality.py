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

# Problems grouped by product type, so combinations make sense
CATALOGUE = {
    "electrical": (["kettle", "toaster", "blender", "hair dryer", "electric toothbrush",
                    "bluetooth speaker", "wireless headphones", "smartwatch", "desk lamp"],
                   ["stopped working after a few days", "won't turn on", "keeps overheating",
                    "battery barely lasts", "makes a loud rattling noise", "stopped charging",
                    "has a burning smell", "keeps cutting out"]),
    "clothing":   (["winter coat", "jeans", "running trainers", "dress", "jumper"],
                   ["stitching is coming apart", "colour is different from the photos",
                    "size is nothing like the size guide", "bobbled after one wash",
                    "zip broke", "sole is peeling off"]),
    "home":       (["sofa", "office chair", "suitcase", "frying pan", "bookshelf"],
                   ["arrived scratched", "has a part missing", "feels cheap and flimsy",
                    "a wheel or leg snapped", "is wobbly even when assembled properly",
                    "doesn't look anything like the website photos"]),
}
TONES = ["polite", "angry", "frustrated but calm", "very brief", "confused", "disappointed"]
STYLES = ["all lowercase with no punctuation", "with a couple of spelling mistakes",
          "formal British English", "casual text-message style", "long and rambling",
          "plain and simple"]
REFERENCES = ["names the product", "says 'it' or 'the item' without naming the product",
              "mentions order number #" , "describes what it is used for instead of naming it"]
OPENINGS = ["I", "My", "So", "Just", "Why", "Can", "Hello", "The", "Not happy", "Ordered"]
WANTS = ["nothing, just reporting it", "what to do next", "a replacement", "a repair",
         "whether the warranty covers it", "their money back"]
WANT_WEIGHTS = [3, 3, 2, 1, 1, 1]          # most customers don't name a remedy

PROMPT = """You write realistic messages that customers send to a UK online shop's support inbox.
Write exactly {n} messages, one per specification below.
Rules:
- one or two sentences, 6 to 35 words
- written as the customer, British English
- use your OWN words: do NOT copy phrases from the specification
- begin each message with the opening word given
- if a problem does not make sense for the product, change it to a realistic one
Return JSON only, in this form: {{"messages": ["...", "..."]}}

Specifications:
{specs}"""


def make_specs(n: int) -> str:
    lines = []
    for i in range(1, n + 1):
        products, problems = random.choice(list(CATALOGUE.values()))
        ref = random.choice(REFERENCES)
        if ref.startswith("mentions order"):
            ref += str(random.randint(100000, 999999))
        want = random.choices(WANTS, weights=WANT_WEIGHTS)[0]
        lines.append(f"{i}. product: {random.choice(products)} | problem: {random.choice(problems)}"
                     f" | tone: {random.choice(TONES)} | style: {random.choice(STYLES)}"
                     f" | message {ref} | customer wants: {want}"
                     f" | opening word: {random.choice(OPENINGS)}")
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
                               format="json", options={"temperature": 1.0, "seed": batch_no})
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