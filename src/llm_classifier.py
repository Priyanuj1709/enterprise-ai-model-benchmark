"""Approach A: Llama 3.2 3B via Ollama, prompt-only (no training).
Prompts are compared on the VALIDATION set only. test.csv is not read here.
Usage:  python src/llm_classifier.py --prompt v1 --limit 50
        python src/llm_classifier.py --prompt v2"""
import argparse
import random
import time
from pathlib import Path

import ollama
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

MODEL = "llama3.2:3b"
OPTIONS = {"temperature": 0, "num_predict": 8, "seed": 0}   # short, repeatable answers
LABELS = ["Delivery", "Refund", "Payment", "Product Quality", "Other"]

SYSTEM = """You classify customer enquiries for a UK online shop into exactly one category.

Categories:
- Delivery: where or when an order will arrive, tracking, delivery options, delivery address
- Refund: refund status, refund policy, getting money back for non-fault reasons
- Payment: failed payments, payment methods, invoices, charges
- Product Quality: the item is faulty, broken, damaged, poor quality or not as described
- Other: accounts, newsletters, placing/changing/cancelling orders, contacting staff, anything else

Reply with the category name only."""

RULES = """

If a message fits two categories:
1. A faulty or damaged item is Product Quality, even if the customer asks for a refund.
2. An item damaged on arrival is Product Quality.
3. A late delivery plus a refund request is Delivery."""


def parse(raw: str):
    """Turn the model's reply into one of LABELS, or None if it can't be matched."""
    t = raw.strip().lower().replace("-", " ").replace("_", " ")
    for lab in LABELS:                      # normal case: reply starts with the label
        if t.startswith(lab.lower()):
            return lab
    for lab in LABELS:                      # fallback: label somewhere in the reply
        if lab.lower() in t:
            return lab
    return None


def build_messages(text, version, examples):
    if version == "v1":
        msgs = [{"role": "system", "content": SYSTEM}]
    else:                                   # v2: rules + few-shot examples as past turns
        msgs = [{"role": "system", "content": SYSTEM + RULES}]
        for ex_text, ex_label in examples:
            msgs += [{"role": "user", "content": ex_text},
                     {"role": "assistant", "content": ex_label}]
    msgs.append({"role": "user", "content": text})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=["v1", "v2"], default="v1")
    ap.add_argument("--limit", type=int, default=None, help="quick test on N messages")
    args = ap.parse_args()

    val = pd.read_csv("data/splits/val.csv")
    if args.limit:
        val = val.sample(args.limit, random_state=0)

    # Few-shot examples come from TRAIN only (2 per category, shuffled)
    train = pd.read_csv("data/splits/train.csv")
    examples = [(r.text, r.label) for lab in LABELS
                for r in train[train["label"] == lab].sample(2, random_state=1).itertuples()]
    random.Random(0).shuffle(examples)

    start = time.perf_counter()
    ollama.chat(model=MODEL, messages=[{"role": "user", "content": "Hi"}], options=OPTIONS)
    print(f"Warm-up (model load): {time.perf_counter() - start:.1f} s")

    rows = []
    for r in tqdm(val.itertuples(), total=len(val), desc=f"Prompt {args.prompt}"):
        t0 = time.perf_counter()
        resp = ollama.chat(model=MODEL, messages=build_messages(r.text, args.prompt, examples),
                           options=OPTIONS)
        seconds = time.perf_counter() - t0
        raw = resp["message"]["content"]
        pred = parse(raw)
        rows.append({"id": r.id, "text": r.text, "label": r.label, "raw": raw,
                     "pred": pred or "Other", "unparsed": pred is None, "seconds": seconds})

    out = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    suffix = f"_n{args.limit}" if args.limit else ""
    out.to_csv(f"results/llm_val_{args.prompt}{suffix}.csv", index=False)

    print(f"\nPrompt {args.prompt} on {len(out)} validation messages")
    print(f"Macro-F1: {f1_score(out['label'], out['pred'], average='macro'):.4f}")
    print(f"Unparsed replies: {out['unparsed'].sum()} ({out['unparsed'].mean():.1%})")
    print(f"Time per message: mean {out['seconds'].mean():.3f} s, "
          f"95th pct {out['seconds'].quantile(0.95):.3f} s")
    print(f"Throughput: {1 / out['seconds'].mean():.1f} messages/s on this GPU\n")
    print(classification_report(out["label"], out["pred"], digits=3, zero_division=0))
    if out["unparsed"].any():
        print("Examples of unparsed replies:", out.loc[out["unparsed"], "raw"].head(5).tolist())


if __name__ == "__main__":
    main()