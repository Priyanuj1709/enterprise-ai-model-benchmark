"""Step 6: the ONE-TIME final evaluation of both approaches.
Runs on the locked test set and the hand-written challenge set. Nothing is tuned here."""
import hashlib
import random
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import ollama
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, precision_recall_fscore_support)
from tqdm import tqdm

from llm_classifier import LABELS, MODEL, OPTIONS, build_messages, parse

# --- 1. Prove the test set is unchanged since it was locked ------------------
expected = Path("data/splits/test.sha256").read_text().strip()
actual = hashlib.sha256(Path("data/splits/test.csv").read_bytes()).hexdigest()
assert actual == expected, "test.csv has changed since it was locked!"
print("Test set fingerprint verified.\n")

DATASETS = {
    "test": pd.read_csv("data/splits/test.csv"),
    "challenge": pd.read_csv("data/challenge/challenge_set.csv"),
}

# --- 2. Load the frozen approaches -------------------------------------------
baseline = joblib.load("models/tfidf_logreg.joblib")

train = pd.read_csv("data/splits/train.csv")          # same few-shot examples as validation
examples = [(r.text, r.label) for lab in LABELS
            for r in train[train["label"] == lab].sample(2, random_state=1).itertuples()]
random.Random(0).shuffle(examples)
ollama.chat(model=MODEL, messages=[{"role": "user", "content": "Hi"}], options=OPTIONS)  # warm-up


def run_baseline(texts):
    preds, secs = [], []
    for t in texts:                                     # one at a time, like the LLM
        t0 = time.perf_counter()
        preds.append(baseline.predict([t])[0])
        secs.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    baseline.predict(texts)                             # batch mode, as it would really run
    batch_ms = (time.perf_counter() - t0) / len(texts) * 1000
    return preds, secs, batch_ms


def run_llm(texts):
    preds, raws, secs, unparsed = [], [], [], []
    for t in tqdm(texts, desc="LLM"):
        t0 = time.perf_counter()
        resp = ollama.chat(model=MODEL, messages=build_messages(t, "v2", examples), options=OPTIONS)
        secs.append(time.perf_counter() - t0)
        raw = resp["message"]["content"]
        p = parse(raw)
        raws.append(raw)
        preds.append(p or "Other")
        unparsed.append(p is None)
    return preds, raws, secs, unparsed


def macro_metrics(y, p):
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=LABELS,
                                                       average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, p), "macro_precision": prec,
            "macro_recall": rec, "macro_f1": f1}


# --- 3. Run everything --------------------------------------------------------
summary, reports = [], []
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

for row, (name, df) in enumerate(DATASETS.items()):
    print(f"\n### {name}: {len(df)} messages")
    texts = df["text"].tolist()
    b_pred, b_sec, b_batch_ms = run_baseline(texts)
    l_pred, l_raw, l_sec, l_unp = run_llm(texts)

    df = df.assign(pred_baseline=b_pred, pred_llm=l_pred, raw_llm=l_raw,
                   unparsed_llm=l_unp, sec_baseline=b_sec, sec_llm=l_sec)
    Path("results").mkdir(exist_ok=True)
    df.to_csv(f"results/{name}_predictions.csv", index=False)

    for col, approach, secs, col_idx in [("pred_baseline", "B: TF-IDF + LogReg", b_sec, 0),
                                         ("pred_llm", "A: Llama 3.2 3B (prompt v2)", l_sec, 1)]:
        m = macro_metrics(df["label"], df[col])
        s = pd.Series(secs) * 1000
        result = {"dataset": name, "approach": approach,
                  **{k: round(v, 4) for k, v in m.items()},
                  "mean_ms": round(s.mean(), 2), "p95_ms": round(s.quantile(0.95), 2),
                  "batch_ms": round(b_batch_ms, 3) if col == "pred_baseline" else None,
                  "unparsed": int(sum(l_unp)) if col == "pred_llm" else 0}
        if "ambiguous" in df.columns:                   # challenge set only
            for flag, key in [("no", "acc_clear"), ("yes", "acc_ambiguous")]:
                sub = df[df["ambiguous"] == flag]
                result[key] = round(accuracy_score(sub["label"], sub[col]), 3)
        summary.append(result)

        reports.append(f"=== {name} | {approach} ===\n"
                       + classification_report(df["label"], df[col], labels=LABELS,
                                               digits=3, zero_division=0))
        ax = axes[row, col_idx]
        ConfusionMatrixDisplay.from_predictions(df["label"], df[col], labels=LABELS, cmap="Blues",
                                                colorbar=False, ax=ax, xticks_rotation=30)
        ax.set_title(f"{name} set - {approach}\nmacro-F1 {m['macro_f1']:.3f}")

# --- 4. Save and print --------------------------------------------------------
summary_df = pd.DataFrame(summary)
summary_df.to_csv("results/summary.csv", index=False)
Path("results/classification_reports.txt").write_text("\n".join(reports))
Path("reports/figures").mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig("reports/figures/confusion_matrices.png", dpi=150)

print("\n".join(reports))
pd.set_option("display.width", 200)
print(summary_df.to_string(index=False))
print("\nSaved results/summary.csv, results/*_predictions.csv, "
      "results/classification_reports.txt, reports/figures/confusion_matrices.png")