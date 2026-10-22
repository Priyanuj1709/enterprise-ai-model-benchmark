"""Approach B: TF-IDF + Logistic Regression.
Settings are chosen on the VALIDATION set only. test.csv is not read here."""
import itertools
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline

train = pd.read_csv("data/splits/train.csv")
val = pd.read_csv("data/splits/val.csv")

GRID = {
    "ngram_range": [(1, 1), (1, 2)],
    "C": [0.1, 1.0, 10.0],
    "class_weight": [None, "balanced"],
}


def build(ngram_range, C, class_weight):
    """TF-IDF features -> Logistic Regression, packaged as one pipeline."""
    return make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=ngram_range, min_df=2, sublinear_tf=True),
        LogisticRegression(C=C, class_weight=class_weight, max_iter=2000),
    )


# 1. Try every combination, score on validation
results = []
for ngram_range, C, class_weight in itertools.product(*GRID.values()):
    model = build(ngram_range, C, class_weight)
    model.fit(train["text"], train["label"])
    f1 = f1_score(val["label"], model.predict(val["text"]), average="macro")
    results.append({"ngram_range": ngram_range, "C": C,
                    "class_weight": class_weight, "val_macro_f1": round(f1, 4)})
    print(results[-1])

best = max(results, key=lambda r: r["val_macro_f1"])
print("\nBest settings:", best)

# 2. Detailed validation report for the best settings
model = build(best["ngram_range"], best["C"], best["class_weight"])
model.fit(train["text"], train["label"])
print("\nValidation report (best settings):")
print(classification_report(val["label"], model.predict(val["text"]), digits=3))

# 3. Final model: best settings, trained on train + validation
full = pd.concat([train, val])
start = time.perf_counter()
final = build(best["ngram_range"], best["C"], best["class_weight"])
final.fit(full["text"], full["label"])
train_seconds = time.perf_counter() - start
print(f"Final model trained on {len(full):,} rows in {train_seconds:.2f} s")

# 4. Save model, tuning table and chosen settings
Path("models").mkdir(exist_ok=True)
joblib.dump(final, "models/tfidf_logreg.joblib")
Path("results").mkdir(exist_ok=True)
pd.DataFrame(results).to_csv("results/baseline_tuning.csv", index=False)
with open("results/baseline_config.json", "w") as f:
    json.dump({"best": best, "train_seconds": round(train_seconds, 3),
               "train_rows": len(full)}, f, indent=2, default=str)
print("Saved models/tfidf_logreg.joblib and results/baseline_*.")