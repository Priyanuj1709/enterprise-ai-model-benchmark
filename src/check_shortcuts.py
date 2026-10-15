"""Look for shortcut patterns that would let a classifier cheat."""
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("data/processed/dataset.csv")

print("Most common FIRST WORD per category:")
for label, group in df.groupby("label"):
    first = group["text"].str.split().str[0].str.lower()
    top = [(w, f"{c / len(group):.0%}") for w, c in Counter(first).most_common(4)]
    print(f"  {label:16} {top}")

print("\nMost common 3-word phrases in Product Quality (share of messages):")
pq = df.loc[df["label"] == "Product Quality", "text"]
vec = CountVectorizer(ngram_range=(3, 3))
counts = vec.fit_transform(pq).sum(axis=0).A1
names = vec.get_feature_names_out()
for i in counts.argsort()[::-1][:15]:
    print(f"  {names[i]:30} {counts[i] / len(pq):.0%}")