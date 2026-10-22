"""One chart for the README: macro-F1 on the test set vs the challenge set, both approaches."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

s = pd.read_csv("results/summary.csv")
datasets = ["test", "challenge"]
tick_labels = ["Test set\n(same sources as training data)", "Challenge set\n(natural UK wording)"]
series = [("A:", "A: Llama 3.2 3B, local (prompt v2)", "#2a78d6"),
          ("B:", "B: TF-IDF + Logistic Regression", "#eb6834")]

fig, ax = plt.subplots(figsize=(8, 4.8))
width = 0.36
for i, (prefix, label, colour) in enumerate(series):
    values = [s[(s["dataset"] == d) & s["approach"].str.startswith(prefix)]["macro_f1"].iloc[0]
              for d in datasets]
    xs = [x + (i - 0.5) * (width + 0.02) for x in range(len(datasets))]   # small gap between bars
    ax.bar(xs, values, width=width, color=colour, label=label)
    for x, v in zip(xs, values):
        ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=10, color="#333333")

ax.set_xticks(range(len(datasets)), tick_labels)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Macro-F1 (higher is better)")
ax.set_title("Classification quality: familiar data vs natural customer wording", loc="left")
ax.yaxis.grid(True, color="#e5e5e5")
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
fig.tight_layout()

Path("reports/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("reports/figures/macro_f1_comparison.png", dpi=150)
print("Saved reports/figures/macro_f1_comparison.png")