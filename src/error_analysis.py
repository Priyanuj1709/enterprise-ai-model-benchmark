"""Step 7a: where and why does each approach go wrong?"""
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 95)


def wilson(correct, n, z=1.96):
    """95% confidence interval for an accuracy measured on n messages."""
    p, d = correct / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return centre - half, centre + half


for name in ["test", "challenge"]:
    df = pd.read_csv(f"results/{name}_predictions.csv")
    b_ok = df["pred_baseline"] == df["label"]
    a_ok = df["pred_llm"] == df["label"]
    print(f"\n================ {name.upper()} ({len(df)} messages) ================")

    for ok, approach in [(b_ok, "TF-IDF"), (a_ok, "LLM")]:
        lo, hi = wilson(ok.sum(), len(df))
        print(f"{approach:7} accuracy {ok.mean():.1%}  (95% CI {lo:.1%} - {hi:.1%})")

    print(f"\nBoth right {(b_ok & a_ok).sum()} | only TF-IDF right {(b_ok & ~a_ok).sum()} | "
          f"only LLM right {(~b_ok & a_ok).sum()} | both wrong {(~b_ok & ~a_ok).sum()}")

    for col, approach in [("pred_baseline", "TF-IDF"), ("pred_llm", "LLM")]:
        wrong = df[df[col] != df["label"]]
        print(f"\nTop mix-ups, {approach} (true -> predicted):")
        print((wrong["label"] + " -> " + wrong[col]).value_counts().head(5).to_string())

    if name == "challenge":
        cols = ["text", "label", "pred_baseline", "pred_llm", "ambiguous"]
        print("\nAll challenge messages the LLM got wrong:")
        print(df.loc[~a_ok, cols].to_string(index=False))
        print("\nTF-IDF mistakes the LLM got right (first 12):")
        print(df.loc[~b_ok & a_ok, cols].head(12).to_string(index=False))
    else:
        print("\nSample of LLM mistakes on the test set (15):")
        print(df.loc[~a_ok, ["text", "label", "pred_llm"]].sample(15, random_state=0)
              .to_string(index=False))