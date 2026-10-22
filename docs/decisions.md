# Decision log

| # | Decision | Why | Trade-off / risk |
|---|---|---|---|
| D1 | Use the Bitext customer-support dataset (26,872 rows) | Free, labelled by intent, licence allows reuse | Synthetic, template-based, American English |
| D2 | Map 26 of the 27 intents to 4 business categories (see build_dataset.py) | Matches the company's categories | Invoice → Payment and cancel_order → Other are judgement calls |
| D3 | Leave out the `complaint` intent | A complaint can belong to any category | Fewer ambiguous examples in training |
| D4 | Replace {{placeholders}} with realistic UK values | Placeholder names like "refund amount" give away the label (shortcut) | Values are invented |
| D5 | Remove 2,237 exact duplicates; keep near-duplicates in the same split | Prevent data leakage between train and test | — |
| D6 | Use an assumed class mix: Delivery 35%, Refund 20%, Payment 15%, Product Quality 10%, Other 20% | Real inboxes are imbalanced; Bitext is not | Assumption, not company data |
| D7 | Generate Product Quality messages with Mistral 7B | Bitext has no such category; different model family from the one evaluated (Llama) | Possible generator-style shortcut; checked with check_shortcuts.py |
| D8 | v1 → v2 generator | v1 copied prompt phrases ("refund or replacement" in ~35%) and all began with the product name | Documented in Git history |
| D9 | Train/validation/test split 70/15/15; test used once at the end | Assignment rule: no tuning on the evaluation set | Smaller training set |

| D11 | Stratified + grouped split (~70/15/15) with StratifiedGroupKFold, seed 42 | Representative test set; near-duplicates never cross splits | Grouping by normalised text may miss paraphrases |
| D12 | Test set locked with a SHA-256 hash; opened once in Step 6 | Assignment rule: no tuning on evaluation data | — |
| D15 | Approach B: TF-IDF + Logistic Regression; 12 settings compared on validation by macro-F1; best refit on train+val | Macro-F1 weights small classes (Product Quality) equally; test set untouched | Small grid, kept simple on purpose |
| D16 | Val macro-F1 0.998 treated as optimistic | Val shares generators with train; real robustness measured on the hand-written challenge set | — |
| D17 | Coefficient inspection: Product Quality relies on generator opening words (just, why, hello, so...) and Payment memorised Bitext invoice numbers | Evidence that val macro-F1 0.998 is inflated by shortcut learning | Not fixed on purpose; impact measured on the hand-written challenge set |
| D18 | Final evaluation run once on locked test set (hash verified) and challenge set; frozen models (saved TF-IDF, prompt v2) | Honest final comparison, no tuning on evaluation data | Single run; LLM timing on a laptop GPU (RTX 4060) |
| D19 | Recommend Approach A (local LLM) with a shadow-mode pilot; re-test B once real labelled enquiries exist | Robust on natural wording (challenge 0.897 vs 0.546); negligible running cost; no labelled data needed | Evidence is synthetic + 60 messages; pilot must confirm on real traffic |