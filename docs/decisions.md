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