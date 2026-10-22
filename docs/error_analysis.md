# Error analysis

## 1. Headline results

| Approach | Test set macro-F1 | Challenge set macro-F1 | Challenge accuracy (95% CI) |
|---|---|---|---|
| B: TF-IDF + Logistic Regression | 0.995 | 0.546 | 56.7% (44.1–68.4%) |
| A: Llama 3.2 3B, prompt v2 | 0.951 | 0.897 | 90.0% (79.9–95.3%) |

The test set comes from the same sources as the training data; the challenge set is
written in natural UK English by a different author. The confidence intervals on the
challenge set do not overlap, so the difference is real despite only 60 messages.

## 2. TF-IDF: high scores from shortcut learning

- Coefficient inspection (D17) showed Product Quality was recognised by the synthetic
  generator's opening words ("just", "why", "ordered", "hello", "so") and Payment by
  memorised Bitext invoice numbers.
- On the challenge set, 17 of 26 TF-IDF errors were predictions of Product Quality,
  e.g. "ordered a hoodie last tuesday and the tracking hasnt moved" -> Product Quality.
- Conclusion: the 0.995 test score measures familiarity with the training sources,
  not understanding of customer messages.

## 3. LLM: most errors sit on boundaries defined by our label mapping

- Test set: 54 of 68 errors are Delivery/Other/Payment boundary cases created by our
  mapping decisions (D2), e.g. "early exit penalties" (mapped to Other, LLM says Payment),
  "I can't update my address" (mapped to Delivery, LLM says Other). Many are defensible.
- Genuine errors: tie-break rule 1 was not always followed
  (e.g. "size nothing like the size guide, want a refund" -> Refund instead of Product Quality).
- Challenge set: 3 of 6 errors confuse Refund with Payment when the message mentions
  a card or PayPal.
- Robustness risks seen during development: a safety refusal and a chatty non-answer
  (prompt v1, D20). Production use needs a fallback queue for unparseable replies.

## 4. Ambiguous messages

| Approach | Clear messages (44) | Ambiguous messages (16) |
|---|---|---|
| TF-IDF | 54.5% | 62.5% |
| LLM | 93.2% | 81.2% |

The LLM is weaker on ambiguous messages, as expected; in practice these should be the
cases most often routed to a human.

## 5. Business implications

- Compute cost is negligible for both (~£40/month for the LLM, ~£0 for TF-IDF);
  misrouting cost dominates and depends on accuracy on real messages.
- The ranking of the approaches does not depend on the assumed cost per misroute.
- Many LLM errors could be reduced by sharper category definitions, which is a
  business decision, not a technical one.

## 6. Limitations

- The challenge set is small (60) and was drafted with an AI, not taken from real customers.
- TF-IDF was trained on synthetic data; trained on real labelled company enquiries it
  would likely perform much better. This was not tested.
- Timings were measured on one laptop GPU (RTX 4060).
- Cost figures are assumptions (see src/cost_model.py) and should be replaced with
  the company's own numbers.