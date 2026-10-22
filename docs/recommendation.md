# Recommendation: automating customer enquiry classification

**To:** Customer Operations leadership
**Question:** Which approach should we use to automatically sort our ~50,000 daily
customer enquiries into Delivery, Refund, Payment, Product Quality and Other?

## Recommendation

**Use the small language model running on our own hardware (Approach A), starting with
a supervised pilot.** Do not adopt the conventional classifier (Approach B) yet: it only
works well once we have thousands of our own enquiries labelled, which we do not have today.

## Why

| | Small language model (A) | Conventional classifier (B) |
|---|---|---|
| Correct on realistic, naturally written messages | **9 in 10** | fewer than 6 in 10 |
| Correct on messages similar to its training examples | 95 in 100 | 99.5 in 100 |
| Time to sort a full day's 50,000 enquiries | under 50 minutes | under 1 second |
| Running cost | about £40 per month (one ~£1,000 PC) | about £0 (existing server) |
| One-off setup | about £2,000 | about £2,000, including labelling 5,000 enquiries |
| Needs our own labelled enquiries before it works | **No** | **Yes** |

- **Quality on real wording is what matters.** Real customers write in their own words.
  The conventional classifier scored almost perfectly on text resembling its training
  examples but was wrong more than 4 times in 10 on naturally written messages.
  The language model stayed right about 9 times in 10.
- **Cost is not the deciding factor.** Both are cheap to run. The real cost is enquiries
  sent to the wrong team. Under our assumptions (2 minutes of staff time to re-route each one),
  the language model would cost roughly **£280,000 per month less** in wasted staff time
  on realistic messages. The exact figure depends on our assumptions, but the ranking does not.
- **Speed is not a constraint.** One standard PC sorts a whole day's enquiries in under an hour.
- **Customer data stays in-house.** The model runs on our own hardware; no enquiries are
  sent to an outside AI provider.

## Safeguards

1. **Human fallback:** unclear or unusual model answers go to a general queue.
2. **Easy correction:** agents re-route wrong tickets with one click; every correction is saved.
3. **Monthly check:** a sample of 200 real enquiries is checked by staff to track accuracy.
4. **Clearer category rules:** most model mistakes were on borderline cases (fees, address
   changes, refunds paid to cards). Operations should confirm where these belong.

## Suggested plan

| When | Step |
|---|---|
| Weeks 1–2 | Shadow mode: the model suggests a category, staff still decide. Measure accuracy on real enquiries. |
| Weeks 3–6 | Go live on 10% of enquiries, then all, if accuracy is at least 85%. |
| Month 3 | Use the saved corrections (thousands of real labelled enquiries) to re-test the conventional classifier. It is faster and cheaper, and may then perform best. Choose based on real data. |

## What this evaluation cannot tell us

- The test messages were synthetic or written for this study, not real customer emails.
  The pilot must confirm the accuracy on real traffic.
- We do not yet know how accurately enquiries are routed today, so we cannot yet say
  how much better or worse automation is than the current process. This should be
  measured during the shadow-mode weeks.