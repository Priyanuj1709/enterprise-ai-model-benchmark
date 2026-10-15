"""Map Bitext intents to 5 business categories, replace placeholders with
realistic values, remove duplicates and create a realistically imbalanced dataset."""
import random
import re
from pathlib import Path
import pandas as pd

SEED = 42                      # fixed seed = same result every run (reproducible)
random.seed(SEED)

INTENT_TO_LABEL = {
    # Delivery
    "delivery_period": "Delivery", "delivery_options": "Delivery",
    "track_order": "Delivery", "change_shipping_address": "Delivery",
    "set_up_shipping_address": "Delivery",
    # Refund
    "get_refund": "Refund", "track_refund": "Refund", "check_refund_policy": "Refund",
    # Payment
    "payment_issue": "Payment", "check_payment_methods": "Payment",
    "get_invoice": "Payment", "check_invoice": "Payment",
    # Other
    "create_account": "Other", "delete_account": "Other", "edit_account": "Other",
    "switch_account": "Other", "recover_password": "Other",
    "registration_problems": "Other", "contact_customer_service": "Other",
    "contact_human_agent": "Other", "newsletter_subscription": "Other",
    "place_order": "Other", "change_order": "Other", "cancel_order": "Other",
    "review": "Other", "check_cancellation_fee": "Other",
    # "complaint" is deliberately NOT mapped: it is ambiguous across categories
}

# How many messages per class (Product Quality = 1,000 is added in Step 2c)
TARGET = {"Delivery": 3500, "Refund": 2000, "Payment": 1500, "Other": 2000}

FILL = {
    "Order Number":    lambda: f"#{random.randint(100000, 999999)}",
    "Invoice Number":  lambda: f"INV-{random.randint(10000, 99999)}",
    "Refund Amount":   lambda: str(random.randint(5, 300)),
    "Currency Symbol": lambda: "£",
    "Person Name":     lambda: random.choice(["Sarah", "James", "Priya", "Tom", "Aisha"]),
    "Delivery City":   lambda: random.choice(["Leeds", "Bristol", "Glasgow", "Cardiff", "Manchester"]),
    "Delivery Country": lambda: "the UK",
}

def fill_placeholders(text: str) -> str:
    def replace(match):
        key = match.group(1).strip()
        return FILL[key]() if key in FILL else key.lower()
    text = re.sub(r"\{\{(.*?)\}\}", replace, text)
    return text.replace(" dollars", " pounds").replace("$", "£")

df = pd.read_csv("data/raw/bitext_customer_support.csv")

# 1. Show which placeholders exist (worth a line in your README)
print("Placeholders found:")
print(df["instruction"].str.findall(r"\{\{(.*?)\}\}").explode().value_counts(), "\n")

# 2. Keep complaints aside for the ambiguous test set later
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)
df[df["intent"] == "complaint"].to_csv(out_dir / "bitext_complaints.csv", index=False)

# 3. Map intents to labels, drop unmapped (complaint)
df["label"] = df["intent"].map(INTENT_TO_LABEL)
df = df.dropna(subset=["label"])

# 4. Remove exact duplicates BEFORE filling placeholders
#    (after filling, random numbers would make duplicates look unique)
before = len(df)
df = df.drop_duplicates(subset="instruction")
print(f"Removed {before - len(df):,} exact duplicates\n")

# 5. Replace placeholders with realistic values
df["text"] = df["instruction"].map(fill_placeholders)

# 6. Sample a realistic class mix
parts = [df[df["label"] == lab].sample(n, random_state=SEED) for lab, n in TARGET.items()]
final = pd.concat(parts)[["text", "label", "intent"]]
final["source"] = "bitext"
final = final.sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle

final.to_csv(out_dir / "bitext_mapped.csv", index=False)
print("Class distribution:")
print(final["label"].value_counts())
print(f"\nSaved {len(final):,} rows to {out_dir / 'bitext_mapped.csv'}")
print("\nExamples after cleaning:")
print(final.sample(5, random_state=1)[["label", "text"]].to_string())