"""Step 7b: estimated monthly operating cost for 50,000 enquiries a day.
Money figures are ASSUMPTIONS (edit them). Speed and accuracy are MEASURED (results/summary.csv)."""
import pandas as pd

# ---------------- Assumptions: edit for the real company ----------------
VOLUME_PER_DAY = 50_000
DAYS_PER_MONTH = 30
ELECTRICITY_GBP_PER_KWH = 0.27          # UK business tariff (check current rate)
GPU_PC_GBP = 1_000                       # PC with an RTX 4060-class GPU
HARDWARE_LIFETIME_MONTHS = 36
PC_IDLE_WATTS, GPU_EXTRA_WATTS = 60, 100 # always-on PC + extra draw while classifying
STAFF_GBP_PER_HOUR = 17                  # advisor pay incl. employer costs
MINUTES_LOST_PER_MISROUTE = 2            # time for staff to spot and re-route a wrong category
ENGINEER_GBP_PER_DAY = 400
SETUP_DAYS = {"A": 5, "B": 4}            # build, integrate, test
LABELLING_SECONDS_PER_MESSAGE = 20
TRAINING_LABELS_NEEDED = {"A": 0, "B": 5_000}   # B needs real labelled enquiries to train
# -------------------------------------------------------------------------

summary = pd.read_csv("results/summary.csv")


def measured(dataset, prefix):
    return summary[(summary["dataset"] == dataset) & summary["approach"].str.startswith(prefix)].iloc[0]


cost_per_misroute = MINUTES_LOST_PER_MISROUTE / 60 * STAFF_GBP_PER_HOUR
rows = {}
for key, prefix, name in [("A", "A:", "A: Llama 3.2 3B (local GPU)"), ("B", "B:", "B: TF-IDF + LogReg (CPU)")]:
    test, challenge = measured("test", prefix), measured("challenge", prefix)
    ms = test["mean_ms"] if key == "A" else test["batch_ms"]      # how each would really run
    compute_hours = VOLUME_PER_DAY * ms / 1000 / 3600

    if key == "A":
        kwh_per_day = PC_IDLE_WATTS / 1000 * 24 + GPU_EXTRA_WATTS / 1000 * compute_hours
        infra = GPU_PC_GBP / HARDWARE_LIFETIME_MONTHS + kwh_per_day * DAYS_PER_MONTH * ELECTRICITY_GBP_PER_KWH
    else:
        infra = 0.0          # a few seconds of CPU a day on an existing server

    setup = (SETUP_DAYS[key] * ENGINEER_GBP_PER_DAY
             + TRAINING_LABELS_NEEDED[key] * LABELLING_SECONDS_PER_MESSAGE / 3600 * STAFF_GBP_PER_HOUR)

    def misroutes(acc):
        return VOLUME_PER_DAY * (1 - acc)

    rows[name] = {
        "Compute time per day": f"{compute_hours * 60:.1f} min" if compute_hours > 0.02
                                else f"{compute_hours * 3600:.1f} s",
        "Infrastructure per month (£)": round(infra),
        "One-off setup (£)": round(setup),
        "Misroutes per day - test-like messages": round(misroutes(test["accuracy"])),
        "Misroute cost per month - test-like (£)":
            round(misroutes(test["accuracy"]) * cost_per_misroute * DAYS_PER_MONTH),
        "Misroutes per day - realistic messages": round(misroutes(challenge["accuracy"])),
        "Misroute cost per month - realistic (£)":
            round(misroutes(challenge["accuracy"]) * cost_per_misroute * DAYS_PER_MONTH),
    }

table = pd.DataFrame(rows)
print(table.to_string())
table.to_csv("results/cost_model.csv")
print(f"\nAssumed cost per misrouted enquiry: £{cost_per_misroute:.2f}")