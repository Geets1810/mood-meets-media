import pandas as pd
import random
from datetime import timedelta

INPUT_FILE = "data/dim_model_version.csv"
OUTPUT_FILE = "data/dim_model_version_repaired.csv"

random.seed(42)

df = pd.read_csv(INPUT_FILE, parse_dates=["activation_date"])

today = pd.Timestamp.today().normalize()

rows = []

for _, row in df.iterrows():

    # Decide if this version is governed
    governed = random.random() < 0.65  # 65% governed

    if governed and pd.notna(row["activation_date"]) and row["activation_date"] <= today:

        row["lifecycle_phase"] = random.choice(["Monitoring", "Active"])
        row["model_status"] = random.choice(["Active", "Under Review"])

        # Assign a realistic review due date
        days_out = random.randint(270, 450)
        row["expected_revalidation_date"] = (
            row["activation_date"] + timedelta(days=days_out)
        ).date()

    else:
        # Not governed yet or retired
        row["lifecycle_phase"] = random.choice(["Development", "Retired"])
        row["model_status"] = "Retired"
        row["expected_revalidation_date"] = None

    rows.append(row)

repaired_df = pd.DataFrame(rows)
repaired_df.to_csv(OUTPUT_FILE, index=False)

print("✅ dim_model_version repaired with governance realism")
