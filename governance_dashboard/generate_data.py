import pandas as pd
from pathlib import Path
from datetime import timedelta
import random

# =====================================================
# Config
# =====================================================

DATA_DIR = Path("data")
OUT_DIR = Path("data_stimulated")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TARGET_REVIEW_ROWS = 200

model_variants = ["Retail", "SME", "Wholesale", "Digital", "Legacy"]

# =====================================================
# Helpers
# =====================================================

def read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")

# =====================================================
# Load source CSVs
# =====================================================

dim_model = read_csv_safe(DATA_DIR / "dim_model.csv")
dim_model_version = read_csv_safe(DATA_DIR / "dim_model_version.csv")
dim_person = read_csv_safe(DATA_DIR / "dim_person.csv")
bridge_assignment = read_csv_safe(DATA_DIR / "bridge_model_assignment.csv")
fact_review = read_csv_safe(DATA_DIR / "fact_model_review.csv")

# =====================================================
# Date parsing
# =====================================================

date_cols = [
    "created_date", "activation_date", "retirement_date",
    "assignment_start_date", "assignment_end_date",
    "review_date", "next_review_due_date"
]

for df in [dim_model, dim_model_version, bridge_assignment, fact_review]:
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

# =====================================================
# FIX 1: Filter invalid model_version rows
# =====================================================

dim_model_version = dim_model_version[
    dim_model_version["model_id"].notna()
].copy()

assert dim_model_version["model_id"].isna().sum() == 0, \
    "dim_model_version contains rows without model_id"

# =====================================================
# 1️⃣ Expand MODELS
# =====================================================

expanded_models = []
model_id_map = {}

for _, row in dim_model.iterrows():
    for suffix in model_variants:
        new_id = f"{row.model_id}-{suffix[:3].upper()}"
        model_id_map[(row.model_id, suffix)] = new_id

        new_row = row.copy()
        new_row["model_id"] = new_id
        new_row["model_name"] = f"{row.model_name} – {suffix}"
        expanded_models.append(new_row)

final_models = pd.DataFrame(expanded_models)

# =====================================================
# 2️⃣ Expand MODEL VERSIONS (guarded)
# =====================================================

expanded_versions = []
version_id_map = {}

for _, v in dim_model_version.iterrows():
    base_model_id = v.model_id

    for suffix in model_variants:
        key = (base_model_id, suffix)
        if key not in model_id_map:
            continue

        new_model_id = model_id_map[key]
        new_version_id = f"{new_model_id}-{v.version_label}"

        version_id_map[(v.model_version_id, suffix)] = new_version_id

        new_v = v.copy()
        new_v["model_id"] = new_model_id
        new_v["model_version_id"] = new_version_id
        expanded_versions.append(new_v)

final_versions = pd.DataFrame(expanded_versions)

# =====================================================
# 3️⃣ Expand ASSIGNMENTS (guarded)
# =====================================================

expanded_assignments = []

for _, a in bridge_assignment.iterrows():
    if pd.isna(a.model_id):
        continue

    for suffix in model_variants:
        key = (a.model_id, suffix)
        if key not in model_id_map:
            continue

        new_model_id = model_id_map[key]

        new_a = a.copy()
        new_a["assignment_id"] = f"{a.assignment_id}-{suffix[:3].upper()}"
        new_a["model_id"] = new_model_id
        expanded_assignments.append(new_a)

final_assignments = pd.DataFrame(expanded_assignments)

# =====================================================
# 4️⃣ Expand REVIEWS (guarded)
# =====================================================

expanded_reviews = []
review_counter = 1

while len(expanded_reviews) < TARGET_REVIEW_ROWS:
    for _, r in fact_review.iterrows():
        for suffix in model_variants:
            key = (r.model_version_id, suffix)
            if key not in version_id_map:
                continue

            new_version_id = version_id_map[key]

            jitter = random.randint(90, 360)

            new_r = r.copy()
            new_r["review_id"] = f"REV-{review_counter:04d}"
            new_r["model_version_id"] = new_version_id
            new_r["review_date"] = r.review_date + timedelta(days=jitter)
            new_r["next_review_due_date"] = (
                r.next_review_due_date + timedelta(days=jitter)
                if pd.notnull(r.next_review_due_date)
                else None
            )

            expanded_reviews.append(new_r)
            review_counter += 1

            if len(expanded_reviews) >= TARGET_REVIEW_ROWS:
                break
        if len(expanded_reviews) >= TARGET_REVIEW_ROWS:
            break

final_reviews = pd.DataFrame(expanded_reviews)

# =====================================================
# Write output
# =====================================================

final_models.to_csv(OUT_DIR / "dim_model.csv", index=False)
final_versions.to_csv(OUT_DIR / "dim_model_version.csv", index=False)
dim_person.to_csv(OUT_DIR / "dim_person.csv", index=False)
final_assignments.to_csv(OUT_DIR / "bridge_model_assignment.csv", index=False)
final_reviews.to_csv(OUT_DIR / "fact_model_review.csv", index=False)

print("✅ Data stimulation complete")
print(f"Models: {len(final_models)}")
print(f"Versions: {len(final_versions)}")
print(f"Assignments: {len(final_assignments)}")
print(f"Reviews: {len(final_reviews)}")
