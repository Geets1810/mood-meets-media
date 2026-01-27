import os
import numpy as np
import pandas as pd

# -----------------------------
# Configuration (locked assumptions)
# -----------------------------
N_SESSIONS = 50000
RANDOM_SEED = 42

BASE_SUCCESS_RATE = 0.85
FAILURE_RATE = 0.15

SOFT_DECLINE_RATE = 0.70
HARD_DECLINE_RATE = 0.30

RECOVERY_RATE_CONTROL = 0.20
RECOVERY_RATE_TREATMENT = 0.35

AVG_RETRIES_CONTROL = 0.6
AVG_RETRIES_TREATMENT = 0.9

AVG_TIME_SUCCESS_CONTROL = 45  # seconds
AVG_TIME_SUCCESS_TREATMENT = 40  # seconds

np.random.seed(RANDOM_SEED)

# -----------------------------
# Generate base session data
# -----------------------------
sessions = pd.DataFrame({
    "session_id": np.arange(1, N_SESSIONS + 1),
    "variant": np.random.choice(["A", "B"], size=N_SESSIONS),
    "device": np.random.choice(["web", "ios", "android"], size=N_SESSIONS, p=[0.5, 0.3, 0.2]),
    "region": np.random.choice(["NA", "EU", "APAC"], size=N_SESSIONS, p=[0.6, 0.25, 0.15]),
    "returning_user": np.random.choice([True, False], size=N_SESSIONS, p=[0.55, 0.45])
})

sessions["initiated"] = 1

# -----------------------------
# Determine failures
# -----------------------------
sessions["failed"] = np.random.rand(N_SESSIONS) < FAILURE_RATE

sessions["first_failure_type"] = np.where(
    sessions["failed"],
    np.random.choice(
        ["soft", "hard"],
        size=N_SESSIONS,
        p=[SOFT_DECLINE_RATE, HARD_DECLINE_RATE]
    ),
    None
)

# -----------------------------
# Determine recovery
# -----------------------------
def recover(row):
    if not row["failed"]:
        return True
    if row["first_failure_type"] == "hard":
        return False

    if row["variant"] == "A":
        return np.random.rand() < RECOVERY_RATE_CONTROL
    else:
        return np.random.rand() < RECOVERY_RATE_TREATMENT

sessions["succeeded"] = sessions.apply(recover, axis=1)

# -----------------------------
# Retry count
# -----------------------------
sessions["retry_count"] = np.where(
    sessions["failed"],
    np.where(
        sessions["variant"] == "A",
        np.random.poisson(AVG_RETRIES_CONTROL, size=N_SESSIONS),
        np.random.poisson(AVG_RETRIES_TREATMENT, size=N_SESSIONS)
    ),
    0
)

# -----------------------------
# Time to success
# -----------------------------
sessions["time_to_success_sec"] = np.where(
    sessions["succeeded"],
    np.where(
        sessions["variant"] == "A",
        np.random.normal(AVG_TIME_SUCCESS_CONTROL, 10, size=N_SESSIONS),
        np.random.normal(AVG_TIME_SUCCESS_TREATMENT, 10, size=N_SESSIONS)
    ),
    np.nan
)

sessions["time_to_success_sec"] = sessions["time_to_success_sec"].clip(lower=5)

# -----------------------------
# Save output (robust path handling)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "session_summary.csv")
sessions.to_csv(OUTPUT_PATH, index=False)

print(f"Synthetic session-level data generated at {OUTPUT_PATH}")
