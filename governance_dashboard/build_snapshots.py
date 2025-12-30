import pandas as pd
from pandas.tseries.offsets import MonthEnd

# -------------------------------------------------------------------
# Load base datasets
# -------------------------------------------------------------------
dim_model = pd.read_csv("data/dim_model.csv")

dim_model_version = pd.read_csv(
    "data/dim_model_version.csv",
    parse_dates=["expected_revalidation_date", "activation_date"]
)

fact_model_review = pd.read_csv(
    "data/fact_model_review.csv",
    parse_dates=["review_date"]
)

bridge_assignment = pd.read_csv("data/bridge_model_assignment.csv")

# -------------------------------------------------------------------
# Prepare latest review per model_version (event → state bridge)
# -------------------------------------------------------------------
latest_reviews = (
    fact_model_review
    .sort_values("review_date")
    .groupby("model_version_id", as_index=False)
    .last()[[
        "model_version_id",
        "review_date",
        "risk_tier_after"
    ]]
    .rename(columns={
        "review_date": "latest_review_date",
        "risk_tier_after": "latest_risk_tier"
    })
)

# -------------------------------------------------------------------
# Attach model + review context
# -------------------------------------------------------------------
base = (
    dim_model_version
    .merge(dim_model, on="model_id", how="left")
    .merge(latest_reviews, on="model_version_id", how="left")
)

# -------------------------------------------------------------------
# Attach assignments (do NOT pre-filter)
# -------------------------------------------------------------------
assignments = (
    bridge_assignment[["model_version_id", "person_id", "assignment_role"]]
)

base = base.merge(assignments, on="model_version_id", how="left")

# -------------------------------------------------------------------
# Derive lead validator person id (nullable by design)
# -------------------------------------------------------------------
base["lead_validator_person_id"] = base.apply(
    lambda r: r.person_id if r.assignment_role == "Lead Validator" else None,
    axis=1
)

# Drop helper columns
base = base.drop(columns=["person_id", "assignment_role"])


# -------------------------------------------------------------------
# Generate snapshot months (MONTH-END ONLY)
# -------------------------------------------------------------------
start_month = (
    base["activation_date"]
    .min()
    .to_period("M")
    .to_timestamp("M")
)

end_month = (
    pd.Timestamp.today()
    .to_period("M")
    .to_timestamp("M")
)

snapshot_months = pd.date_range(
    start=start_month,
    end=end_month,
    freq="M"
)

# -------------------------------------------------------------------
# Build monthly snapshots
# -------------------------------------------------------------------
snapshots = []

for snapshot_month in snapshot_months:
    snap = base.copy()

    # Only include model versions active as of snapshot month
    snap = snap[snap["activation_date"] <= snapshot_month]

    # ---------------------------------------------------------------
    # Review status as-of month end
    # ---------------------------------------------------------------
    snap["review_status"] = snap.apply(
        lambda r: "Closed"
        if pd.notna(r.latest_review_date)
        and r.latest_review_date <= snapshot_month
        else "Open",
        axis=1
    )
    # Review status as-of snapshot month
    snap["review_status"] = snap.apply(
    lambda r: "Closed"
    if pd.notna(r.latest_review_date)
       and r.latest_review_date <= snapshot_month
    else "Open",
    axis=1
    )

    # ---------------------------------------------------------------
    # Overdue flag
    # ---------------------------------------------------------------
    snap["overdue_flag"] = snap.apply(
        lambda r: "Y"
        if r.review_status == "Open"
        and r.expected_revalidation_date < snapshot_month
        else "N",
        axis=1
    )

    # ---------------------------------------------------------------
    # Days overdue (month-end bounded)
    # ---------------------------------------------------------------
    snap["days_overdue"] = snap.apply(
        lambda r: (snapshot_month - r.expected_revalidation_date).days
        if r.overdue_flag == "Y"
        else 0,
        axis=1
    )

    # ---------------------------------------------------------------
    # Risk tier as-of snapshot
    # ---------------------------------------------------------------
    snap["current_risk_tier"] = snap["latest_risk_tier"]


    # Snapshot month
    snap["snapshot_month"] = snapshot_month

    snapshots.append(
        snap[[
            "snapshot_month",
            "model_version_id",
            "business_domain",
            "lead_validator_person_id",
            "review_status",
            "overdue_flag",
            "days_overdue",
            "current_risk_tier"
        ]]
    )

# -------------------------------------------------------------------
# Final monthly snapshot table
# -------------------------------------------------------------------
fact_model_review_backlog_monthly = pd.concat(
    snapshots,
    ignore_index=True
)

# Write output
fact_model_review_backlog_monthly.to_csv(
    "data/fact_model_review_backlog_monthly.csv",
    index=False
)

print("Monthly governance snapshot generated successfully.")
