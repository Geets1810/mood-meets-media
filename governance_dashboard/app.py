import duckdb
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import duckdb
import streamlit as st

@st.cache_resource
def get_connection():
    return duckdb.connect("data/governance.duckdb")

con = get_connection()


@st.cache_resource
def load_tables():
    con.execute("""
        CREATE OR REPLACE TABLE dim_model AS
        SELECT * FROM read_csv_auto('data/dim_model.csv')
    """)

    con.execute("""
        CREATE OR REPLACE TABLE dim_model_version AS
        SELECT * FROM read_csv_auto('data/dim_model_version.csv')
    """)

    con.execute("""
        CREATE OR REPLACE TABLE dim_person AS
        SELECT * FROM read_csv_auto('data/dim_person.csv')
    """)

    con.execute("""
        CREATE OR REPLACE TABLE bridge_model_assignment AS
        SELECT * FROM read_csv_auto('data/bridge_model_assignment.csv')
    """)

    con.execute("""
        CREATE OR REPLACE TABLE fact_model_review AS
        SELECT * FROM read_csv_auto('data/fact_model_review.csv')
    """)

    con.execute("""
        CREATE OR REPLACE TABLE fact_model_review_backlog_monthly AS
        SELECT * FROM read_csv_auto('data/fact_model_review_backlog_monthly.csv')
    """)

#load_tables()

# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(
    page_title="Model Review Backlog Dashboard",
    layout="wide"
)

st.title("Model Review Backlog Dashboard")
st.caption("Month-end snapshot view of model governance health")

# --------------------------------------------------
# DuckDB connection
# --------------------------------------------------
con = duckdb.connect("data/governance.duckdb")

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Filters")

snapshot_months = con.execute("""
    SELECT DISTINCT snapshot_month
    FROM fact_model_review_backlog_monthly
    ORDER BY snapshot_month
""").df()["snapshot_month"].tolist()

selected_snapshot = st.sidebar.selectbox(
    "Snapshot Month",
    snapshot_months,
    index=len(snapshot_months) - 1
)

domains = con.execute("""
    SELECT DISTINCT business_domain
    FROM fact_model_review_backlog_monthly
""").df()["business_domain"].tolist()

selected_domains = st.sidebar.multiselect(
    "Business Domain",
    domains,
    default=domains
)

if not selected_domains:
    selected_domains = domains

domain_filter = ",".join([f"'{d}'" for d in selected_domains])

# ==================================================
# ROW 1
# ==================================================
col1, col2 = st.columns(2)

# --------------------------------------------------
# Metric 1: Overdue Reviews – Trend (LINE)
# --------------------------------------------------
with col1:
    st.subheader("Overdue Reviews – Monthly Trend")

    overdue_trend = con.execute(f"""
        SELECT snapshot_month, COUNT(*) AS overdue_count
        FROM fact_model_review_backlog_monthly
        WHERE overdue_flag = 'Y'
          AND business_domain IN ({domain_filter})
        GROUP BY snapshot_month
        ORDER BY snapshot_month
    """).df()

    st.line_chart(overdue_trend.set_index("snapshot_month"))

# --------------------------------------------------
# Metric 2: Open vs Closed (STACKED AREA)
# --------------------------------------------------
with col2:
    st.subheader("Backlog Stabilization (Open vs Closed)")

    status_trend = con.execute(f"""
        SELECT snapshot_month, review_status, COUNT(*) AS cnt
        FROM fact_model_review_backlog_monthly
        WHERE business_domain IN ({domain_filter})
        GROUP BY snapshot_month, review_status
        ORDER BY snapshot_month
    """).df()

    pivot = status_trend.pivot(
        index="snapshot_month",
        columns="review_status",
        values="cnt"
    ).fillna(0)

    st.area_chart(pivot)

# ==================================================
# ROW 2
# ==================================================
col3, col4 = st.columns(2)

# --------------------------------------------------
# Metric 3: Median Days Overdue (HORIZONTAL BAR)
# --------------------------------------------------
with col3:
    st.subheader("Median Days Overdue by Business Domain")

    median_df = con.execute(f"""
        SELECT business_domain,
               MEDIAN(days_overdue) AS median_days_overdue
        FROM fact_model_review_backlog_monthly
        WHERE overdue_flag = 'Y'
          AND snapshot_month = '{selected_snapshot}'
          AND business_domain IN ({domain_filter})
        GROUP BY business_domain
        ORDER BY median_days_overdue DESC
    """).df()

    fig, ax = plt.subplots()
    ax.barh(
        median_df["business_domain"],
        median_df["median_days_overdue"]
    )
    ax.set_xlabel("Median Days Overdue")
    ax.invert_yaxis()
    st.pyplot(fig)

# --------------------------------------------------
# Metric 4: SLA Buckets (DONUT)
# --------------------------------------------------
with col4:
    st.subheader("SLA Breach Buckets")

    sla_df = con.execute(f"""
        SELECT
            CASE
                WHEN days_overdue <= 30 THEN '0–30'
                WHEN days_overdue <= 60 THEN '31–60'
                WHEN days_overdue <= 90 THEN '61–90'
                ELSE '90+'
            END AS sla_bucket,
            COUNT(*) AS cnt
        FROM fact_model_review_backlog_monthly
        WHERE overdue_flag = 'Y'
          AND snapshot_month = '{selected_snapshot}'
          AND business_domain IN ({domain_filter})
        GROUP BY sla_bucket
        ORDER BY cnt DESC
    """).df()

    fig, ax = plt.subplots()
    ax.pie(
        sla_df["cnt"],
        labels=sla_df["sla_bucket"],
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"width": 0.4}
    )
    ax.set_title("SLA Distribution")
    st.pyplot(fig)

# ==================================================
# ROW 3
# ==================================================
col5, col6 = st.columns(2)

# --------------------------------------------------
# Metric 5: Validator Backlog (HORIZONTAL BAR)
# --------------------------------------------------
with col5:
    st.subheader("Open Backlog by Lead Validator")

    validator_df = con.execute(f"""
        SELECT
            COALESCE(CAST(lead_validator_person_id AS VARCHAR), 'Unassigned') AS validator,
            COUNT(*) AS open_reviews
        FROM fact_model_review_backlog_monthly
        WHERE review_status = 'Open'
          AND snapshot_month = '{selected_snapshot}'
          AND business_domain IN ({domain_filter})
        GROUP BY validator
        ORDER BY open_reviews DESC
        LIMIT 10
    """).df()

    fig, ax = plt.subplots()
    ax.barh(
        validator_df["validator"],
        validator_df["open_reviews"]
    )
    ax.set_xlabel("Open Reviews")
    ax.invert_yaxis()
    st.pyplot(fig)

# --------------------------------------------------
# Metric 6: Risk Tier Distribution (DONUT)
# --------------------------------------------------
with col6:
    st.subheader("Risk Tier Distribution")

    risk_df = con.execute(f"""
        SELECT current_risk_tier, COUNT(*) AS cnt
        FROM fact_model_review_backlog_monthly
        WHERE snapshot_month = '{selected_snapshot}'
          AND business_domain IN ({domain_filter})
        GROUP BY current_risk_tier
    """).df()

    fig, ax = plt.subplots()
    ax.pie(
        risk_df["cnt"],
        labels=risk_df["current_risk_tier"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.4}
    )
    ax.set_title("Risk Tier Mix")
    st.pyplot(fig)

# ==================================================
# RAW DATA TABLE
# ==================================================
st.subheader("Snapshot Data (Filtered)")

raw_df = con.execute(f"""
    SELECT *
    FROM fact_model_review_backlog_monthly
    WHERE snapshot_month = '{selected_snapshot}'
      AND business_domain IN ({domain_filter})
    ORDER BY days_overdue DESC
""").df()

st.dataframe(raw_df, use_container_width=True)

st.download_button(
    label="⬇️ Download CSV",
    data=raw_df.to_csv(index=False),
    file_name=f"model_review_snapshot_{selected_snapshot}.csv",
    mime="text/csv"
)

con.close()
