import streamlit as st
import duckdb
import pandas as pd
import altair as alt

st.set_page_config(page_title="Model Governance Dashboard", layout="wide")

# --------------------------------------------------
# DuckDB Connection (SINGLE, READ-ONLY, CACHED)
# --------------------------------------------------
@st.cache_resource
def get_connection():
    return duckdb.connect("data/governance.duckdb", read_only=True)

con = get_connection()

def get_df(query, params=None):
    if params:
        return con.execute(query, params).df()
    return con.execute(query).df()

# --------------------------------------------------
# Sidebar Filters
# --------------------------------------------------
st.sidebar.header("Filters")

snapshot_dates = get_df("""
    SELECT DISTINCT snapshot_month
    FROM fact_model_review_backlog_monthly
    ORDER BY snapshot_month
""")["snapshot_month"]

selected_snapshot = st.sidebar.selectbox(
    "Snapshot Month",
    snapshot_dates,
    index=len(snapshot_dates) - 1
)

business_domains = get_df("""
    SELECT DISTINCT business_domain
    FROM dim_model
    ORDER BY business_domain
""")["business_domain"]

selected_domains = st.sidebar.multiselect(
    "Business Domain",
    business_domains,
    default=business_domains.tolist()
)

# Temporary table for filtering
con.execute("DROP TABLE IF EXISTS selected_domains_tbl")
con.execute("""
    CREATE TEMP TABLE selected_domains_tbl AS
    SELECT * FROM UNNEST(?)
""", [selected_domains])

# --------------------------------------------------
# Base Filtered Dataset
# --------------------------------------------------
base_df = get_df(f"""
    SELECT *
    FROM fact_model_review_backlog_monthly
    WHERE snapshot_month = '{selected_snapshot}'
      AND business_domain IN (SELECT * FROM selected_domains_tbl)
""")

# --------------------------------------------------
# METRIC 1: Overdue Trend (Line)
# --------------------------------------------------
st.subheader("Overdue Review Trend")

overdue_trend = get_df("""
    SELECT
        snapshot_month,
        COUNT(*) AS overdue_count
    FROM fact_model_review_backlog_monthly
    WHERE overdue_flag = 'Y'
      AND business_domain IN (SELECT * FROM selected_domains_tbl)
    GROUP BY snapshot_month
    ORDER BY snapshot_month
""")

st.line_chart(overdue_trend.set_index("snapshot_month"))

# --------------------------------------------------
# METRIC 2: Open vs Closed (Stacked Area)
# --------------------------------------------------
st.subheader("Open vs Closed Reviews")

open_closed = get_df("""
    SELECT
        snapshot_month,
        review_status,
        COUNT(*) AS count
    FROM fact_model_review_backlog_monthly
    WHERE business_domain IN (SELECT * FROM selected_domains_tbl)
    GROUP BY snapshot_month, review_status
""")

st.area_chart(
    open_closed.pivot(
        index="snapshot_month",
        columns="review_status",
        values="count"
    )
)

# --------------------------------------------------
# METRIC 3: Median Days Overdue by Domain (Horizontal Bar)
# --------------------------------------------------
st.subheader("Median Days Overdue by Domain")

median_overdue = get_df("""
    SELECT
        business_domain,
        MEDIAN(days_overdue) AS median_days_overdue
    FROM fact_model_review_backlog_monthly
    WHERE overdue_flag = 'Y'
      AND snapshot_month = ?
      AND business_domain IN (SELECT * FROM selected_domains_tbl)
    GROUP BY business_domain
    ORDER BY median_days_overdue DESC
""", [selected_snapshot])

st.bar_chart(
    median_overdue.set_index("business_domain"),
    horizontal=True
)

# --------------------------------------------------
# METRIC 4: SLA Breach Buckets (Donut)
# --------------------------------------------------
st.subheader("SLA Breach Buckets")

sla_buckets = get_df("""
    SELECT
        CASE
            WHEN days_overdue <= 30 THEN '0–30'
            WHEN days_overdue <= 60 THEN '31–60'
            WHEN days_overdue <= 90 THEN '61–90'
            ELSE '90+'
        END AS sla_bucket,
        COUNT(*) AS count
    FROM fact_model_review_backlog_monthly
    WHERE overdue_flag = 'Y'
      AND snapshot_month = ?
      AND business_domain IN (SELECT * FROM selected_domains_tbl)
    GROUP BY sla_bucket
""", [selected_snapshot])

sla_chart = alt.Chart(sla_buckets).mark_arc(innerRadius=50).encode(
    theta="count:Q",
    color="sla_bucket:N"
)

st.altair_chart(sla_chart, use_container_width=True)

# --------------------------------------------------
# METRIC 5: Backlog by Lead Validator (Horizontal Bar)
# --------------------------------------------------
st.subheader("Open Backlog by Lead Validator")

validator_backlog = get_df("""
    SELECT
        p.person_name AS lead_validator,
        COUNT(*) AS open_reviews
    FROM fact_model_review_backlog_monthly f
    LEFT JOIN dim_person p
      ON f.lead_validator_person_id = p.person_id
    WHERE f.review_status = 'Open'
      AND f.snapshot_month = ?
      AND f.business_domain IN (SELECT * FROM selected_domains_tbl)
    GROUP BY p.person_name
    ORDER BY open_reviews DESC
""", [selected_snapshot])

st.bar_chart(
    validator_backlog.set_index("lead_validator"),
    horizontal=True
)

# --------------------------------------------------
# METRIC 6: Risk Tier Distribution (Donut)
# --------------------------------------------------
st.subheader("Risk Tier Distribution")

risk_dist = get_df("""
    SELECT
        current_risk_tier,
        COUNT(*) AS count
    FROM fact_model_review_backlog_monthly
    WHERE snapshot_month = ?
      AND business_domain IN (SELECT * FROM selected_domains_tbl)
    GROUP BY current_risk_tier
""", [selected_snapshot])

risk_chart = alt.Chart(risk_dist).mark_arc(innerRadius=50).encode(
    theta="count:Q",
    color="current_risk_tier:N"
)

st.altair_chart(risk_chart, use_container_width=True)

# --------------------------------------------------
# Raw Data + Download
# --------------------------------------------------
st.subheader("Raw Snapshot Data")

st.dataframe(base_df)

csv = base_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Snapshot Data",
    csv,
    "model_governance_snapshot.csv",
    "text/csv"
)
