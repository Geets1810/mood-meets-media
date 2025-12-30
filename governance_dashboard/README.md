# Model Governance Health Dashboard

This project is a bite-sized simulation of the model governance analytics I build as an Analytics Engineer.

## What this shows
- Snapshot-based monitoring of model review compliance
- Realistic governance behavior (on-time reviews, delayed reviews, extensions)
- Actionable views for validators and risk leadership

## Key design decisions
- Daily snapshot fact table to support accurate as-of analysis
- Overdue status resolves after review completion (not cumulative)
- Simulated data reflects operational behavior, not random noise

## Stack
- Python (pandas)
- DuckDB
- Streamlit

## Dashboard highlights
- Overdue review trend (stabilization after intervention)
- Backlog by Lead Validator
- Overdue exposure by business domain

## Note on data
All data is simulated to reflect governance patterns observed in regulated financial environments.
