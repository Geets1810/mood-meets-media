-- =========================
-- Core Dimensions
-- =========================

CREATE TABLE IF NOT EXISTS dim_person (
  person_id VARCHAR(10),
  person_name VARCHAR(100),
  email VARCHAR(150),
  role VARCHAR(50),
  department VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS dim_model (
  model_id VARCHAR(10),
  model_name VARCHAR(150),
  business_domain VARCHAR(50),
  use_case VARCHAR(100),
  criticality_tier VARCHAR(20),
  regulatory_scope VARCHAR(20),
  created_date DATE,
  decommissioned_date DATE
);

CREATE TABLE IF NOT EXISTS dim_model_version (
  model_version_id VARCHAR(15),
  model_id VARCHAR(10),
  version_label VARCHAR(20),
  model_type VARCHAR(30),
  created_date DATE,
  activation_date DATE,
  retirement_date DATE,
  lifecycle_phase VARCHAR(30),
  model_status VARCHAR(50),
  management_status VARCHAR(50),
  current_risk_tier VARCHAR(20),
  expected_revalidation_date DATE,
  last_review_id VARCHAR(20)
);

-- =========================
-- Bridge Tables
-- =========================

CREATE TABLE IF NOT EXISTS bridge_model_assignment (
  model_assignment_id VARCHAR(20),
  model_version_id VARCHAR(15),
  person_id VARCHAR(10),
  assignment_role VARCHAR(50),
  assignment_start_date DATE,
  assignment_end_date DATE
);

-- =========================
-- Fact Tables
-- =========================

CREATE TABLE IF NOT EXISTS fact_model_review (
  review_id VARCHAR(20),
  model_version_id VARCHAR(15),
  review_date DATE,
  review_cycle VARCHAR(30),
  review_outcome VARCHAR(50),
  risk_tier_before VARCHAR(20),
  risk_tier_after VARCHAR(20),
  issues_found_count INTEGER,
  material_issue_flag BOOLEAN,
  next_review_due_date DATE,
  reviewed_by_id VARCHAR(10),
  comments VARCHAR
);

-- =========================
-- Governance Extensions (NEW)
-- =========================

CREATE TABLE IF NOT EXISTS fact_revalidation_extension (
  model_version_id VARCHAR(15),
  extension_date DATE,
  old_due_date DATE,
  new_due_date DATE,
  extension_reason VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS fact_model_governance_snapshot (
  snapshot_date DATE,
  model_version_id VARCHAR(15),
  business_domain VARCHAR(50),
  lead_validator_name VARCHAR(100),
  effective_due_date DATE,
  is_overdue BOOLEAN,
  days_overdue INTEGER,
  current_risk_tier VARCHAR(20),
  management_status VARCHAR(50)
);
