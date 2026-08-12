-- ============================================================================
-- Insurance Claims Anomaly Detection - Database Schema
-- Dialect: SQLite
--
-- Source data:
--   insurance_data.csv  -> insurance_claims   (10,000 rows)
--   employee_data.csv   -> agents             (1,200 rows)
--   vendor_data.csv     -> vendors            (600 rows)
--
-- Column names, nullability, and category values below were derived by
-- profiling the actual CSV files (empty-string counts, distinct value
-- counts, and field widths), not assumed from naming conventions alone.
-- ============================================================================

PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------------------------
-- Table: agents
-- Business purpose: One row per licensed insurance agent who can be assigned
-- to a claim. Used to detect anomalies such as a single agent handling an
-- abnormally high volume/value of claims, or claims tied to agents outside
-- their normal territory (agent CITY/STATE vs. claim INCIDENT_STATE).
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS agents;
CREATE TABLE agents (
    agent_id            TEXT PRIMARY KEY
                             CHECK (agent_id LIKE 'AGENT%'),
    agent_name          TEXT NOT NULL,
    date_of_joining     DATE NOT NULL,               -- agent hire date; short tenure can correlate with fraud risk
    address_line1       TEXT,
    address_line2       TEXT,                        -- sparsely populated (apt/suite), nullable
    city                TEXT,                        -- nullable: a small number of source rows omit city
    state               TEXT CHECK (length(state) = 2),
    postal_code         TEXT,                        -- stored as TEXT to preserve leading zeros (e.g. '05677')
    emp_routing_number  TEXT,                         -- bank routing number for agent payouts (sensitive)
    emp_acct_number     TEXT                          -- bank account number for agent payouts (sensitive)
);

CREATE INDEX idx_agents_state ON agents (state);

-- ----------------------------------------------------------------------------
-- Table: vendors
-- Business purpose: One row per third-party service/repair vendor that can
-- be assigned to fulfill a claim (e.g. auto body shop, medical provider).
-- Used to detect anomalies such as a vendor tied to a disproportionate share
-- of high-value or denied claims (a common collusion/kickback fraud pattern).
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS vendors;
CREATE TABLE vendors (
    vendor_id       TEXT PRIMARY KEY
                        CHECK (vendor_id LIKE 'VNDR%'),
    vendor_name     TEXT NOT NULL,
    address_line1   TEXT,
    address_line2   TEXT,                             -- sparsely populated (apt/suite), nullable
    city            TEXT,                              -- nullable: a small number of source rows omit city
    state           TEXT CHECK (length(state) = 2),
    postal_code     TEXT                               -- stored as TEXT to preserve leading zeros
);

CREATE INDEX idx_vendors_state ON vendors (state);

-- ----------------------------------------------------------------------------
-- Table: insurance_claims
-- Business purpose: One row per insurance claim transaction, combining
-- policy, customer, incident, and financial details. This is the primary
-- fact table for anomaly detection: unusual CLAIM_AMOUNT relative to
-- PREMIUM_AMOUNT, suspiciously short gaps between POLICY_EFF_DT and
-- LOSS_DT, delayed reporting (REPORT_DT - LOSS_DT), late-night incidents,
-- and concentration of claims by AGENT_ID / VENDOR_ID are all analyses
-- this table is designed to support.
-- ----------------------------------------------------------------------------
-- NOTE: column order below intentionally matches insurance_data.csv's header
-- order so the file can be loaded with a plain positional CSV import
-- (e.g. sqlite3 `.import`) without a column-mapping step.
DROP TABLE IF EXISTS insurance_claims;
CREATE TABLE insurance_claims (
    txn_date_time                DATETIME NOT NULL,     -- timestamp the transaction/claim record was created
    transaction_id              TEXT PRIMARY KEY
                                     CHECK (transaction_id LIKE 'TXN%'),
    customer_id                  TEXT NOT NULL,
    policy_number                TEXT NOT NULL,
    policy_eff_dt                DATE NOT NULL,         -- policy effective (start) date
    loss_dt                      DATE NOT NULL,         -- date the insured loss/incident occurred
    report_dt                    DATE NOT NULL,         -- date the claim was reported to the insurer
    insurance_type                TEXT NOT NULL
                                     CHECK (insurance_type IN
                                        ('Property', 'Mobile', 'Health', 'Life', 'Travel', 'Motor')),
    premium_amount                NUMERIC(12, 2) NOT NULL CHECK (premium_amount >= 0),
    claim_amount                  NUMERIC(12, 2) NOT NULL CHECK (claim_amount >= 0),

    -- Customer / policyholder details
    customer_name                 TEXT,
    address_line1                 TEXT,
    address_line2                 TEXT,                 -- sparsely populated (apt/suite), nullable
    city                          TEXT,                 -- nullable: a small number of source rows omit city
    state                         TEXT CHECK (length(state) = 2),
    postal_code                   TEXT,                 -- stored as TEXT to preserve leading zeros
    ssn                           TEXT,                  -- sensitive PII; mask/encrypt at rest in production
    marital_status                 TEXT CHECK (marital_status IN ('Y', 'N')),
    age                            INTEGER CHECK (age >= 0),
    tenure                         INTEGER CHECK (tenure >= 0),   -- customer relationship tenure, in months
    employment_status              TEXT CHECK (employment_status IN ('Y', 'N')),
    no_of_family_members           INTEGER CHECK (no_of_family_members >= 0),
    risk_segmentation              TEXT CHECK (risk_segmentation IN ('L', 'M', 'H')),   -- Low / Medium / High risk tier
    house_type                     TEXT CHECK (house_type IN ('Own', 'Rent', 'Mortgage')),
    social_class                   TEXT CHECK (social_class IN ('LI', 'MI', 'HI')),      -- Low / Middle / High income
    routing_number                 TEXT,                  -- customer bank routing number (sensitive)
    acct_number                    TEXT,                  -- customer bank account number (sensitive)
    customer_education_level       TEXT CHECK (customer_education_level IN
                                        ('High School', 'College', 'Bachelor', 'Masters', 'MD', 'PhD', 'NA')),

    -- Claim / incident details
    claim_status                   TEXT NOT NULL CHECK (claim_status IN ('A', 'D')),    -- Approved / Denied
    incident_severity              TEXT CHECK (incident_severity IN
                                        ('Total Loss', 'Major Loss', 'Minor Loss')),
    authority_contacted            TEXT CHECK (authority_contacted IN
                                        ('Police', 'Ambulance', 'Other', 'None')),
    any_injury                     INTEGER CHECK (any_injury IN (0, 1)),                 -- boolean flag
    police_report_available        INTEGER CHECK (police_report_available IN (0, 1)),    -- boolean flag
    incident_state                 TEXT CHECK (length(incident_state) = 2),
    incident_city                  TEXT,                   -- nullable: a small number of source rows omit city
    incident_hour_of_the_day       INTEGER CHECK (incident_hour_of_the_day BETWEEN 0 AND 23),

    -- Relationships
    agent_id                       TEXT NOT NULL,
    vendor_id                      TEXT,                    -- nullable: not every claim involves a vendor

    CONSTRAINT fk_claims_agent
        FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT fk_claims_vendor
        FOREIGN KEY (vendor_id) REFERENCES vendors (vendor_id)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT chk_claims_dates
        CHECK (report_dt >= loss_dt)
);

-- Frequently queried / joined columns
CREATE INDEX idx_claims_loss_dt        ON insurance_claims (loss_dt);
CREATE INDEX idx_claims_report_dt      ON insurance_claims (report_dt);
CREATE INDEX idx_claims_txn_date_time  ON insurance_claims (txn_date_time);
CREATE INDEX idx_claims_agent_id       ON insurance_claims (agent_id);
CREATE INDEX idx_claims_vendor_id      ON insurance_claims (vendor_id);
CREATE INDEX idx_claims_claim_amount   ON insurance_claims (claim_amount);
CREATE INDEX idx_claims_customer_id    ON insurance_claims (customer_id);
CREATE INDEX idx_claims_policy_number  ON insurance_claims (policy_number);
CREATE INDEX idx_claims_insurance_type ON insurance_claims (insurance_type);
CREATE INDEX idx_claims_status         ON insurance_claims (claim_status);

-- Composite index to speed up common anomaly-detection query shapes:
-- "high-value claims per agent/vendor over a date range"
CREATE INDEX idx_claims_agent_amount   ON insurance_claims (agent_id, claim_amount);
CREATE INDEX idx_claims_vendor_amount  ON insurance_claims (vendor_id, claim_amount);
