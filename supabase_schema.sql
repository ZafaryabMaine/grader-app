-- SMASH Human-Eval Grader — Supabase schema
-- Run this in the Supabase SQL Editor to create the required tables.
-- Assumes app_users table already exists from previous grader versions.

-- ============================================================
-- packet_metadata: loaded once by researcher via upload script
-- Contains all internal + blind fields. Annotator app queries
-- only the blind subset (judge_id, target_kind, intended_source_surface,
-- intended_target_surface, quant_pred, adversarial_pred).
-- ============================================================

CREATE TABLE IF NOT EXISTS packet_metadata (
    judge_id                TEXT PRIMARY KEY,
    row_id                  TEXT NOT NULL UNIQUE,
    sample_id               TEXT NOT NULL,
    method                  TEXT NOT NULL,
    method_family           TEXT NOT NULL,
    model                   TEXT NOT NULL,
    preset                  TEXT NOT NULL,
    dataset                 TEXT NOT NULL,
    target_kind             TEXT NOT NULL,
    intended_source_surface TEXT NOT NULL,
    intended_target_surface TEXT NOT NULL,
    quant_pred              TEXT NOT NULL,
    adversarial_pred        TEXT NOT NULL,
    reference               TEXT,
    clean_pred              TEXT,
    budget_type             TEXT,
    budget_value            INTEGER,
    is_targeted             SMALLINT NOT NULL DEFAULT 0,
    smash_step_accepted     SMALLINT DEFAULT 0,
    is_overlap              SMALLINT NOT NULL DEFAULT 0,
    packet_version          TEXT NOT NULL,
    source_run_dir          TEXT,
    source_jsonl            TEXT,
    auto_decision_type      TEXT DEFAULT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_packet_metadata_sample
    ON packet_metadata (sample_id);
CREATE INDEX IF NOT EXISTS idx_packet_metadata_method
    ON packet_metadata (method);
CREATE INDEX IF NOT EXISTS idx_packet_metadata_model
    ON packet_metadata (model);
CREATE INDEX IF NOT EXISTS idx_packet_metadata_target_kind
    ON packet_metadata (target_kind);
CREATE INDEX IF NOT EXISTS idx_packet_metadata_packet_version
    ON packet_metadata (packet_version);


-- ============================================================
-- annotations_v4: written by annotators via the Streamlit app.
-- Primary key is (username, judge_id) so each annotator can
-- grade the same row independently (for inter-rater agreement).
-- ============================================================

CREATE TABLE IF NOT EXISTS annotations_v4 (
    username                TEXT NOT NULL,
    judge_id                TEXT NOT NULL REFERENCES packet_metadata(judge_id),
    timestamp               TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_disappeared      TEXT NOT NULL CHECK (source_disappeared IN ('Yes', 'No', 'Unsure')),
    target_appeared         TEXT NOT NULL CHECK (target_appeared IN ('Yes', 'No', 'Unsure')),
    extra_meaning_changed   TEXT NOT NULL CHECK (extra_meaning_changed IN ('Yes', 'No', 'Unsure')),
    obvious_artifact        TEXT NOT NULL CHECK (obvious_artifact IN ('Yes', 'No', 'Unsure')),
    annotator_note          TEXT DEFAULT '',
    clean_success           TEXT NOT NULL CHECK (clean_success IN ('Yes', 'No')),
    partial_success         TEXT NOT NULL CHECK (partial_success IN ('Yes', 'No')),
    target_miss             TEXT NOT NULL CHECK (target_miss IN ('Yes', 'No')),
    PRIMARY KEY (username, judge_id)
);

CREATE INDEX IF NOT EXISTS idx_annotations_v4_judge
    ON annotations_v4 (judge_id);
CREATE INDEX IF NOT EXISTS idx_annotations_v4_user
    ON annotations_v4 (username);


-- ============================================================
-- app_users: if not already present from a previous version,
-- uncomment and run the block below.
-- ============================================================

-- CREATE TABLE IF NOT EXISTS app_users (
--     username        TEXT PRIMARY KEY,
--     password_hash   TEXT NOT NULL,
--     is_active       BOOLEAN NOT NULL DEFAULT TRUE
-- );
