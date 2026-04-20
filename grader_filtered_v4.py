"""
grader_filtered_v4.py — SMASH Human-Eval Grader (NeurIPS packet design)

Version: v4 (2026-04-20)
Based on: grader_filtered.py (stable entrypoint, unchanged)

Changes from v3/stable:
  - Canonical paper-packet schema (input_v3 contract) as primary path
  - 5-question NeurIPS rubric (source_disappeared, target_appeared,
    extra_meaning_changed, obvious_artifact, plausible_but_wrong)
  - Blind vs non-blind view support
  - Legacy CSV ingestion as compatibility layer (clearly separated)
  - Fixed auto-decide logic (identical decode ≠ success when target intended)
  - quant_pred as grading anchor (not reference or AnchorTranscript)

Promotion: Do NOT replace grader_filtered.py until explicitly approved.
"""

import difflib
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import bcrypt
import pandas as pd
import streamlit as st
from supabase import create_client

# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(layout="wide", page_title="SMASH Human-Eval Grader")

# ============================================================
# SCHEMA CONTRACTS
# ============================================================

# --- Canonical paper-packet columns (input_v3 contract) ---
CANONICAL_COLUMNS = [
    "judge_id", "row_id", "sample_id", "method", "method_family", "model",
    "preset", "preset_canonical", "dataset", "budget_type", "budget_value",
    "target_kind", "intended_source_surface", "intended_target_surface",
    "intended_target_etype", "reference", "clean_pred", "quant_pred",
    "adversarial_pred", "is_targeted", "smash_step_accepted", "is_overlap",
    "source_run_dir", "source_jsonl",
]

# Minimum columns required for canonical mode
CANONICAL_REQUIRED = {
    "judge_id", "target_kind", "intended_source_surface",
    "intended_target_surface", "quant_pred", "adversarial_pred",
}

# Columns shown to annotator in blind mode
BLIND_COLUMNS = [
    "judge_id", "target_kind", "intended_source_surface",
    "intended_target_surface", "reference", "quant_pred", "adversarial_pred",
]

# Columns NEVER shown to annotator (leak method identity or provenance)
FORBIDDEN_IN_BLIND = {
    "row_id", "sample_id", "method", "method_family", "model", "preset",
    "preset_canonical", "dataset", "budget_type", "budget_value",
    "is_targeted", "clean_pred", "smash_step_accepted", "is_overlap",
    "source_run_dir", "source_jsonl", "git_sha", "notes",
}

# --- Legacy column rename map (compatibility layer) ---
LEGACY_COLUMN_RENAME = {
    "AnchorTranscript": "quant_pred",
    "AdversarialTranscript": "adversarial_pred",
    "Method": "method",
    "UtteranceID": "sample_id",
    "BlindItemID": "row_id",
    "BudgetBits": "budget_value",
}

# Legacy method name mapping
LEGACY_METHOD_MAP = {
    "PBS/BFA": "pbs_ce",
    "PBS+Constraint": "pbs_ce_gated",
    "SMASH": "smash_masked",
    "TGRF-CE": "tgrf_ce",
    "TGRF-MSB": "tgrf_msb",
    "TGRF-Uniform": "tgrf_uniform",
}

# Legacy model name mapping
LEGACY_MODEL_MAP = {
    "wspS": "whisper_small",
    "wspL": "whisper_large_v3",
    "seamless": "seamless",
}

# Legacy dataset mapping
LEGACY_DATASET_MAP = {
    "syn": "synthetic_wa",
    "lib": "librispeech_test_clean",
    "med": "multimed",
}

# --- Allowed values for validation ---
ALLOWED_METHODS = {
    "smash_masked", "pbs_ce", "pbs_ce_gated",
    "silentstriker_native", "silentstriker_ce", "silentstriker_ce_gated",
    "urbf", "quant_only", "tgrf_ce", "tgrf_msb", "tgrf_uniform",
}

ALLOWED_METHOD_FAMILIES = {"SMASH", "PBS", "SilentStriker", "URBF", "Sanity", "TGRF"}

METHOD_TO_FAMILY = {
    "smash_masked": "SMASH",
    "pbs_ce": "PBS", "pbs_ce_gated": "PBS",
    "silentstriker_native": "SilentStriker",
    "silentstriker_ce": "SilentStriker",
    "silentstriker_ce_gated": "SilentStriker",
    "urbf": "URBF", "quant_only": "Sanity",
    "tgrf_ce": "TGRF", "tgrf_msb": "TGRF", "tgrf_uniform": "TGRF",
}

ALLOWED_MODELS = {"whisper_small", "whisper_large_v3", "seamless"}

ALLOWED_TARGET_KINDS = {"num", "neg", "ent"}

TARGETED_METHODS = {"smash_masked", "silentstriker_native"}

# --- Annotation rubric ---
ANNOTATION_FIELDS_V4 = [
    "source_disappeared", "target_appeared",
    "extra_meaning_changed", "obvious_artifact", "plausible_but_wrong",
]

# Legacy 3-question rubric (used when intended_source/target unavailable)
ANNOTATION_FIELDS_LEGACY = [
    "intended_edit_achieved", "extra_meaning_changed", "obvious_artifact",
]

TARGET_KIND_LABELS = {
    "num": "Number",
    "neg": "Negation",
    "ent": "Entity",
    "none": "None",
    "unknown": "Unknown",
}


# ============================================================
# STATE MANAGEMENT
# ============================================================
SESSION_DEFAULTS = {
    "authenticated": False,
    "username": None,
    "current_idx": 0,
    "annotations": {},
    "user_db_rows": {},
    "loaded_db_username": None,
    "csv_signature": None,
    "data_loaded": False,
    "full_df": None,
    "display_df": None,
    "input_mode": None,  # "canonical" or "legacy"
    "rubric_mode": None,  # "v4" or "legacy"
}
for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def show_diff(text1: str, text2: str) -> str:
    """Character-level diff with HTML highlighting."""
    result = ""
    codes = difflib.SequenceMatcher(None, text1, text2).get_opcodes()
    for tag, i1, i2, j1, j2 in codes:
        if tag == "equal":
            result += text2[j1:j2]
        elif tag == "insert":
            result += (
                f"<span style='background-color: #e8f5e9; color: #2e7d32; "
                f"font-weight: bold;'>{text2[j1:j2]}</span>"
            )
        elif tag == "replace":
            result += (
                f"<span style='background-color: #fff8e1; color: #f57f17; "
                f"font-weight: bold;'>{text2[j1:j2]}</span>"
            )
        elif tag == "delete":
            result += (
                f"<span style='background-color: #ffebee; color: #c62828; "
                f"text-decoration: line-through;'>{text1[i1:i2]}</span>"
            )
    return result


def to_int_if_possible(value):
    if value in (None, ""):
        return value
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return value


# ============================================================
# SUPABASE
# ============================================================

def get_supabase_client():
    url = st.secrets.get("supabase_url")
    key = st.secrets.get("supabase_key")
    if not url or not key:
        raise KeyError("Missing supabase_url or supabase_key in .streamlit/secrets.toml")
    return create_client(url, key)


# ============================================================
# AUTH
# ============================================================

def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(
        plain_password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except Exception:
        return False


def get_user_by_username(username: str):
    try:
        client = get_supabase_client()
        result = (
            client.table("app_users")
            .select("*")
            .ilike("username", username)
            .limit(1)
            .execute()
        )
        if not result.data:
            return None
        return result.data[0]
    except Exception as exc:
        st.error("Failed to query user table.")
        st.exception(exc)
        return None


def create_app_user(username: str, plain_password: str, is_active: bool = True):
    client = get_supabase_client()
    row = {
        "username": username.strip(),
        "password_hash": hash_password(plain_password),
        "is_active": is_active,
    }
    return client.table("app_users").insert(row).execute()


def seed_users(user_list):
    """Bulk-create users. user_list = [(username, plain_password), ...]"""
    client = get_supabase_client()
    rows = [
        {
            "username": u.strip(),
            "password_hash": hash_password(p),
            "is_active": True,
        }
        for u, p in user_list
    ]
    return client.table("app_users").upsert(rows, on_conflict="username").execute()


def load_user_supabase_rows(username):
    try:
        client = get_supabase_client()
        result = client.table("annotations_v4").select("*").eq("username", username).execute()
    except Exception as exc:
        st.error("Supabase connection failed — see details below:")
        st.exception(exc)
        st.session_state.user_db_rows = {}
        st.session_state.loaded_db_username = username
        return

    user_rows = {}
    for record in result.data:
        jid = record.get("judge_id")
        if jid:
            user_rows[jid] = {"data": record}

    st.session_state.user_db_rows = user_rows
    st.session_state.loaded_db_username = username


def hydrate_annotations_from_supabase(df):
    """Restore annotations from Supabase into session state."""
    annotations = {}
    saved_rows = st.session_state.user_db_rows
    for display_idx, row in df.iterrows():
        jid = row.get("judge_id", "")
        if jid in saved_rows:
            annotations[display_idx] = saved_rows[jid]["data"]
    st.session_state.annotations = annotations


def persist_annotation(username, annotation_record):
    judge_id = annotation_record.get("judge_id", "")
    timestamp = datetime.now(timezone.utc).isoformat()

    row_payload = {
        "username": username,
        "judge_id": judge_id,
        "timestamp": timestamp,
        "source_disappeared": str(annotation_record.get("source_disappeared", "")),
        "target_appeared": str(annotation_record.get("target_appeared", "")),
        "extra_meaning_changed": str(annotation_record.get("extra_meaning_changed", "")),
        "obvious_artifact": str(annotation_record.get("obvious_artifact", "")),
        "plausible_but_wrong": str(annotation_record.get("plausible_but_wrong", "")),
        "annotator_note": str(annotation_record.get("annotator_note", "")),
        "clean_success": str(annotation_record.get("clean_success", "")),
        "partial_success": str(annotation_record.get("partial_success", "")),
        "target_miss": str(annotation_record.get("target_miss", "")),
        "completion_status": "annotated",
    }

    try:
        client = get_supabase_client()
        client.table("annotations_v4").upsert(
            row_payload, on_conflict="username,judge_id"
        ).execute()
        st.session_state.user_db_rows[judge_id] = {"data": row_payload}
        return True
    except Exception as exc:
        st.warning(
            f"Could not sync to Supabase: {exc}. Your in-session work is still available."
        )
        return False


# ============================================================
# FORMAT DETECTION & INGESTION
# ============================================================

def detect_input_format(df: pd.DataFrame) -> str:
    """Determine whether input is canonical (v3+) or legacy format."""
    cols = set(df.columns)
    # Canonical: has judge_id and adversarial_pred
    if "judge_id" in cols and "adversarial_pred" in cols:
        return "canonical"
    # Legacy: has AnchorTranscript or AdversarialTranscript
    if "AnchorTranscript" in cols or "AdversarialTranscript" in cols:
        return "legacy"
    # Fallback: check for renamed columns
    if "quant_pred" in cols and "adversarial_pred" in cols:
        return "canonical"
    return "legacy"


def ingest_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a canonical-format CSV."""
    missing = CANONICAL_REQUIRED - set(df.columns)
    if missing:
        st.error(f"Canonical CSV missing required columns: {sorted(missing)}")
        st.stop()

    # Fill optional columns with defaults
    if "row_id" not in df.columns:
        df["row_id"] = df["judge_id"]
    if "sample_id" not in df.columns:
        df["sample_id"] = df["judge_id"]
    if "method" not in df.columns:
        df["method"] = "unknown"
    if "method_family" not in df.columns:
        df["method_family"] = df["method"].map(METHOD_TO_FAMILY).fillna("Unknown")
    if "model" not in df.columns:
        df["model"] = "unknown"
    if "preset" not in df.columns:
        df["preset"] = "unknown"
    if "dataset" not in df.columns:
        df["dataset"] = "unknown"
    if "reference" not in df.columns:
        df["reference"] = df["quant_pred"]
    if "is_targeted" not in df.columns:
        df["is_targeted"] = df["method"].isin(TARGETED_METHODS).astype(int)
    if "smash_step_accepted" not in df.columns:
        df["smash_step_accepted"] = 0
    if "is_overlap" not in df.columns:
        df["is_overlap"] = 0
    if "budget_type" not in df.columns:
        df["budget_type"] = "unknown"
    if "budget_value" not in df.columns:
        df["budget_value"] = 0

    # Normalize target_kind
    df["target_kind"] = df["target_kind"].astype(str).str.strip().str.lower()

    # Internal metadata flag
    df["_input_mode"] = "canonical"
    df["_target_kind_source"] = "explicit"

    return df


# --- Legacy detection heuristics (backward compat only) ---

_NEG_WORDS = frozenset([
    "no", "not", "never", "neither", "nor", "nobody", "nothing", "nowhere",
    "without", "hardly", "barely", "scarcely", "cannot",
])
_NEG_CONTRACTION_RX = re.compile(r"\bn't\b", re.IGNORECASE)

_NUM_RX = re.compile(
    r"\b(\d[\d,]*(?:\.\d+)?|\d+/\d+)\b"
    r"|\b(zero|one|two|three|four|five|six|seven|eight|nine|ten"
    r"|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen"
    r"|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy"
    r"|eighty|ninety|hundred|thousand|million|billion"
    r"|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.IGNORECASE,
)

_PERSON_FIRST_LEGACY = frozenset([
    "john", "mary", "james", "robert", "michael", "william", "david",
    "richard", "joseph", "thomas", "charles", "christopher", "daniel",
    "matthew", "anthony", "mark", "donald", "steven", "paul", "andrew",
    "emma", "olivia", "ava", "sophia", "isabella", "mia", "amelia",
    "harper", "evelyn", "abigail", "avery", "dylan", "elena", "camden",
    "lucia", "marisol", "priya", "sofia",
])
_PERSON_LAST_LEGACY = frozenset([
    "smith", "johnson", "williams", "brown", "jones", "garcia", "miller",
    "davis", "rodriguez", "martinez", "hernandez", "lopez", "gonzalez",
    "wilson", "anderson", "taylor", "moore", "jackson", "martin", "lee",
    "perez", "thompson", "white", "harris", "sanchez", "clark", "ramirez",
    "lewis", "robinson", "chen", "park", "kwon", "vega", "nair", "torres",
])
_DRUGS_LEGACY = frozenset([
    "acetaminophen", "paracetamol", "ibuprofen", "naproxen", "aspirin",
    "metformin", "insulin", "lisinopril", "losartan", "amlodipine",
    "metoprolol", "atorvastatin", "simvastatin", "omeprazole", "amoxicillin",
    "azithromycin", "doxycycline", "prednisone", "albuterol", "warfarin",
    "heparin", "sertraline", "fluoxetine", "gabapentin", "morphine",
    "oxycodone", "levothyroxine", "ondansetron",
])
_PLACES_LEGACY = frozenset([
    "california", "texas", "florida", "new york", "washington",
    "massachusetts", "illinois", "united states", "usa", "canada", "mexico",
    "united kingdom", "uk", "france", "germany", "spain", "italy", "china",
    "japan", "india", "australia", "new york city", "los angeles",
    "san francisco", "boston", "chicago", "seattle", "miami", "houston",
    "dallas", "atlanta", "atlas insurance", "crescent theater",
    "eastport pharmacy", "harborview labs", "harborview studios",
    "helix robotics", "keystone airport", "lakeside inn", "mapleton clinic",
    "northlake university", "nova medical", "orion books", "redwood city",
    "riverdale hospital", "silver pine", "summit media", "zeta telecom",
])
_ENTITY_POOL_LEGACY = _DRUGS_LEGACY | _PLACES_LEGACY
_TITLE_RX_LEGACY = re.compile(
    r"\b(dr|mr|ms|mrs|prof)\.?\s+([a-z][a-z'\-]{2,})\b", re.IGNORECASE
)


def _neg_keys(text: str):
    t = text.lower()
    keys = [w for w in re.findall(r"\b\w+\b", t) if w in _NEG_WORDS]
    keys += _NEG_CONTRACTION_RX.findall(t)
    return Counter(keys)


def _num_mentions(text: str):
    return Counter(m.group(0).lower() for m in _NUM_RX.finditer(text))


def _entity_mentions(text: str):
    t = text.lower()
    found = []
    for phrase in _ENTITY_POOL_LEGACY:
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            found.append(phrase)
    words = re.findall(r"\b[a-z][a-z'\-]{2,}\b", t)
    for w in words:
        if w in _PERSON_FIRST_LEGACY:
            found.append(w)
        if w in _PERSON_LAST_LEGACY:
            found.append(w)
    for m in _TITLE_RX_LEGACY.finditer(text):
        found.append(m.group(2).lower())
    return Counter(found)


def detect_target_type_heuristic(original: str, adversarial: str) -> str:
    """Heuristic target detection from text diff. LEGACY FALLBACK ONLY."""
    if original == adversarial:
        return "none"
    try:
        if _neg_keys(original) != _neg_keys(adversarial):
            return "neg"
        if _num_mentions(original) != _num_mentions(adversarial):
            return "num"
        if _entity_mentions(original) != _entity_mentions(adversarial):
            return "ent"
    except Exception:
        pass
    return "unknown"


def ingest_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a legacy-format CSV into the canonical internal model."""
    # Rename columns
    df = df.rename(columns=LEGACY_COLUMN_RENAME)

    # Generate judge_id if missing
    if "judge_id" not in df.columns:
        df["judge_id"] = [f"L{i:05d}" for i in range(len(df))]

    # Normalize row_id / sample_id
    if "row_id" not in df.columns:
        df["row_id"] = df.get("sample_id", pd.Series(df.index.astype(str)))
    if "sample_id" not in df.columns:
        df["sample_id"] = df["row_id"]

    # Normalize method names
    if "method" in df.columns:
        df["method"] = df["method"].map(
            lambda v: LEGACY_METHOD_MAP.get(str(v).strip(), str(v).strip().lower())
        )
    else:
        df["method"] = "unknown"

    df["method_family"] = df["method"].map(METHOD_TO_FAMILY).fillna("Unknown")

    # Normalize model
    if "Model" in df.columns:
        df["model"] = df["Model"].map(
            lambda v: LEGACY_MODEL_MAP.get(str(v).strip(), str(v).strip().lower())
        )
    elif "model" not in df.columns:
        df["model"] = "unknown"

    # Normalize dataset
    if "Dataset" in df.columns:
        df["dataset"] = df["Dataset"].map(
            lambda v: LEGACY_DATASET_MAP.get(str(v).strip(), str(v).strip().lower())
        )
    elif "dataset" not in df.columns:
        df["dataset"] = "unknown"

    # Preset from RunTag
    if "RunTag" in df.columns:
        df["preset"] = df["RunTag"].apply(
            lambda v: "wdl2" if "wdl2" in str(v).lower() else str(v).strip().lower()
        )
    elif "preset" not in df.columns:
        df["preset"] = "unknown"

    # Budget
    if "budget_value" in df.columns:
        df["budget_value"] = pd.to_numeric(df["budget_value"], errors="coerce").fillna(0).astype(int)
    else:
        df["budget_value"] = 0
    df["budget_type"] = df["method"].apply(
        lambda m: "k_per_module" if "silentstriker" in m else "bit_budget"
    )

    # Text fields
    if "quant_pred" not in df.columns:
        df["quant_pred"] = ""
    if "adversarial_pred" not in df.columns:
        df["adversarial_pred"] = ""

    # Reference: not available in legacy; use quant_pred as proxy
    if "reference" not in df.columns:
        df["reference"] = df["quant_pred"]

    # Target kind: heuristic detection as fallback
    has_target = (
        "target_kind" in df.columns
        and not df["target_kind"].isna().all()
        and not (df["target_kind"].astype(str).str.strip() == "").all()
    )
    if has_target:
        df["target_kind"] = df["target_kind"].astype(str).str.strip().str.lower()
        df["_target_kind_source"] = "explicit"
    else:
        df["target_kind"] = df.apply(
            lambda r: detect_target_type_heuristic(
                str(r.get("quant_pred", "")),
                str(r.get("adversarial_pred", "")),
            ),
            axis=1,
        )
        df["_target_kind_source"] = "inferred"

    # Intended source/target: NOT available in legacy
    if "intended_source_surface" not in df.columns:
        df["intended_source_surface"] = ""
    if "intended_target_surface" not in df.columns:
        df["intended_target_surface"] = ""

    # Flags
    df["is_targeted"] = df["method"].isin(TARGETED_METHODS).astype(int)
    df["smash_step_accepted"] = 0
    df["is_overlap"] = 0
    df["_input_mode"] = "legacy"

    return df


# ============================================================
# ROW CLASSIFICATION (replaces auto_decide_rows)
# ============================================================

def classify_rows(df: pd.DataFrame):
    """
    Classify rows into display vs auto-decided.

    Canonical mode:
      - If quant_pred == adversarial_pred AND target_kind in {num,neg,ent}:
        no_decode_change (attack failed to move decode)
      - If quant_pred == adversarial_pred AND target_kind == "none":
        no_change_expected (control / sanity row)
      - All other rows: presented to annotator.

    Legacy mode (fallback):
      - Same logic but with _target_kind_source="inferred" caveat.
    """
    full_df = df.copy()

    for col in ["auto_decision", "auto_reason", "presented_to_annotator", "completion_status"]:
        full_df[col] = ""

    mask_identical = full_df["quant_pred"].astype(str) == full_df["adversarial_pred"].astype(str)

    for idx in full_df[mask_identical].index:
        tk = str(full_df.at[idx, "target_kind"]).strip().lower()
        full_df.at[idx, "presented_to_annotator"] = "No"
        full_df.at[idx, "completion_status"] = "auto_decided"

        if tk in ALLOWED_TARGET_KINDS:
            # Attack intended an edit but decode didn't change → failure
            full_df.at[idx, "auto_decision"] = "no_decode_change"
            full_df.at[idx, "auto_reason"] = (
                f"identical_decode_with_intended_target_{tk}"
            )
        elif tk == "none":
            # No edit intended, no change happened → expected
            full_df.at[idx, "auto_decision"] = "no_change_expected"
            full_df.at[idx, "auto_reason"] = "control_row_no_edit_intended"
        else:
            # Unknown target, identical decode → inconclusive
            full_df.at[idx, "auto_decision"] = "inconclusive"
            full_df.at[idx, "auto_reason"] = "identical_decode_unknown_target"

    display_df = full_df[~mask_identical].copy().reset_index()
    return full_df, display_df


# ============================================================
# DATA LOADING
# ============================================================

def load_source_csv():
    csv_path = st.secrets.get("csv_file_path")
    csv_url = st.secrets.get("csv_url")

    if not csv_path and not csv_url:
        st.error("Missing csv_file_path or csv_url in .streamlit/secrets.toml")
        st.stop()

    try:
        raw_df = pd.read_csv(csv_path if csv_path else csv_url)
    except Exception as exc:
        st.error(f"Failed to load CSV: {exc}")
        st.stop()

    signature = hashlib.md5(
        pd.util.hash_pandas_object(raw_df).values.tobytes()
    ).hexdigest()

    # Detect format and ingest
    input_mode = detect_input_format(raw_df)
    if input_mode == "canonical":
        normalized = ingest_canonical(raw_df)
        rubric_mode = "v4"
    else:
        normalized = ingest_legacy(raw_df)
        # Use v4 rubric only if intended_source/target are available
        has_intent = (
            normalized["intended_source_surface"].astype(str).str.strip() != ""
        ).any()
        rubric_mode = "v4" if has_intent else "legacy"

    full_df, display_df = classify_rows(normalized)

    st.session_state.input_mode = input_mode
    st.session_state.rubric_mode = rubric_mode

    return full_df, display_df, signature


# ============================================================
# OUTPUT / REPORT
# ============================================================

def compute_verdicts_v4(source_disappeared, target_appeared, extra_meaning, artifact, plausible_wrong):
    """Compute derived verdicts from the 5-question v4 rubric."""
    clean = (
        source_disappeared == "Yes"
        and target_appeared == "Yes"
        and extra_meaning == "No"
        and artifact == "No"
        and plausible_wrong == "No"
    )
    partial = (
        source_disappeared == "Yes"
        and target_appeared == "Yes"
        and not clean
    )
    miss = (source_disappeared == "No" or target_appeared == "No")
    return {
        "clean_success": "Yes" if clean else "No",
        "partial_success": "Yes" if partial else "No",
        "target_miss": "Yes" if miss else "No",
    }


def compute_verdicts_legacy(intended_edit, extra_meaning, artifact):
    """Compute derived verdicts from the 3-question legacy rubric."""
    strict = (intended_edit == "Yes" and extra_meaning == "No" and artifact == "No")
    partial = (intended_edit == "Yes" and not strict)
    failure = (intended_edit == "No")
    return {
        "clean_success": "Yes" if strict else "No",
        "partial_success": "Yes" if partial else "No",
        "target_miss": "Yes" if failure else "No",
    }


def generate_summary_report(output_df: pd.DataFrame, username: str) -> str:
    """Generate a markdown summary report."""
    total = len(output_df)
    auto = (output_df["completion_status"] == "auto_decided").sum()
    manual = (output_df["completion_status"] == "annotated").sum()

    # Count verdicts across auto-decided and annotated
    no_dc = (output_df.get("auto_decision", pd.Series(dtype=str)) == "no_decode_change").sum()
    cs = (output_df.get("clean_success", pd.Series(dtype=str)) == "Yes").sum()
    ps = (output_df.get("partial_success", pd.Series(dtype=str)) == "Yes").sum()
    tm = (output_df.get("target_miss", pd.Series(dtype=str)) == "Yes").sum()

    def pct(n, d):
        return f"{100 * n / d:.1f}%" if d else "N/A"

    lines = [
        "# SMASH Human-Eval Annotation Summary", "",
        f"- **Annotator:** {username}",
        f"- **Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"- **Input mode:** {st.session_state.input_mode}",
        f"- **Rubric mode:** {st.session_state.rubric_mode}", "",
        "## Row Counts",
        f"- Total source rows: {total}",
        f"- Auto-decided rows (not shown): {auto}",
        f"  - no_decode_change (attack failed): {no_dc}",
        f"- Manually annotated rows: {manual}", "",
        "## Outcomes (annotated rows only)",
        f"- clean_success: {cs} ({pct(cs, manual)})",
        f"- partial_success: {ps} ({pct(ps, manual)})",
        f"- target_miss: {tm} ({pct(tm, manual)})", "",
    ]

    # Breakdown by method
    if "method" in output_df.columns:
        lines += ["## Breakdown by method", ""]
        for method, grp in output_df.groupby("method"):
            g_cs = (grp.get("clean_success", pd.Series(dtype=str)) == "Yes").sum()
            g_nodc = (grp.get("auto_decision", pd.Series(dtype=str)) == "no_decode_change").sum()
            lines.append(
                f"- **{method}**: {len(grp)} rows | "
                f"clean_success={g_cs} | no_decode_change={g_nodc}"
            )
        lines.append("")

    # Breakdown by target_kind
    if "target_kind" in output_df.columns:
        lines += ["## Breakdown by target_kind", ""]
        for tk, grp in output_df.groupby("target_kind"):
            g_cs = (grp.get("clean_success", pd.Series(dtype=str)) == "Yes").sum()
            g_nodc = (grp.get("auto_decision", pd.Series(dtype=str)) == "no_decode_change").sum()
            lines.append(
                f"- **{tk}**: {len(grp)} rows | "
                f"clean_success={g_cs} | no_decode_change={g_nodc}"
            )
        lines.append("")

    # Breakdown by model (if multiple)
    if "model" in output_df.columns and output_df["model"].nunique() > 1:
        lines += ["## Breakdown by model", ""]
        for model, grp in output_df.groupby("model"):
            g_cs = (grp.get("clean_success", pd.Series(dtype=str)) == "Yes").sum()
            g_nodc = (grp.get("auto_decision", pd.Series(dtype=str)) == "no_decode_change").sum()
            lines.append(
                f"- **{model}**: {len(grp)} rows | "
                f"clean_success={g_cs} | no_decode_change={g_nodc}"
            )
        lines.append("")

    return "\n".join(lines)


def build_output_csv(full_df: pd.DataFrame, annotations: dict, display_df: pd.DataFrame) -> pd.DataFrame:
    """Build the final output CSV with annotations merged in."""
    output = full_df.copy()

    ann_cols = [
        "source_disappeared", "target_appeared", "extra_meaning_changed",
        "obvious_artifact", "plausible_but_wrong", "annotator_note",
        "clean_success", "partial_success", "target_miss",
        "annotator_username", "annotation_timestamp",
    ]
    for col in ann_cols:
        if col not in output.columns:
            output[col] = ""

    for display_idx, ann in annotations.items():
        try:
            orig_idx = int(display_df.iloc[display_idx]["index"])
        except (KeyError, IndexError, ValueError):
            continue
        for col in ann_cols:
            if col in ann:
                output.at[orig_idx, col] = "" if ann[col] is None else str(ann[col])
        output.at[orig_idx, "presented_to_annotator"] = "Yes"
        output.at[orig_idx, "completion_status"] = "annotated"

    return output


# ============================================================
# SESSION HELPERS
# ============================================================

def reset_user_session():
    for key in SESSION_DEFAULTS:
        st.session_state[key] = SESSION_DEFAULTS[key]


def first_unannotated_index(total_rows):
    for idx in range(total_rows):
        if idx not in st.session_state.annotations:
            return idx
    return max(total_rows - 1, 0)


def show_login():
    st.title("Login")
    st.caption("Sign in to continue to the SMASH Human-Eval Grader.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            username_clean = (username or "").strip()
            if not username_clean or not password:
                st.error("Enter both username and password.")
                return

            user_row = get_user_by_username(username_clean)
            if not user_row:
                st.error("Invalid username or password.")
                return
            if not user_row.get("is_active", True):
                st.error("This account is disabled.")
                return
            if not verify_password(password, user_row.get("password_hash", "")):
                st.error("Invalid username or password.")
                return

            reset_user_session()
            st.session_state.authenticated = True
            st.session_state.username = user_row["username"]
            load_user_supabase_rows(user_row["username"])
            st.rerun()


# ============================================================
# MAIN APP
# ============================================================

# --- AUTH GATE ---
if not st.session_state.authenticated:
    show_login()
    st.stop()

# --- SIDEBAR AUTH ---
st.sidebar.success(f"Logged in as: {st.session_state.username}")
if st.sidebar.button("Logout"):
    reset_user_session()
    st.rerun()

st.title("SMASH Human-Eval Grader")
st.caption("NeurIPS-style human evaluation of targeted adversarial ASR edits.")

# --- AUTO-LOAD CSV ---
if not st.session_state.data_loaded:
    try:
        with st.spinner("Loading dataset..."):
            full_df, display_df, signature = load_source_csv()
        new_sig = (st.session_state.username, signature)
        if st.session_state.csv_signature != new_sig:
            if st.session_state.loaded_db_username != st.session_state.username:
                load_user_supabase_rows(st.session_state.username)
            hydrate_annotations_from_supabase(display_df)
            st.session_state.current_idx = first_unannotated_index(len(display_df))
            st.session_state.csv_signature = new_sig
        st.session_state.full_df = full_df
        st.session_state.display_df = display_df
        st.session_state.data_loaded = True
    except Exception as _load_exc:
        st.error("Error during data loading — see details below:")
        st.exception(_load_exc)
        st.stop()

full_df = st.session_state.full_df
display_df = st.session_state.display_df
total_rows = len(display_df)
completed_rows = len(st.session_state.annotations)

# --- MODE INDICATOR ---
input_mode = st.session_state.input_mode or "unknown"
rubric_mode = st.session_state.rubric_mode or "unknown"

if input_mode == "legacy":
    st.sidebar.warning(
        "Legacy CSV detected. Target metadata is inferred heuristically. "
        "For paper-grade evaluation, use the canonical input_v3 format."
    )

# --- SIDEBAR PROGRESS ---
st.sidebar.info(f"{completed_rows} of {total_rows} rows completed")

# --- SIDEBAR INSTRUCTIONS ---
with st.sidebar.expander("Annotation Instructions", expanded=False):
    if rubric_mode == "v4":
        st.markdown("""
**5-Question Rubric (NeurIPS paper eval)**

**Q1. Source disappeared?**
Did the *intended source surface* disappear from the adversarial decode?
- Compare `quant_pred` with `adversarial_pred`
- Look for the token/phrase shown as "Intended source"

**Q2. Target appeared?**
Did the *intended target surface* appear at the right position in the adversarial decode?
- Look for the token/phrase shown as "Intended target"
- Must be at approximately the same position as the source was

**Q3. Extra meaning changed?**
Did anything else in meaning change beyond the intended edit?

**Q4. Obvious artifact?**
Is the sentence clearly broken, degenerated, or corrupted?
- Repeated words/phrases
- Nonsense fragments
- Corrupted symbols / broken encoding
- Visibly incomplete or truncated text

**Q5. Plausible but wrong?**
Is the sentence grammatically fine but semantically off?
- Reads naturally but conveys different meaning unrelated to the intended edit

---
**Ignore** (unless meaning clearly changes):
- Punctuation differences
- Capitalization differences
- Spacing differences
- "ten" vs "10" — treat as equivalent
""")
    else:
        st.markdown("""
**3-Question Rubric (Legacy mode)**

**Intended edit achieved?**
Did the adversarial sentence make the change it was supposed to make?
- *number* — did the intended number change happen?
- *negation* — did polarity flip as intended?
- *entity* — did the intended entity change happen?
- *none* — was meaning correctly left unchanged?

**Extra meaning changed?**
Did anything else in meaning change beyond the intended edit?

**Obvious artifact?**
Is the sentence clearly broken, degenerated, or corrupted?

---
**Ignore:** punctuation, capitalization, spacing, "ten" vs "10".
""")

# --- SIDEBAR EXPORT ---
if st.session_state.annotations:
    st.sidebar.divider()
    st.sidebar.header("Download Responses")
    interim_df = build_output_csv(full_df, st.session_state.annotations, display_df)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    st.sidebar.download_button(
        "Download Annotations CSV",
        interim_df.to_csv(index=False),
        f"annotations_{st.session_state.username}_{timestamp_str}.csv",
        "text/csv",
        use_container_width=True,
    )

    if completed_rows == total_rows and total_rows > 0:
        report_text = generate_summary_report(interim_df, st.session_state.username)
        st.sidebar.download_button(
            "Download Summary Report",
            report_text,
            f"report_{st.session_state.username}_{timestamp_str}.md",
            "text/markdown",
            use_container_width=True,
        )
        st.sidebar.success("All rows annotated!")

# --- EMPTY STATE ---
if total_rows == 0:
    auto_count = len(full_df)
    no_dc = (full_df.get("auto_decision", pd.Series(dtype=str)) == "no_decode_change").sum()
    st.info(
        f"All {auto_count} rows were auto-classified. "
        f"{no_dc} had no decode change (attack failed). "
        "Nothing to manually annotate."
    )
    st.stop()

# --- BOUNDS CHECK ---
if st.session_state.current_idx >= total_rows:
    st.session_state.current_idx = total_rows - 1

idx = st.session_state.current_idx
row = display_df.iloc[idx]

# --- NAVIGATION ---
col_nav1, col_nav2 = st.columns([1, 3])
with col_nav1:
    if st.button("Previous") and idx > 0:
        st.session_state.current_idx -= 1
        st.rerun()
with col_nav2:
    auto_count = len(full_df) - total_rows
    st.write(
        f"**Sample {idx + 1} of {total_rows}** "
        f"(from {len(full_df)} total rows; {auto_count} auto-decided)"
    )
    st.progress((idx + 1) / total_rows)

st.divider()

# --- ROW METADATA (blind-safe) ---
judge_id = str(row.get("judge_id", f"#{idx}"))
target_kind_raw = str(row.get("target_kind", "unknown"))
intended_src = str(row.get("intended_source_surface", ""))
intended_tgt = str(row.get("intended_target_surface", ""))

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Judge ID", judge_id)
with col_m2:
    st.metric("Target Kind", TARGET_KIND_LABELS.get(target_kind_raw, target_kind_raw))
with col_m3:
    if intended_src and intended_tgt:
        st.metric("Intended Edit", f"{intended_src} → {intended_tgt}")
    elif intended_src:
        st.metric("Intended Source", intended_src)

st.divider()

# --- TEXT DISPLAY ---
quant_txt = str(row.get("quant_pred", ""))
adversarial_txt = str(row.get("adversarial_pred", ""))
reference_txt = str(row.get("reference", ""))

# Show reference if different from quant_pred
if reference_txt and reference_txt != quant_txt:
    with st.expander("Reference (human transcript)", expanded=False):
        st.info(reference_txt)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Baseline (quant_pred)")
    st.info(quant_txt)
with col2:
    st.subheader("Adversarial (changes highlighted)")
    diff_html = show_diff(quant_txt, adversarial_txt)
    st.markdown(
        f"<div style='padding: 10px; border: 1px solid #ccc; "
        f"border-radius: 5px;'>{diff_html}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# --- ANNOTATION FORM ---
existing_data = st.session_state.annotations.get(idx, {})


def _radio_index(field_name):
    val = existing_data.get(field_name)
    if val == "Yes":
        return 0
    if val == "No":
        return 1
    if val == "N/A":
        return 2
    return None


with st.form("annotation_form"):
    if rubric_mode == "v4":
        # --- 5-question v4 rubric ---
        q1, q2 = st.columns(2)
        with q1:
            source_disappeared = st.radio(
                "Q1. Source disappeared?",
                ["Yes", "No", "N/A"],
                index=_radio_index("source_disappeared"),
                horizontal=True,
                help=f"Did '{intended_src}' disappear from the adversarial output?",
            )
        with q2:
            target_appeared = st.radio(
                "Q2. Target appeared?",
                ["Yes", "No", "N/A"],
                index=_radio_index("target_appeared"),
                horizontal=True,
                help=f"Did '{intended_tgt}' appear at the right position?",
            )

        q3, q4, q5 = st.columns(3)
        with q3:
            extra_meaning = st.radio(
                "Q3. Extra meaning changed?",
                ["Yes", "No"],
                index=_radio_index("extra_meaning_changed"),
                horizontal=True,
            )
        with q4:
            artifact = st.radio(
                "Q4. Obvious artifact?",
                ["Yes", "No"],
                index=_radio_index("obvious_artifact"),
                horizontal=True,
            )
        with q5:
            plausible_wrong = st.radio(
                "Q5. Plausible but wrong?",
                ["Yes", "No"],
                index=_radio_index("plausible_but_wrong"),
                horizontal=True,
            )
    else:
        # --- 3-question legacy rubric ---
        q1, q2, q3 = st.columns(3)
        with q1:
            intended_edit = st.radio(
                "Intended edit achieved?",
                ["Yes", "No"],
                index=_radio_index("intended_edit_achieved"),
                horizontal=True,
            )
        with q2:
            extra_meaning = st.radio(
                "Extra meaning changed?",
                ["Yes", "No"],
                index=_radio_index("extra_meaning_changed"),
                horizontal=True,
            )
        with q3:
            artifact = st.radio(
                "Obvious artifact?",
                ["Yes", "No"],
                index=_radio_index("obvious_artifact"),
                horizontal=True,
            )
        # Set v4 fields to N/A for legacy
        source_disappeared = None
        target_appeared = None
        plausible_wrong = None
        intended_edit = intended_edit if "intended_edit" in dir() else None

    annotator_note = st.text_input(
        "Annotator note (optional)",
        value=existing_data.get("annotator_note", ""),
    )

    submitted = st.form_submit_button("Save & Next", use_container_width=True)

    if submitted:
        if rubric_mode == "v4":
            unanswered = []
            if source_disappeared is None:
                unanswered.append("Q1")
            if target_appeared is None:
                unanswered.append("Q2")
            if extra_meaning is None:
                unanswered.append("Q3")
            if artifact is None:
                unanswered.append("Q4")
            if plausible_wrong is None:
                unanswered.append("Q5")

            if unanswered:
                st.error(f"Please answer all questions: {', '.join(unanswered)}")
            else:
                verdicts = compute_verdicts_v4(
                    source_disappeared, target_appeared,
                    extra_meaning, artifact, plausible_wrong,
                )
                annotation_record = {
                    "judge_id": judge_id,
                    "source_disappeared": source_disappeared,
                    "target_appeared": target_appeared,
                    "extra_meaning_changed": extra_meaning,
                    "obvious_artifact": artifact,
                    "plausible_but_wrong": plausible_wrong,
                    "annotator_note": annotator_note,
                    **verdicts,
                    "annotator_username": st.session_state.username,
                    "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
                }
                st.session_state.annotations[idx] = annotation_record
                persist_annotation(st.session_state.username, annotation_record)

                if idx < total_rows - 1:
                    st.session_state.current_idx += 1
                st.rerun()
        else:
            # Legacy rubric
            unanswered = []
            if intended_edit is None:
                unanswered.append("Intended edit")
            if extra_meaning is None:
                unanswered.append("Extra meaning")
            if artifact is None:
                unanswered.append("Artifact")

            if unanswered:
                st.error(f"Please answer: {', '.join(unanswered)}")
            else:
                verdicts = compute_verdicts_legacy(intended_edit, extra_meaning, artifact)
                annotation_record = {
                    "judge_id": judge_id,
                    "intended_edit_achieved": intended_edit,
                    "extra_meaning_changed": extra_meaning,
                    "obvious_artifact": artifact,
                    "source_disappeared": "",
                    "target_appeared": "",
                    "plausible_but_wrong": "",
                    "annotator_note": annotator_note,
                    **verdicts,
                    "annotator_username": st.session_state.username,
                    "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
                }
                st.session_state.annotations[idx] = annotation_record
                persist_annotation(st.session_state.username, annotation_record)

                if idx < total_rows - 1:
                    st.session_state.current_idx += 1
                st.rerun()
