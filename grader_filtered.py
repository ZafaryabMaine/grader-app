"""
SMASH Human-Eval Grader — NeurIPS paper evaluation instrument.

Annotator-facing Streamlit app with strong blinding.
Reads blind columns from Supabase `packet_metadata`.
Writes annotations to Supabase `annotations_v4`.
Hidden metadata (method, model, preset, etc.) is never fetched or displayed.

Architecture:
  - packet_metadata table: loaded by researcher via separate script
  - annotations_v4 table: written by this app
  - Export/reporting: researcher joins the two tables offline
"""

import difflib
from datetime import datetime, timezone

import bcrypt
import streamlit as st
from supabase import create_client

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    layout="centered",
    page_title="SMASH Human-Eval",
    page_icon=None,
)

# Columns fetched from packet_metadata (blind subset only)
BLIND_COLUMNS = [
    "judge_id",
    "target_kind",
    "intended_source_surface",
    "intended_target_surface",
    "quant_pred",
    "adversarial_pred",
]

TARGET_KIND_DISPLAY = {
    "num": "Number",
    "neg": "Negation",
    "ent": "Entity",
}

# ============================================================
# SESSION STATE
# ============================================================

_DEFAULTS = {
    "authenticated": False,
    "username": None,
    "current_idx": 0,
    "annotations": {},
    "rows": None,
    "total_rows": 0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ============================================================
# SUPABASE
# ============================================================

def _get_client():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    return create_client(url, key)


# ============================================================
# AUTH
# ============================================================

def _verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def _show_login():
    st.title("SMASH Human-Eval")
    st.caption("Sign in to begin annotation.")

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in", use_container_width=True)

    if submit:
        username = (username or "").strip()
        if not username or not password:
            st.error("Enter both username and password.")
            return

        client = _get_client()
        result = (
            client.table("app_users")
            .select("username, password_hash, is_active")
            .ilike("username", username)
            .limit(1)
            .execute()
        )
        if not result.data:
            st.error("Invalid credentials.")
            return

        user = result.data[0]
        if not user.get("is_active", True):
            st.error("Account disabled.")
            return
        if not _verify_password(password, user["password_hash"]):
            st.error("Invalid credentials.")
            return

        st.session_state.authenticated = True
        st.session_state.username = user["username"]
        st.rerun()


def _logout():
    for k in _DEFAULTS:
        st.session_state[k] = _DEFAULTS[k]
    st.rerun()


# ============================================================
# DATA LOADING
# ============================================================

def _load_rows():
    """Fetch blind columns from packet_metadata."""
    client = _get_client()
    columns = ",".join(BLIND_COLUMNS)

    query = client.table("packet_metadata").select(columns)

    # Filter by packet version if configured
    pv = st.secrets.get("packet_version")
    if pv:
        query = query.eq("packet_version", pv)

    query = query.order("judge_id")
    result = query.execute()
    return result.data or []


def _load_existing_annotations(username: str):
    """Load this annotator's existing annotations."""
    client = _get_client()
    result = (
        client.table("annotations_v4")
        .select("judge_id, source_disappeared, target_appeared, "
                "extra_meaning_changed, obvious_artifact, plausible_but_wrong, "
                "annotator_note")
        .eq("username", username)
        .execute()
    )
    annotations = {}
    rows = st.session_state.rows or []
    judge_id_to_idx = {r["judge_id"]: i for i, r in enumerate(rows)}
    for rec in (result.data or []):
        idx = judge_id_to_idx.get(rec["judge_id"])
        if idx is not None:
            annotations[idx] = rec
    return annotations


# ============================================================
# ANNOTATION PERSISTENCE
# ============================================================

def _save_annotation(username: str, judge_id: str, record: dict):
    """Upsert annotation to Supabase."""
    payload = {
        "username": username,
        "judge_id": judge_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_disappeared": record["source_disappeared"],
        "target_appeared": record["target_appeared"],
        "extra_meaning_changed": record["extra_meaning_changed"],
        "obvious_artifact": record["obvious_artifact"],
        "plausible_but_wrong": record["plausible_but_wrong"],
        "annotator_note": record.get("annotator_note", ""),
        "clean_success": record["clean_success"],
        "partial_success": record["partial_success"],
        "target_miss": record["target_miss"],
    }
    client = _get_client()
    client.table("annotations_v4").upsert(
        payload, on_conflict="username,judge_id"
    ).execute()


# ============================================================
# VERDICT COMPUTATION
# ============================================================

def _compute_verdicts(q1, q2, q3, q4, q5):
    clean = (
        q1 == "Yes" and q2 == "Yes"
        and q3 == "No" and q4 == "No" and q5 == "No"
    )
    partial = (q1 == "Yes" and q2 == "Yes" and not clean)
    miss = (q1 != "Yes" or q2 != "Yes")
    return {
        "clean_success": "Yes" if clean else "No",
        "partial_success": "Yes" if partial else "No",
        "target_miss": "Yes" if miss else "No",
    }


# ============================================================
# DIFF DISPLAY
# ============================================================

def _render_diff(baseline: str, adversarial: str) -> str:
    """Character-level diff with theme-safe inline styling."""
    parts = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, baseline, adversarial
    ).get_opcodes():
        if tag == "equal":
            parts.append(adversarial[j1:j2])
        elif tag == "insert":
            parts.append(
                f"<span style='background:rgba(76,175,80,0.25);font-weight:600;"
                f"border-radius:2px;padding:0 1px'>{adversarial[j1:j2]}</span>"
            )
        elif tag == "replace":
            parts.append(
                f"<span style='background:rgba(255,193,7,0.3);font-weight:600;"
                f"border-radius:2px;padding:0 1px'>{adversarial[j1:j2]}</span>"
            )
        elif tag == "delete":
            parts.append(
                f"<span style='background:rgba(244,67,54,0.2);"
                f"text-decoration:line-through;border-radius:2px;"
                f"padding:0 1px'>{baseline[i1:i2]}</span>"
            )
    return "".join(parts)


# ============================================================
# AUTH GATE
# ============================================================

if not st.session_state.authenticated:
    _show_login()
    st.stop()

# ============================================================
# LOAD DATA (once per session)
# ============================================================

if st.session_state.rows is None:
    with st.spinner("Loading..."):
        rows = _load_rows()
        st.session_state.rows = rows
        st.session_state.total_rows = len(rows)
        st.session_state.annotations = _load_existing_annotations(
            st.session_state.username
        )
        # Jump to first unannotated
        for i in range(len(rows)):
            if i not in st.session_state.annotations:
                st.session_state.current_idx = i
                break

rows = st.session_state.rows
total = st.session_state.total_rows
annotations = st.session_state.annotations
completed = len(annotations)

# ============================================================
# EMPTY STATE
# ============================================================

if total == 0:
    st.title("SMASH Human-Eval")
    st.info("No rows available for annotation.")
    st.stop()

# ============================================================
# SIDEBAR — minimal
# ============================================================

with st.sidebar:
    st.caption(f"Signed in as **{st.session_state.username}**")
    if st.button("Sign out", use_container_width=True):
        _logout()

    st.progress(completed / total if total else 0)
    st.caption(f"{completed} / {total} annotated")

    if completed == total:
        st.success("All rows annotated.")

    # Download own annotations (blind export)
    if annotations:
        import csv
        import io

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=[
            "judge_id", "source_disappeared", "target_appeared",
            "extra_meaning_changed", "obvious_artifact", "plausible_but_wrong",
            "annotator_note", "clean_success", "partial_success", "target_miss",
        ])
        writer.writeheader()
        for _dl_idx in sorted(annotations.keys()):
            ann = annotations[_dl_idx]
            writer.writerow({
                "judge_id": ann.get("judge_id", rows[_dl_idx]["judge_id"]),
                "source_disappeared": ann.get("source_disappeared", ""),
                "target_appeared": ann.get("target_appeared", ""),
                "extra_meaning_changed": ann.get("extra_meaning_changed", ""),
                "obvious_artifact": ann.get("obvious_artifact", ""),
                "plausible_but_wrong": ann.get("plausible_but_wrong", ""),
                "annotator_note": ann.get("annotator_note", ""),
                "clean_success": ann.get("clean_success", ""),
                "partial_success": ann.get("partial_success", ""),
                "target_miss": ann.get("target_miss", ""),
            })

        st.download_button(
            "Download my annotations",
            buf.getvalue(),
            f"annotations_{st.session_state.username}.csv",
            "text/csv",
            use_container_width=True,
        )

# ============================================================
# BOUNDS CHECK
# ============================================================

if st.session_state.current_idx >= total:
    st.session_state.current_idx = total - 1
if st.session_state.current_idx < 0:
    st.session_state.current_idx = 0

idx = st.session_state.current_idx
row = rows[idx]

# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3 = st.columns([1, 2, 1])
with nav1:
    if st.button("← Prev", disabled=(idx == 0), use_container_width=True):
        st.session_state.current_idx -= 1
        st.rerun()
with nav2:
    st.markdown(
        f"<div style='text-align:center;padding:4px 0;font-size:0.85em;opacity:0.7'>"
        f"<strong>{idx + 1}</strong> / {total}</div>",
        unsafe_allow_html=True,
    )
with nav3:
    if st.button("Next →", disabled=(idx == total - 1), use_container_width=True):
        st.session_state.current_idx += 1
        st.rerun()

# ============================================================
# ROW DISPLAY — compact header + text blocks
# ============================================================

judge_id = row["judge_id"]
target_kind = row["target_kind"]
source_surf = row["intended_source_surface"]
target_surf = row["intended_target_surface"]
kind_label = TARGET_KIND_DISPLAY.get(target_kind, target_kind)

# Single-line header strip
st.markdown(
    f"<div style='font-size:0.82em;opacity:0.6;margin-top:4px'>"
    f"{judge_id} &nbsp;&middot;&nbsp; {kind_label}: "
    f"<code>{source_surf}</code> &rarr; <code>{target_surf}</code></div>",
    unsafe_allow_html=True,
)

baseline = row["quant_pred"]
adversarial = row["adversarial_pred"]
diff_html = _render_diff(baseline, adversarial)

# Theme-safe text blocks — transparent bg, subtle border via currentColor
_text_box = (
    "padding:10px 14px;border-radius:5px;font-size:1.02em;line-height:1.55;"
    "border:1px solid currentColor;opacity:0.92;margin:4px 0"
)

st.markdown(
    f"<div style='font-size:0.78em;opacity:0.5;margin-top:8px;margin-bottom:2px'>"
    f"BASELINE</div>"
    f"<div style='{_text_box}'>{baseline}</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='font-size:0.78em;opacity:0.5;margin-top:6px;margin-bottom:2px'>"
    f"ADVERSARIAL</div>"
    f"<div style='{_text_box}'>{diff_html}</div>",
    unsafe_allow_html=True,
)

# ============================================================
# ANNOTATION FORM
# ============================================================

existing = annotations.get(idx, {})


def _idx_for(field, options):
    """Return radio index for a previously-saved value."""
    val = existing.get(field)
    if val in options:
        return options.index(val)
    return None


with st.form("annotate", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        q1 = st.radio(
            f'Q1. Did "{source_surf}" disappear?',
            ["Yes", "No", "N/A"],
            index=_idx_for("source_disappeared", ["Yes", "No", "N/A"]),
            horizontal=True,
            help="Is the intended source token/phrase gone from the adversarial text?",
        )
    with c2:
        q2 = st.radio(
            f'Q2. Did "{target_surf}" appear correctly?',
            ["Yes", "No", "N/A"],
            index=_idx_for("target_appeared", ["Yes", "No", "N/A"]),
            horizontal=True,
            help="Did the intended target appear at roughly the right position?",
        )

    c3, c4, c5 = st.columns(3)
    with c3:
        q3 = st.radio(
            "Q3. Extra meaning changed?",
            ["Yes", "No"],
            index=_idx_for("extra_meaning_changed", ["Yes", "No"]),
            horizontal=True,
            help="Did anything else change in meaning beyond the intended edit?",
        )
    with c4:
        q4 = st.radio(
            "Q4. Obvious artifact?",
            ["Yes", "No"],
            index=_idx_for("obvious_artifact", ["Yes", "No"]),
            horizontal=True,
            help="Repeated words, nonsense, truncation, corrupted text, degeneration.",
        )
    with c5:
        q5 = st.radio(
            "Q5. Plausible but wrong?",
            ["Yes", "No"],
            index=_idx_for("plausible_but_wrong", ["Yes", "No"]),
            horizontal=True,
            help="Grammatically fine but semantically off, unrelated to the intended edit.",
        )

    note = st.text_input(
        "Note (optional)",
        value=existing.get("annotator_note", ""),
    )

    st.caption("*Ignore: punctuation, capitalization, spacing, 'ten' vs '10'.*")
    submitted = st.form_submit_button("Save & Next", use_container_width=True)

if submitted:
    # Validate all answered
    missing = []
    if q1 is None:
        missing.append("Q1")
    if q2 is None:
        missing.append("Q2")
    if q3 is None:
        missing.append("Q3")
    if q4 is None:
        missing.append("Q4")
    if q5 is None:
        missing.append("Q5")

    if missing:
        st.error(f"Please answer: {', '.join(missing)}")
    else:
        verdicts = _compute_verdicts(q1, q2, q3, q4, q5)
        record = {
            "judge_id": judge_id,
            "source_disappeared": q1,
            "target_appeared": q2,
            "extra_meaning_changed": q3,
            "obvious_artifact": q4,
            "plausible_but_wrong": q5,
            "annotator_note": note,
            **verdicts,
        }
        # Persist
        try:
            _save_annotation(st.session_state.username, judge_id, record)
        except Exception as exc:
            st.warning(f"Save failed: {exc}. Annotation stored locally.")

        st.session_state.annotations[idx] = record

        # Advance to next unannotated
        if idx < total - 1:
            next_idx = idx + 1
            for i in range(idx + 1, total):
                if i not in st.session_state.annotations:
                    next_idx = i
                    break
            st.session_state.current_idx = next_idx
        st.rerun()
