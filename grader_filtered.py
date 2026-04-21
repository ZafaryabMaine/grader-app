"""
SMASH Human-Eval Grader — NeurIPS paper evaluation instrument.

Annotator-facing Streamlit app with strong blinding.
Reads blind columns from Supabase `packet_metadata`.
Writes annotations to Supabase `annotations_v4`.
Hidden metadata (method, model, preset, etc.) is never fetched or displayed.

Architecture:
  - packet_metadata table: loaded by researcher via separate script
  - annotations_v4 table: written by this app (human submissions only)
  - Export/reporting: researcher joins the two tables offline
"""

import csv
import difflib
import io
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

# Compact form styling — tighten Streamlit's default radio/label padding
st.markdown("""<style>
/* Tighten radio button group spacing */
div[data-testid="stForm"] .stRadio > div {gap: 0.3rem;}
div[data-testid="stForm"] .stRadio > label {margin-bottom: 0.15rem; font-size: 0.92em;}
div[data-testid="stForm"] .stRadio > div[role="radiogroup"] {gap: 0.4rem;}
/* Tighten column gaps inside the form */
div[data-testid="stForm"] [data-testid="stHorizontalBlock"] {gap: 0.5rem;}
/* Reduce form internal padding */
div[data-testid="stForm"] > div:first-child {padding-top: 0.6rem; padding-bottom: 0.3rem;}
/* Compact text input */
div[data-testid="stForm"] .stTextInput {margin-top: -0.3rem;}
/* Slightly smaller submit button top margin */
div[data-testid="stForm"] .stFormSubmitButton {margin-top: 0.2rem;}
</style>""", unsafe_allow_html=True)

# Columns fetched from packet_metadata for annotation display
BLIND_COLUMNS = [
    "judge_id",
    "target_kind",
    "intended_source_surface",
    "intended_target_surface",
    "quant_pred",
    "adversarial_pred",
    "auto_decision_type",  # used to filter; never shown to annotator
]

TARGET_KIND_DISPLAY = {
    "num": "Number",
    "neg": "Negation",
    "ent": "Entity",
}

EXPORT_FIELDS = [
    "judge_id", "source_disappeared", "target_appeared",
    "extra_meaning_changed", "obvious_artifact", "plausible_but_wrong",
    "annotator_note", "clean_success", "partial_success", "target_miss",
]

# ============================================================
# SESSION STATE
# ============================================================

_DEFAULTS = {
    "authenticated": False,
    "username": None,
    "current_idx": 0,
    "annotations": {},       # keyed by presentable index
    "all_rows": None,        # all rows from packet_metadata
    "presentable": None,     # list of indices into all_rows (excludes auto-decided)
    "auto_decided_count": 0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ============================================================
# SUPABASE
# ============================================================

def _get_client():
    return create_client(st.secrets["supabase_url"], st.secrets["supabase_key"])


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
    """Fetch blind columns + auto_decision_type from packet_metadata."""
    client = _get_client()
    query = client.table("packet_metadata").select(",".join(BLIND_COLUMNS))
    pv = st.secrets.get("packet_version")
    if pv:
        query = query.eq("packet_version", pv)
    query = query.order("judge_id")
    result = query.execute()
    return result.data or []


def _load_existing_annotations(username: str, presentable_rows):
    """Load this annotator's existing annotations. Returns dict keyed by presentable index."""
    client = _get_client()
    result = (
        client.table("annotations_v4")
        .select("judge_id, source_disappeared, target_appeared, "
                "extra_meaning_changed, obvious_artifact, plausible_but_wrong, "
                "annotator_note")
        .eq("username", username)
        .execute()
    )
    jid_to_pidx = {r["judge_id"]: pi for pi, r in enumerate(presentable_rows)}
    annotations = {}
    for rec in (result.data or []):
        pidx = jid_to_pidx.get(rec["judge_id"])
        if pidx is not None:
            annotations[pidx] = rec
    return annotations


# ============================================================
# PERSISTENCE
# ============================================================

def _save_annotation(username: str, judge_id: str, record: dict):
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
# VERDICTS
# ============================================================

def _compute_verdicts(q1, q2, q3, q4, q5):
    # "Can't tell" on Q1/Q2 is treated as not-Yes (conservative)
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
# DIFF RENDERING
# ============================================================

def _render_diff(baseline: str, adversarial: str) -> str:
    """Highlight changes in the adversarial text only.

    Shows the actual adversarial output with inserted/replaced spans
    highlighted. Deleted text from the baseline is NOT injected — the
    annotator sees the real adversarial sentence, not a mixed artifact.
    """
    _hl = (
        "background:rgba(255,193,7,0.28);font-weight:600;"
        "border-radius:2px;padding:0 2px"
    )
    parts = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, baseline, adversarial
    ).get_opcodes():
        if tag == "equal":
            parts.append(adversarial[j1:j2])
        elif tag in ("insert", "replace"):
            parts.append(f"<span style='{_hl}'>{adversarial[j1:j2]}</span>")
        # "delete" — baseline text removed; do not inject into adversarial
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

if st.session_state.all_rows is None:
    with st.spinner("Loading..."):
        all_rows = _load_rows()

        # Separate presentable vs auto-decided rows
        presentable = []
        auto_count = 0
        for r in all_rows:
            if (r.get("auto_decision_type") or "").strip():
                auto_count += 1
            else:
                presentable.append(r)

        st.session_state.all_rows = all_rows
        st.session_state.presentable = presentable
        st.session_state.auto_decided_count = auto_count

        # Load existing annotations and resume
        st.session_state.annotations = _load_existing_annotations(
            st.session_state.username, presentable
        )
        # Jump to first unannotated presentable row
        st.session_state.current_idx = 0
        for i in range(len(presentable)):
            if i not in st.session_state.annotations:
                st.session_state.current_idx = i
                break

presentable = st.session_state.presentable
total = len(presentable)
auto_count = st.session_state.auto_decided_count
annotations = st.session_state.annotations
completed = len(annotations)

# ============================================================
# EMPTY STATE
# ============================================================

if total == 0:
    st.title("SMASH Human-Eval")
    if auto_count > 0:
        st.info(f"All {auto_count} rows were auto-decided (identical output). Nothing to annotate.")
    else:
        st.info("No rows available for annotation.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.caption(f"Signed in as **{st.session_state.username}**")
    if st.button("Sign out", use_container_width=True):
        _logout()

    st.progress(completed / total if total else 0)
    prog_text = f"{completed} / {total} annotated"
    if auto_count > 0:
        prog_text += f"  \n{auto_count} auto-decided"
    st.caption(prog_text)

    if completed == total:
        st.success("All rows annotated.")

    if annotations:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        for _di in sorted(annotations.keys()):
            ann = annotations[_di]
            writer.writerow({
                "judge_id": ann.get("judge_id", presentable[_di]["judge_id"]),
                **{f: ann.get(f, "") for f in EXPORT_FIELDS if f != "judge_id"},
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
row = presentable[idx]

# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3 = st.columns([1, 3, 1])
with nav1:
    if st.button("← Prev", disabled=(idx == 0), use_container_width=True):
        st.session_state.current_idx -= 1
        st.rerun()
with nav2:
    st.markdown(
        f"<div style='text-align:center;padding:6px 0;font-size:0.82em;opacity:0.5'>"
        f"{idx + 1} of {total}</div>",
        unsafe_allow_html=True,
    )
with nav3:
    if st.button("Next →", disabled=(idx == total - 1), use_container_width=True):
        st.session_state.current_idx += 1
        st.rerun()

# ============================================================
# ROW DISPLAY
# ============================================================

judge_id = row["judge_id"]
target_kind = row["target_kind"]
source_surf = row["intended_source_surface"]
target_surf = row["intended_target_surface"]
kind_label = TARGET_KIND_DISPLAY.get(target_kind, target_kind)

# Task context: intended edit prominent, judge_id secondary
st.markdown(
    f"<div style='margin:4px 0 6px 0'>"
    f"<span style='font-size:0.7em;opacity:0.35;float:right'>{judge_id}</span>"
    f"<span style='font-size:0.95em'>Expected {kind_label.lower()} edit: "
    f"replace <strong>&ldquo;{source_surf}&rdquo;</strong> "
    f"with <strong>&ldquo;{target_surf}&rdquo;</strong></span>"
    f"</div>",
    unsafe_allow_html=True,
)

baseline = row["quant_pred"]
adversarial = row["adversarial_pred"]
diff_html = _render_diff(baseline, adversarial)

_lbl = (
    "font-size:0.7em;opacity:0.4;margin-bottom:2px;"
    "text-transform:uppercase;letter-spacing:0.04em"
)
_box = (
    "padding:10px 14px;border-radius:6px;font-size:1em;line-height:1.6;"
    "border:1px solid rgba(128,128,128,0.2)"
)

st.markdown(
    f"<div style='{_lbl};margin-top:6px'>Baseline (before attack)</div>"
    f"<div style='{_box}'>{baseline}</div>"
    f"<div style='{_lbl};margin-top:8px'>Adversarial (after attack)</div>"
    f"<div style='{_box}'>{diff_html}</div>",
    unsafe_allow_html=True,
)

# ============================================================
# ANNOTATION FORM
# ============================================================

existing = annotations.get(idx, {})


def _ri(field, options):
    val = existing.get(field)
    return options.index(val) if val in options else None


with st.form("annotate", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        q1 = st.radio(
            f'Did "{source_surf}" disappear?',
            ["Yes", "No", "Can't tell"],
            index=_ri("source_disappeared", ["Yes", "No", "Can't tell"]),
            horizontal=True,
            help="Look for this word or phrase in the Baseline. Is it missing from the Adversarial text?",
        )
    with c2:
        q2 = st.radio(
            f'Did "{target_surf}" appear in the right place?',
            ["Yes", "No", "Can't tell"],
            index=_ri("target_appeared", ["Yes", "No", "Can't tell"]),
            horizontal=True,
            help="Does this word or phrase show up in the Adversarial text where the source used to be?",
        )

    c3, c4, c5 = st.columns(3)
    with c3:
        q3 = st.radio(
            "Did anything else in meaning change?",
            ["Yes", "No"],
            index=_ri("extra_meaning_changed", ["Yes", "No"]),
            horizontal=True,
            help="Aside from the intended edit, did the rest of the sentence change in meaning?",
        )
    with c4:
        q4 = st.radio(
            "Does the output look broken?",
            ["Yes", "No"],
            index=_ri("obvious_artifact", ["Yes", "No"]),
            horizontal=True,
            help="Repeated words, gibberish, cut-off text, garbled characters, or obvious nonsense.",
        )
    with c5:
        q5 = st.radio(
            "Sounds normal but says the wrong thing?",
            ["Yes", "No"],
            index=_ri("plausible_but_wrong", ["Yes", "No"]),
            horizontal=True,
            help="The sentence reads fine but says something different or incorrect beyond the intended edit.",
        )

    st.caption('*Ignore punctuation, capitalization, spacing, and "ten" vs "10".*')

    note = st.text_input(
        "Note (optional)",
        value=existing.get("annotator_note", ""),
    )

    submitted = st.form_submit_button("Save & Next", use_container_width=True)

if submitted:
    missing = [f"Q{i+1}" for i, v in enumerate([q1, q2, q3, q4, q5]) if v is None]
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
        try:
            _save_annotation(st.session_state.username, judge_id, record)
        except Exception as exc:
            st.warning(f"Save failed: {exc}. Annotation stored locally.")

        st.session_state.annotations[idx] = record

        # Advance to next unannotated presentable row
        next_idx = idx
        for i in range(idx + 1, total):
            if i not in st.session_state.annotations:
                next_idx = i
                break
        else:
            # All remaining are annotated; stay or wrap
            next_idx = min(idx + 1, total - 1)
        st.session_state.current_idx = next_idx
        st.rerun()
