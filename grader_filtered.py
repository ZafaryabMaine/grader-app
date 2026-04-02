import difflib
import hashlib
import re
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from supabase import create_client

# --- APP CONFIG ---
st.set_page_config(layout="wide", page_title="Adversarial Edit Annotator")


# --- CONSTANTS ---
USERS = {"alice": "pass1", "bob": "pass2"}

ANNOTATION_FIELDS = ["intended_edit_achieved", "extra_meaning_changed", "obvious_artifact"]

COLUMN_RENAME_MAP = {
    "AnchorTranscript": "original_sentence",
    "AdversarialTranscript": "adversarial_sentence",
    "Method": "method_name",
    "UtteranceID": "row_id",
}

TARGET_TYPE_LABELS = {
    "number": "Number",
    "negation": "Negation",
    "entity": "Entity",
    "none": "None",
}

TARGET_TYPE_NORMALIZE = {"num": "number", "neg": "negation", "ent": "entity", "none": "none"}

# --- STATE MANAGEMENT ---
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
}
for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- HELPER: TEXT DIFF ---
def show_diff(text1, text2):
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
    return result


def to_int_if_possible(value):
    if value in (None, ""):
        return value
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return value


# --- SUPABASE ---

def get_supabase_client():
    url = st.secrets.get("supabase_url")
    key = st.secrets.get("supabase_key")
    if not url or not key:
        raise KeyError("Missing supabase_url or supabase_key in .streamlit/secrets.toml")
    return create_client(url, key)


def load_user_supabase_rows(username):
    try:
        client = get_supabase_client()
        result = client.table("annotations").select("*").eq("username", username).execute()
    except Exception as exc:
        st.warning(f"Supabase unavailable. Progress sync disabled: {exc}")
        st.session_state.user_db_rows = {}
        st.session_state.loaded_db_username = username
        return

    user_rows = {}
    for record in result.data:
        raw_index = to_int_if_possible(record.get("index"))
        if raw_index in (None, ""):
            continue
        user_rows[raw_index] = {"data": record}

    st.session_state.user_db_rows = user_rows
    st.session_state.loaded_db_username = username


def build_annotation_from_db_record(df_row, db_record):
    row_data = df_row.to_dict()
    saved = dict(db_record)
    saved.pop("username", None)
    saved.pop("timestamp", None)
    saved.pop("id", None)
    return {
        **row_data,
        "intended_edit_achieved": saved.get("intended_edit_achieved") or None,
        "extra_meaning_changed":  saved.get("extra_meaning_changed")  or None,
        "obvious_artifact":       saved.get("obvious_artifact")       or None,
        "annotator_note":         saved.get("annotator_note", ""),
        "strict_success":         saved.get("strict_success", ""),
        "partial_success":        saved.get("partial_success", ""),
        "failure":                saved.get("failure", ""),
        "presented_to_annotator": saved.get("presented_to_annotator", "Yes"),
        "completion_status":      saved.get("completion_status", "annotated"),
    }


def hydrate_annotations_from_supabase(df):
    annotations = {}
    saved_rows = st.session_state.user_db_rows
    for filtered_idx, row in df.iterrows():
        raw_index = to_int_if_possible(row.get("index"))
        if raw_index in saved_rows:
            annotations[filtered_idx] = build_annotation_from_db_record(
                row, saved_rows[raw_index]["data"]
            )
    st.session_state.annotations = annotations


def first_unannotated_index(total_rows):
    for idx in range(total_rows):
        if idx not in st.session_state.annotations:
            return idx
    return max(total_rows - 1, 0)


def persist_annotation(username, annotation_record):
    raw_index = to_int_if_possible(annotation_record.get("index"))
    timestamp = datetime.now(timezone.utc).isoformat()

    row_payload = {
        "username": username,
        "timestamp": timestamp,
        "index": raw_index,
        "row_id": str(annotation_record.get("row_id", "")),
        "intended_edit_achieved": annotation_record.get("intended_edit_achieved", ""),
        "extra_meaning_changed":  annotation_record.get("extra_meaning_changed", ""),
        "obvious_artifact":       annotation_record.get("obvious_artifact", ""),
        "annotator_note":         annotation_record.get("annotator_note", ""),
        "strict_success":         annotation_record.get("strict_success", ""),
        "partial_success":        annotation_record.get("partial_success", ""),
        "failure":                annotation_record.get("failure", ""),
        "presented_to_annotator": annotation_record.get("presented_to_annotator", "Yes"),
        "completion_status":      annotation_record.get("completion_status", "annotated"),
    }

    try:
        client = get_supabase_client()
        client.table("annotations").upsert(
            row_payload, on_conflict="username,index"
        ).execute()
        st.session_state.user_db_rows[raw_index] = {"data": row_payload}
        return True
    except Exception as exc:
        st.warning(
            f"Could not sync to Supabase: {exc}. Your in-session work is still available."
        )
        return False


# --- SELF-CONTAINED DETECTION ---

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

_PERSON_FIRST = frozenset([
    "john","mary","james","robert","michael","william","david","richard","joseph","thomas",
    "charles","christopher","daniel","matthew","anthony","mark","donald","steven","paul",
    "andrew","emma","olivia","ava","sophia","isabella","mia","amelia","harper","evelyn",
    "abigail","avery","dylan","elena","camden","lucia","marisol","priya","sofia",
])
_PERSON_LAST = frozenset([
    "smith","johnson","williams","brown","jones","garcia","miller","davis","rodriguez",
    "martinez","hernandez","lopez","gonzalez","wilson","anderson","taylor","moore",
    "jackson","martin","lee","perez","thompson","white","harris","sanchez","clark",
    "ramirez","lewis","robinson","chen","park","kwon","vega","nair","torres",
])
_DRUGS = frozenset([
    "acetaminophen","paracetamol","ibuprofen","naproxen","aspirin","metformin","insulin",
    "lisinopril","losartan","amlodipine","metoprolol","atorvastatin","simvastatin",
    "omeprazole","amoxicillin","azithromycin","doxycycline","prednisone","albuterol",
    "warfarin","heparin","sertraline","fluoxetine","gabapentin","morphine","oxycodone",
    "levothyroxine","ondansetron",
])
_PLACES = frozenset([
    "california","texas","florida","new york","washington","massachusetts","illinois",
    "united states","usa","canada","mexico","united kingdom","uk","france","germany",
    "spain","italy","china","japan","india","australia","new york city","los angeles",
    "san francisco","boston","chicago","seattle","miami","houston","dallas","atlanta",
    "atlas insurance","crescent theater","eastport pharmacy","harborview labs",
    "harborview studios","helix robotics","keystone airport","lakeside inn",
    "mapleton clinic","northlake university","nova medical","orion books","redwood city",
    "riverdale hospital","silver pine","summit media","zeta telecom",
])
_ENTITY_POOL = _DRUGS | _PLACES
_TITLE_RX = re.compile(r"\b(dr|mr|ms|mrs|prof)\.?\s+([a-z][a-z'\-]{2,})\b", re.IGNORECASE)


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
    for phrase in _ENTITY_POOL:
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            found.append(phrase)
    words = re.findall(r"\b[a-z][a-z'\-]{2,}\b", t)
    for w in words:
        if w in _PERSON_FIRST:
            found.append(w)
        if w in _PERSON_LAST:
            found.append(w)
    for m in _TITLE_RX.finditer(text):
        found.append(m.group(2).lower())
    return Counter(found)


def detect_target_type(original: str, adversarial: str) -> str:
    """Detect what type of change occurred. Priority: negation > number > entity > none."""
    if original == adversarial:
        return "none"
    try:
        if _neg_keys(original) != _neg_keys(adversarial):
            return "negation"
        if _num_mentions(original) != _num_mentions(adversarial):
            return "number"
        if _entity_mentions(original) != _entity_mentions(adversarial):
            return "entity"
    except Exception:
        pass
    return "none"


# --- DATA LOADING ---

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_RENAME_MAP)

    if "row_id" not in df.columns:
        df["row_id"] = df.index.astype(str)

    if "method_name" not in df.columns:
        df["method_name"] = ""
        st.warning("Column 'method_name' not found in CSV; defaulting to empty.")

    if "target_type" not in df.columns or df["target_type"].isna().all() or (df["target_type"].astype(str).str.strip() == "").all():
        df["target_type"] = df.apply(
            lambda r: detect_target_type(
                str(r.get("original_sentence", "")),
                str(r.get("adversarial_sentence", "")),
            ),
            axis=1,
        )
    else:
        df["target_type"] = df["target_type"].apply(
            lambda v: TARGET_TYPE_NORMALIZE.get(str(v).strip().lower(), str(v).strip().lower())
        )

    return df


def auto_decide_rows(df: pd.DataFrame):
    full_df = df.copy()

    for col in ["auto_decision", "auto_reason", "presented_to_annotator", "completion_status"]:
        full_df[col] = ""

    mask_identical = full_df["original_sentence"] == full_df["adversarial_sentence"]

    for idx in full_df[mask_identical].index:
        tt = full_df.at[idx, "target_type"]
        full_df.at[idx, "presented_to_annotator"] = "No"
        full_df.at[idx, "completion_status"] = "auto_decided"
        if tt == "none":
            full_df.at[idx, "auto_decision"] = "strict_success"
            full_df.at[idx, "auto_reason"] = "identical_sentence_with_none_target"
        else:
            full_df.at[idx, "auto_decision"] = "failure"
            full_df.at[idx, "auto_reason"] = "identical_sentence_with_expected_target_edit"

    display_df = full_df[~mask_identical].copy().reset_index()
    return full_df, display_df


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

    normalized = normalize_columns(raw_df)
    full_df, display_df = auto_decide_rows(normalized)

    return full_df, display_df, signature


# --- OUTPUT / REPORT ---

def build_output_csv(full_df: pd.DataFrame, annotations: dict, display_df: pd.DataFrame) -> pd.DataFrame:
    output = full_df.copy()

    output_ann_cols = [
        "intended_edit_achieved", "extra_meaning_changed", "obvious_artifact",
        "annotator_note", "strict_success", "partial_success", "failure",
        "annotator_username", "annotation_timestamp",
    ]
    for col in output_ann_cols:
        if col not in output.columns:
            output[col] = ""

    for display_idx, ann in annotations.items():
        try:
            orig_idx = int(display_df.iloc[display_idx]["index"])
        except (KeyError, IndexError, ValueError):
            continue
        for col in output_ann_cols:
            if col in ann:
                output.at[orig_idx, col] = ann[col]
        output.at[orig_idx, "presented_to_annotator"] = ann.get("presented_to_annotator", "Yes")
        output.at[orig_idx, "completion_status"] = ann.get("completion_status", "annotated")

    return output


def generate_summary_report(output_df: pd.DataFrame, username: str) -> str:
    total = len(output_df)
    auto = (output_df["completion_status"] == "auto_decided").sum()
    manual = (output_df["completion_status"] == "annotated").sum()

    ss = (output_df.get("strict_success", pd.Series(dtype=str)) == "Yes").sum() + \
         (output_df.get("auto_decision", pd.Series(dtype=str)) == "strict_success").sum()
    ps = (output_df.get("partial_success", pd.Series(dtype=str)) == "Yes").sum()
    fl = (output_df.get("failure", pd.Series(dtype=str)) == "Yes").sum() + \
         (output_df.get("auto_decision", pd.Series(dtype=str)) == "failure").sum()

    ann_rows = output_df[output_df["completion_status"] == "annotated"]
    iea_yes = (ann_rows["intended_edit_achieved"] == "Yes").sum() if len(ann_rows) else 0
    emc_yes = (ann_rows["extra_meaning_changed"] == "Yes").sum() if len(ann_rows) else 0
    oa_yes  = (ann_rows["obvious_artifact"] == "Yes").sum() if len(ann_rows) else 0

    def pct(n, d):
        return f"{100 * n / d:.1f}%" if d else "N/A"

    lines = [
        "# Annotation Summary Report", "",
        f"- **Annotator:** {username}",
        f"- **Generated:** {datetime.now(timezone.utc).isoformat()}", "",
        "## Row Counts",
        f"- Total source rows: {total}",
        f"- Auto-decided rows (not shown): {auto}",
        f"- Manually annotated rows: {manual}", "",
        "## Outcomes (all rows)",
        f"- strict_success: {ss} ({pct(ss, total)})",
        f"- partial_success: {ps} ({pct(ps, total)})",
        f"- failure: {fl} ({pct(fl, total)})", "",
        "## Manual Annotation Rates",
        f"- intended_edit_achieved = Yes: {iea_yes} / {manual} ({pct(iea_yes, manual)})",
        f"- extra_meaning_changed = Yes:  {emc_yes} / {manual} ({pct(emc_yes, manual)})",
        f"- obvious_artifact = Yes:       {oa_yes} / {manual} ({pct(oa_yes, manual)})", "",
    ]

    if "method_name" in output_df.columns:
        lines += ["## Breakdown by method_name", ""]
        for method, grp in output_df.groupby("method_name"):
            g_ss = (grp.get("strict_success", pd.Series(dtype=str)) == "Yes").sum() + \
                   (grp.get("auto_decision", pd.Series(dtype=str)) == "strict_success").sum()
            g_fl = (grp.get("failure", pd.Series(dtype=str)) == "Yes").sum() + \
                   (grp.get("auto_decision", pd.Series(dtype=str)) == "failure").sum()
            lines.append(f"- **{method}**: {len(grp)} rows | strict_success={g_ss} | failure={g_fl}")
        lines.append("")

    if "target_type" in output_df.columns:
        lines += ["## Breakdown by target_type", ""]
        for tt, grp in output_df.groupby("target_type"):
            g_ss = (grp.get("strict_success", pd.Series(dtype=str)) == "Yes").sum() + \
                   (grp.get("auto_decision", pd.Series(dtype=str)) == "strict_success").sum()
            g_fl = (grp.get("failure", pd.Series(dtype=str)) == "Yes").sum() + \
                   (grp.get("auto_decision", pd.Series(dtype=str)) == "failure").sum()
            lines.append(f"- **{tt}**: {len(grp)} rows | strict_success={g_ss} | failure={g_fl}")
        lines.append("")

    if len(ann_rows):
        lines += ["## Y/N/N Combination Counts (manual annotations)", ""]
        combos = [
            ("Y N N", "Yes", "No",  "No",  "clean success"),
            ("Y Y N", "Yes", "Yes", "No",  "target hit + collateral damage"),
            ("Y N Y", "Yes", "No",  "Yes", "target hit + obvious artifact"),
            ("Y Y Y", "Yes", "Yes", "Yes", "target hit + collateral + artifact"),
            ("N N N", "No",  "No",  "No",  "miss, otherwise clean"),
            ("N Y N", "No",  "Yes", "No",  "wrong/unintended change"),
            ("N N Y", "No",  "No",  "Yes", "miss + obvious artifact"),
            ("N Y Y", "No",  "Yes", "Yes", "wrong change + obvious artifact"),
        ]
        for label, iea, emc, oa, desc in combos:
            count = (
                (ann_rows["intended_edit_achieved"] == iea) &
                (ann_rows["extra_meaning_changed"]  == emc) &
                (ann_rows["obvious_artifact"]       == oa)
            ).sum()
            lines.append(f"- **{label}** ({desc}): {count}")
        lines.append("")

    return "\n".join(lines)


# --- SESSION ---

def reset_user_session():
    for key in SESSION_DEFAULTS:
        st.session_state[key] = SESSION_DEFAULTS[key]


def show_login():
    st.title("Login")
    st.caption("Sign in to continue to the annotator.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if USERS.get(username) == password:
                reset_user_session()
                st.session_state.authenticated = True
                st.session_state.username = username
                load_user_supabase_rows(username)
                st.rerun()
            else:
                st.error("Invalid username or password.")


# --- AUTH GATE ---
if not st.session_state.authenticated:
    show_login()
    st.stop()

# --- SIDEBAR AUTH ---
st.sidebar.success(f"Logged in as: {st.session_state.username}")
if st.sidebar.button("Logout"):
    reset_user_session()
    st.rerun()

st.title("Adversarial Edit Annotator")
st.caption("Human evaluation of targeted adversarial sentence edits.")

# --- AUTO-LOAD CSV ---
if not st.session_state.data_loaded:
    with st.spinner("Loading dataset..."):
        full_df, display_df, signature = load_source_csv()
    new_sig = (st.session_state.username, signature)
    if st.session_state.csv_signature != new_sig:
        if st.session_state.loaded_db_username != st.session_state.username:
            load_user_supabase_rows(st.session_state.username)
        hydrate_annotations_from_supabase(display_df)
        st.session_state.current_idx = first_unannotated_index(len(display_df)) if len(display_df) else 0
        st.session_state.csv_signature = new_sig
    st.session_state.full_df = full_df
    st.session_state.display_df = display_df
    st.session_state.data_loaded = True

full_df = st.session_state.full_df
display_df = st.session_state.display_df
total_rows = len(display_df)
completed_rows = len(st.session_state.annotations)

# --- SIDEBAR PROGRESS + INSTRUCTIONS ---
st.sidebar.info(f"{completed_rows} of {total_rows} rows completed")

with st.sidebar.expander("Annotation Instructions", expanded=False):
    st.markdown("""
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

**What to ignore (unless meaning clearly changes):**
- Punctuation differences (comma, period, quote)
- Capitalization differences
- Spacing differences
- "ten" vs "10" — treat as equivalent

**Obvious artifact includes:**
- Repeated words or phrases
- Nonsense fragments
- Corrupted symbols / broken encoding
- Visibly incomplete or truncated text
- Degeneration or very noticeable manipulation
""")

# --- SIDEBAR EXPORT (always available once any annotations exist) ---
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
    st.success(
        f"All {len(full_df)} rows have identical original and adversarial sentences — "
        "nothing to manually annotate."
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

# --- ROW METADATA ---
meta1, meta2, meta3, meta4 = st.columns(4)
with meta1:
    st.metric("Row ID", str(row.get("row_id", "—")))
with meta2:
    st.metric("Method", str(row.get("method_name", "—")))
with meta3:
    target_raw = str(row.get("target_type", ""))
    st.metric("Expected Target", TARGET_TYPE_LABELS.get(target_raw, target_raw or "—"))
with meta4:
    surface = row.get("target_surface_in_original", "")
    if surface and str(surface) not in ("", "nan"):
        st.metric("Target Surface", str(surface))

st.divider()

# --- DIFF DISPLAY ---
original_txt = str(row["original_sentence"])
adversarial_txt = str(row["adversarial_sentence"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.info(original_txt)
with col2:
    st.subheader("Adversarial (changes highlighted)")
    diff_html = show_diff(original_txt, adversarial_txt)
    st.markdown(
        f"<div style='padding: 10px; border: 1px solid #ccc; border-radius: 5px;'>{diff_html}</div>",
        unsafe_allow_html=True,
    )

st.divider()

# --- ANNOTATION FORM ---
existing_data = st.session_state.annotations.get(idx, {})


def _radio_index(field):
    val = existing_data.get(field)
    if val == "Yes":
        return 0
    if val == "No":
        return 1
    return None


with st.form("annotation_form"):
    q1, q2, q3 = st.columns(3)
    with q1:
        intended = st.radio(
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

    annotator_note = st.text_input(
        "Annotator note (optional)",
        value=existing_data.get("annotator_note", ""),
    )

    submitted = st.form_submit_button("Save & Next", use_container_width=True)

    if submitted:
        unanswered = [
            name for name, val in [
                ("Intended edit achieved", intended),
                ("Extra meaning changed", extra_meaning),
                ("Obvious artifact", artifact),
            ]
            if val is None
        ]
        if unanswered:
            st.error(f"Please answer all required questions: {', '.join(unanswered)}")
        else:
            strict = (intended == "Yes" and extra_meaning == "No" and artifact == "No")
            partial = (intended == "Yes" and not strict)
            failure_flag = (intended == "No")

            annotation_record = {
                **row.to_dict(),
                "intended_edit_achieved": intended,
                "extra_meaning_changed": extra_meaning,
                "obvious_artifact": artifact,
                "annotator_note": annotator_note,
                "strict_success": "Yes" if strict else "No",
                "partial_success": "Yes" if partial else "No",
                "failure": "Yes" if failure_flag else "No",
                "presented_to_annotator": "Yes",
                "completion_status": "annotated",
                "annotator_username": st.session_state.username,
                "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            st.session_state.annotations[idx] = annotation_record
            persist_annotation(st.session_state.username, annotation_record)

            if idx < total_rows - 1:
                st.session_state.current_idx += 1
            st.rerun()
