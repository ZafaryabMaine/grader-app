# old version
import difflib
from datetime import datetime, timezone

import gspread
import pandas as pd
import streamlit as st
from oauth2client.service_account import ServiceAccountCredentials

# --- APP CONFIG ---
st.set_page_config(layout="wide", page_title="Adversarial Diff Grader")


# --- HARDCODED USERS (replace manually as needed) ---
USERS = {"alice": "pass1", "bob": "pass2"}
RESPONSES_TAB = "responses"
ANNOTATION_INT_FIELDS = [
    "Readable",
    "GrammarOK",
    "MeaningChanged",
    "UnintendedChange",
    "GarbageArtifacts",
]

# --- STATE MANAGEMENT ---
SESSION_DEFAULTS = {
    "authenticated": False,
    "username": None,
    "current_idx": 0,
    "annotations": {},
    "user_sheet_rows": {},
    "loaded_sheet_username": None,
    "csv_signature": None,
}
for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- HELPER: TEXT DIFF ---
def show_diff(text1, text2):
    """Highlights differences between two strings."""
    result = ""
    codes = difflib.SequenceMatcher(None, text1, text2).get_opcodes()
    for tag, i1, i2, j1, j2 in codes:
        if tag == "equal":
            result += text2[j1:j2]
        elif tag == "insert":
            result += (
                f"<span style='background-color: #d4edda; color: #155724; "
                f"font-weight: bold;'>{text2[j1:j2]}</span>"
            )
        elif tag == "replace":
            result += (
                f"<span style='background-color: #fff3cd; color: #856404; "
                f"font-weight: bold;'>{text2[j1:j2]}</span>"
            )
    return result


def normalize_sheet_value(value):
    if pd.isna(value):
        return ""
    return value


def to_int_if_possible(value):
    if value in (None, ""):
        return value
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return value


def column_letter(col_num):
    result = ""
    while col_num > 0:
        col_num, remainder = divmod(col_num - 1, 26)
        result = chr(65 + remainder) + result
    return result


def get_sheet_credentials_dict():
    if "gcp_service_account" not in st.secrets:
        raise KeyError("Missing [gcp_service_account] in .streamlit/secrets.toml")
    return dict(st.secrets["gcp_service_account"])


def get_responses_worksheet():
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = ServiceAccountCredentials.from_json_keyfile_dict(
        get_sheet_credentials_dict(), scopes
    )
    client = gspread.authorize(credentials)

    sheet_id = st.secrets.get("google_sheet_id")
    sheet_name = st.secrets.get("google_sheet_name")

    if sheet_id:
        workbook = client.open_by_key(sheet_id)
    elif sheet_name:
        workbook = client.open(sheet_name)
    else:
        raise KeyError(
            "Missing google_sheet_id or google_sheet_name in .streamlit/secrets.toml"
        )

    return workbook.worksheet(RESPONSES_TAB)


def ensure_sheet_headers(worksheet, required_headers):
    existing_headers = worksheet.row_values(1)
    if not existing_headers:
        worksheet.update("A1", [required_headers])
        return required_headers

    missing_headers = [header for header in required_headers if header not in existing_headers]
    if missing_headers:
        existing_headers = existing_headers + missing_headers
        worksheet.update("A1", [existing_headers])
    return existing_headers


def load_user_sheet_rows(username):
    try:
        worksheet = get_responses_worksheet()
        all_values = worksheet.get_all_values()
    except Exception as exc:
        st.warning(f"Google Sheets is temporarily unavailable. Progress sync disabled: {exc}")
        st.session_state.user_sheet_rows = {}
        st.session_state.loaded_sheet_username = username
        return

    user_rows = {}
    if all_values:
        headers = all_values[0]
        for row_num, row in enumerate(all_values[1:], start=2):
            row_dict = {
                headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))
            }
            if row_dict.get("username") != username:
                continue

            raw_index = to_int_if_possible(row_dict.get("index"))
            if raw_index in (None, ""):
                continue

            for field in ANNOTATION_INT_FIELDS:
                row_dict[field] = to_int_if_possible(row_dict.get(field))

            user_rows[raw_index] = {
                "sheet_row_num": row_num,
                "data": row_dict,
            }

    st.session_state.user_sheet_rows = user_rows
    st.session_state.loaded_sheet_username = username


def build_annotation_from_sheet_record(df_row, sheet_record):
    row_data = df_row.to_dict()
    saved_data = dict(sheet_record)
    saved_data.pop("username", None)
    saved_data.pop("timestamp", None)
    return {
        **row_data,
        "Readable": to_int_if_possible(saved_data.get("Readable")),
        "GrammarOK": to_int_if_possible(saved_data.get("GrammarOK")),
        "MeaningChanged": to_int_if_possible(saved_data.get("MeaningChanged")),
        "ChangeType": saved_data.get("ChangeType", "num"),
        "UnintendedChange": to_int_if_possible(saved_data.get("UnintendedChange")),
        "GarbageArtifacts": to_int_if_possible(saved_data.get("GarbageArtifacts")),
        "Rationale": saved_data.get("Rationale", ""),
    }


def hydrate_annotations_from_sheet(df):
    annotations = {}
    saved_rows = st.session_state.user_sheet_rows
    for filtered_idx, row in df.iterrows():
        raw_index = to_int_if_possible(row.get("index"))
        if raw_index in saved_rows:
            annotations[filtered_idx] = build_annotation_from_sheet_record(
                row, saved_rows[raw_index]["data"]
            )
    st.session_state.annotations = annotations


def first_unannotated_index(total_rows):
    for idx in range(total_rows):
        if idx not in st.session_state.annotations:
            return idx
    return max(total_rows - 1, 0)


def persist_annotation(username, annotation_record):
    timestamp = datetime.now(timezone.utc).isoformat()
    row_payload = {
        "username": username,
        "timestamp": timestamp,
        **annotation_record,
    }

    required_headers = list(row_payload.keys())
    raw_index = to_int_if_possible(annotation_record.get("index"))

    try:
        worksheet = get_responses_worksheet()
        headers = ensure_sheet_headers(worksheet, required_headers)
        row_values = [normalize_sheet_value(row_payload.get(header, "")) for header in headers]

        if raw_index in st.session_state.user_sheet_rows:
            sheet_row_num = st.session_state.user_sheet_rows[raw_index]["sheet_row_num"]
            range_ref = f"A{sheet_row_num}:{column_letter(len(headers))}{sheet_row_num}"
            worksheet.update(range_ref, [row_values])
        else:
            worksheet.append_row(row_values, value_input_option="USER_ENTERED")
            sheet_row_num = len(worksheet.get_all_values())

        st.session_state.user_sheet_rows[raw_index] = {
            "sheet_row_num": sheet_row_num,
            "data": row_payload,
        }
        return True
    except Exception as exc:
        st.warning(
            f"Could not sync this annotation to Google Sheets right now: {exc}. "
            "Your in-session work is still available."
        )
        return False


def reset_user_session():
    for key in [
        "authenticated",
        "username",
        "current_idx",
        "annotations",
        "user_sheet_rows",
        "loaded_sheet_username",
        "csv_signature",
    ]:
        st.session_state[key] = SESSION_DEFAULTS[key]


def show_login():
    st.title("🔐 Login")
    st.caption("Sign in to continue to the grader.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if USERS.get(username) == password:
                reset_user_session()
                st.session_state.authenticated = True
                st.session_state.username = username
                load_user_sheet_rows(username)
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

st.title("🔍 Filtered Adversarial Grader")
st.caption("Only showing rows where Anchor and Adversarial transcripts differ.")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)

    # FILTER: Only rows where transcripts are different
    df = raw_df[
        raw_df["AnchorTranscript"] != raw_df["AdversarialTranscript"]
    ].copy().reset_index()

    file_signature = (
        st.session_state.username,
        uploaded_file.name,
        uploaded_file.size,
        len(df),
    )

    if st.session_state.csv_signature != file_signature:
        if st.session_state.loaded_sheet_username != st.session_state.username:
            load_user_sheet_rows(st.session_state.username)
        hydrate_annotations_from_sheet(df)
        st.session_state.current_idx = first_unannotated_index(len(df)) if len(df) else 0
        st.session_state.csv_signature = file_signature

    total_rows = len(df)
    completed_rows = len(st.session_state.annotations)
    st.sidebar.info(f"{completed_rows} of {total_rows} rows completed")

    if total_rows == 0:
        st.success("All rows are identical! Nothing to annotate.")
    else:
        if st.session_state.current_idx >= total_rows:
            st.session_state.current_idx = total_rows - 1

        idx = st.session_state.current_idx

        # --- NAVIGATION ---
        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
        with col_nav1:
            if st.button("⬅️ Previous") and idx > 0:
                st.session_state.current_idx -= 1
                st.rerun()
        with col_nav2:
            st.write(
                f"**Sample {idx + 1} of {total_rows}** "
                f"(Filtered from {len(raw_df)} total)"
            )
            st.progress((idx + 1) / total_rows)
        with col_nav3:
            if st.button("Skip ➡️") and idx < total_rows - 1:
                st.session_state.current_idx += 1
                st.rerun()

        st.divider()

        # --- DISPLAY AREA ---
        anchor_txt = str(df.iloc[idx]["AnchorTranscript"])
        adversarial_txt = str(df.iloc[idx]["AdversarialTranscript"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⚓ Anchor")
            st.info(anchor_txt)

        with col2:
            st.subheader("⚡ Adversarial (Diff Highlighted)")
            diff_html = show_diff(anchor_txt, adversarial_txt)
            st.markdown(
                (
                    "<div style='padding: 10px; border: 1px solid #ccc; "
                    f"border-radius: 5px;'>{diff_html}</div>"
                ),
                unsafe_allow_html=True,
            )

        st.divider()

        # --- ANNOTATION FORM ---
        existing_data = st.session_state.annotations.get(idx, {})

        with st.form("annotation_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                readable = st.radio(
                    "Readable?",
                    [1, 0],
                    index=0 if existing_data.get("Readable") == 1 else 1,
                    horizontal=True,
                )
                grammar = st.radio(
                    "Grammar OK?",
                    [1, 0],
                    index=0 if existing_data.get("GrammarOK") == 1 else 1,
                    horizontal=True,
                )
            with c2:
                meaning_changed = st.radio(
                    "Meaning Changed?",
                    [1, 0],
                    index=0 if existing_data.get("MeaningChanged", 1) == 1 else 1,
                    horizontal=True,
                )
                change_type = st.selectbox(
                    "Change Type",
                    ["num", "neg", "ent", "multi", "none"],
                    index=["num", "neg", "ent", "multi", "none"].index(
                        existing_data.get("ChangeType", "num")
                    ),
                )
            with c3:
                unintended = st.radio(
                    "Unintended?",
                    [1, 0],
                    index=0 if existing_data.get("UnintendedChange", 1) == 1 else 1,
                    horizontal=True,
                )
                garbage = st.radio(
                    "Garbage?",
                    [0, 1],
                    index=0 if existing_data.get("GarbageArtifacts") == 0 else 1,
                    horizontal=True,
                )

            rationale = st.text_input(
                "Rationale (Short)", value=existing_data.get("Rationale", "")
            )

            if st.form_submit_button("✅ Save & Next", use_container_width=True):
                annotation_record = {
                    **df.iloc[idx].to_dict(),
                    "Readable": readable,
                    "GrammarOK": grammar,
                    "MeaningChanged": meaning_changed,
                    "ChangeType": "none" if meaning_changed == 0 else change_type,
                    "UnintendedChange": unintended,
                    "GarbageArtifacts": garbage,
                    "Rationale": rationale,
                }
                st.session_state.annotations[idx] = annotation_record
                persist_annotation(st.session_state.username, annotation_record)

                if idx < total_rows - 1:
                    st.session_state.current_idx += 1
                st.rerun()

    # --- SIDEBAR DOWNLOAD ---
    if st.session_state.annotations:
        st.sidebar.header("Export")
        final_df = pd.DataFrame(list(st.session_state.annotations.values()))
        st.sidebar.download_button(
            "📥 Download Results",
            final_df.to_csv(index=False),
            "annotated_diffs.csv",
            "text/csv",
        )
