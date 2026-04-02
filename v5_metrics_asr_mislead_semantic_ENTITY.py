#!/usr/bin/env python3
"""
ASR metrics for "misleading but plausible" transcript attacks.

Design goals:
- Single source of truth for normalization and negation canonicalization (src/semantics).
- Semantic numeric matching: "three" == "3" (avoid surface-only wins).
- Conservative entity detection for real ASR transcripts (heuristic + small lexicons).
- Stable output schema for logging (clean/quant/attack rows share keys).

This module is used both by:
  - the SMASH runner (decode-time objective + auto-target selection)
  - offline log analysis
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from src.semantics.canon import normalize_for_wer, canon_entity_surface
from src.semantics.negation import extract_negation_keys, NEG_KEYS, NEG_SPAN_RX


# ============================================================
# WER / CER
# ============================================================



def _canon_entity(surface: str) -> str:
    """Canonicalize entity surfaces to avoid counting case-only or punctuation-only changes."""
    s = normalize_for_wer(surface or "")
    # keep alnum and spaces only
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _edit_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def compute_wer(ref: str, hyp: str) -> float:
    """WER on whitespace-tokenized strings."""
    ref = ref or ""
    hyp = hyp or ""
    try:
        from jiwer import wer as _wer  # type: ignore
        return float(_wer(ref, hyp))
    except Exception:
        r = ref.split()
        h = hyp.split()
        if len(r) == 0:
            return 0.0 if len(h) == 0 else 1.0
        return float(_edit_distance(r, h) / max(len(r), 1))


def compute_cer(ref: str, hyp: str) -> float:
    """CER on raw strings (fallback to edit distance if python-Levenshtein unavailable)."""
    r = ref or ""
    h = hyp or ""
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    try:
        import Levenshtein  # type: ignore
        return float(Levenshtein.distance(r, h) / max(len(r), 1))
    except Exception:
        return float(_edit_distance(list(r), list(h)) / max(len(r), 1))


# ============================================================
# Negation
# ============================================================

def negation_signature(text: str) -> List[str]:
    """Canonical negation cues as a list (keeps duplicates)."""
    return list(extract_negation_keys(text))


def negation_flipped(base: str, hyp: str) -> Tuple[bool, Dict[str, Any]]:
    b = negation_signature(base)
    h = negation_signature(hyp)
    changed = Counter(b) != Counter(h)
    return bool(changed), {"neg_base": b, "neg_hyp": h, "negation_flip": int(changed)}


def find_negation_spans(text: str) -> List[Dict[str, Any]]:
    """Char-span detector used by runner auto-targeting."""
    t = text or ""
    out: List[Dict[str, Any]] = []
    for m in NEG_SPAN_RX.finditer(t):
        s, e = m.span()
        out.append({"kind": "neg", "start": int(s), "end": int(e), "surface": t[s:e]})
    return out


# ============================================================
# Numbers (semantic + span finding)
# ============================================================

_WORD_RX = re.compile(r"[A-Za-z']+")

_NUM_DIGIT_RX = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_NUM_SLASH_RX = re.compile(r"\b\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\b")
_NUM_RANGE_RX = re.compile(r"\b\d+(?:\.\d+)?\s*(?:-|–)\s*\d+(?:\.\d+)?\b")
_NUM_RANGE_TO_RX = re.compile(r"\b\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?\b", re.IGNORECASE)

# For mention extraction (unit-aware)
_NUM_WITH_UNIT_RX = re.compile(
    r"(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*(?P<unit>%|percent|mg|mcg|g|kg|lb|lbs|ml|l|cc|mmhg|bpm|hr|hrs|cm|mm|m)(?![A-Za-z]))?",
    re.IGNORECASE,
)

# Alnum "codes" like SNP1 / UO1 / 2P / B12
_CODE_RX = re.compile(r"\b(?:[a-z]{1,6}\d{1,4}|\d{1,4}[a-z]{1,3})\b", re.IGNORECASE)


_NUM_SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_NUM_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}
_NUM_SCALE = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}


def _parse_number_words(words: List[Tuple[str, int, int]], i: int) -> Tuple[Optional[int], int]:
    """Parse a conservative number-word phrase from words[i:]. Returns (value, next_index)."""
    total = 0
    current = 0
    j = i
    seen = False

    while j < len(words):
        w = (words[j][0] or "").lower()
        if w == "and":
            if seen:
                j += 1
                continue
            break

        if w in _ORDINALS:
            current += int(_ORDINALS[w])
            seen = True
            j += 1
            break

        if w in _NUM_SMALL:
            current += int(_NUM_SMALL[w])
            seen = True
            j += 1
            continue

        if w in _NUM_TENS:
            current += int(_NUM_TENS[w])
            seen = True
            j += 1
            continue

        if w == "hundred":
            if current == 0:
                current = 1
            current *= 100
            seen = True
            j += 1
            continue

        if w in _NUM_SCALE:
            scale = int(_NUM_SCALE[w])
            if current == 0:
                current = 1
            total += current * scale
            current = 0
            seen = True
            j += 1
            continue

        break

    if not seen:
        return None, i
    return int(total + current), j


def _canon_num(num_s: str) -> str:
    canon = (num_s or "").replace(",", "")
    if "." in canon:
        a, b = canon.split(".", 1)
        b = b.rstrip("0")
        canon = a if b == "" else f"{a}.{b}"
    return canon


def find_numeric_spans(text: str) -> List[Dict[str, Any]]:
    """Char-span numeric detector used by runner auto-targeting.

    Returns dicts with:
      kind in {"slash","range","num"}
      start, end, surface
      value (int) only for kind=="num" when parseable
    """
    out: List[Dict[str, Any]] = []
    t = text or ""

    for rx, kind in [(_NUM_SLASH_RX, "slash"), (_NUM_RANGE_RX, "range"), (_NUM_RANGE_TO_RX, "range"), (_NUM_DIGIT_RX, "num")]:
        for m in rx.finditer(t):
            s, e = m.span()
            surf = t[s:e]
            val = None
            if kind == "num":
                digits = re.findall(r"\d+", surf.replace(",", ""))
                if digits:
                    try:
                        val = int(digits[0])
                    except Exception:
                        val = None
            out.append({"kind": kind, "start": int(s), "end": int(e), "surface": surf, "value": val})

    # Number-word spans (conservative)
    words = [(m.group(0), m.start(), m.end()) for m in _WORD_RX.finditer(t)]
    i = 0
    while i < len(words):
        val, j = _parse_number_words(words, i)
        if val is None:
            i += 1
            continue
        s = words[i][1]
        e = words[j - 1][2]
        out.append({"kind": "num", "start": int(s), "end": int(e), "surface": t[s:e], "value": int(val)})
        i = j

    # Prefer structured numeric spans first (slash/range), then longer
    def _pri(k: str) -> int:
        return 0 if k in ("slash", "range") else 1

    out.sort(key=lambda d: (_pri(d.get("kind", "")), -(d["end"] - d["start"]), d["start"]))
    return out


def extract_numeric_mentions(text: str) -> List[Tuple[str, Optional[str], str]]:
    """Return list of semantic numeric mentions: (value, unit, kind).

    kind in {"num","slash","range","code"}.
    """
    t = (text or "").replace("\u00a0", " ")
    out: List[Tuple[str, Optional[str], str]] = []

    # keep order: slash/range first
    for m in _NUM_SLASH_RX.finditer(t):
        s = m.group(0)
        parts = [p.strip() for p in re.split(r"/", s)]
        if len(parts) == 2:
            out.append((f"{_canon_num(parts[0])}/{_canon_num(parts[1])}", None, "slash"))

    for m in _NUM_RANGE_RX.finditer(t):
        s = m.group(0)
        parts = [p.strip() for p in re.split(r"(?:-|–)", s)]
        if len(parts) == 2:
            out.append((f"{_canon_num(parts[0])}-{_canon_num(parts[1])}", None, "range"))
    for m in _NUM_RANGE_TO_RX.finditer(t):
        s = m.group(0)
        parts = [p.strip() for p in re.split(r"\bto\b", s, flags=re.IGNORECASE)]
        if len(parts) == 2:
            out.append((f"{_canon_num(parts[0])}-{_canon_num(parts[1])}", None, "range"))

    # digit+unit
    for m in _NUM_WITH_UNIT_RX.finditer(t.lower()):
        num_s = m.group("num")
        unit = m.group("unit")
        if not num_s:
            continue
        canon = _canon_num(num_s)
        if unit:
            unit = unit.lower()
            if unit == "percent":
                unit = "%"
        out.append((canon, unit, "num"))

    # number-words => semantic numeric mentions
    for sp in find_numeric_spans(t):
        if sp.get("kind") != "num":
            continue
        surf = (sp.get("surface") or "").strip()
        if not surf:
            continue
        # If it's digit span, already added above; only add word spans here.
        if any(ch.isdigit() for ch in surf):
            continue
        v = sp.get("value", None)
        if isinstance(v, int):
                        # Attach unit token immediately following the number-word span (e.g., "five mg").
            unit = None
            tail = t[sp.get("end", 0):].lower()
            m_unit = re.match(r"^\s*(%|percent|mg|mcg|g|kg|lb|lbs|ml|l|cc|mmhg|bpm|hr|hrs|cm|mm|m)\b", tail)
            if m_unit:
                unit = m_unit.group(1)
                if unit == "percent":
                    unit = "%"
            out.append((str(int(v)), unit, "num"))

    # codes
    for code in _CODE_RX.findall(t):
        c = (code or "").lower()
        if c.isdigit():
            continue
        out.append((c, None, "code"))

    return out


def _numeric_surface_mentions(text: str) -> List[str]:
    """Surface-level numeric mentions (for diagnosing surface-only diffs)."""
    t = (text or "").replace("\u00a0", " ")
    out: List[str] = []
    for sp in find_numeric_spans(t):
        surf = normalize_for_wer(sp.get("surface", "")).replace(" ", "")
        if surf:
            out.append(surf)
    # Also include codes as surfaces
    for code in _CODE_RX.findall(t):
        c = normalize_for_wer(code).replace(" ", "")
        if c:
            out.append(c)
    return out


def numeric_mentions_changed(base: str, hyp: str) -> Tuple[bool, Dict[str, Any]]:
    b = extract_numeric_mentions(base)
    h = extract_numeric_mentions(hyp)

    b_counts: Counter = Counter(b)
    h_counts: Counter = Counter(h)

    all_keys = set(b_counts) | set(h_counts)
    unmatched_base = 0
    unmatched_hyp = 0
    kind_delta = {"num": 0, "slash": 0, "range": 0, "code": 0}
    for k in all_keys:
        cb = int(b_counts.get(k, 0))
        ch = int(h_counts.get(k, 0))
        if cb != ch:
            kind_delta[k[2]] += abs(cb - ch)
        if cb > ch:
            unmatched_base += (cb - ch)
        elif ch > cb:
            unmatched_hyp += (ch - cb)

    changed = (unmatched_base > 0) or (unmatched_hyp > 0)

    return bool(changed), {
        "mentions_base": list(b),
        "mentions_hyp": list(h),
        "mentions_unmatched_base": int(unmatched_base),
        "mentions_unmatched_hyp": int(unmatched_hyp),
        "mentions_changed": int(changed),
        "mentions_kind_delta": {k: int(v) for k, v in kind_delta.items()},
    }


# ============================================================
# Entities (conservative, real transcripts)
# ============================================================

# Minimal pools; extend these in your own code if you want higher recall.
_PERSON_FIRST = {
    "john","mary","james","robert","michael","william","david","richard","joseph","thomas",
    "charles","christopher","daniel","matthew","anthony","mark","donald","steven","paul","andrew",
    "emma","olivia","ava","sophia","isabella","mia","amelia","harper","evelyn","abigail",
    "avery",
    "dylan",
    "elena",
    "camden",
    "lucia",
    "marisol",
    "priya",
    "sofia",

}
_PERSON_LAST = {
    "smith","johnson","williams","brown","jones","garcia","miller","davis","rodriguez","martinez",
    "hernandez","lopez","gonzalez","wilson","anderson","taylor","moore","jackson","martin",
    "lee","perez","thompson","white","harris","sanchez","clark","ramirez","lewis","robinson",
    "chen",
    "park",
    "kwon",
    "vega",
    "nair",
    "torres",

}
_DRUGS = {
    "acetaminophen","paracetamol","ibuprofen","naproxen","aspirin","metformin","insulin",
    "lisinopril","losartan","valsartan","amlodipine","metoprolol","atorvastatin","simvastatin","rosuvastatin",
    "omeprazole","pantoprazole","amoxicillin","penicillin","azithromycin","doxycycline","vancomycin","ceftriaxone",
    "prednisone","dexamethasone","albuterol","warfarin","heparin","apixaban","rivaroxaban","clopidogrel",
    "sertraline","fluoxetine","citalopram","gabapentin","pregabalin","morphine","oxycodone","fentanyl",
    "levothyroxine","allopurinol","tamsulosin","ondansetron",
}
_PLACES = {
    "california","texas","florida","new york","washington","massachusetts","illinois",
    "united states","usa","canada","mexico","united kingdom","uk","france","germany","spain","italy",
    "china","japan","india","australia",
    "new york city","los angeles","san francisco","boston","chicago","seattle","miami","houston","dallas","atlanta",
    "atlas insurance",
    "crescent theater",
    "eastport pharmacy",
    "harborview labs",
    "harborview studios",
    "helix robotics",
    "keystone airport",
    "lakeside inn",
    "mapleton clinic",
    "northlake university",
    "nova medical",
    "orion books",
    "redwood city",
    "riverdale hospital",
    "silver pine",
    "summit media",
    "zeta telecom",

}
_ACRONYMS = {
    "cdc","fda","nih","who","icd","ehr","icu","er","ed","ct","mri","cva","dvt","pe","bp","hr","bpm","ecg","ekg",
}

_DRUG_SUFFIX_RX = re.compile(
    r"\b[a-z]{4,}(?:pril|sartan|olol|statin|prazole|mab|nib|vir|navir|cycline|mycin|caine|dipine)\b",
    re.I,
)

_TITLE_NAME_RX = re.compile(r"\b(dr|mr|ms|mrs|prof|professor)\.?\s+([A-Za-z][A-Za-z'\-]{2,})\b", re.I)
_FULLNAME_RX = re.compile(r"\b([A-Za-z][A-Za-z'\-]{2,})\s+([A-Za-z][A-Za-z'\-]{2,})\b")
_ACRONYM_RX = re.compile(r"\b[A-Z]{2,6}\b")

ENTITY_POOLS = {
    "person_first": sorted(_PERSON_FIRST),
    "person_last": sorted(_PERSON_LAST),
    "drug": sorted(_DRUGS),
    "place": sorted(_PLACES),
    "acronym": sorted(_ACRONYMS),
}


def find_entity_spans(text: str, *, types: Optional[set] = None) -> List[Dict[str, Any]]:
    """Conservative entity spans: kind='ent', etype in {person,drug,place,acronym}."""
    t = text or ""
    types = {x.strip().lower() for x in (types or {"person","drug","place","acronym"}) if str(x).strip()}
    out: List[Dict[str, Any]] = []

    def _add(etype: str, s: int, e: int):
        if s < 0 or e <= s:
            return
        out.append({"kind": "ent", "etype": etype, "start": int(s), "end": int(e), "surface": t[s:e]})

    if "place" in types:
        for p in _PLACES:
            rx = re.compile(rf"\b{re.escape(p)}\b", re.I)
            for m in rx.finditer(t):
                _add("place", *m.span())

    if "drug" in types:
        for d in _DRUGS:
            rx = re.compile(rf"\b{re.escape(d)}\b", re.I)
            for m in rx.finditer(t):
                _add("drug", *m.span())
        for m in _DRUG_SUFFIX_RX.finditer(t):
            _add("drug", *m.span())

    if "person" in types:
        for m in _TITLE_NAME_RX.finditer(t):
            s, e = m.span(2)
            _add("person", s, e)

        for m in _FULLNAME_RX.finditer(t):
            w1, w2 = (m.group(1) or ""), (m.group(2) or "")
            if w1.lower() in _PERSON_FIRST and w2.lower() in _PERSON_LAST:
                _add("person", *m.span())

        for m in re.finditer(r"\b[A-Za-z][A-Za-z'\-]{2,}\b", t):
            w = t[m.start():m.end()]
            if w.lower() in _PERSON_FIRST:
                _add("person", *m.span())

    if "acronym" in types:
        for m in _ACRONYM_RX.finditer(t):
            _add("acronym", *m.span())
        for a in _ACRONYMS:
            rx = re.compile(rf"\b{re.escape(a)}\b", re.I)
            for m in rx.finditer(t):
                _add("acronym", *m.span())

    # De-dup: prefer longer overlapping spans
    out.sort(key=lambda d: (-(d["end"] - d["start"]), d["start"]))
    kept: List[Dict[str, Any]] = []
    for d in out:
        s, e = d["start"], d["end"]
        if any((s < k["end"] and e > k["start"]) for k in kept):
            continue
        kept.append(d)
    kept.sort(key=lambda d: d["start"])
    return kept


def entity_mentions_changed(base: str, hyp: str) -> Tuple[bool, Dict[str, Any]]:
    b_sp = find_entity_spans(base)
    h_sp = find_entity_spans(hyp)

    b_items = [(_canon_entity(d["surface"]), d.get("etype", "ent")) for d in b_sp if _canon_entity(d["surface"])]
    h_items = [(_canon_entity(d["surface"]), d.get("etype", "ent")) for d in h_sp if _canon_entity(d["surface"])]

    b_counts = Counter(b_items)
    h_counts = Counter(h_items)

    unmatched_b = list((b_counts - h_counts).elements())
    unmatched_h = list((h_counts - b_counts).elements())

    kind_delta: Dict[str, int] = {"person": 0, "drug": 0, "place": 0, "acronym": 0}
    for _, k in unmatched_b:
        if k in kind_delta:
            kind_delta[k] += 1
    for _, k in unmatched_h:
        if k in kind_delta:
            kind_delta[k] += 1

    changed = int(bool(unmatched_b or unmatched_h))
    stats: Dict[str, Any] = {
        "ent_base": [[s, k] for (s, k) in sorted(list(b_counts.elements()))],
        "ent_hyp": [[s, k] for (s, k) in sorted(list(h_counts.elements()))],
        "ent_unmatched_base": int(len(unmatched_b)),
        "ent_unmatched_hyp": int(len(unmatched_h)),
        "ent_changed": int(changed),
        "ent_kind_delta": kind_delta,
        "entity_changed": int(changed),
    }
    return bool(changed), stats

#in case it is required for deduplication
# from collections import Counter
# from typing import Any, Dict, Tuple

# def entity_mentions_changed(base: str, hyp: str) -> Tuple[bool, Dict[str, Any]]:
#     b_sp = find_entity_spans(base)
#     h_sp = find_entity_spans(hyp)

#     b_items = [(_canon_entity(d["surface"]), d.get("etype", "ent")) for d in b_sp if _canon_entity(d["surface"])]
#     h_items = [(_canon_entity(d["surface"]), d.get("etype", "ent")) for d in h_sp if _canon_entity(d["surface"])]

#     # Counters for *diagnostics*
#     b_counts = Counter(b_items)
#     h_counts = Counter(h_items)

#     # NEW: set-based comparison for "changed" so duplicates don't count as change
#     b_set = set(b_counts.keys())
#     h_set = set(h_counts.keys())

#     unmatched_b_set = sorted(list(b_set - h_set))
#     unmatched_h_set = sorted(list(h_set - b_set))

#     # Optional diagnostics: how much duplication happened in hyp/base
#     dup_b = sum(max(0, c - 1) for c in b_counts.values())
#     dup_h = sum(max(0, c - 1) for c in h_counts.values())

#     kind_delta: Dict[str, int] = {"person": 0, "drug": 0, "place": 0, "acronym": 0}
#     for _, k in unmatched_b_set:
#         if k in kind_delta:
#             kind_delta[k] += 1
#     for _, k in unmatched_h_set:
#         if k in kind_delta:
#             kind_delta[k] += 1

#     changed = int(bool(unmatched_b_set or unmatched_h_set))

#     stats: Dict[str, Any] = {
#         # Keep existing outputs (but now they are "all mentions" including duplicates)
#         "ent_base": [[s, k] for (s, k) in sorted(list(b_counts.elements()))],
#         "ent_hyp": [[s, k] for (s, k) in sorted(list(h_counts.elements()))],

#         # Keep key names, but redefine unmatched counts to mean "unique entity mismatches"
#         "ent_unmatched_base": int(len(unmatched_b_set)),
#         "ent_unmatched_hyp": int(len(unmatched_h_set)),

#         "ent_changed": int(changed),
#         "ent_kind_delta": kind_delta,
#         "entity_changed": int(changed),

#         # NEW optional fields (won't break old code if you ignore them)
#         "ent_dup_base": int(dup_b),
#         "ent_dup_hyp": int(dup_h),
#     }
#     return bool(changed), stats


# ============================================================
# Directionality & content drift
# ============================================================

_STOPWORDS = {
    "the","a","an","and","or","but","to","of","in","on","for","with","as","at",
    "is","are","was","were","be","been","being","i","you","he","she","it","we","they",
    "this","that","these","those","there","here","who","whom","which","what","when","where","why","how",
    "do","does","did","done","doing","have","has","had","having","will","would","can","could","should","may","might","must",
}

_SHORT_IMPORTANT = {
    # common medical/clinical abbreviations
    "cdc","covid","rna","dna","mrna","sars","mers",
    "dvt","pe","icu","er","ed","pcr","crp","inr","ecg","ekg","ct","mri","us",
    "egfr","bmi","hr","rr","bp","uo1","snp","snp1","cc",
}

_DIR_PAIRS = [
    ("increase","decrease"),
    ("increased","decreased"),
    ("increasing","decreasing"),
    ("higher","lower"),
    ("more","less"),
    ("up","down"),
    ("left","right"),
    ("anterior","posterior"),
    ("proximal","distal"),
    ("benign","malignant"),
    ("acute","chronic"),
    ("present","absent"),
    ("positive","negative"),
]

_WORD_RE = re.compile(r"[a-z0-9']+")


def _simple_stem(w: str) -> str:
    w = w.lower()
    if len(w) <= 4:
        return w
    for suf in ("ing","edly","ed","ly","es","s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def directionality_flips(base: str, hyp: str) -> Tuple[int, Dict[str, Any]]:
    bset = set(_simple_stem(w) for w in _WORD_RE.findall(normalize_for_wer(base)))
    hset = set(_simple_stem(w) for w in _WORD_RE.findall(normalize_for_wer(hyp)))
    flips = []
    for a, b in _DIR_PAIRS:
        sa = _simple_stem(a)
        sb = _simple_stem(b)
        if (sa in bset) and (sb in hset) and (sb not in bset):
            flips.append((a, b))
        if (sb in bset) and (sa in hset) and (sa not in bset):
            flips.append((b, a))
    uniq = []
    seen = set()
    for x in flips:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return len(uniq), {"dir_flips": uniq, "dir_flip_count": int(len(uniq))}


def content_keywords(text: str, k: int = 12) -> List[str]:
    toks = _WORD_RE.findall(normalize_for_wer(text))
    out: List[str] = []
    seen = set()
    neg_set = set(NEG_KEYS)
    for w in toks:
        if w.isdigit():
            continue
        if w in _STOPWORDS or w in neg_set:
            continue
        is_alnum_code = (any(c.isalpha() for c in w) and any(c.isdigit() for c in w) and len(w) >= 3)
        if not (len(w) >= 4 or w in _SHORT_IMPORTANT or is_alnum_code):
            continue
        sw = _simple_stem(w)
        if sw in seen:
            continue
        seen.add(sw)
        out.append(w)

    # acronyms first, then longer tokens
    out.sort(key=lambda x: (0 if x in _SHORT_IMPORTANT else 1, -len(x)))
    return out[:k]


def keyword_drops(base: str, hyp: str, k: int = 12) -> Tuple[int, Dict[str, Any]]:
    kb = content_keywords(base, k=k)
    hset = set(_simple_stem(w) for w in _WORD_RE.findall(normalize_for_wer(hyp)))
    dropped = [w for w in kb if _simple_stem(w) not in hset]
    return len(dropped), {"kw_base": kb, "kw_dropped": dropped, "kw_drop_count": int(len(dropped))}


# ============================================================
# Mislead score
# ============================================================

def mislead_score(base: str, hyp: str) -> Tuple[float, Dict[str, Any]]:
    """Score semantic pivots between baseline and hypothesis."""
    score = 0.0
    flags: List[str] = []

    n_changed, n_stats = numeric_mentions_changed(base, hyp)
    if n_changed:
        score += 2.0
        flags.append("numeric")

    neg_changed, neg_stats = negation_flipped(base, hyp)
    if neg_changed:
        score += 2.0
        flags.append("negation")

    ent_changed, ent_stats = entity_mentions_changed(base, hyp)
    if ent_changed:
        score += 2.0
        flags.append("entity")

    d_count, d_stats = directionality_flips(base, hyp)
    if d_count > 0:
        score += 1.0
        flags.append("direction")

    kw_drop, kw_stats = keyword_drops(base, hyp, k=12)
    drops = kw_stats.get("kw_dropped", [])
    drops_short = any((d in _SHORT_IMPORTANT) for d in drops)
    if kw_drop >= 2 or drops_short:
        score += 1.0
        flags.append("content")

    # Surface-only diagnostics for numerics (semantic unchanged but surfaces changed)
    b_surf = _numeric_surface_mentions(base)
    h_surf = _numeric_surface_mentions(hyp)
    num_surface_only = int((sorted(b_surf) != sorted(h_surf)) and (not n_changed))

    stats: Dict[str, Any] = {}
    stats.update(n_stats)
    stats.update(neg_stats)
    stats.update(ent_stats)
    stats.update(d_stats)
    stats.update(kw_stats)
    stats["mislead_flags"] = flags
    stats["mislead_score"] = float(score)
    stats["critical_score"] = float(score)  # alias used by older runners
    stats["num_surface_only"] = int(num_surface_only)

    # semantic number lists (canonical values from extract_numeric_mentions)
    def _num_sem_list(ms: List[Tuple[str, Optional[str], str]]) -> List[str]:
        outl: List[str] = []
        for v, unit, kind in ms:
            if kind not in ("num", "slash", "range"):
                continue
            outl.append(f"{v}" if unit is None else f"{v} {unit}".strip())
        return outl

    stats["num_sem_base"] = _num_sem_list(stats.get("mentions_base", []))
    stats["num_sem_hyp"] = _num_sem_list(stats.get("mentions_hyp", []))
    stats["num_sem_changed"] = int(sorted(stats["num_sem_base"]) != sorted(stats["num_sem_hyp"]))

    return float(score), stats


# ============================================================
# Public API
# ============================================================

_PLACEHOLDER_BASELINE_KEYS: Dict[str, Any] = {
    # semantic deltas (baseline-relative) placeholders
    "mentions_base": [],
    "mentions_hyp": [],
    "mentions_unmatched_base": 0,
    "mentions_unmatched_hyp": 0,
    "mentions_changed": 0,
    "mentions_kind_delta": {"num": 0, "slash": 0, "range": 0, "code": 0},
    "neg_base": [],
    "neg_hyp": [],
    "negation_flip": 0,
    "dir_flips": [],
    "dir_flip_count": 0,
    "ent_base": [],
    "ent_hyp": [],
    "ent_unmatched_base": 0,
    "ent_unmatched_hyp": 0,
    "ent_changed": 0,
    "ent_kind_delta": {"person": 0, "drug": 0, "place": 0, "acronym": 0},
    "entity_changed": 0,
    "kw_base": [],
    "kw_dropped": [],
    "kw_drop_count": 0,
    "mislead_flags": [],
    "mislead_score": 0.0,
    "critical_score": 0.0,
    "num_sem_base": [],
    "num_sem_hyp": [],
    "num_sem_changed": 0,
    "num_surface_only": 0,
    # legacy aliases
    "mislead_flags_semantic": [],
    "mislead_score_semantic": 0.0,
    "critical_score_semantic": 0.0,
}


def compute_asr_metrics(
    reference: str,
    hyp: str,
    baseline: Optional[str] = None,
    *,
    include_semantic: Optional[bool] = None,
) -> Dict[str, Any]:
    """Compute ASR metrics.

    Always returns:
      - wer_ref, cer_ref

    If baseline is provided, also returns:
      - wer_base, cer_base
      - semantic pivot metrics (mislead_score, num/neg/entity, etc.)
    """
    out: Dict[str, Any] = {}

    ref_n = normalize_for_wer(reference)
    hyp_n = normalize_for_wer(hyp)
    out["wer_ref"] = compute_wer(ref_n, hyp_n)
    out["cer_ref"] = compute_cer(reference or "", hyp or "")

    if baseline is None:
        out.update(_PLACEHOLDER_BASELINE_KEYS)
        return out

    base_n = normalize_for_wer(baseline)
    out["wer_base"] = compute_wer(base_n, hyp_n)
    out["cer_base"] = compute_cer(baseline or "", hyp or "")

    if include_semantic is None:
        include_semantic = True

    if include_semantic:
        _score, stats = mislead_score(baseline, hyp)
        out.update(stats)
        # stable aliases for runner
        out["mislead_flags_semantic"] = list(out.get("mislead_flags", []))
        out["mislead_score_semantic"] = float(out.get("mislead_score", 0.0))
        out["critical_score_semantic"] = float(out.get("critical_score", 0.0))
    else:
        out.update(_PLACEHOLDER_BASELINE_KEYS)

    return out


__all__ = [
    "normalize_for_wer",
    "compute_asr_metrics",
    "mislead_score",
    "extract_numeric_mentions",
    "numeric_mentions_changed",
    "negation_signature",
    "negation_flipped",
    "find_negation_spans",
    "find_numeric_spans",
    "find_entity_spans",
    "ENTITY_POOLS",
]
