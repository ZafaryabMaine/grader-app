#!/usr/bin/env python3
"""
Build the final grader-ready packet from data/input.csv.

Logic:
  1. Map input.csv columns to packet_metadata schema
  2. For presentable rows with empty target_kind/intended_source/intended_target:
     - If same sample_id has known context elsewhere in the dataset: restore it
     - Otherwise: mark as auto_decision_type = "missing_context" (excluded from annotator queue)
  3. Write data/packet_final.csv matching packet_metadata schema

Output: data/packet_final.csv
"""

import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_FILE = os.path.join(REPO_ROOT, "data", "input.csv")
OUTPUT_FILE = os.path.join(REPO_ROOT, "data", "packet_final.csv")

CANONICAL_COLUMNS = [
    "judge_id",
    "row_id",
    "sample_id",
    "method",
    "method_family",
    "model",
    "preset",
    "dataset",
    "target_kind",
    "intended_source_surface",
    "intended_target_surface",
    "quant_pred",
    "adversarial_pred",
    "reference",
    "clean_pred",
    "budget_type",
    "budget_value",
    "is_targeted",
    "smash_step_accepted",
    "is_overlap",
    "packet_version",
    "source_run_dir",
    "source_jsonl",
    "auto_decision_type",
]


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))
    print(f"Source: {INPUT_FILE}")
    print(f"Source rows: {len(source_rows)}")

    # Build sample_id → (target_kind, source, target) lookup from populated rows
    sample_context = {}
    for r in source_rows:
        sid = r["sample_id"]
        tk = (r.get("target_kind") or "").strip()
        src = (r.get("intended_source_surface") or "").strip()
        tgt = (r.get("intended_target_surface") or "").strip()
        if tk and src and tgt and sid not in sample_context:
            sample_context[sid] = (tk, src, tgt)

    output_rows = []
    restored_count = 0
    excluded_count = 0

    for src in source_rows:
        method_family = src.get("method_family", "")
        is_targeted = 1 if method_family.startswith("targeted_") else 0

        budget_raw = (src.get("budget") or "0").strip()
        try:
            budget_value = int(budget_raw)
        except ValueError:
            budget_value = 0

        if "ss_" in method_family:
            budget_type = "k_per_module"
        else:
            budget_type = "bit_budget"

        auto_decision = (src.get("auto_decision_type") or "").strip()
        target_kind = (src.get("target_kind") or "").strip()
        intended_src = (src.get("intended_source_surface") or "").strip()
        intended_tgt = (src.get("intended_target_surface") or "").strip()

        # Handle missing context on presentable rows
        if not auto_decision and not target_kind:
            sid = src["sample_id"]
            if sid in sample_context:
                # Restore from same sample_id
                target_kind, intended_src, intended_tgt = sample_context[sid]
                restored_count += 1
            else:
                # Cannot restore — exclude from annotator queue
                auto_decision = "missing_context"
                excluded_count += 1

        out = {
            "judge_id": src["judge_id"],
            "row_id": src.get("row_id", src["judge_id"]),
            "sample_id": src.get("sample_id", ""),
            "method": src.get("method", ""),
            "method_family": method_family,
            "model": src.get("model", ""),
            "preset": src.get("target_preset", ""),
            "dataset": src.get("dataset", ""),
            "target_kind": target_kind,
            "intended_source_surface": intended_src,
            "intended_target_surface": intended_tgt,
            "quant_pred": src.get("quant_pred", ""),
            "adversarial_pred": src.get("adversarial_pred", ""),
            "reference": src.get("reference", ""),
            "clean_pred": src.get("clean_pred", ""),
            "budget_type": budget_type,
            "budget_value": budget_value,
            "is_targeted": is_targeted,
            "smash_step_accepted": 0,
            "is_overlap": int(src.get("is_overlap") or 0),
            "packet_version": src.get("packet_version", "v4"),
            "source_run_dir": src.get("sample_file", ""),
            "source_jsonl": "",
            "auto_decision_type": auto_decision,
        }
        output_rows.append(out)

    # Validate
    judge_ids = [r["judge_id"] for r in output_rows]
    assert len(set(judge_ids)) == len(judge_ids), \
        f"Duplicate judge_ids! {len(judge_ids)} total, {len(set(judge_ids))} unique"
    assert all(r["quant_pred"] for r in output_rows), "Empty quant_pred!"
    assert all(r["adversarial_pred"] for r in output_rows), "Empty adversarial_pred!"

    # Stats
    auto_identical = sum(1 for r in output_rows if r["auto_decision_type"] == "identical_decode")
    auto_missing = sum(1 for r in output_rows if r["auto_decision_type"] == "missing_context")
    auto_total = sum(1 for r in output_rows if (r["auto_decision_type"] or "").strip())
    presentable = len(output_rows) - auto_total
    overlap = sum(1 for r in output_rows if r["is_overlap"] == 1)

    # Verify all presentable rows have context
    for r in output_rows:
        if not (r["auto_decision_type"] or "").strip():
            assert r["target_kind"], f"Presentable row {r['judge_id']} still has empty target_kind!"
            assert r["intended_source_surface"], f"Presentable row {r['judge_id']} has empty intended_source!"
            assert r["intended_target_surface"], f"Presentable row {r['judge_id']} has empty intended_target!"

    # Write
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Total rows: {len(output_rows)}")
    print(f"Columns: {len(CANONICAL_COLUMNS)}")
    print(f"")
    print(f"Auto-decided (identical_decode): {auto_identical}")
    print(f"Auto-decided (missing_context):  {auto_missing}")
    print(f"Auto-decided total:              {auto_total}")
    print(f"Presentable to annotators:       {presentable}")
    print(f"  (of which restored context:    {restored_count})")
    print(f"Overlap rows:                    {overlap}")
    print(f"Packet version:                  {output_rows[0]['packet_version']}")


if __name__ == "__main__":
    main()
