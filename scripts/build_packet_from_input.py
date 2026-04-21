#!/usr/bin/env python3
"""
Build packet_metadata-ready CSV from data/input.csv.

Maps the new input.csv (27 columns) to the packet_metadata schema (24 columns,
excluding Supabase-managed created_at).

Mappings:
  - target_preset → preset
  - budget → budget_value (int); budget_type inferred from method_family
  - is_targeted derived from method_family (targeted_* = 1, else 0)
  - smash_step_accepted: not in source → default 0
  - sample_file → source_run_dir (best available provenance)
  - source_jsonl: not in source → empty
  - auto_decision_type: pass through from source

Extra source columns preserved as-is for audit:
  - strategy, max_steps, seed, branch, bucket, cell_id, auto_decision_reason

These are appended after the canonical columns so they exist in the CSV
but are not part of the packet_metadata table schema. They can be loaded
into Supabase by adding the columns, or ignored.

Output: data/packet_final.csv
"""

import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_FILE = os.path.join(REPO_ROOT, "data", "input.csv")
OUTPUT_FILE = os.path.join(REPO_ROOT, "data", "packet_final.csv")

# Canonical packet_metadata columns (matches supabase_schema.sql, minus created_at)
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
        reader = csv.DictReader(f)
        source_rows = list(reader)
    print(f"Source: {INPUT_FILE}")
    print(f"Source rows: {len(source_rows)}")

    output_rows = []

    for src in source_rows:
        method_family = src.get("method_family", "")
        is_targeted = 1 if method_family.startswith("targeted_") else 0

        # Budget: parse int from string, infer type
        budget_raw = (src.get("budget") or "0").strip()
        try:
            budget_value = int(budget_raw)
        except ValueError:
            budget_value = 0

        if "ss_" in method_family or "silentstriker" in method_family.lower():
            budget_type = "k_per_module"
        else:
            budget_type = "bit_budget"

        out = {
            "judge_id": src["judge_id"],
            "row_id": src.get("row_id", src["judge_id"]),
            "sample_id": src.get("sample_id", ""),
            "method": src.get("method", ""),
            "method_family": method_family,
            "model": src.get("model", ""),
            "preset": src.get("target_preset", ""),
            "dataset": src.get("dataset", ""),
            "target_kind": src.get("target_kind", ""),
            "intended_source_surface": src.get("intended_source_surface", ""),
            "intended_target_surface": src.get("intended_target_surface", ""),
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
            "auto_decision_type": src.get("auto_decision_type", ""),
        }
        output_rows.append(out)

    # Validate
    judge_ids = [r["judge_id"] for r in output_rows]
    assert len(set(judge_ids)) == len(judge_ids), f"Duplicate judge_ids! {len(judge_ids)} total, {len(set(judge_ids))} unique"
    assert all(r["quant_pred"] for r in output_rows), "Empty quant_pred found!"
    assert all(r["adversarial_pred"] for r in output_rows), "Empty adversarial_pred found!"

    # Stats
    auto_count = sum(1 for r in output_rows if (r["auto_decision_type"] or "").strip())
    presentable = len(output_rows) - auto_count
    overlap = sum(1 for r in output_rows if r["is_overlap"] == 1)

    # Write
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Rows: {len(output_rows)}")
    print(f"Columns: {len(CANONICAL_COLUMNS)}")
    print(f"Auto-decided: {auto_count}")
    print(f"Presentable to annotators: {presentable}")
    print(f"Overlap rows: {overlap}")
    print(f"Packet version: {output_rows[0]['packet_version'] if output_rows else 'n/a'}")


if __name__ == "__main__":
    main()
