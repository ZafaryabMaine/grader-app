#!/usr/bin/env python3
"""
Build the packet_metadata-ready CSV from input_v3 source files.

Input:
  - docs/paper_readiness/input_v3.csv (162 rows)
  - docs/paper_readiness/input_v3_judge_overlap.csv (37 overlap rows)

Output:
  - data/packet_v3_final.csv (162 rows, packet_metadata schema)

Transformations:
  1. Map all input_v3 columns to packet_metadata schema
  2. Generate row_id as {sample_id}::{method}::{model}::{preset}
  3. Fix 5 rows with empty adversarial_pred (set to quant_pred)
  4. Derive is_overlap from overlap file (match base judge_ids)
  5. Add packet_version = "v3"
  6. Drop columns not in packet_metadata (sample_path, preset_canonical, intended_target_etype)
"""

import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_V3 = os.path.join(REPO_ROOT, "docs", "paper_readiness", "input_v3.csv")
OVERLAP_FILE = os.path.join(REPO_ROOT, "docs", "paper_readiness", "input_v3_judge_overlap.csv")
OUTPUT_FILE = os.path.join(REPO_ROOT, "data", "packet_v3_final.csv")

# packet_metadata column order (excluding created_at which Supabase auto-fills)
OUTPUT_COLUMNS = [
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
]


def main():
    # Load overlap base IDs
    overlap_base_ids = set()
    with open(OVERLAP_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            base_id = row["judge_id"].replace("_OVL", "")
            overlap_base_ids.add(base_id)
    print(f"Overlap base IDs loaded: {len(overlap_base_ids)}")

    # Load and transform input_v3
    with open(INPUT_V3, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        source_rows = list(reader)
    print(f"Source rows loaded: {len(source_rows)}")

    output_rows = []
    fixes_applied = 0

    for src in source_rows:
        judge_id = src["judge_id"]
        sample_id = src["sample_id"]
        method = src["method"]
        model = src["model"]
        preset = src["preset"]
        quant_pred = src["quant_pred"]
        adversarial_pred = (src.get("adversarial_pred") or "").strip()

        # Fix: empty adversarial_pred → use quant_pred (gate-rejected, no change)
        if not adversarial_pred:
            adversarial_pred = quant_pred
            fixes_applied += 1

        # Generate row_id
        row_id = f"{sample_id}::{method}::{model}::{preset}"

        # Derive is_overlap
        is_overlap = 1 if judge_id in overlap_base_ids else 0

        out = {
            "judge_id": judge_id,
            "row_id": row_id,
            "sample_id": sample_id,
            "method": method,
            "method_family": src.get("method_family", ""),
            "model": model,
            "preset": preset,
            "dataset": src.get("dataset", ""),
            "target_kind": src["target_kind"],
            "intended_source_surface": src["intended_source_surface"],
            "intended_target_surface": src["intended_target_surface"],
            "quant_pred": quant_pred,
            "adversarial_pred": adversarial_pred,
            "reference": src.get("reference", ""),
            "clean_pred": src.get("clean_pred", ""),
            "budget_type": src.get("budget_type", ""),
            "budget_value": int(src.get("budget_value") or 0),
            "is_targeted": int(src.get("is_targeted") or 0),
            "smash_step_accepted": int(src.get("smash_step_accepted") or 0),
            "is_overlap": is_overlap,
            "packet_version": "v3",
            "source_run_dir": src.get("source_run_dir", ""),
            "source_jsonl": src.get("source_jsonl", ""),
        }
        output_rows.append(out)

    # Validate
    row_ids = [r["row_id"] for r in output_rows]
    judge_ids = [r["judge_id"] for r in output_rows]
    assert len(set(row_ids)) == len(row_ids), "Duplicate row_ids found!"
    assert len(set(judge_ids)) == len(judge_ids), "Duplicate judge_ids found!"
    assert all(r["quant_pred"] for r in output_rows), "Empty quant_pred found!"
    assert all(r["adversarial_pred"] for r in output_rows), "Empty adversarial_pred found!"

    overlap_count = sum(1 for r in output_rows if r["is_overlap"] == 1)

    # Write
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Rows: {len(output_rows)}")
    print(f"Columns: {len(OUTPUT_COLUMNS)}")
    print(f"Empty adversarial_pred fixed: {fixes_applied}")
    print(f"Overlap rows marked: {overlap_count}")
    print(f"Identical quant==adv: {sum(1 for r in output_rows if r['quant_pred'] == r['adversarial_pred'])}")
    print(f"smash_step_accepted=1: {sum(1 for r in output_rows if r['smash_step_accepted'] == 1)}")


if __name__ == "__main__":
    main()
