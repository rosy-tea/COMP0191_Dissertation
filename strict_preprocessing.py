"""Build a strict, auditable PVK analysis cohort from the raw database.

This is an independent branch.  It never overwrites the original 3,953-row
``df_base.csv``.  The strict cohort requires a complete, position-preserving
material--thickness match for ETL, HTL, back contact, and perovskite layers.

Recorded material categories such as ``Unknown`` and ``None`` are preserved.
Thickness placeholders (blank, nan, None, Unknown) remain in position and make
the corresponding total thickness unavailable; they are never silently
dropped and summed as though the stack were complete.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SEPARATOR_RE = re.compile(r"[|∣｜]")
NUMERIC_MISSING = {"", "nan", "none", "unknown", "na", "n/a"}

LAYER_SPECS = {
    "ETL": ("etl.stack_sequence", "etl.thickness", "sum"),
    "HTL": ("htl.stack_sequence", "htl.thickness_list", "sum"),
    "BC": (
        "backcontact.stack_sequence",
        "backcontact.thickness_list",
        "sum",
    ),
    "PVK": (
        "perovskite.composition_long_form",
        "perovskite.thickness",
        "max",
    ),
}

THICKNESS_BOUNDS_NM = {
    "ETL": (5.0, 1000.0),
    "HTL": (5.0, 1000.0),
    "BC": (10.0, 1000.0),
    "PVK": (50.0, 2000.0),
}

JV_BOUNDS = {
    "jv.default_Jsc": (0.0, 45.0),
    "jv.default_Voc": (0.0, 1.3),
    "jv.default_FF": (0.3, 0.9),
}


def split_preserving_positions(value: object) -> list[str]:
    """Split a database list without deleting placeholders or categories."""

    if pd.isna(value):
        return []
    return [token.strip() for token in SEPARATOR_RE.split(str(value))]


def parse_numeric_token(token: object) -> float:
    text = str(token).strip()
    if text.lower() in NUMERIC_MISSING:
        return np.nan
    try:
        value = float(text)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def complete_numeric_list(tokens: list[str]) -> bool:
    return bool(tokens) and all(pd.notna(parse_numeric_token(x)) for x in tokens)


def aggregate_thickness(tokens: list[str], method: str) -> float:
    if not complete_numeric_list(tokens):
        return np.nan
    values = [parse_numeric_token(x) for x in tokens]
    if method == "sum":
        return float(sum(values))
    if method == "max":
        return float(max(values))
    raise ValueError(f"Unknown aggregation method: {method}")


def add_layer_fields(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    for prefix, (material_col, thickness_col, aggregation) in LAYER_SPECS.items():
        material_tokens = out[material_col].map(split_preserving_positions)
        thickness_tokens = out[thickness_col].map(split_preserving_positions)

        if prefix != "PVK":
            out[f"{prefix}_layers"] = material_tokens
            out[f"{prefix}_thickness_values"] = thickness_tokens
            out[f"{prefix}_n_layers"] = material_tokens.map(len)
            out[f"{prefix}_n_thickness"] = thickness_tokens.map(len)

        count_match = (
            material_tokens.map(len).gt(0)
            & material_tokens.map(len).eq(thickness_tokens.map(len))
        )
        numeric_complete = thickness_tokens.map(complete_numeric_list)
        valid = count_match & numeric_complete

        clean = thickness_tokens.map(
            lambda tokens: aggregate_thickness(tokens, aggregation)
        ).where(valid)

        if prefix == "PVK":
            out["PVK_thickness_clean"] = clean
            out["PVK_thickness_note"] = np.select(
                [
                    ~count_match,
                    count_match & ~numeric_complete,
                    valid & thickness_tokens.map(len).gt(1),
                ],
                ["count_mismatch", "incomplete_numeric", "multiple_values_max"],
                default="single_value",
            )
            out["PVK_valid"] = valid
        else:
            out[f"{prefix}_thickness_clean"] = clean
            out[f"{prefix}_valid"] = valid

    out["all_layer_match"] = (
        out["ETL_valid"]
        & out["HTL_valid"]
        & out["BC_valid"]
        & out["PVK_valid"]
    )
    return out


def append_stage(
    rows: list[dict[str, object]],
    stage: str,
    mask: pd.Series,
    previous_count: int,
    note: str,
) -> int:
    remaining = int(mask.sum())
    rows.append(
        {
            "stage": stage,
            "remaining_rows": remaining,
            "removed_at_stage": previous_count - remaining,
            "note": note,
        }
    )
    return remaining


def build_strict_cohort(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    original_columns = list(raw.columns)
    audit_rows: list[dict[str, object]] = [
        {
            "stage": "Raw database",
            "remaining_rows": len(raw),
            "removed_at_stage": 0,
            "note": f"{len(original_columns)} original descriptors",
        }
    ]

    # Exact duplicates only.  Do not rank by PCE and do not collapse distinct
    # devices merely because they share a reference or baseline descriptors.
    exact_duplicate = raw.duplicated(subset=original_columns, keep="first")
    working = raw.loc[~exact_duplicate].copy()
    previous = append_stage(
        audit_rows,
        "Exact-row deduplication",
        pd.Series(True, index=working.index),
        len(raw),
        "Drop only rows identical across all 387 raw descriptors; never keep the highest PCE",
    )

    working = add_layer_fields(working)
    layer_mask = working["all_layer_match"]
    previous = append_stage(
        audit_rows,
        "Complete layer-thickness validation",
        layer_mask,
        previous,
        "ETL, HTL, BC and PVK token counts align and every thickness token is numeric",
    )

    mask = layer_mask.copy()
    for prefix, (lower, upper) in THICKNESS_BOUNDS_NM.items():
        mask &= working[f"{prefix}_thickness_clean"].between(lower, upper)
    previous = append_stage(
        audit_rows,
        "Thickness bounds",
        mask,
        previous,
        "ETL/HTL 5-1000 nm; BC 10-1000 nm; PVK 50-2000 nm",
    )

    working["perovskite_bandgap_clean"] = pd.to_numeric(
        working["perovskite.band_gap"], errors="coerce"
    )
    mask &= working["perovskite_bandgap_clean"].between(1.0, 2.5)
    previous = append_stage(
        audit_rows,
        "Bandgap bounds",
        mask,
        previous,
        "Numeric PVK bandgap in 1.0-2.5 eV",
    )

    for column, (lower, upper) in JV_BOUNDS.items():
        numeric = pd.to_numeric(working[column], errors="coerce")
        working[column] = numeric
        mask &= numeric.between(lower, upper)
    previous = append_stage(
        audit_rows,
        "JV-component bounds",
        mask,
        previous,
        "Jsc 0-45 mA/cm2; Voc 0-1.3 V; FF 0.3-0.9",
    )

    working["jv.default_PCE"] = pd.to_numeric(
        working["jv.default_PCE"], errors="coerce"
    )
    mask &= working["jv.default_PCE"].notna()
    previous = append_stage(
        audit_rows,
        "PCE availability",
        mask,
        previous,
        "Require a numeric default PCE; no component-PCE identity filter applied",
    )

    working["baseline_complete"] = mask
    strict = working.loc[mask].copy()

    # Quality audit only: default PCE may be stabilised while Jsc/Voc/FF are
    # scan-derived.  Record inconsistency instead of silently deleting it here.
    calculated_pce = (
        strict["jv.default_Jsc"]
        * strict["jv.default_Voc"]
        * strict["jv.default_FF"]
    )
    absolute_error = (strict["jv.default_PCE"] - calculated_pce).abs()
    relative_error = absolute_error / strict["jv.default_PCE"].replace(0, np.nan)

    direction_columns = [
        "jv.default_Jsc_scan_direction",
        "jv.default_Voc_scan_direction",
        "jv.default_FF_scan_direction",
        "jv.default_PCE_scan_direction",
    ]
    directions = strict[direction_columns].astype("string")
    component_directions = directions[direction_columns[:3]]
    explicit_component_tuple = (
        component_directions.notna().all(axis=1)
        & component_directions.nunique(axis=1).eq(1)
    )
    pce_direction_differs = (
        explicit_component_tuple
        & directions[direction_columns[3]].notna()
        & directions[direction_columns[3]].ne(directions[direction_columns[0]])
    )

    quality = {
        "raw_shape": [int(raw.shape[0]), int(raw.shape[1])],
        "strict_shape": [int(strict.shape[0]), int(strict.shape[1])],
        "exact_raw_duplicates_removed": int(exact_duplicate.sum()),
        "unique_doi_groups": int(strict["ref.DOI_number"].nunique(dropna=True)),
        "component_pce_abs_error_gt_0_5": int((absolute_error > 0.5).sum()),
        "component_pce_abs_error_gt_1_0": int((absolute_error > 1.0).sum()),
        "component_pce_rel_error_gt_5pct": int((relative_error > 0.05).sum()),
        "component_pce_rel_error_gt_10pct": int((relative_error > 0.10).sum()),
        "component_pce_rel_error_gt_20pct": int((relative_error > 0.20).sum()),
        "nonpositive_pce_rows": int((strict["jv.default_PCE"] <= 0).sum()),
        "all_default_scan_directions_same_or_all_missing": int(
            directions.nunique(axis=1, dropna=False).eq(1).sum()
        ),
        "pce_direction_stabilised": int(
            directions[direction_columns[3]]
            .str.lower()
            .eq("stabilised")
            .fillna(False)
            .sum()
        ),
        "pce_direction_differs_from_explicit_component_tuple": int(
            pce_direction_differs.sum()
        ),
        "deduplication_policy": "Exact duplicate raw rows only",
        "component_pce_consistency_policy": "Audit only; not a cohort filter",
    }

    # Preserve the same 387 raw + 24 preprocessing-column schema used by the
    # downstream notebooks, while correcting the semantics of those fields.
    helper_columns = [
        "ETL_layers",
        "ETL_thickness_values",
        "ETL_n_layers",
        "ETL_n_thickness",
        "ETL_valid",
        "HTL_layers",
        "HTL_thickness_values",
        "HTL_n_layers",
        "HTL_n_thickness",
        "HTL_valid",
        "BC_layers",
        "BC_thickness_values",
        "BC_n_layers",
        "BC_n_thickness",
        "BC_valid",
        "ETL_thickness_clean",
        "HTL_thickness_clean",
        "BC_thickness_clean",
        "PVK_thickness_clean",
        "PVK_thickness_note",
        "PVK_valid",
        "all_layer_match",
        "perovskite_bandgap_clean",
        "baseline_complete",
    ]
    strict = strict[original_columns + helper_columns]
    quality["strict_shape"] = [int(strict.shape[0]), int(strict.shape[1])]

    return strict, pd.DataFrame(audit_rows), quality


def main() -> None:
    # All files are read from / written to the folder containing this script
    folder = Path(__file__).resolve().parent

    #input_path = folder / "perovskite_data.csv"
    output_path = folder / "df_base_strict.csv"
    audit_csv = folder / "strict_preprocessing_stage_counts.csv"
    quality_json = folder / "strict_preprocessing_quality.json"

    raw = pd.read_csv("perovskite_data.csv", low_memory=False)
    strict, audit, quality = build_strict_cohort(raw)

    strict.to_csv(output_path, index=False)
    audit.to_csv(audit_csv, index=False)
    quality_json.write_text(
        json.dumps(quality, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(audit.to_string(index=False))
    print(json.dumps(quality, indent=2, ensure_ascii=False))
    print(f"\nStrict dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
