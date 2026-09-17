from pathlib import Path

import numpy as np
import pandas as pd

from opex_event_config import (
    OPEX_EVENTS,
    OPEX_MANAGEMENT_NOTES,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_OPEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "actual_opex.csv"
)

FINAL_OPEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "actual_opex.csv"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "ground_truth_opex_events.csv"
)

MANAGEMENT_NOTES_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "opex_management_notes_2026-08.md"
)


# ============================================================
# LOAD BASELINE
# ============================================================

def load_baseline() -> pd.DataFrame:
    """
    Load validated baseline Actual OPEX.
    """

    if not BASELINE_OPEX_PATH.exists():

        raise FileNotFoundError(
            f"Baseline OPEX dataset not found: "
            f"{BASELINE_OPEX_PATH}"
        )

    return pd.read_csv(
        BASELINE_OPEX_PATH,
        parse_dates=["date"],
    )


# ============================================================
# EVENT MASK
# ============================================================

def build_event_mask(
    df: pd.DataFrame,
    event,
) -> pd.Series:
    """
    Identify the exact OPEX row affected by an event.
    """

    return (
        (df["date"] == pd.Timestamp(event.date))
        & (
            df["cost_centre"]
            == event.cost_centre
        )
        & (
            df["account"]
            == event.account
        )
    )


# ============================================================
# APPLY ONE EVENT
# ============================================================

def apply_event(
    df: pd.DataFrame,
    event,
) -> dict:
    """
    Apply a controlled OPEX event and return
    realized impact for hidden ground truth.
    """

    mask = build_event_mask(
        df,
        event,
    )

    rows_affected = int(
        mask.sum()
    )

    if rows_affected != 1:

        raise ValueError(
            f"{event.event_id}: expected exactly "
            f"one matching OPEX row, "
            f"found {rows_affected}."
        )

    before_amount = float(
        df.loc[
            mask,
            "amount",
        ].iloc[0]
    )

    # --------------------------------------------------------
    # Apply event
    # --------------------------------------------------------

    after_amount = (
        before_amount
        + event.amount_change
    )

    if after_amount < 0:

        raise ValueError(
            f"{event.event_id}: event would create "
            "negative OPEX."
        )

    after_amount = round(
        after_amount,
        2,
    )

    df.loc[
        mask,
        "amount",
    ] = after_amount

    # --------------------------------------------------------
    # Ground truth record
    # --------------------------------------------------------

    return {
        "event_id": event.event_id,
        "date": event.date,
        "cost_centre": event.cost_centre,
        "account": event.account,
        "description": event.description,
        "rows_affected": rows_affected,
        "baseline_amount": round(
            before_amount,
            2,
        ),
        "amount_change": round(
            event.amount_change,
            2,
        ),
        "final_amount": after_amount,
        "ebitda_impact": round(
            -event.amount_change,
            2,
        ),
        "evidence_available":
            event.evidence_available,
        "evidence_id":
            event.evidence_id,
    }


# ============================================================
# FINAL DATA VALIDATION
# ============================================================

def validate_final_opex(
    baseline: pd.DataFrame,
    final: pd.DataFrame,
) -> None:
    """
    Ensure event injection preserves OPEX data integrity.
    """

    if len(baseline) != len(final):

        raise ValueError(
            "OPEX row count changed during "
            "event injection."
        )

    if final.isna().any().any():

        raise ValueError(
            "Missing values detected in "
            "final OPEX dataset."
        )

    if (
        final["amount"] < 0
    ).any():

        raise ValueError(
            "Negative OPEX amount detected."
        )

    if (
        final["headcount"] < 0
    ).any():

        raise ValueError(
            "Negative headcount detected."
        )

    # --------------------------------------------------------
    # Payroll must remain driver-reconciled.
    #
    # Our controlled OPEX events currently affect only
    # non-payroll lines, so payroll should still reconcile
    # exactly.
    # --------------------------------------------------------

    payroll = (
        final[
            final["account"] == "Payroll"
        ]
        .copy()
    )

    expected_payroll = (
        payroll["headcount"]
        * payroll["average_employee_cost"]
    ).round(2)

    if not np.allclose(
        payroll["amount"],
        expected_payroll,
        atol=0.01,
    ):

        raise ValueError(
            "Payroll reconciliation failed "
            "after OPEX event injection."
        )

    # --------------------------------------------------------
    # Grain uniqueness
    # --------------------------------------------------------

    grain = [
        "date",
        "cost_centre",
        "account",
    ]

    if final.duplicated(
        grain
    ).any():

        raise ValueError(
            "Duplicate OPEX grain detected."
        )


# ============================================================
# MANAGEMENT EVIDENCE
# ============================================================

def write_management_notes() -> None:
    """
    Write the OPEX evidence document that will later
    be available to the retrieval / AI layer.

    Undocumented events are deliberately excluded.
    """

    MANAGEMENT_NOTES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "# Northstar Systems AS",
        "",
        "## August 2026 OPEX Management Notes",
        "",
        (
            "These notes contain selected operational "
            "explanations provided by management."
        ),
        "",
    ]

    for evidence_id, note in (
        OPEX_MANAGEMENT_NOTES.items()
    ):

        lines.extend(
            [
                (
                    f"### {evidence_id} — "
                    f"{note['title']}"
                ),
                "",
                note["text"],
                "",
            ]
        )

    MANAGEMENT_NOTES_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# EVENT SUMMARY
# ============================================================

def print_event_summary(
    ground_truth: pd.DataFrame,
) -> None:
    """
    Print realized OPEX and EBITDA effect.
    """

    print(
        "\nControlled OPEX Events"
    )

    print(
        "=" * 95
    )

    for row in ground_truth.itertuples():

        evidence = (
            row.evidence_id
            if pd.notna(row.evidence_id)
            else "NONE"
        )

        print(
            f"{row.event_id} | "
            f"{row.cost_centre:<15} | "
            f"{row.account:<12} | "
            f"OPEX impact NOK "
            f"{row.amount_change / 1_000_000:>5.2f}m | "
            f"EBITDA impact NOK "
            f"{row.ebitda_impact / 1_000_000:>6.2f}m | "
            f"Evidence: {evidence}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    baseline = load_baseline()

    final = baseline.copy()

    ground_truth_records = []

    for event in OPEX_EVENTS:

        record = apply_event(
            final,
            event,
        )

        ground_truth_records.append(
            record
        )

    ground_truth = pd.DataFrame(
        ground_truth_records
    )

    validate_final_opex(
        baseline,
        final,
    )

    # --------------------------------------------------------
    # Save final OPEX
    # --------------------------------------------------------

    FINAL_OPEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        FINAL_OPEX_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save hidden ground truth
    # --------------------------------------------------------

    GROUND_TRUTH_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ground_truth.to_csv(
        GROUND_TRUTH_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save management evidence
    # --------------------------------------------------------

    write_management_notes()

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print_event_summary(
        ground_truth
    )

    print(
        "\nFinal Actual OPEX dataset:"
    )

    print(
        FINAL_OPEX_PATH
    )

    print(
        "\nHidden OPEX ground truth:"
    )

    print(
        GROUND_TRUTH_PATH
    )

    print(
        "\nOPEX management evidence:"
    )

    print(
        MANAGEMENT_NOTES_PATH
    )

    print(
        "\nValidation: PASSED"
    )


if __name__ == "__main__":
    main()