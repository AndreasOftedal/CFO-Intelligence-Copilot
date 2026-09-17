from pathlib import Path

import numpy as np
import pandas as pd

from event_config import (
    BUSINESS_EVENTS,
    MANAGEMENT_NOTES,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "actual_sales.csv"
)

FINAL_ACTUAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "actual_sales.csv"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
    / "ground_truth_events.csv"
)

MANAGEMENT_NOTES_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "monthly_management_notes_2026-08.md"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_baseline() -> pd.DataFrame:
    """
    Load the validated baseline Actual sales dataset.
    """

    if not BASELINE_PATH.exists():

        raise FileNotFoundError(
            f"Baseline dataset not found: "
            f"{BASELINE_PATH}"
        )

    return pd.read_csv(
        BASELINE_PATH,
        parse_dates=["date"],
    )


# ============================================================
# EVENT FILTERING
# ============================================================

def build_event_mask(
    df: pd.DataFrame,
    event,
) -> pd.Series:
    """
    Build the row filter associated with a business event.
    """

    mask = (
        (df["date"] == pd.Timestamp(event.date))
        & (df["country"] == event.country)
    )

    if event.product is not None:

        mask = (
            mask
            & (df["product"] == event.product)
        )

    if event.segment is not None:

        mask = (
            mask
            & (df["segment"] == event.segment)
        )

    return mask


# ============================================================
# FINANCIAL RECALCULATION
# ============================================================

def recalculate_financials(
    df: pd.DataFrame,
    mask: pd.Series,
) -> None:
    """
    Recalculate all dependent financial values after
    commercial drivers have changed.
    """

    df.loc[
        mask,
        "gross_sales",
    ] = (
        df.loc[
            mask,
            "units",
        ]
        * df.loc[
            mask,
            "list_price",
        ]
    ).round(2)

    df.loc[
        mask,
        "discount_value",
    ] = (
        df.loc[
            mask,
            "gross_sales",
        ]
        * df.loc[
            mask,
            "discount_rate",
        ]
    ).round(2)

    df.loc[
        mask,
        "net_revenue",
    ] = (
        df.loc[
            mask,
            "gross_sales",
        ]
        - df.loc[
            mask,
            "discount_value",
        ]
    ).round(2)

    df.loc[
        mask,
        "cogs",
    ] = (
        df.loc[
            mask,
            "units",
        ]
        * df.loc[
            mask,
            "unit_cost",
        ]
    ).round(2)

    df.loc[
        mask,
        "gross_profit",
    ] = (
        df.loc[
            mask,
            "net_revenue",
        ]
        - df.loc[
            mask,
            "cogs",
        ]
    ).round(2)

    nonzero_revenue = (
        mask
        & (df["net_revenue"] != 0)
    )

    df.loc[
        nonzero_revenue,
        "gross_margin_pct",
    ] = (
        df.loc[
            nonzero_revenue,
            "gross_profit",
        ]
        / df.loc[
            nonzero_revenue,
            "net_revenue",
        ]
    ).round(6)

    zero_revenue = (
        mask
        & (df["net_revenue"] == 0)
    )

    df.loc[
        zero_revenue,
        "gross_margin_pct",
    ] = 0.0


# ============================================================
# EVENT APPLICATION
# ============================================================

def apply_event(
    df: pd.DataFrame,
    event,
) -> dict:
    """
    Apply one controlled business event and return its
    realized financial impact for the hidden ground truth.
    """

    mask = build_event_mask(
        df,
        event,
    )

    rows_affected = int(
        mask.sum()
    )

    if rows_affected == 0:

        raise ValueError(
            f"{event.event_id}: event matched zero rows."
        )

    # --------------------------------------------------------
    # Capture financial position BEFORE event
    # --------------------------------------------------------

    before = (
        df.loc[
            mask,
            [
                "units",
                "net_revenue",
                "cogs",
                "gross_profit",
            ],
        ]
        .sum()
    )

    # --------------------------------------------------------
    # Apply driver change
    # --------------------------------------------------------

    if event.driver == "unit_cost":

        df.loc[
            mask,
            "unit_cost",
        ] = (
            df.loc[
                mask,
                "unit_cost",
            ]
            * event.magnitude
        ).round(2)

    elif event.driver == "discount_rate":

        new_discount = (
            df.loc[
                mask,
                "discount_rate",
            ]
            + event.magnitude
        )

        new_discount = np.clip(
            new_discount,
            0.0,
            0.25,
        )

        df.loc[
            mask,
            "discount_rate",
        ] = np.round(
            new_discount,
            6,
        )

    elif event.driver == "units":

        new_units = (
            df.loc[
                mask,
                "units",
            ]
            * event.magnitude
        )

        df.loc[
            mask,
            "units",
        ] = (
            new_units
            .round()
            .astype(int)
        )

    else:

        raise ValueError(
            f"{event.event_id}: unsupported driver "
            f"{event.driver}."
        )

    # --------------------------------------------------------
    # Recalculate dependent financial values
    # --------------------------------------------------------

    recalculate_financials(
        df,
        mask,
    )

    # --------------------------------------------------------
    # Capture financial position AFTER event
    # --------------------------------------------------------

    after = (
        df.loc[
            mask,
            [
                "units",
                "net_revenue",
                "cogs",
                "gross_profit",
            ],
        ]
        .sum()
    )

    # --------------------------------------------------------
    # Store realized impact
    # --------------------------------------------------------

    return {
        "event_id": event.event_id,
        "date": event.date,
        "country": event.country,
        "product": event.product,
        "segment": event.segment,
        "driver": event.driver,
        "operation": event.operation,
        "magnitude": event.magnitude,
        "description": event.description,
        "rows_affected": rows_affected,
        "units_change": int(
            after["units"]
            - before["units"]
        ),
        "revenue_impact": round(
            after["net_revenue"]
            - before["net_revenue"],
            2,
        ),
        "cogs_impact": round(
            after["cogs"]
            - before["cogs"],
            2,
        ),
        "gross_profit_impact": round(
            after["gross_profit"]
            - before["gross_profit"],
            2,
        ),
        "evidence_available": (
            event.evidence_available
        ),
        "evidence_id": (
            event.evidence_id
        ),
    }


# ============================================================
# FINAL DATA VALIDATION
# ============================================================

def validate_final_data(
    baseline: pd.DataFrame,
    final: pd.DataFrame,
) -> None:
    """
    Ensure event injection did not break financial integrity.
    """

    if len(baseline) != len(final):

        raise ValueError(
            "Row count changed during event injection."
        )

    if final.isna().any().any():

        raise ValueError(
            "Missing values detected in final Actual data."
        )

    if (
        final["units"] < 0
    ).any():

        raise ValueError(
            "Negative unit volume detected."
        )

    if not (
        final["discount_rate"]
        .between(
            0,
            0.25,
        )
        .all()
    ):

        raise ValueError(
            "Invalid discount rate detected."
        )

    calculated_gross_sales = (
        final["units"]
        * final["list_price"]
    ).round(2)

    if not np.allclose(
        final["gross_sales"],
        calculated_gross_sales,
        atol=0.01,
    ):

        raise ValueError(
            "Gross sales reconciliation failed."
        )

    calculated_discount = (
        final["gross_sales"]
        * final["discount_rate"]
    ).round(2)

    if not np.allclose(
        final["discount_value"],
        calculated_discount,
        atol=0.01,
    ):

        raise ValueError(
            "Discount reconciliation failed."
        )

    calculated_revenue = (
        final["gross_sales"]
        - final["discount_value"]
    ).round(2)

    if not np.allclose(
        final["net_revenue"],
        calculated_revenue,
        atol=0.01,
    ):

        raise ValueError(
            "Revenue reconciliation failed."
        )

    calculated_cogs = (
        final["units"]
        * final["unit_cost"]
    ).round(2)

    if not np.allclose(
        final["cogs"],
        calculated_cogs,
        atol=0.01,
    ):

        raise ValueError(
            "COGS reconciliation failed."
        )

    calculated_gp = (
        final["net_revenue"]
        - final["cogs"]
    ).round(2)

    if not np.allclose(
        final["gross_profit"],
        calculated_gp,
        atol=0.01,
    ):

        raise ValueError(
            "Gross profit reconciliation failed."
        )


# ============================================================
# MANAGEMENT NOTE DOCUMENT
# ============================================================

def write_management_notes() -> None:
    """
    Create the management document that will later be
    available to the retrieval / AI layer.
    """

    MANAGEMENT_NOTES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "# Northstar Systems AS",
        "",
        "## August 2026 Management Notes",
        "",
        (
            "These notes contain selected operational "
            "explanations provided by management."
        ),
        "",
    ]

    for evidence_id, note in MANAGEMENT_NOTES.items():

        lines.extend(
            [
                f"### {evidence_id} — {note['title']}",
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
# SUMMARY OUTPUT
# ============================================================

def print_event_summary(
    ground_truth: pd.DataFrame,
) -> None:
    """
    Print the realized financial impact of each event.
    """

    print(
        "\nControlled Business Events"
    )

    print(
        "=" * 90
    )

    for row in ground_truth.itertuples():

        evidence = (
            row.evidence_id
            if pd.notna(row.evidence_id)
            else "NONE"
        )

        print(
            f"{row.event_id} | "
            f"{row.driver:<13} | "
            f"Revenue impact "
            f"NOK {row.revenue_impact / 1_000_000:>7.2f}m | "
            f"GP impact "
            f"NOK {row.gross_profit_impact / 1_000_000:>7.2f}m | "
            f"Evidence: {evidence}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    baseline = load_baseline()

    final = baseline.copy()

    ground_truth_records = []

    for event in BUSINESS_EVENTS:

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

    validate_final_data(
        baseline,
        final,
    )

    FINAL_ACTUAL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GROUND_TRUTH_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        FINAL_ACTUAL_PATH,
        index=False,
    )

    ground_truth.to_csv(
        GROUND_TRUTH_PATH,
        index=False,
    )

    write_management_notes()

    print_event_summary(
        ground_truth
    )

    print(
        "\nFinal Actual dataset:"
    )

    print(
        FINAL_ACTUAL_PATH
    )

    print(
        "\nHidden ground truth:"
    )

    print(
        GROUND_TRUTH_PATH
    )

    print(
        "\nManagement evidence:"
    )

    print(
        MANAGEMENT_NOTES_PATH
    )

    print(
        "\nValidation: PASSED"
    )


if __name__ == "__main__":
    main()