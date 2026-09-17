from pathlib import Path

import numpy as np
import pandas as pd

from opex_config import (
    ANNUAL_SALARY_INFLATION,
    HEADCOUNT_ASSUMPTIONS,
    NON_PAYROLL_ASSUMPTIONS,
    NON_PAYROLL_VOLATILITY,
    OPEX_ACTUAL_END,
    OPEX_ACTUAL_START,
    OPEX_RANDOM_SEED,
    OPEX_SEASONALITY,
    PAYROLL_VOLATILITY,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "actual_opex.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def years_since_start(
    date: pd.Timestamp,
) -> float:
    """
    Return fractional years since January 2024.
    """

    return (
        (date.year - 2024)
        + ((date.month - 1) / 12)
    )


# ============================================================
# PAYROLL GENERATION
# ============================================================

def generate_payroll_rows(
    date: pd.Timestamp,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Generate monthly Payroll expense by cost centre.

    Payroll is driver-based:

        headcount x average monthly employee cost
    """

    elapsed_years = years_since_start(
        date
    )

    rows = []

    for assumption in HEADCOUNT_ASSUMPTIONS:

        # ----------------------------------------------------
        # Headcount
        # ----------------------------------------------------

        expected_headcount = (
            assumption.base_headcount_2024
            * (
                1
                + assumption.annual_headcount_growth
            ) ** elapsed_years
        )

        headcount = max(
            1,
            int(
                round(
                    expected_headcount
                )
            ),
        )

        # ----------------------------------------------------
        # Average employee cost
        # ----------------------------------------------------

        employee_cost = (
            assumption.average_monthly_employee_cost_2024
            * (
                1
                + ANNUAL_SALARY_INFLATION
            ) ** elapsed_years
        )

        payroll_noise = rng.normal(
            loc=1.0,
            scale=PAYROLL_VOLATILITY,
        )

        employee_cost = (
            employee_cost
            * payroll_noise
        )

        employee_cost = round(
            employee_cost,
            2,
        )

        # ----------------------------------------------------
        # Payroll amount
        # ----------------------------------------------------

        amount = (
            headcount
            * employee_cost
        )

        amount = round(
            amount,
            2,
        )

        rows.append(
            {
                "date": date,
                "scenario": "Actual",
                "cost_centre": assumption.cost_centre,
                "account": "Payroll",
                "amount": amount,
                "headcount": headcount,
                "average_employee_cost": employee_cost,
                "seasonality_profile": "payroll",
            }
        )

    return rows


# ============================================================
# NON-PAYROLL GENERATION
# ============================================================

def generate_non_payroll_rows(
    date: pd.Timestamp,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Generate monthly non-payroll operating expenses.

    Expense is based on:

        base cost
        x annual growth
        x account seasonality
        x limited operational volatility
    """

    elapsed_years = years_since_start(
        date
    )

    rows = []

    for assumption in NON_PAYROLL_ASSUMPTIONS:

        annual_growth_factor = (
            1
            + assumption.annual_growth_rate
        ) ** elapsed_years

        seasonality = (
            OPEX_SEASONALITY[
                assumption.seasonality_profile
            ][
                date.month
            ]
        )

        volatility = rng.normal(
            loc=1.0,
            scale=NON_PAYROLL_VOLATILITY,
        )

        amount = (
            assumption.monthly_base_cost_2024
            * annual_growth_factor
            * seasonality
            * volatility
        )

        amount = max(
            0.0,
            amount,
        )

        amount = round(
            amount,
            2,
        )

        rows.append(
            {
                "date": date,
                "scenario": "Actual",
                "cost_centre": assumption.cost_centre,
                "account": assumption.account,
                "amount": amount,
                "headcount": 0,
                "average_employee_cost": 0.0,
                "seasonality_profile":
                    assumption.seasonality_profile,
            }
        )

    return rows


# ============================================================
# OPEX GENERATOR
# ============================================================

def generate_actual_opex() -> pd.DataFrame:
    """
    Generate Actual OPEX from January 2024 through August 2026.
    """

    rng = np.random.default_rng(
        OPEX_RANDOM_SEED
    )

    dates = pd.date_range(
        start=OPEX_ACTUAL_START,
        end=OPEX_ACTUAL_END,
        freq="MS",
    )

    rows = []

    for date in dates:

        rows.extend(
            generate_payroll_rows(
                date,
                rng,
            )
        )

        rows.extend(
            generate_non_payroll_rows(
                date,
                rng,
            )
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_opex_data(
    df: pd.DataFrame,
) -> None:
    """
    Validate OPEX data before saving.
    """

    required_columns = {
        "date",
        "scenario",
        "cost_centre",
        "account",
        "amount",
        "headcount",
        "average_employee_cost",
        "seasonality_profile",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing OPEX columns: "
            f"{sorted(missing_columns)}"
        )

    if df.isna().any().any():

        raise ValueError(
            "OPEX dataset contains missing values."
        )

    if (
        df["amount"] < 0
    ).any():

        raise ValueError(
            "Negative OPEX amount detected."
        )

    if (
        df["headcount"] < 0
    ).any():

        raise ValueError(
            "Negative headcount detected."
        )

    if set(
        df["scenario"].unique()
    ) != {"Actual"}:

        raise ValueError(
            "Unexpected OPEX scenario detected."
        )

    # --------------------------------------------------------
    # Payroll reconciliation
    # --------------------------------------------------------

    payroll = (
        df[
            df["account"] == "Payroll"
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
            "Payroll reconciliation failed."
        )

    # --------------------------------------------------------
    # Non-payroll driver fields
    # --------------------------------------------------------

    non_payroll = (
        df[
            df["account"] != "Payroll"
        ]
    )

    if not (
        non_payroll["headcount"] == 0
    ).all():

        raise ValueError(
            "Non-payroll rows contain headcount."
        )

    if not (
        non_payroll[
            "average_employee_cost"
        ] == 0
    ).all():

        raise ValueError(
            "Non-payroll rows contain "
            "employee-cost drivers."
        )

    # --------------------------------------------------------
    # Duplicate grain
    # --------------------------------------------------------

    grain = [
        "date",
        "cost_centre",
        "account",
    ]

    if df.duplicated(
        grain
    ).any():

        duplicates = (
            df[
                df.duplicated(
                    grain,
                    keep=False,
                )
            ][grain]
        )

        raise ValueError(
            "Duplicate OPEX grain detected:\n"
            f"{duplicates}"
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print annual OPEX and headcount sanity checks.
    """

    annual_opex = (
        df.assign(
            year=df["date"].dt.year
        )
        .groupby(
            "year",
            as_index=False,
        )
        .agg(
            total_opex=(
                "amount",
                "sum",
            ),
        )
    )

    payroll = (
        df[
            df["account"] == "Payroll"
        ]
        .copy()
    )

    payroll["year"] = (
        payroll["date"].dt.year
    )

    annual_payroll = (
        payroll.groupby(
            "year",
            as_index=False,
        )
        .agg(
            payroll=(
                "amount",
                "sum",
            ),
        )
    )

    annual = annual_opex.merge(
        annual_payroll,
        on="year",
        how="left",
    )

    print(
        "\nNorthstar Systems AS"
    )

    print(
        "Actual OPEX Dataset"
    )

    print(
        "=" * 70
    )

    print(
        f"Rows generated: "
        f"{len(df):,}"
    )

    print(
        "Period: "
        f"{df['date'].min().date()} "
        "to "
        f"{df['date'].max().date()}"
    )

    print(
        "\nAnnual OPEX summary:"
    )

    for row in annual.itertuples():

        non_payroll = (
            row.total_opex
            - row.payroll
        )

        print(
            f"{row.year}: "
            f"Total OPEX NOK "
            f"{row.total_opex / 1_000_000:,.1f}m | "
            f"Payroll NOK "
            f"{row.payroll / 1_000_000:,.1f}m | "
            f"Non-payroll NOK "
            f"{non_payroll / 1_000_000:,.1f}m"
        )

    # --------------------------------------------------------
    # Latest headcount
    # --------------------------------------------------------

    latest_date = (
        payroll["date"].max()
    )

    latest_headcount = (
        payroll[
            payroll["date"]
            == latest_date
        ][
            "headcount"
        ]
        .sum()
    )

    print(
        f"\nHeadcount at "
        f"{latest_date.date()}: "
        f"{latest_headcount:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    df = generate_actual_opex()

    validate_opex_data(
        df
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_summary(
        df
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nValidation: PASSED"
    )


if __name__ == "__main__":
    main()