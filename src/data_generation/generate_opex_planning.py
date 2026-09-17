from pathlib import Path

import numpy as np
import pandas as pd

from opex_planning_config import (
    ACTUAL_THROUGH,
    BUDGET_EMPLOYEE_COST_INCREASE,
    BUDGET_HEADCOUNT_GROWTH,
    BUDGET_NON_PAYROLL_GROWTH,
    FORECAST_EMPLOYEE_COST_REVISION,
    FORECAST_FUTURE_START,
    FORECAST_HEADCOUNT_REVISION,
    FORECAST_NON_PAYROLL_REVISION,
    FORECAST_SPECIFIC_OPEX_ADJUSTMENTS,
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

FINAL_ACTUAL_OPEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "actual_opex.csv"
)

BUDGET_OPEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "budget_opex.csv"
)

FORECAST_OPEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "latest_forecast_opex.csv"
)

BUDGET_SALES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "budget_sales.csv"
)

FORECAST_SALES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "latest_forecast_sales.csv"
)


# ============================================================
# DATA MODEL
# ============================================================

GRAIN_COLUMNS = [
    "date",
    "cost_centre",
    "account",
]

COMPARE_COLUMNS = [
    "date",
    "cost_centre",
    "account",
    "amount",
    "headcount",
    "average_employee_cost",
    "seasonality_profile",
]


# ============================================================
# LOAD SOURCE DATA
# ============================================================

def load_source_data():
    """
    Load baseline OPEX, final Actual OPEX and
    previously generated sales planning datasets.
    """

    required_paths = [
        BASELINE_OPEX_PATH,
        FINAL_ACTUAL_OPEX_PATH,
        BUDGET_SALES_PATH,
        FORECAST_SALES_PATH,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required dataset not found: {path}"
            )

    baseline_opex = pd.read_csv(
        BASELINE_OPEX_PATH,
        parse_dates=["date"],
    )

    final_actual_opex = pd.read_csv(
        FINAL_ACTUAL_OPEX_PATH,
        parse_dates=["date"],
    )

    budget_sales = pd.read_csv(
        BUDGET_SALES_PATH,
        parse_dates=["date"],
    )

    forecast_sales = pd.read_csv(
        FORECAST_SALES_PATH,
        parse_dates=["date"],
    )

    return (
        baseline_opex,
        final_actual_opex,
        budget_sales,
        forecast_sales,
    )


# ============================================================
# BUDGET GENERATION
# ============================================================

def generate_budget_opex(
    baseline_opex: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate full-year 2026 OPEX Budget.

    Budget is based on corresponding 2025 Actual
    monthly cost structure plus explicit planning
    assumptions.
    """

    actual_2025 = (
        baseline_opex[
            baseline_opex["date"].dt.year == 2025
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(actual_2025) == 0:

        raise ValueError(
            "No 2025 OPEX data available "
            "for Budget generation."
        )

    budget = actual_2025.copy()

    # --------------------------------------------------------
    # Shift prior-year months into 2026
    # --------------------------------------------------------

    budget["date"] = (
        budget["date"]
        + pd.DateOffset(years=1)
    )

    budget["scenario"] = "Budget"

    # --------------------------------------------------------
    # Payroll Budget
    # --------------------------------------------------------

    payroll_mask = (
        budget["account"] == "Payroll"
    )

    payroll_growth = (
        budget.loc[
            payroll_mask,
            "cost_centre",
        ]
        .map(BUDGET_HEADCOUNT_GROWTH)
    )

    if payroll_growth.isna().any():

        raise ValueError(
            "Missing Budget headcount growth "
            "for one or more cost centres."
        )

    budget.loc[
        payroll_mask,
        "headcount",
    ] = (
        budget.loc[
            payroll_mask,
            "headcount",
        ]
        * (1 + payroll_growth)
    ).round().astype(int)

    budget.loc[
        payroll_mask,
        "average_employee_cost",
    ] = (
        budget.loc[
            payroll_mask,
            "average_employee_cost",
        ]
        * (1 + BUDGET_EMPLOYEE_COST_INCREASE)
    ).round(2)

    budget.loc[
        payroll_mask,
        "amount",
    ] = (
        budget.loc[
            payroll_mask,
            "headcount",
        ]
        * budget.loc[
            payroll_mask,
            "average_employee_cost",
        ]
    ).round(2)

    # --------------------------------------------------------
    # Non-payroll Budget
    # --------------------------------------------------------

    non_payroll_mask = (
        budget["account"] != "Payroll"
    )

    non_payroll_growth = (
        budget.loc[
            non_payroll_mask,
            "account",
        ]
        .map(BUDGET_NON_PAYROLL_GROWTH)
    )

    if non_payroll_growth.isna().any():

        raise ValueError(
            "Missing Budget non-payroll growth "
            "for one or more accounts."
        )

    budget.loc[
        non_payroll_mask,
        "amount",
    ] = (
        budget.loc[
            non_payroll_mask,
            "amount",
        ]
        * (1 + non_payroll_growth)
    ).round(2)

    return budget


# ============================================================
# LATEST FORECAST GENERATION
# ============================================================

def generate_latest_forecast_opex(
    final_actual_opex: pd.DataFrame,
    budget_opex: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate Latest Forecast OPEX.

    January-August:
        exact final Actual OPEX.

    September-December:
        Budget updated using latest management
        assumptions.
    """

    actual_cutoff = pd.Timestamp(
        ACTUAL_THROUGH
    )

    future_start = pd.Timestamp(
        FORECAST_FUTURE_START
    )

    # --------------------------------------------------------
    # Jan-Aug = Actual
    # --------------------------------------------------------

    forecast_actual = (
        final_actual_opex[
            (
                final_actual_opex["date"].dt.year
                == 2026
            )
            & (
                final_actual_opex["date"]
                <= actual_cutoff
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    forecast_actual["scenario"] = (
        "Latest Forecast"
    )

    # --------------------------------------------------------
    # Sep-Dec = revised Budget
    # --------------------------------------------------------

    forecast_future = (
        budget_opex[
            budget_opex["date"]
            >= future_start
        ]
        .copy()
        .reset_index(drop=True)
    )

    forecast_future["scenario"] = (
        "Latest Forecast"
    )

    # --------------------------------------------------------
    # Payroll revisions
    # --------------------------------------------------------

    payroll_mask = (
        forecast_future["account"]
        == "Payroll"
    )

    headcount_revision = (
        forecast_future.loc[
            payroll_mask,
            "cost_centre",
        ]
        .map(FORECAST_HEADCOUNT_REVISION)
    )

    if headcount_revision.isna().any():

        raise ValueError(
            "Missing Forecast headcount revision."
        )

    forecast_future.loc[
        payroll_mask,
        "headcount",
    ] = (
        forecast_future.loc[
            payroll_mask,
            "headcount",
        ]
        * (1 + headcount_revision)
    ).round().astype(int)

    forecast_future.loc[
        payroll_mask,
        "average_employee_cost",
    ] = (
        forecast_future.loc[
            payroll_mask,
            "average_employee_cost",
        ]
        * (
            1
            + FORECAST_EMPLOYEE_COST_REVISION
        )
    ).round(2)

    forecast_future.loc[
        payroll_mask,
        "amount",
    ] = (
        forecast_future.loc[
            payroll_mask,
            "headcount",
        ]
        * forecast_future.loc[
            payroll_mask,
            "average_employee_cost",
        ]
    ).round(2)

    # --------------------------------------------------------
    # Non-payroll revisions
    # --------------------------------------------------------

    non_payroll_mask = (
        forecast_future["account"]
        != "Payroll"
    )

    expense_revision = (
        forecast_future.loc[
            non_payroll_mask,
            "account",
        ]
        .map(FORECAST_NON_PAYROLL_REVISION)
    )

    if expense_revision.isna().any():

        raise ValueError(
            "Missing Forecast non-payroll revision."
        )

    forecast_future.loc[
        non_payroll_mask,
        "amount",
    ] = (
        forecast_future.loc[
            non_payroll_mask,
            "amount",
        ]
        * (1 + expense_revision)
    ).round(2)

    # --------------------------------------------------------
    # Specific documented forecast adjustments
    # --------------------------------------------------------

    for adjustment in (
        FORECAST_SPECIFIC_OPEX_ADJUSTMENTS
    ):

        adjustment_date = pd.Timestamp(
            adjustment["date"]
        )

        mask = (
            (
                forecast_future["date"]
                == adjustment_date
            )
            & (
                forecast_future["cost_centre"]
                == adjustment["cost_centre"]
            )
            & (
                forecast_future["account"]
                == adjustment["account"]
            )
        )

        rows_affected = int(
            mask.sum()
        )

        if rows_affected != 1:

            raise ValueError(
                f"{adjustment['adjustment_id']}: "
                f"expected exactly one matching row, "
                f"found {rows_affected}."
            )

        forecast_future.loc[
            mask,
            "amount",
        ] = (
            forecast_future.loc[
                mask,
                "amount",
            ]
            + adjustment["amount_change"]
        ).round(2)

    # --------------------------------------------------------
    # Combine Actual + future
    # --------------------------------------------------------

    latest_forecast = pd.concat(
        [
            forecast_actual,
            forecast_future,
        ],
        ignore_index=True,
    )

    latest_forecast = (
        latest_forecast
        .sort_values(
            GRAIN_COLUMNS
        )
        .reset_index(drop=True)
    )

    return latest_forecast


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_opex_integrity(
    df: pd.DataFrame,
    scenario_name: str,
) -> None:
    """
    Validate fundamental OPEX identities.
    """

    if df.isna().any().any():

        raise ValueError(
            f"{scenario_name}: missing values."
        )

    if (
        df["amount"] < 0
    ).any():

        raise ValueError(
            f"{scenario_name}: negative OPEX."
        )

    if (
        df["headcount"] < 0
    ).any():

        raise ValueError(
            f"{scenario_name}: negative headcount."
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
            f"{scenario_name}: "
            "Payroll reconciliation failed."
        )

    # --------------------------------------------------------
    # Duplicate grain
    # --------------------------------------------------------

    if df.duplicated(
        GRAIN_COLUMNS
    ).any():

        raise ValueError(
            f"{scenario_name}: duplicate grain rows."
        )


def validate_budget_opex(
    budget: pd.DataFrame,
    source_2025: pd.DataFrame,
) -> None:
    """
    Validate full-year Budget.
    """

    if len(budget) != len(source_2025):

        raise ValueError(
            "Budget OPEX row count does not match "
            "2025 source structure."
        )

    if set(
        budget["scenario"].unique()
    ) != {"Budget"}:

        raise ValueError(
            "Budget OPEX contains invalid scenario."
        )

    if set(
        budget["date"].dt.year.unique()
    ) != {2026}:

        raise ValueError(
            "Budget OPEX must contain only 2026."
        )

    if set(
        budget["date"].dt.month.unique()
    ) != set(range(1, 13)):

        raise ValueError(
            "Budget OPEX does not contain all months."
        )

    validate_opex_integrity(
        budget,
        "Budget OPEX",
    )


def validate_forecast_opex(
    latest_forecast: pd.DataFrame,
    final_actual: pd.DataFrame,
    budget: pd.DataFrame,
) -> None:
    """
    Validate Latest Forecast.

    Most important control:

    January-August Forecast must equal
    January-August final Actual exactly,
    excluding scenario label.
    """

    if len(latest_forecast) != len(budget):

        raise ValueError(
            "Latest Forecast OPEX row count "
            "does not match Budget."
        )

    if set(
        latest_forecast["scenario"].unique()
    ) != {"Latest Forecast"}:

        raise ValueError(
            "Latest Forecast OPEX contains "
            "invalid scenario values."
        )

    if set(
        latest_forecast["date"].dt.month.unique()
    ) != set(range(1, 13)):

        raise ValueError(
            "Latest Forecast OPEX does not "
            "contain all months."
        )

    validate_opex_integrity(
        latest_forecast,
        "Latest Forecast OPEX",
    )

    # --------------------------------------------------------
    # Jan-Aug Actual reconciliation
    # --------------------------------------------------------

    cutoff = pd.Timestamp(
        ACTUAL_THROUGH
    )

    actual_ytd = (
        final_actual[
            (
                final_actual["date"].dt.year
                == 2026
            )
            & (
                final_actual["date"]
                <= cutoff
            )
        ]
        .copy()
        .sort_values(
            GRAIN_COLUMNS
        )
        .reset_index(drop=True)
    )

    forecast_ytd = (
        latest_forecast[
            latest_forecast["date"]
            <= cutoff
        ]
        .copy()
        .sort_values(
            GRAIN_COLUMNS
        )
        .reset_index(drop=True)
    )

    if len(actual_ytd) != len(forecast_ytd):

        raise ValueError(
            "Forecast OPEX YTD row count "
            "does not match Actual."
        )

    actual_compare = (
        actual_ytd[
            COMPARE_COLUMNS
        ]
        .reset_index(drop=True)
    )

    forecast_compare = (
        forecast_ytd[
            COMPARE_COLUMNS
        ]
        .reset_index(drop=True)
    )

    if not actual_compare.equals(
        forecast_compare
    ):

        raise ValueError(
            "Latest Forecast OPEX Jan-Aug "
            "does not exactly equal Actual OPEX."
        )


# ============================================================
# FULL P&L SUMMARY
# ============================================================

def calculate_pnl(
    sales: pd.DataFrame,
    opex: pd.DataFrame,
) -> dict:
    """
    Calculate high-level operating P&L.
    """

    revenue = float(
        sales["net_revenue"].sum()
    )

    gross_profit = float(
        sales["gross_profit"].sum()
    )

    total_opex = float(
        opex["amount"].sum()
    )

    ebitda = (
        gross_profit
        - total_opex
    )

    gross_margin = (
        gross_profit / revenue
        if revenue != 0
        else 0.0
    )

    ebitda_margin = (
        ebitda / revenue
        if revenue != 0
        else 0.0
    )

    return {
        "revenue": revenue,
        "gross_profit": gross_profit,
        "gross_margin": gross_margin,
        "opex": total_opex,
        "ebitda": ebitda,
        "ebitda_margin": ebitda_margin,
    }


def print_pnl_line(
    name: str,
    pnl: dict,
) -> None:

    print(
        f"{name:<17} | "
        f"Revenue NOK "
        f"{pnl['revenue'] / 1_000_000:>8.1f}m | "
        f"GP NOK "
        f"{pnl['gross_profit'] / 1_000_000:>7.1f}m | "
        f"OPEX NOK "
        f"{pnl['opex'] / 1_000_000:>7.1f}m | "
        f"EBITDA NOK "
        f"{pnl['ebitda'] / 1_000_000:>7.1f}m | "
        f"EBITDA Margin "
        f"{pnl['ebitda_margin']:>6.1%}"
    )


def print_summary(
    budget_sales: pd.DataFrame,
    forecast_sales: pd.DataFrame,
    budget_opex: pd.DataFrame,
    forecast_opex: pd.DataFrame,
) -> None:
    """
    Print integrated Budget vs Forecast P&L.
    """

    budget_pnl = calculate_pnl(
        budget_sales,
        budget_opex,
    )

    forecast_pnl = calculate_pnl(
        forecast_sales,
        forecast_opex,
    )

    revenue_variance = (
        forecast_pnl["revenue"]
        - budget_pnl["revenue"]
    )

    gp_variance = (
        forecast_pnl["gross_profit"]
        - budget_pnl["gross_profit"]
    )

    opex_variance = (
        forecast_pnl["opex"]
        - budget_pnl["opex"]
    )

    ebitda_variance = (
        forecast_pnl["ebitda"]
        - budget_pnl["ebitda"]
    )

    ebitda_margin_variance_pp = (
        forecast_pnl["ebitda_margin"]
        - budget_pnl["ebitda_margin"]
    ) * 100

    print(
        "\nNorthstar Systems AS"
    )

    print(
        "2026 Integrated P&L — Budget vs Latest Forecast"
    )

    print(
        "=" * 120
    )

    print_pnl_line(
        "Budget",
        budget_pnl,
    )

    print_pnl_line(
        "Latest Forecast",
        forecast_pnl,
    )

    print(
        "\nLatest Forecast vs Budget"
    )

    print(
        "-" * 80
    )

    print(
        f"Revenue variance:      NOK "
        f"{revenue_variance / 1_000_000:>7.1f}m"
    )

    print(
        f"Gross Profit variance: NOK "
        f"{gp_variance / 1_000_000:>7.1f}m"
    )

    print(
        f"OPEX variance:         NOK "
        f"{opex_variance / 1_000_000:>7.1f}m"
    )

    print(
        f"EBITDA variance:       NOK "
        f"{ebitda_variance / 1_000_000:>7.1f}m"
    )

    print(
        f"EBITDA margin change:      "
        f"{ebitda_margin_variance_pp:>7.2f} pp"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    (
        baseline_opex,
        final_actual_opex,
        budget_sales,
        forecast_sales,
    ) = load_source_data()

    # --------------------------------------------------------
    # Generate Budget OPEX
    # --------------------------------------------------------

    budget_opex = generate_budget_opex(
        baseline_opex
    )

    source_2025 = (
        baseline_opex[
            baseline_opex["date"].dt.year == 2025
        ]
        .copy()
    )

    validate_budget_opex(
        budget_opex,
        source_2025,
    )

    # --------------------------------------------------------
    # Generate Latest Forecast OPEX
    # --------------------------------------------------------

    forecast_opex = (
        generate_latest_forecast_opex(
            final_actual_opex,
            budget_opex,
        )
    )

    validate_forecast_opex(
        forecast_opex,
        final_actual_opex,
        budget_opex,
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    BUDGET_OPEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    budget_opex.to_csv(
        BUDGET_OPEX_PATH,
        index=False,
    )

    forecast_opex.to_csv(
        FORECAST_OPEX_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Integrated P&L
    # --------------------------------------------------------

    print_summary(
        budget_sales,
        forecast_sales,
        budget_opex,
        forecast_opex,
    )

    print(
        "\nBudget OPEX saved to:"
    )

    print(
        BUDGET_OPEX_PATH
    )

    print(
        "\nLatest Forecast OPEX saved to:"
    )

    print(
        FORECAST_OPEX_PATH
    )

    print(
        "\nActual Jan-Aug OPEX Forecast reconciliation: PASSED"
    )

    print(
        "OPEX financial validation: PASSED"
    )


if __name__ == "__main__":
    main()