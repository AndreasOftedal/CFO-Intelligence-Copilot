from pathlib import Path

import numpy as np
import pandas as pd

from planning_config import (
    ACTUAL_THROUGH,
    BUDGET_COST_INCREASE,
    BUDGET_DISCOUNT_CHANGE,
    BUDGET_PRICE_INCREASE,
    BUDGET_VOLUME_GROWTH,
    FORECAST_COST_REVISION,
    FORECAST_DISCOUNT_CHANGE,
    FORECAST_FUTURE_START,
    FORECAST_SPECIFIC_ADJUSTMENTS,
    FORECAST_VOLUME_REVISION,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_ACTUAL_PATH = (
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

BUDGET_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "budget_sales.csv"
)

FORECAST_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "latest_forecast_sales.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

FINANCIAL_COLUMNS = [
    "units",
    "list_price",
    "discount_rate",
    "gross_sales",
    "discount_value",
    "net_revenue",
    "unit_cost",
    "cogs",
    "gross_profit",
    "gross_margin_pct",
]

GRAIN_COLUMNS = [
    "date",
    "country",
    "product",
    "product_family",
    "segment",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def round_to_nearest_50(
    value: float,
) -> float:
    """
    Round commercial prices to NOK 50 increments.
    """

    return round(value / 50) * 50


def recalculate_financials(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recalculate all financial outputs from the
    underlying commercial drivers.

    Financial values are never trusted if their
    underlying drivers have changed.
    """

    df = df.copy()

    df["gross_sales"] = (
        df["units"]
        * df["list_price"]
    ).round(2)

    df["discount_value"] = (
        df["gross_sales"]
        * df["discount_rate"]
    ).round(2)

    df["net_revenue"] = (
        df["gross_sales"]
        - df["discount_value"]
    ).round(2)

    df["cogs"] = (
        df["units"]
        * df["unit_cost"]
    ).round(2)

    df["gross_profit"] = (
        df["net_revenue"]
        - df["cogs"]
    ).round(2)

    nonzero_revenue = (
        df["net_revenue"] != 0
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

    df.loc[
        ~nonzero_revenue,
        "gross_margin_pct",
    ] = 0.0

    return df


# ============================================================
# LOAD SOURCE DATA
# ============================================================

def load_source_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load:

    1. Baseline Actual data for building Budget.
    2. Final Actual data, including controlled events,
       for building Latest Forecast.
    """

    if not BASELINE_ACTUAL_PATH.exists():

        raise FileNotFoundError(
            f"Missing baseline Actual dataset: "
            f"{BASELINE_ACTUAL_PATH}"
        )

    if not FINAL_ACTUAL_PATH.exists():

        raise FileNotFoundError(
            f"Missing final Actual dataset: "
            f"{FINAL_ACTUAL_PATH}"
        )

    baseline_actual = pd.read_csv(
        BASELINE_ACTUAL_PATH,
        parse_dates=["date"],
    )

    final_actual = pd.read_csv(
        FINAL_ACTUAL_PATH,
        parse_dates=["date"],
    )

    return (
        baseline_actual,
        final_actual,
    )


# ============================================================
# BUDGET GENERATION
# ============================================================

def generate_budget(
    baseline_actual: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate the full-year 2026 Budget.

    The Budget is based on 2025 Actual commercial
    performance and management assumptions prepared
    before 2026.

    Method:

        2025 monthly Actual
        + planned market volume growth
        + planned price increase
        + planned cost inflation
        + planned discount discipline
        = 2026 Budget
    """

    actual_2025 = (
        baseline_actual[
            baseline_actual["date"].dt.year == 2025
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(actual_2025) == 0:

        raise ValueError(
            "No 2025 Actual data available "
            "for Budget generation."
        )

    budget = actual_2025.copy()

    # --------------------------------------------------------
    # Move prior-year months into 2026
    # --------------------------------------------------------

    budget["date"] = (
        budget["date"]
        + pd.DateOffset(years=1)
    )

    budget["scenario"] = "Budget"

    # --------------------------------------------------------
    # Volume planning
    # --------------------------------------------------------

    volume_growth = (
        budget["country"]
        .map(BUDGET_VOLUME_GROWTH)
    )

    if volume_growth.isna().any():

        raise ValueError(
            "Budget volume growth missing "
            "for one or more markets."
        )

    budget["units"] = (
        budget["units"]
        * (1 + volume_growth)
    ).round().astype(int)

    # --------------------------------------------------------
    # Price planning
    # --------------------------------------------------------

    budget["list_price"] = (
        budget["list_price"]
        * (1 + BUDGET_PRICE_INCREASE)
    )

    budget["list_price"] = (
        budget["list_price"]
        .apply(round_to_nearest_50)
        .round(2)
    )

    # --------------------------------------------------------
    # Discount planning
    # --------------------------------------------------------

    discount_change = (
        budget["country"]
        .map(BUDGET_DISCOUNT_CHANGE)
    )

    if discount_change.isna().any():

        raise ValueError(
            "Budget discount assumption missing "
            "for one or more markets."
        )

    budget["discount_rate"] = (
        budget["discount_rate"]
        + discount_change
    )

    budget["discount_rate"] = (
        np.clip(
            budget["discount_rate"],
            0.0,
            0.25,
        )
    )

    budget["discount_rate"] = (
        budget["discount_rate"]
        .round(6)
    )

    # --------------------------------------------------------
    # Cost planning
    # --------------------------------------------------------

    budget["unit_cost"] = (
        budget["unit_cost"]
        * (1 + BUDGET_COST_INCREASE)
    ).round(2)

    # --------------------------------------------------------
    # Recalculate financial outputs
    # --------------------------------------------------------

    budget = recalculate_financials(
        budget
    )

    return budget


# ============================================================
# LATEST FORECAST GENERATION
# ============================================================

def generate_latest_forecast(
    final_actual: pd.DataFrame,
    budget: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate Latest Forecast as of September 2026.

    January-August:
        exactly equal to final Actual.

    September-December:
        Budget updated using latest management
        expectations.
    """

    actual_cutoff = pd.Timestamp(
        ACTUAL_THROUGH
    )

    forecast_future_start = pd.Timestamp(
        FORECAST_FUTURE_START
    )

    # --------------------------------------------------------
    # Historical portion:
    # Actual Jan-Aug
    # --------------------------------------------------------

    forecast_actual = (
        final_actual[
            (
                final_actual["date"].dt.year == 2026
            )
            & (
                final_actual["date"]
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
    # Future portion:
    # Budget Sep-Dec
    # --------------------------------------------------------

    forecast_future = (
        budget[
            budget["date"]
            >= forecast_future_start
        ]
        .copy()
        .reset_index(drop=True)
    )

    forecast_future["scenario"] = (
        "Latest Forecast"
    )

    # --------------------------------------------------------
    # Market-level volume revision
    # --------------------------------------------------------

    volume_revision = (
        forecast_future["country"]
        .map(FORECAST_VOLUME_REVISION)
    )

    if volume_revision.isna().any():

        raise ValueError(
            "Forecast volume revision missing "
            "for one or more markets."
        )

    forecast_future["units"] = (
        forecast_future["units"]
        * (1 + volume_revision)
    ).round().astype(int)

    # --------------------------------------------------------
    # Market-level discount revision
    # --------------------------------------------------------

    discount_revision = (
        forecast_future["country"]
        .map(FORECAST_DISCOUNT_CHANGE)
    )

    if discount_revision.isna().any():

        raise ValueError(
            "Forecast discount revision missing "
            "for one or more markets."
        )

    forecast_future["discount_rate"] = (
        forecast_future["discount_rate"]
        + discount_revision
    )

    forecast_future["discount_rate"] = (
        np.clip(
            forecast_future["discount_rate"],
            0.0,
            0.25,
        )
        .round(6)
    )

    # --------------------------------------------------------
    # Market-level cost revision
    # --------------------------------------------------------

    cost_revision = (
        forecast_future["country"]
        .map(FORECAST_COST_REVISION)
    )

    if cost_revision.isna().any():

        raise ValueError(
            "Forecast cost revision missing "
            "for one or more markets."
        )

    forecast_future["unit_cost"] = (
        forecast_future["unit_cost"]
        * (1 + cost_revision)
    ).round(2)

    # --------------------------------------------------------
    # Specific timing adjustments
    # --------------------------------------------------------

    for adjustment in (
        FORECAST_SPECIFIC_ADJUSTMENTS
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
                forecast_future["country"]
                == adjustment["country"]
            )
            & (
                forecast_future["product"]
                == adjustment["product"]
            )
            & (
                forecast_future["segment"]
                == adjustment["segment"]
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

        if adjustment["driver"] == "units":

            forecast_future.loc[
                mask,
                "units",
            ] = (
                forecast_future.loc[
                    mask,
                    "units",
                ]
                * adjustment["factor"]
            ).round().astype(int)

        else:

            raise ValueError(
                f"{adjustment['adjustment_id']}: "
                "unsupported adjustment driver."
            )

    # --------------------------------------------------------
    # Recalculate all future financial outputs
    # --------------------------------------------------------

    forecast_future = recalculate_financials(
        forecast_future
    )

    # --------------------------------------------------------
    # Combine Actual and future estimate
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
# FINANCIAL VALIDATION
# ============================================================

def validate_financial_integrity(
    df: pd.DataFrame,
    scenario_name: str,
) -> None:
    """
    Validate all deterministic financial identities.
    """

    if df.isna().any().any():

        raise ValueError(
            f"{scenario_name}: missing values detected."
        )

    if (
        df["units"] < 0
    ).any():

        raise ValueError(
            f"{scenario_name}: negative units detected."
        )

    if not (
        df["discount_rate"]
        .between(
            0,
            0.25,
        )
        .all()
    ):

        raise ValueError(
            f"{scenario_name}: invalid discount rates."
        )

    expected_gross_sales = (
        df["units"]
        * df["list_price"]
    ).round(2)

    if not np.allclose(
        df["gross_sales"],
        expected_gross_sales,
        atol=0.01,
    ):

        raise ValueError(
            f"{scenario_name}: "
            "gross sales reconciliation failed."
        )

    expected_discount = (
        df["gross_sales"]
        * df["discount_rate"]
    ).round(2)

    if not np.allclose(
        df["discount_value"],
        expected_discount,
        atol=0.01,
    ):

        raise ValueError(
            f"{scenario_name}: "
            "discount reconciliation failed."
        )

    expected_revenue = (
        df["gross_sales"]
        - df["discount_value"]
    ).round(2)

    if not np.allclose(
        df["net_revenue"],
        expected_revenue,
        atol=0.01,
    ):

        raise ValueError(
            f"{scenario_name}: "
            "net revenue reconciliation failed."
        )

    expected_cogs = (
        df["units"]
        * df["unit_cost"]
    ).round(2)

    if not np.allclose(
        df["cogs"],
        expected_cogs,
        atol=0.01,
    ):

        raise ValueError(
            f"{scenario_name}: "
            "COGS reconciliation failed."
        )

    expected_gp = (
        df["net_revenue"]
        - df["cogs"]
    ).round(2)

    if not np.allclose(
        df["gross_profit"],
        expected_gp,
        atol=0.01,
    ):

        raise ValueError(
            f"{scenario_name}: "
            "gross profit reconciliation failed."
        )


# ============================================================
# BUDGET VALIDATION
# ============================================================

def validate_budget(
    budget: pd.DataFrame,
) -> None:
    """
    Validate Budget completeness and integrity.
    """

    expected_rows = (
        12
        * 6
        * 8
        * 4
    )

    if len(budget) != expected_rows:

        raise ValueError(
            f"Budget: expected "
            f"{expected_rows:,} rows, "
            f"found {len(budget):,}."
        )

    if set(
        budget["scenario"].unique()
    ) != {"Budget"}:

        raise ValueError(
            "Budget contains invalid scenario values."
        )

    if set(
        budget["date"].dt.year.unique()
    ) != {2026}:

        raise ValueError(
            "Budget must contain only 2026."
        )

    months = set(
        budget["date"].dt.month.unique()
    )

    if months != set(range(1, 13)):

        raise ValueError(
            "Budget does not contain all 12 months."
        )

    if budget.duplicated(
        GRAIN_COLUMNS
    ).any():

        raise ValueError(
            "Budget contains duplicate grain rows."
        )

    validate_financial_integrity(
        budget,
        "Budget",
    )


# ============================================================
# FORECAST VALIDATION
# ============================================================

def validate_forecast(
    latest_forecast: pd.DataFrame,
    final_actual: pd.DataFrame,
) -> None:
    """
    Validate Latest Forecast.

    Most important control:

    January-August Latest Forecast must equal
    January-August Actual exactly for all financial
    and commercial driver columns.
    """

    expected_rows = (
        12
        * 6
        * 8
        * 4
    )

    if len(latest_forecast) != expected_rows:

        raise ValueError(
            f"Latest Forecast: expected "
            f"{expected_rows:,} rows, "
            f"found {len(latest_forecast):,}."
        )

    if set(
        latest_forecast["scenario"].unique()
    ) != {"Latest Forecast"}:

        raise ValueError(
            "Latest Forecast contains invalid "
            "scenario values."
        )

    months = set(
        latest_forecast["date"].dt.month.unique()
    )

    if months != set(range(1, 13)):

        raise ValueError(
            "Latest Forecast does not contain "
            "all 12 months."
        )

    if latest_forecast.duplicated(
        GRAIN_COLUMNS
    ).any():

        raise ValueError(
            "Latest Forecast contains "
            "duplicate grain rows."
        )

    validate_financial_integrity(
        latest_forecast,
        "Latest Forecast",
    )

    # --------------------------------------------------------
    # Actual Jan-Aug reconciliation
    # --------------------------------------------------------

    actual_cutoff = pd.Timestamp(
        ACTUAL_THROUGH
    )

    actual_ytd = (
        final_actual[
            (
                final_actual["date"].dt.year == 2026
            )
            & (
                final_actual["date"]
                <= actual_cutoff
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
            <= actual_cutoff
        ]
        .copy()
        .sort_values(
            GRAIN_COLUMNS
        )
        .reset_index(drop=True)
    )

    if len(actual_ytd) != len(forecast_ytd):

        raise ValueError(
            "Latest Forecast YTD row count "
            "does not match Actual."
        )

    comparison_columns = (
        GRAIN_COLUMNS
        + FINANCIAL_COLUMNS
    )

    actual_compare = (
        actual_ytd[
            comparison_columns
        ]
        .reset_index(drop=True)
    )

    forecast_compare = (
        forecast_ytd[
            comparison_columns
        ]
        .reset_index(drop=True)
    )

    if not actual_compare.equals(
        forecast_compare
    ):

        raise ValueError(
            "Latest Forecast January-August "
            "does not exactly equal Actual."
        )


# ============================================================
# SUMMARY OUTPUT
# ============================================================

def scenario_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate full-year high-level financial metrics.
    """

    revenue = df[
        "net_revenue"
    ].sum()

    gross_profit = df[
        "gross_profit"
    ].sum()

    units = df[
        "units"
    ].sum()

    gross_margin = (
        gross_profit / revenue
        if revenue != 0
        else 0.0
    )

    return {
        "revenue": revenue,
        "gross_profit": gross_profit,
        "units": units,
        "gross_margin": gross_margin,
    }


def print_summary(
    budget: pd.DataFrame,
    latest_forecast: pd.DataFrame,
) -> None:
    """
    Print Budget vs Latest Forecast comparison.
    """

    budget_summary = scenario_summary(
        budget
    )

    forecast_summary = scenario_summary(
        latest_forecast
    )

    revenue_variance = (
        forecast_summary["revenue"]
        - budget_summary["revenue"]
    )

    gp_variance = (
        forecast_summary["gross_profit"]
        - budget_summary["gross_profit"]
    )

    print(
        "\nNorthstar Systems AS"
    )

    print(
        "2026 Planning Engine"
    )

    print(
        "=" * 75
    )

    print(
        "\nBudget:"
    )

    print(
        f"Revenue: NOK "
        f"{budget_summary['revenue'] / 1_000_000:,.1f}m"
    )

    print(
        f"Gross Profit: NOK "
        f"{budget_summary['gross_profit'] / 1_000_000:,.1f}m"
    )

    print(
        f"Gross Margin: "
        f"{budget_summary['gross_margin']:.1%}"
    )

    print(
        f"Units: "
        f"{budget_summary['units']:,.0f}"
    )

    print(
        "\nLatest Forecast:"
    )

    print(
        f"Revenue: NOK "
        f"{forecast_summary['revenue'] / 1_000_000:,.1f}m"
    )

    print(
        f"Gross Profit: NOK "
        f"{forecast_summary['gross_profit'] / 1_000_000:,.1f}m"
    )

    print(
        f"Gross Margin: "
        f"{forecast_summary['gross_margin']:.1%}"
    )

    print(
        f"Units: "
        f"{forecast_summary['units']:,.0f}"
    )

    print(
        "\nForecast vs Budget:"
    )

    print(
        f"Revenue variance: NOK "
        f"{revenue_variance / 1_000_000:,.1f}m"
    )

    print(
        f"Gross Profit variance: NOK "
        f"{gp_variance / 1_000_000:,.1f}m"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    (
        baseline_actual,
        final_actual,
    ) = load_source_data()

    # --------------------------------------------------------
    # Generate Budget
    # --------------------------------------------------------

    budget = generate_budget(
        baseline_actual
    )

    validate_budget(
        budget
    )

    # --------------------------------------------------------
    # Generate Latest Forecast
    # --------------------------------------------------------

    latest_forecast = (
        generate_latest_forecast(
            final_actual,
            budget,
        )
    )

    validate_forecast(
        latest_forecast,
        final_actual,
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    BUDGET_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    budget.to_csv(
        BUDGET_OUTPUT_PATH,
        index=False,
    )

    latest_forecast.to_csv(
        FORECAST_OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Output summary
    # --------------------------------------------------------

    print_summary(
        budget,
        latest_forecast,
    )

    print(
        "\nBudget saved to:"
    )

    print(
        BUDGET_OUTPUT_PATH
    )

    print(
        "\nLatest Forecast saved to:"
    )

    print(
        FORECAST_OUTPUT_PATH
    )

    print(
        "\nActual Jan-Aug Forecast reconciliation: PASSED"
    )

    print(
        "Financial validation: PASSED"
    )


if __name__ == "__main__":
    main()