from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BUDGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "budget_opex.csv"
)

FORECAST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "latest_forecast_opex.csv"
)


# ============================================================
# LOAD
# ============================================================

def load_data():

    budget = pd.read_csv(
        BUDGET_PATH,
        parse_dates=["date"],
    )

    forecast = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["date"],
    )

    return budget, forecast


# ============================================================
# COMPARISON
# ============================================================

def compare(
    budget: pd.DataFrame,
    forecast: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:

    budget_summary = (
        budget.groupby(
            group_column,
            as_index=False,
        )
        .agg(
            budget=("amount", "sum"),
        )
    )

    forecast_summary = (
        forecast.groupby(
            group_column,
            as_index=False,
        )
        .agg(
            forecast=("amount", "sum"),
        )
    )

    result = budget_summary.merge(
        forecast_summary,
        on=group_column,
        how="outer",
    )

    result["variance"] = (
        result["forecast"]
        - result["budget"]
    )

    result["variance_pct"] = (
        result["variance"]
        / result["budget"]
    )

    return result.sort_values(
        "variance",
        ascending=False,
    )


# ============================================================
# OUTPUT
# ============================================================

def print_comparison(
    title: str,
    comparison: pd.DataFrame,
    label_column: str,
) -> None:

    print(f"\n{title}")
    print("=" * 80)

    for row in comparison.itertuples():

        label = getattr(
            row,
            label_column,
        )

        print(
            f"{label:<25} | "
            f"Budget NOK "
            f"{row.budget / 1_000_000:>7.1f}m | "
            f"Forecast NOK "
            f"{row.forecast / 1_000_000:>7.1f}m | "
            f"Variance NOK "
            f"{row.variance / 1_000_000:>6.1f}m | "
            f"{row.variance_pct:>6.1%}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    budget, forecast = load_data()

    print(
        "\nNorthstar Systems AS"
    )

    print(
        "2026 OPEX Planning — Business Sanity Check"
    )

    account_comparison = compare(
        budget,
        forecast,
        "account",
    )

    cost_centre_comparison = compare(
        budget,
        forecast,
        "cost_centre",
    )

    print_comparison(
        "Latest Forecast vs Budget — OPEX Account",
        account_comparison,
        "account",
    )

    print_comparison(
        "Latest Forecast vs Budget — Cost Centre",
        cost_centre_comparison,
        "cost_centre",
    )


if __name__ == "__main__":
    main()