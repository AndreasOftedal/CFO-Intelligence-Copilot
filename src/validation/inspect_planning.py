from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACTUAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "actual_sales.csv"
)

BUDGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "budget_sales.csv"
)

FORECAST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "latest_forecast_sales.csv"
)


def load_data():
    actual = pd.read_csv(
        ACTUAL_PATH,
        parse_dates=["date"],
    )

    budget = pd.read_csv(
        BUDGET_PATH,
        parse_dates=["date"],
    )

    forecast = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["date"],
    )

    return actual, budget, forecast


def summarize(df, group_column):
    summary = (
        df.groupby(
            group_column,
            as_index=False,
        )
        .agg(
            revenue=("net_revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            units=("units", "sum"),
        )
    )

    summary["gross_margin_pct"] = (
        summary["gross_profit"]
        / summary["revenue"]
    )

    return summary


def compare_scenarios(
    budget,
    forecast,
    group_column,
):
    budget_summary = summarize(
        budget,
        group_column,
    )

    forecast_summary = summarize(
        forecast,
        group_column,
    )

    comparison = budget_summary.merge(
        forecast_summary,
        on=group_column,
        suffixes=("_budget", "_forecast"),
    )

    comparison["revenue_variance"] = (
        comparison["revenue_forecast"]
        - comparison["revenue_budget"]
    )

    comparison["revenue_variance_pct"] = (
        comparison["revenue_variance"]
        / comparison["revenue_budget"]
    )

    comparison["gp_variance"] = (
        comparison["gross_profit_forecast"]
        - comparison["gross_profit_budget"]
    )

    comparison["gp_variance_pct"] = (
        comparison["gp_variance"]
        / comparison["gross_profit_budget"]
    )

    comparison["margin_variance_pp"] = (
        comparison["gross_margin_pct_forecast"]
        - comparison["gross_margin_pct_budget"]
    ) * 100

    comparison["units_variance_pct"] = (
        (
            comparison["units_forecast"]
            - comparison["units_budget"]
        )
        / comparison["units_budget"]
    )

    return comparison.sort_values(
        "revenue_variance"
    )


def print_comparison(
    title,
    comparison,
    label_column,
):
    print(f"\n{title}")
    print("=" * 115)

    for row in comparison.itertuples():

        label = getattr(
            row,
            label_column,
        )

        print(
            f"{label:<25} | "
            f"Revenue Var "
            f"NOK {row.revenue_variance / 1_000_000:>7.1f}m "
            f"({row.revenue_variance_pct:>6.1%}) | "
            f"GP Var "
            f"NOK {row.gp_variance / 1_000_000:>7.1f}m "
            f"({row.gp_variance_pct:>6.1%}) | "
            f"Margin "
            f"{row.margin_variance_pp:>6.2f}pp | "
            f"Units "
            f"{row.units_variance_pct:>6.1%}"
        )


def ytd_actual_vs_budget(
    actual,
    budget,
):
    actual_ytd = (
        actual[
            (
                actual["date"].dt.year == 2026
            )
            & (
                actual["date"].dt.month <= 8
            )
        ]
        .copy()
    )

    budget_ytd = (
        budget[
            budget["date"].dt.month <= 8
        ]
        .copy()
    )

    return compare_scenarios(
        budget_ytd,
        actual_ytd,
        "country",
    )


def main():
    actual, budget, forecast = load_data()

    print(
        "\nNorthstar Systems AS"
    )

    print(
        "2026 Planning — Business Sanity Check"
    )

    country_comparison = compare_scenarios(
        budget,
        forecast,
        "country",
    )

    product_comparison = compare_scenarios(
        budget,
        forecast,
        "product",
    )

    ytd_comparison = ytd_actual_vs_budget(
        actual,
        budget,
    )

    print_comparison(
        "Full-Year Latest Forecast vs Budget — Country",
        country_comparison,
        "country",
    )

    print_comparison(
        "Full-Year Latest Forecast vs Budget — Product",
        product_comparison,
        "product",
    )

    print_comparison(
        "Jan-Aug Actual vs Budget — Country",
        ytd_comparison,
        "country",
    )


if __name__ == "__main__":
    main()