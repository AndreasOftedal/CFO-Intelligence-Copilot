from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OPEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "actual_opex.csv"
)

SALES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "actual_sales.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    opex = pd.read_csv(
        OPEX_PATH,
        parse_dates=["date"],
    )

    sales = pd.read_csv(
        SALES_PATH,
        parse_dates=["date"],
    )

    return opex, sales


# ============================================================
# OPEX SUMMARIES
# ============================================================

def summarize_opex(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:

    summary = (
        df.groupby(
            group_column,
            as_index=False,
        )
        .agg(
            amount=("amount", "sum"),
        )
        .sort_values(
            "amount",
            ascending=False,
        )
    )

    total = summary["amount"].sum()

    summary["share_pct"] = (
        summary["amount"]
        / total
    )

    return summary


def print_opex_summary(
    title: str,
    summary: pd.DataFrame,
    label_column: str,
) -> None:

    print(f"\n{title}")
    print("=" * 75)

    for row in summary.itertuples():

        label = getattr(
            row,
            label_column,
        )

        print(
            f"{label:<25} | "
            f"NOK "
            f"{row.amount / 1_000_000:>8.1f}m | "
            f"{row.share_pct:>6.1%}"
        )


# ============================================================
# HEADCOUNT
# ============================================================

def print_headcount(
    opex_2025: pd.DataFrame,
) -> None:

    payroll = (
        opex_2025[
            opex_2025["account"] == "Payroll"
        ]
        .copy()
    )

    latest_date = (
        payroll["date"].max()
    )

    latest = (
        payroll[
            payroll["date"] == latest_date
        ]
        .sort_values(
            "headcount",
            ascending=False,
        )
    )

    print(
        f"\nHeadcount by Cost Centre "
        f"— {latest_date.date()}"
    )

    print(
        "=" * 75
    )

    for row in latest.itertuples():

        print(
            f"{row.cost_centre:<25} | "
            f"HC {row.headcount:>4,.0f} | "
            f"Avg monthly employee cost "
            f"NOK "
            f"{row.average_employee_cost:>10,.0f}"
        )

    print(
        "-" * 75
    )

    print(
        f"{'TOTAL':<25} | "
        f"HC "
        f"{latest['headcount'].sum():>4,.0f}"
    )


# ============================================================
# EBITDA
# ============================================================

def calculate_ebitda_summary(
    opex: pd.DataFrame,
    sales: pd.DataFrame,
) -> pd.DataFrame:

    sales_summary = (
        sales.assign(
            year=sales["date"].dt.year
        )
        .groupby(
            "year",
            as_index=False,
        )
        .agg(
            revenue=("net_revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
        )
    )

    opex_summary = (
        opex.assign(
            year=opex["date"].dt.year
        )
        .groupby(
            "year",
            as_index=False,
        )
        .agg(
            opex=("amount", "sum"),
        )
    )

    summary = (
        sales_summary.merge(
            opex_summary,
            on="year",
            how="inner",
        )
    )

    summary["ebitda"] = (
        summary["gross_profit"]
        - summary["opex"]
    )

    summary["ebitda_margin_pct"] = (
        summary["ebitda"]
        / summary["revenue"]
    )

    summary["opex_pct_revenue"] = (
        summary["opex"]
        / summary["revenue"]
    )

    return summary


def print_ebitda_summary(
    summary: pd.DataFrame,
) -> None:

    print(
        "\nOperating Performance"
    )

    print(
        "=" * 100
    )

    for row in summary.itertuples():

        period_label = (
            "YTD Jan-Aug"
            if row.year == 2026
            else "Full Year"
        )

        print(
            f"{row.year} {period_label:<11} | "
            f"Revenue NOK "
            f"{row.revenue / 1_000_000:>8.1f}m | "
            f"GP NOK "
            f"{row.gross_profit / 1_000_000:>7.1f}m | "
            f"OPEX NOK "
            f"{row.opex / 1_000_000:>7.1f}m | "
            f"EBITDA NOK "
            f"{row.ebitda / 1_000_000:>7.1f}m | "
            f"Margin "
            f"{row.ebitda_margin_pct:>6.1%}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    opex, sales = load_data()

    # Use 2025 because it is the latest complete year.
    opex_2025 = (
        opex[
            opex["date"].dt.year == 2025
        ]
        .copy()
    )

    print(
        "\nNorthstar Systems AS"
    )

    print(
        "2025 Actual OPEX — Business Sanity Check"
    )

    account_summary = summarize_opex(
        opex_2025,
        "account",
    )

    cost_centre_summary = summarize_opex(
        opex_2025,
        "cost_centre",
    )

    print_opex_summary(
        "OPEX by Account",
        account_summary,
        "account",
    )

    print_opex_summary(
        "OPEX by Cost Centre",
        cost_centre_summary,
        "cost_centre",
    )

    print_headcount(
        opex_2025
    )

    operating_summary = (
        calculate_ebitda_summary(
            opex,
            sales,
        )
    )

    print_ebitda_summary(
        operating_summary
    )


if __name__ == "__main__":
    main()