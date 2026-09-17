from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "actual_sales.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    return df


def financial_summary(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:

    summary = (
        df.groupby(
            group_column,
            as_index=False,
        )
        .agg(
            revenue=("net_revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            units=("units", "sum"),
            discount_value=("discount_value", "sum"),
            gross_sales=("gross_sales", "sum"),
        )
    )

    summary["gross_margin_pct"] = (
        summary["gross_profit"]
        / summary["revenue"]
    )

    summary["discount_rate_pct"] = (
        summary["discount_value"]
        / summary["gross_sales"]
    )

    return summary.sort_values(
        "revenue",
        ascending=False,
    )


def print_summary(
    title: str,
    summary: pd.DataFrame,
    label_column: str,
) -> None:

    print(f"\n{title}")
    print("=" * 85)

    for row in summary.itertuples():
        label = getattr(
            row,
            label_column,
        )

        print(
            f"{label:<25} | "
            f"Revenue NOK "
            f"{row.revenue / 1_000_000:>8.1f}m | "
            f"GM {row.gross_margin_pct:>6.1%} | "
            f"Discount {row.discount_rate_pct:>6.1%} | "
            f"Units {row.units:>8,.0f}"
        )


def main() -> None:

    df = load_data()

    # Use 2025 because it is our latest complete year.
    df_2025 = df[
        df["date"].dt.year == 2025
    ].copy()

    print("\nNorthstar Systems AS")
    print("2025 Actual Sales — Business Sanity Check")

    country_summary = financial_summary(
        df_2025,
        "country",
    )

    product_summary = financial_summary(
        df_2025,
        "product",
    )

    segment_summary = financial_summary(
        df_2025,
        "segment",
    )

    print_summary(
        "Revenue by Country",
        country_summary,
        "country",
    )

    print_summary(
        "Revenue by Product",
        product_summary,
        "product",
    )

    print_summary(
        "Revenue by Customer Segment",
        segment_summary,
        "segment",
    )


if __name__ == "__main__":
    main()