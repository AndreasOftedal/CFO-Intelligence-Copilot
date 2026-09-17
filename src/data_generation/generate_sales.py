from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ACTUAL_END,
    ACTUAL_START,
    MARKETS,
    MARKET_PRODUCT_DEMAND,
    MARKET_SEGMENT_MIX,
    MONTHLY_SEASONALITY,
    PRODUCTS,
    RANDOM_SEED,
    REQUIRED_SALES_COLUMNS,
    SEGMENTS,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "actual_sales.csv"
)


# ============================================================
# GENERATION ASSUMPTIONS
# ============================================================

# Northstar raises list prices by approximately 2.5% annually.
ANNUAL_PRICE_INFLATION = 0.025

# Procurement costs increase slightly faster than selling prices.
ANNUAL_COST_INFLATION = 0.035

# Random operational variation around the underlying assumptions.
DEMAND_VOLATILITY = 0.08
DISCOUNT_VOLATILITY = 0.012
UNIT_COST_VOLATILITY = 0.012


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def years_since_start(date: pd.Timestamp) -> float:
    """
    Return fractional years since January 2024.

    Examples:
        January 2024 -> 0.00
        January 2025 -> 1.00
        July 2025    -> 1.50
    """

    return (
        (date.year - 2024)
        + ((date.month - 1) / 12)
    )


def round_to_nearest_50(value: float) -> float:
    """
    Round commercial list prices to realistic NOK 50 increments.
    """

    return round(value / 50) * 50


# ============================================================
# ACTUAL SALES GENERATOR
# ============================================================

def generate_actual_sales() -> pd.DataFrame:
    """
    Generate monthly synthetic Actual sales data
    for Northstar Systems AS.

    Dataset grain:

        month
        x country
        x product
        x customer segment

    Financial values are derived from commercial drivers:

        units
        x list price
        -> gross sales
        -> discount
        -> net revenue
        -> COGS
        -> gross profit

    Random variation is deliberately limited to:

        - demand
        - discounting
        - procurement cost

    RANDOM_SEED makes the generated dataset reproducible.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    dates = pd.date_range(
        start=ACTUAL_START,
        end=ACTUAL_END,
        freq="MS",
    )

    rows = []

    # ========================================================
    # MONTH LEVEL
    # ========================================================

    for date in dates:

        elapsed_years = years_since_start(
            date
        )

        price_inflation_factor = (
            1 + ANNUAL_PRICE_INFLATION
        ) ** elapsed_years

        cost_inflation_factor = (
            1 + ANNUAL_COST_INFLATION
        ) ** elapsed_years

        seasonality = (
            MONTHLY_SEASONALITY[
                date.month
            ]
        )

        # ====================================================
        # MARKET LEVEL
        # ====================================================

        for market in MARKETS:

            market_growth_factor = (
                1
                + market.annual_growth_rate
            ) ** elapsed_years

            # ================================================
            # PRODUCT LEVEL
            # ================================================

            for product in PRODUCTS:

                product_demand_factor = (
                    MARKET_PRODUCT_DEMAND[
                        market.name
                    ][
                        product.name
                    ]
                )

                # --------------------------------------------
                # List price
                # --------------------------------------------

                list_price = (
                    product.base_list_price
                    * market.price_factor
                    * price_inflation_factor
                )

                list_price = round_to_nearest_50(
                    list_price
                )

                list_price = round(
                    list_price,
                    2,
                )

                # --------------------------------------------
                # Unit cost
                #
                # Unit cost is set once for a given
                # month-market-product combination.
                #
                # All customer segments therefore face the
                # same underlying product cost.
                # --------------------------------------------

                unit_cost_noise = rng.normal(
                    loc=1.0,
                    scale=UNIT_COST_VOLATILITY,
                )

                unit_cost = (
                    product.base_unit_cost
                    * cost_inflation_factor
                    * unit_cost_noise
                )

                # Store the commercial driver at two decimals
                # BEFORE downstream calculations.
                unit_cost = round(
                    unit_cost,
                    2,
                )

                # ============================================
                # CUSTOMER SEGMENT LEVEL
                # ============================================

                for segment in SEGMENTS:

                    segment_share = (
                        MARKET_SEGMENT_MIX[
                            market.name
                        ][
                            segment.name
                        ]
                    )

                    # ----------------------------------------
                    # Expected unit demand
                    # ----------------------------------------

                    expected_units = (
                        product.monthly_base_units
                        * market.volume_factor
                        * segment_share
                        * product_demand_factor
                        * seasonality
                        * market_growth_factor
                    )

                    # ----------------------------------------
                    # Demand variation
                    # ----------------------------------------

                    demand_noise = rng.normal(
                        loc=1.0,
                        scale=DEMAND_VOLATILITY,
                    )

                    units = max(
                        0,
                        int(
                            round(
                                expected_units
                                * demand_noise
                            )
                        ),
                    )

                    # ----------------------------------------
                    # Discount rate
                    # ----------------------------------------

                    discount_rate = (
                        segment.base_discount_rate
                        + rng.normal(
                            loc=0.0,
                            scale=DISCOUNT_VOLATILITY,
                        )
                    )

                    discount_rate = float(
                        np.clip(
                            discount_rate,
                            0.0,
                            0.25,
                        )
                    )

                    # IMPORTANT:
                    #
                    # Round the commercial driver BEFORE
                    # calculating downstream financial values.
                    #
                    # This ensures:
                    #
                    # gross_sales x discount_rate
                    # reconciles exactly with discount_value.
                    discount_rate = round(
                        discount_rate,
                        6,
                    )

                    # ========================================
                    # FINANCIAL CALCULATIONS
                    # ========================================

                    gross_sales = (
                        units
                        * list_price
                    )

                    gross_sales = round(
                        gross_sales,
                        2,
                    )

                    discount_value = (
                        gross_sales
                        * discount_rate
                    )

                    discount_value = round(
                        discount_value,
                        2,
                    )

                    net_revenue = (
                        gross_sales
                        - discount_value
                    )

                    net_revenue = round(
                        net_revenue,
                        2,
                    )

                    cogs = (
                        units
                        * unit_cost
                    )

                    cogs = round(
                        cogs,
                        2,
                    )

                    gross_profit = (
                        net_revenue
                        - cogs
                    )

                    gross_profit = round(
                        gross_profit,
                        2,
                    )

                    if net_revenue != 0:

                        gross_margin_pct = (
                            gross_profit
                            / net_revenue
                        )

                    else:

                        gross_margin_pct = 0.0

                    gross_margin_pct = round(
                        gross_margin_pct,
                        6,
                    )

                    # ========================================
                    # STORE OBSERVATION
                    # ========================================

                    rows.append(
                        {
                            "date": date,
                            "scenario": "Actual",
                            "country": market.name,
                            "product": product.name,
                            "product_family": product.family,
                            "segment": segment.name,
                            "units": units,
                            "list_price": list_price,
                            "discount_rate": discount_rate,
                            "gross_sales": gross_sales,
                            "discount_value": discount_value,
                            "net_revenue": net_revenue,
                            "unit_cost": unit_cost,
                            "cogs": cogs,
                            "gross_profit": gross_profit,
                            "gross_margin_pct": gross_margin_pct,
                        }
                    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_sales_data(
    df: pd.DataFrame,
) -> None:
    """
    Run deterministic quality controls before saving.

    If a core accounting or commercial identity does not
    reconcile, generation stops immediately.
    """

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    missing_columns = [
        column
        for column
        in REQUIRED_SALES_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    # ========================================================
    # EXPECTED NUMBER OF ROWS
    # ========================================================

    expected_months = len(
        pd.date_range(
            start=ACTUAL_START,
            end=ACTUAL_END,
            freq="MS",
        )
    )

    expected_rows = (
        expected_months
        * len(MARKETS)
        * len(PRODUCTS)
        * len(SEGMENTS)
    )

    if len(df) != expected_rows:

        raise ValueError(
            f"Expected {expected_rows:,} rows, "
            f"but generated {len(df):,}."
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    if df.isna().any().any():

        raise ValueError(
            "Dataset contains missing values."
        )

    # ========================================================
    # VOLUME VALIDATION
    # ========================================================

    if (
        df["units"] < 0
    ).any():

        raise ValueError(
            "Negative unit volumes detected."
        )

    # ========================================================
    # PRICE / COST VALIDATION
    # ========================================================

    if (
        df["list_price"] <= 0
    ).any():

        raise ValueError(
            "Invalid list prices detected."
        )

    if (
        df["unit_cost"] <= 0
    ).any():

        raise ValueError(
            "Invalid unit costs detected."
        )

    # ========================================================
    # DISCOUNT VALIDATION
    # ========================================================

    if not (
        df["discount_rate"]
        .between(
            0,
            0.25,
        )
        .all()
    ):

        raise ValueError(
            "Invalid discount rates detected."
        )

    # ========================================================
    # GROSS SALES RECONCILIATION
    # ========================================================

    calculated_gross_sales = (
        df["units"]
        * df["list_price"]
    )

    if not np.allclose(
        df["gross_sales"],
        calculated_gross_sales,
        atol=0.01,
    ):

        raise ValueError(
            "Gross sales calculation "
            "failed validation."
        )

    # ========================================================
    # DISCOUNT VALUE RECONCILIATION
    # ========================================================

    calculated_discount_value = (
        df["gross_sales"]
        * df["discount_rate"]
    ).round(2)

    if not np.allclose(
        df["discount_value"],
        calculated_discount_value,
        atol=0.01,
    ):

        raise ValueError(
            "Discount value calculation "
            "failed validation."
        )

    # ========================================================
    # NET REVENUE RECONCILIATION
    # ========================================================

    calculated_net_revenue = (
        df["gross_sales"]
        - df["discount_value"]
    ).round(2)

    if not np.allclose(
        df["net_revenue"],
        calculated_net_revenue,
        atol=0.01,
    ):

        raise ValueError(
            "Net revenue calculation "
            "failed validation."
        )

    # ========================================================
    # COGS RECONCILIATION
    # ========================================================

    calculated_cogs = (
        df["units"]
        * df["unit_cost"]
    ).round(2)

    if not np.allclose(
        df["cogs"],
        calculated_cogs,
        atol=0.01,
    ):

        raise ValueError(
            "COGS calculation "
            "failed validation."
        )

    # ========================================================
    # GROSS PROFIT RECONCILIATION
    # ========================================================

    calculated_gross_profit = (
        df["net_revenue"]
        - df["cogs"]
    ).round(2)

    if not np.allclose(
        df["gross_profit"],
        calculated_gross_profit,
        atol=0.01,
    ):

        raise ValueError(
            "Gross profit calculation "
            "failed validation."
        )

    # ========================================================
    # GROSS MARGIN RECONCILIATION
    # ========================================================

    nonzero_revenue = (
        df["net_revenue"] != 0
    )

    calculated_margin = (
        df.loc[
            nonzero_revenue,
            "gross_profit",
        ]
        / df.loc[
            nonzero_revenue,
            "net_revenue",
        ]
    ).round(6)

    if not np.allclose(
        df.loc[
            nonzero_revenue,
            "gross_margin_pct",
        ],
        calculated_margin,
        atol=0.000001,
    ):

        raise ValueError(
            "Gross margin calculation "
            "failed validation."
        )

    # ========================================================
    # SCENARIO VALIDATION
    # ========================================================

    if set(
        df["scenario"].unique()
    ) != {"Actual"}:

        raise ValueError(
            "Unexpected scenario values detected."
        )


# ============================================================
# SUMMARY OUTPUT
# ============================================================

def print_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print a compact financial sanity check.
    """

    summary = (
        df.assign(
            year=df["date"].dt.year
        )
        .groupby(
            "year",
            as_index=False,
        )
        .agg(
            revenue=(
                "net_revenue",
                "sum",
            ),
            gross_profit=(
                "gross_profit",
                "sum",
            ),
            units=(
                "units",
                "sum",
            ),
        )
    )

    summary[
        "gross_margin_pct"
    ] = (
        summary["gross_profit"]
        / summary["revenue"]
    )

    print(
        "\nNorthstar Systems AS"
    )

    print(
        "Actual Sales Dataset"
    )

    print(
        "-" * 50
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
        "\nAnnual summary:"
    )

    for row in summary.itertuples():

        print(
            f"{row.year}: "
            f"Revenue NOK "
            f"{row.revenue / 1_000_000:,.1f}m | "
            f"Gross Profit NOK "
            f"{row.gross_profit / 1_000_000:,.1f}m | "
            f"GM "
            f"{row.gross_margin_pct:.1%} | "
            f"Units "
            f"{row.units:,.0f}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    df = generate_actual_sales()

    validate_sales_data(
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
        f"\nSaved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        "\nValidation: PASSED"
    )


if __name__ == "__main__":
    main()