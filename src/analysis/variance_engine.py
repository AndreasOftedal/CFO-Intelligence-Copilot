import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_config import (
    ACTUAL_THROUGH,
    ANALYSIS_OUTPUT_DIRECTORY,
    COMPARISON_ACTUAL_YTD_VS_BUDGET,
    COMPARISON_FORECAST_VS_BUDGET,
    FORECAST_ANALYSIS_FILENAME,
    MATERIALITY_ABSOLUTE_NOK,
    MATERIALITY_EBITDA_SHARE,
    RECONCILIATION_TOLERANCE_NOK,
    TOP_N_COST_CENTRES,
    TOP_N_COUNTRIES,
    TOP_N_DRIVERS,
    TOP_N_OPEX_ACCOUNTS,
    TOP_N_PRODUCTS,
    YTD_ANALYSIS_FILENAME,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACTUAL_SALES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "actual_sales.csv"
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

ACTUAL_OPEX_PATH = (
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

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / ANALYSIS_OUTPUT_DIRECTORY
)


# ============================================================
# GRAINS
# ============================================================

SALES_GRAIN = [
    "date",
    "country",
    "product",
    "product_family",
    "segment",
]

OPEX_GRAIN = [
    "date",
    "cost_centre",
    "account",
]


# ============================================================
# INPUT COLUMNS
# ============================================================

SALES_DRIVER_COLUMNS = [
    "units",
    "list_price",
    "discount_rate",
    "unit_cost",
    "net_revenue",
    "gross_profit",
]

OPEX_DRIVER_COLUMNS = [
    "amount",
    "headcount",
    "average_employee_cost",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_csv(
    path: Path,
) -> pd.DataFrame:
    """
    Load a finance dataset and parse the date column.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Required dataset not found: {path}"
        )

    return pd.read_csv(
        path,
        parse_dates=["date"],
    )


def load_all_data():
    """
    Load all datasets required by the analysis engine.
    """

    return {
        "actual_sales": load_csv(
            ACTUAL_SALES_PATH
        ),
        "budget_sales": load_csv(
            BUDGET_SALES_PATH
        ),
        "forecast_sales": load_csv(
            FORECAST_SALES_PATH
        ),
        "actual_opex": load_csv(
            ACTUAL_OPEX_PATH
        ),
        "budget_opex": load_csv(
            BUDGET_OPEX_PATH
        ),
        "forecast_opex": load_csv(
            FORECAST_OPEX_PATH
        ),
    }


# ============================================================
# FINANCIAL CALCULATION HELPERS
# ============================================================

def calculate_revenue(
    units: pd.Series,
    list_price: pd.Series,
    discount_rate: pd.Series,
) -> pd.Series:
    """
    Reproduce the exact revenue calculation used by
    the source finance engine.

    Rounding is intentionally performed at the same
    stages as the underlying dataset.
    """

    gross_sales = (
        units
        * list_price
    ).round(2)

    discount_value = (
        gross_sales
        * discount_rate
    ).round(2)

    net_revenue = (
        gross_sales
        - discount_value
    ).round(2)

    return net_revenue


def calculate_cogs(
    units: pd.Series,
    unit_cost: pd.Series,
) -> pd.Series:
    """
    Reproduce source COGS calculation.
    """

    return (
        units
        * unit_cost
    ).round(2)


def calculate_gross_profit(
    units: pd.Series,
    list_price: pd.Series,
    discount_rate: pd.Series,
    unit_cost: pd.Series,
) -> pd.Series:
    """
    Calculate Gross Profit using the same accounting
    identities and rounding logic as the source model.
    """

    revenue = calculate_revenue(
        units,
        list_price,
        discount_rate,
    )

    cogs = calculate_cogs(
        units,
        unit_cost,
    )

    return (
        revenue
        - cogs
    ).round(2)


# ============================================================
# DATA ALIGNMENT
# ============================================================

def merge_sales_scenarios(
    base: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align two sales scenarios at identical financial grain.
    """

    base_columns = (
        SALES_GRAIN
        + SALES_DRIVER_COLUMNS
    )

    comparison_columns = (
        SALES_GRAIN
        + SALES_DRIVER_COLUMNS
    )

    merged = (
        base[
            base_columns
        ]
        .merge(
            comparison[
                comparison_columns
            ],
            on=SALES_GRAIN,
            how="outer",
            suffixes=(
                "_base",
                "_comparison",
            ),
            indicator=True,
        )
    )

    if not (
        merged["_merge"] == "both"
    ).all():

        unmatched = (
            merged[
                merged["_merge"] != "both"
            ][
                SALES_GRAIN
                + ["_merge"]
            ]
        )

        raise ValueError(
            "Sales scenarios do not share "
            "identical analytical grain:\n"
            f"{unmatched.head(20)}"
        )

    merged = (
        merged
        .drop(
            columns="_merge"
        )
        .sort_values(
            SALES_GRAIN
        )
        .reset_index(drop=True)
    )

    return merged


def merge_opex_scenarios(
    base: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align two OPEX scenarios at identical financial grain.
    """

    base_columns = (
        OPEX_GRAIN
        + OPEX_DRIVER_COLUMNS
    )

    comparison_columns = (
        OPEX_GRAIN
        + OPEX_DRIVER_COLUMNS
    )

    merged = (
        base[
            base_columns
        ]
        .merge(
            comparison[
                comparison_columns
            ],
            on=OPEX_GRAIN,
            how="outer",
            suffixes=(
                "_base",
                "_comparison",
            ),
            indicator=True,
        )
    )

    if not (
        merged["_merge"] == "both"
    ).all():

        unmatched = (
            merged[
                merged["_merge"] != "both"
            ][
                OPEX_GRAIN
                + ["_merge"]
            ]
        )

        raise ValueError(
            "OPEX scenarios do not share "
            "identical analytical grain:\n"
            f"{unmatched.head(20)}"
        )

    merged = (
        merged
        .drop(
            columns="_merge"
        )
        .sort_values(
            OPEX_GRAIN
        )
        .reset_index(drop=True)
    )

    return merged


# ============================================================
# COMMERCIAL BRIDGE
# ============================================================

def calculate_commercial_bridge(
    merged: pd.DataFrame,
) -> dict:
    """
    Sequentially decompose Revenue and Gross Profit
    variance.

    Sequence:

        0. Base scenario
        1. Volume & mix
        2. List price
        3. Discount
        4. Unit cost

    The bridge operates at detailed grain and therefore
    captures mix changes naturally within the volume/mix
    contribution.
    """

    q_base = merged[
        "units_base"
    ]

    q_comparison = merged[
        "units_comparison"
    ]

    p_base = merged[
        "list_price_base"
    ]

    p_comparison = merged[
        "list_price_comparison"
    ]

    d_base = merged[
        "discount_rate_base"
    ]

    d_comparison = merged[
        "discount_rate_comparison"
    ]

    c_base = merged[
        "unit_cost_base"
    ]

    c_comparison = merged[
        "unit_cost_comparison"
    ]

    # --------------------------------------------------------
    # Revenue states
    # --------------------------------------------------------

    revenue_0 = calculate_revenue(
        q_base,
        p_base,
        d_base,
    )

    revenue_1 = calculate_revenue(
        q_comparison,
        p_base,
        d_base,
    )

    revenue_2 = calculate_revenue(
        q_comparison,
        p_comparison,
        d_base,
    )

    revenue_3 = calculate_revenue(
        q_comparison,
        p_comparison,
        d_comparison,
    )

    # --------------------------------------------------------
    # Gross Profit states
    # --------------------------------------------------------

    gp_0 = calculate_gross_profit(
        q_base,
        p_base,
        d_base,
        c_base,
    )

    gp_1 = calculate_gross_profit(
        q_comparison,
        p_base,
        d_base,
        c_base,
    )

    gp_2 = calculate_gross_profit(
        q_comparison,
        p_comparison,
        d_base,
        c_base,
    )

    gp_3 = calculate_gross_profit(
        q_comparison,
        p_comparison,
        d_comparison,
        c_base,
    )

    gp_4 = calculate_gross_profit(
        q_comparison,
        p_comparison,
        d_comparison,
        c_comparison,
    )

    # --------------------------------------------------------
    # Validate calculated endpoints against stored finance data
    # --------------------------------------------------------

    stored_base_revenue = float(
        merged[
            "net_revenue_base"
        ].sum()
    )

    stored_comparison_revenue = float(
        merged[
            "net_revenue_comparison"
        ].sum()
    )

    stored_base_gp = float(
        merged[
            "gross_profit_base"
        ].sum()
    )

    stored_comparison_gp = float(
        merged[
            "gross_profit_comparison"
        ].sum()
    )

    if not np.isclose(
        revenue_0.sum(),
        stored_base_revenue,
        atol=RECONCILIATION_TOLERANCE_NOK,
    ):

        raise ValueError(
            "Base Revenue calculation does not "
            "reconcile to source data."
        )

    if not np.isclose(
        revenue_3.sum(),
        stored_comparison_revenue,
        atol=RECONCILIATION_TOLERANCE_NOK,
    ):

        raise ValueError(
            "Comparison Revenue calculation does not "
            "reconcile to source data."
        )

    if not np.isclose(
        gp_0.sum(),
        stored_base_gp,
        atol=RECONCILIATION_TOLERANCE_NOK,
    ):

        raise ValueError(
            "Base Gross Profit calculation does not "
            "reconcile to source data."
        )

    if not np.isclose(
        gp_4.sum(),
        stored_comparison_gp,
        atol=RECONCILIATION_TOLERANCE_NOK,
    ):

        raise ValueError(
            "Comparison Gross Profit calculation does not "
            "reconcile to source data."
        )

    # --------------------------------------------------------
    # Revenue bridge
    # --------------------------------------------------------

    revenue_bridge = {
        "volume_mix": float(
            (
                revenue_1
                - revenue_0
            ).sum()
        ),
        "list_price": float(
            (
                revenue_2
                - revenue_1
            ).sum()
        ),
        "discount": float(
            (
                revenue_3
                - revenue_2
            ).sum()
        ),
    }

    # --------------------------------------------------------
    # Gross Profit bridge
    # --------------------------------------------------------

    gp_bridge = {
        "volume_mix": float(
            (
                gp_1
                - gp_0
            ).sum()
        ),
        "list_price": float(
            (
                gp_2
                - gp_1
            ).sum()
        ),
        "discount": float(
            (
                gp_3
                - gp_2
            ).sum()
        ),
        "unit_cost": float(
            (
                gp_4
                - gp_3
            ).sum()
        ),
    }

    revenue_variance = (
        stored_comparison_revenue
        - stored_base_revenue
    )

    gp_variance = (
        stored_comparison_gp
        - stored_base_gp
    )

    revenue_bridge_total = sum(
        revenue_bridge.values()
    )

    gp_bridge_total = sum(
        gp_bridge.values()
    )

    revenue_residual = (
        revenue_variance
        - revenue_bridge_total
    )

    gp_residual = (
        gp_variance
        - gp_bridge_total
    )

    if abs(
        revenue_residual
    ) > RECONCILIATION_TOLERANCE_NOK:

        raise ValueError(
            "Revenue bridge failed reconciliation. "
            f"Residual: NOK {revenue_residual:,.2f}"
        )

    if abs(
        gp_residual
    ) > RECONCILIATION_TOLERANCE_NOK:

        raise ValueError(
            "Gross Profit bridge failed reconciliation. "
            f"Residual: NOK {gp_residual:,.2f}"
        )

    return {
        "revenue_bridge": revenue_bridge,
        "gross_profit_bridge": gp_bridge,
        "base_revenue": stored_base_revenue,
        "comparison_revenue":
            stored_comparison_revenue,
        "revenue_variance": revenue_variance,
        "base_gross_profit": stored_base_gp,
        "comparison_gross_profit":
            stored_comparison_gp,
        "gross_profit_variance": gp_variance,
        "revenue_reconciliation_residual":
            revenue_residual,
        "gross_profit_reconciliation_residual":
            gp_residual,
    }


# ============================================================
# OPEX BRIDGE
# ============================================================

def calculate_opex_bridge(
    merged: pd.DataFrame,
) -> dict:
    """
    Decompose OPEX variance into:

        Headcount
        Employee cost
        Non-payroll

    Positive OPEX variance is unfavorable to EBITDA.
    """

    payroll = (
        merged[
            merged["account"] == "Payroll"
        ]
        .copy()
    )

    non_payroll = (
        merged[
            merged["account"] != "Payroll"
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Payroll states
    # --------------------------------------------------------

    payroll_0 = (
        payroll["headcount_base"]
        * payroll[
            "average_employee_cost_base"
        ]
    ).round(2)

    payroll_1 = (
        payroll[
            "headcount_comparison"
        ]
        * payroll[
            "average_employee_cost_base"
        ]
    ).round(2)

    payroll_2 = (
        payroll[
            "headcount_comparison"
        ]
        * payroll[
            "average_employee_cost_comparison"
        ]
    ).round(2)

    stored_base_payroll = float(
        payroll[
            "amount_base"
        ].sum()
    )

    stored_comparison_payroll = float(
        payroll[
            "amount_comparison"
        ].sum()
    )

    if not np.isclose(
        payroll_0.sum(),
        stored_base_payroll,
        atol=RECONCILIATION_TOLERANCE_NOK,
    ):

        raise ValueError(
            "Base Payroll calculation failed "
            "reconciliation."
        )

    if not np.isclose(
        payroll_2.sum(),
        stored_comparison_payroll,
        atol=RECONCILIATION_TOLERANCE_NOK,
    ):

        raise ValueError(
            "Comparison Payroll calculation failed "
            "reconciliation."
        )

    headcount_impact = float(
        (
            payroll_1
            - payroll_0
        ).sum()
    )

    employee_cost_impact = float(
        (
            payroll_2
            - payroll_1
        ).sum()
    )

    # --------------------------------------------------------
    # Non-payroll
    # --------------------------------------------------------

    non_payroll_impact = float(
        (
            non_payroll[
                "amount_comparison"
            ]
            - non_payroll[
                "amount_base"
            ]
        ).sum()
    )

    bridge = {
        "headcount": headcount_impact,
        "employee_cost":
            employee_cost_impact,
        "non_payroll":
            non_payroll_impact,
    }

    base_opex = float(
        merged[
            "amount_base"
        ].sum()
    )

    comparison_opex = float(
        merged[
            "amount_comparison"
        ].sum()
    )

    opex_variance = (
        comparison_opex
        - base_opex
    )

    bridge_total = sum(
        bridge.values()
    )

    residual = (
        opex_variance
        - bridge_total
    )

    if abs(
        residual
    ) > RECONCILIATION_TOLERANCE_NOK:

        raise ValueError(
            "OPEX bridge failed reconciliation. "
            f"Residual: NOK {residual:,.2f}"
        )

    return {
        "bridge": bridge,
        "base_opex": base_opex,
        "comparison_opex":
            comparison_opex,
        "opex_variance":
            opex_variance,
        "reconciliation_residual":
            residual,
    }


# ============================================================
# COMMERCIAL GROUP ANALYSIS
# ============================================================

def rank_commercial_groups(
    merged: pd.DataFrame,
    group_column: str,
    top_n: int,
) -> list[dict]:
    """
    Rank countries or products by absolute Gross Profit
    variance and identify the largest underlying driver.
    """

    records = []

    for value, group in merged.groupby(
        group_column
    ):

        bridge = calculate_commercial_bridge(
            group
        )

        gp_drivers = (
            bridge[
                "gross_profit_bridge"
            ]
        )

        largest_driver = max(
            gp_drivers,
            key=lambda key: abs(
                gp_drivers[key]
            ),
        )

        records.append(
            {
                group_column: str(value),
                "revenue_variance_nok": float(
                    bridge[
                        "revenue_variance"
                    ]
                ),
                "gross_profit_variance_nok": float(
                    bridge[
                        "gross_profit_variance"
                    ]
                ),
                "largest_gp_driver":
                    largest_driver,
                "largest_gp_driver_impact_nok":
                    float(
                        gp_drivers[
                            largest_driver
                        ]
                    ),
            }
        )

    records.sort(
        key=lambda record: abs(
            record[
                "gross_profit_variance_nok"
            ]
        ),
        reverse=True,
    )

    return records[
        :top_n
    ]


# ============================================================
# OPEX GROUP ANALYSIS
# ============================================================

def rank_opex_groups(
    merged: pd.DataFrame,
    group_column: str,
    top_n: int,
) -> list[dict]:
    """
    Rank OPEX accounts or cost centres by absolute variance.
    """

    records = []

    grouped = (
        merged.groupby(
            group_column,
            as_index=False,
        )
        .agg(
            base_opex=(
                "amount_base",
                "sum",
            ),
            comparison_opex=(
                "amount_comparison",
                "sum",
            ),
        )
    )

    grouped["variance"] = (
        grouped[
            "comparison_opex"
        ]
        - grouped[
            "base_opex"
        ]
    )

    for row in grouped.itertuples():

        records.append(
            {
                group_column: str(
                    getattr(
                        row,
                        group_column,
                    )
                ),
                "base_opex_nok": float(
                    row.base_opex
                ),
                "comparison_opex_nok": float(
                    row.comparison_opex
                ),
                "opex_variance_nok": float(
                    row.variance
                ),
                "ebitda_impact_nok": float(
                    -row.variance
                ),
            }
        )

    records.sort(
        key=lambda record: abs(
            record[
                "opex_variance_nok"
            ]
        ),
        reverse=True,
    )

    return records[
        :top_n
    ]


# ============================================================
# MATERIALITY
# ============================================================

def calculate_materiality_threshold(
    ebitda_variance: float,
) -> float:
    """
    Dynamic management materiality threshold.

    A driver must be at least:

        absolute NOK threshold

    and large enough relative to total EBITDA variance.
    """

    relative_threshold = (
        abs(ebitda_variance)
        * MATERIALITY_EBITDA_SHARE
    )

    return max(
        MATERIALITY_ABSOLUTE_NOK,
        relative_threshold,
    )


def is_material(
    value: float,
    threshold: float,
) -> bool:

    return (
        abs(value)
        >= threshold
    )


# ============================================================
# EBITDA BRIDGE
# ============================================================

def build_ebitda_bridge(
    commercial: dict,
    opex: dict,
    materiality_threshold: float,
) -> list[dict]:
    """
    Build the final reconciled EBITDA bridge.

    Gross Profit drivers feed EBITDA directly.

    OPEX increases reduce EBITDA, therefore their sign is
    inverted in the EBITDA bridge.
    """

    bridge = []

    # --------------------------------------------------------
    # Commercial / Gross Profit drivers
    # --------------------------------------------------------

    for driver, impact in (
        commercial[
            "gross_profit_bridge"
        ].items()
    ):

        bridge.append(
            {
                "driver": driver,
                "category": "commercial",
                "impact_nok": float(
                    impact
                ),
                "material": is_material(
                    impact,
                    materiality_threshold,
                ),
            }
        )

    # --------------------------------------------------------
    # OPEX drivers
    # --------------------------------------------------------

    for driver, opex_impact in (
        opex["bridge"].items()
    ):

        ebitda_impact = (
            -opex_impact
        )

        bridge.append(
            {
                "driver": driver,
                "category": "opex",
                "impact_nok": float(
                    ebitda_impact
                ),
                "material": is_material(
                    ebitda_impact,
                    materiality_threshold,
                ),
            }
        )

    bridge.sort(
        key=lambda record: abs(
            record["impact_nok"]
        ),
        reverse=True,
    )

    return bridge


# ============================================================
# ANALYSIS BUILDER
# ============================================================

def build_analysis(
    comparison_name: str,
    base_label: str,
    comparison_label: str,
    base_sales: pd.DataFrame,
    comparison_sales: pd.DataFrame,
    base_opex: pd.DataFrame,
    comparison_opex: pd.DataFrame,
) -> dict:
    """
    Build one complete deterministic FP&A analysis.
    """

    merged_sales = (
        merge_sales_scenarios(
            base_sales,
            comparison_sales,
        )
    )

    merged_opex = (
        merge_opex_scenarios(
            base_opex,
            comparison_opex,
        )
    )

    commercial = (
        calculate_commercial_bridge(
            merged_sales
        )
    )

    opex = calculate_opex_bridge(
        merged_opex
    )

    base_ebitda = (
        commercial[
            "base_gross_profit"
        ]
        - opex[
            "base_opex"
        ]
    )

    comparison_ebitda = (
        commercial[
            "comparison_gross_profit"
        ]
        - opex[
            "comparison_opex"
        ]
    )

    ebitda_variance = (
        comparison_ebitda
        - base_ebitda
    )

    base_ebitda_margin = (
        base_ebitda
        / commercial[
            "base_revenue"
        ]
    )

    comparison_ebitda_margin = (
        comparison_ebitda
        / commercial[
            "comparison_revenue"
        ]
    )

    materiality_threshold = (
        calculate_materiality_threshold(
            ebitda_variance
        )
    )

    ebitda_bridge = (
        build_ebitda_bridge(
            commercial,
            opex,
            materiality_threshold,
        )
    )

    bridge_total = sum(
        item["impact_nok"]
        for item in ebitda_bridge
    )

    ebitda_residual = (
        ebitda_variance
        - bridge_total
    )

    if abs(
        ebitda_residual
    ) > RECONCILIATION_TOLERANCE_NOK:

        raise ValueError(
            f"{comparison_name}: EBITDA bridge "
            "failed reconciliation. "
            f"Residual NOK "
            f"{ebitda_residual:,.2f}"
        )

    # --------------------------------------------------------
    # Materiality flags for detailed rankings
    # --------------------------------------------------------

    countries = rank_commercial_groups(
        merged_sales,
        "country",
        TOP_N_COUNTRIES,
    )

    products = rank_commercial_groups(
        merged_sales,
        "product",
        TOP_N_PRODUCTS,
    )

    opex_accounts = rank_opex_groups(
        merged_opex,
        "account",
        TOP_N_OPEX_ACCOUNTS,
    )

    cost_centres = rank_opex_groups(
        merged_opex,
        "cost_centre",
        TOP_N_COST_CENTRES,
    )

    for record in countries:

        record["material"] = is_material(
            record[
                "gross_profit_variance_nok"
            ],
            materiality_threshold,
        )

    for record in products:

        record["material"] = is_material(
            record[
                "gross_profit_variance_nok"
            ],
            materiality_threshold,
        )

    for record in opex_accounts:

        record["material"] = is_material(
            record[
                "ebitda_impact_nok"
            ],
            materiality_threshold,
        )

    for record in cost_centres:

        record["material"] = is_material(
            record[
                "ebitda_impact_nok"
            ],
            materiality_threshold,
        )

    top_drivers = (
        ebitda_bridge[
            :TOP_N_DRIVERS
        ]
    )

    return {
        "metadata": {
            "comparison":
                comparison_name,
            "base_scenario":
                base_label,
            "comparison_scenario":
                comparison_label,
            "finance_engine":
                "deterministic",
        },

        "summary": {
            "base_revenue_nok": float(
                commercial[
                    "base_revenue"
                ]
            ),
            "comparison_revenue_nok": float(
                commercial[
                    "comparison_revenue"
                ]
            ),
            "revenue_variance_nok": float(
                commercial[
                    "revenue_variance"
                ]
            ),

            "base_gross_profit_nok": float(
                commercial[
                    "base_gross_profit"
                ]
            ),
            "comparison_gross_profit_nok": float(
                commercial[
                    "comparison_gross_profit"
                ]
            ),
            "gross_profit_variance_nok": float(
                commercial[
                    "gross_profit_variance"
                ]
            ),

            "base_opex_nok": float(
                opex[
                    "base_opex"
                ]
            ),
            "comparison_opex_nok": float(
                opex[
                    "comparison_opex"
                ]
            ),
            "opex_variance_nok": float(
                opex[
                    "opex_variance"
                ]
            ),

            "base_ebitda_nok": float(
                base_ebitda
            ),
            "comparison_ebitda_nok": float(
                comparison_ebitda
            ),
            "ebitda_variance_nok": float(
                ebitda_variance
            ),

            "base_ebitda_margin": float(
                base_ebitda_margin
            ),
            "comparison_ebitda_margin": float(
                comparison_ebitda_margin
            ),
            "ebitda_margin_variance_pp": float(
                (
                    comparison_ebitda_margin
                    - base_ebitda_margin
                )
                * 100
            ),
        },

        "materiality": {
            "absolute_threshold_nok": float(
                MATERIALITY_ABSOLUTE_NOK
            ),
            "ebitda_share_threshold": float(
                MATERIALITY_EBITDA_SHARE
            ),
            "effective_threshold_nok": float(
                materiality_threshold
            ),
        },

        "commercial_bridge": {
            "revenue": {
                key: float(value)
                for key, value
                in commercial[
                    "revenue_bridge"
                ].items()
            },
            "gross_profit": {
                key: float(value)
                for key, value
                in commercial[
                    "gross_profit_bridge"
                ].items()
            },
            "revenue_reconciliation_residual_nok":
                float(
                    commercial[
                        "revenue_reconciliation_residual"
                    ]
                ),
            "gross_profit_reconciliation_residual_nok":
                float(
                    commercial[
                        "gross_profit_reconciliation_residual"
                    ]
                ),
        },

        "opex_bridge": {
            key: {
                "opex_variance_nok": float(
                    value
                ),
                "ebitda_impact_nok": float(
                    -value
                ),
            }
            for key, value
            in opex[
                "bridge"
            ].items()
        },

        "ebitda_bridge":
            ebitda_bridge,

        "top_ebitda_drivers":
            top_drivers,

        "rankings": {
            "countries":
                countries,
            "products":
                products,
            "opex_accounts":
                opex_accounts,
            "cost_centres":
                cost_centres,
        },

        "reconciliation": {
            "ebitda_bridge_total_nok": float(
                bridge_total
            ),
            "reported_ebitda_variance_nok": float(
                ebitda_variance
            ),
            "residual_nok": float(
                ebitda_residual
            ),
            "passed": bool(
                abs(
                    ebitda_residual
                )
                <= RECONCILIATION_TOLERANCE_NOK
            ),
        },
    }


# ============================================================
# JSON OUTPUT
# ============================================================

def save_analysis(
    analysis: dict,
    filename: str,
) -> Path:
    """
    Save structured deterministic analysis as JSON.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_DIRECTORY
        / filename
    )

    path.write_text(
        json.dumps(
            analysis,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def format_nok_m(
    value: float,
) -> str:

    return (
        f"NOK "
        f"{value / 1_000_000:,.1f}m"
    )


def driver_label(
    driver: str,
) -> str:

    labels = {
        "volume_mix": "Volume & mix",
        "list_price": "List price",
        "discount": "Discount",
        "unit_cost": "Unit cost",
        "headcount": "Headcount",
        "employee_cost": "Employee cost",
        "non_payroll": "Non-payroll OPEX",
    }

    return labels.get(
        driver,
        driver,
    )


def print_analysis(
    analysis: dict,
) -> None:
    """
    Print management-readable deterministic analysis.
    """

    metadata = analysis[
        "metadata"
    ]

    summary = analysis[
        "summary"
    ]

    materiality = analysis[
        "materiality"
    ]

    print(
        "\n"
        + "=" * 100
    )

    print(
        metadata[
            "comparison"
        ]
    )

    print(
        "=" * 100
    )

    print(
        f"Revenue variance:      "
        f"{format_nok_m(summary['revenue_variance_nok'])}"
    )

    print(
        f"Gross Profit variance: "
        f"{format_nok_m(summary['gross_profit_variance_nok'])}"
    )

    print(
        f"OPEX variance:         "
        f"{format_nok_m(summary['opex_variance_nok'])}"
    )

    print(
        f"EBITDA variance:       "
        f"{format_nok_m(summary['ebitda_variance_nok'])}"
    )

    print(
        f"EBITDA margin change:  "
        f"{summary['ebitda_margin_variance_pp']:.2f} pp"
    )

    print(
        f"Materiality threshold: "
        f"{format_nok_m(materiality['effective_threshold_nok'])}"
    )

    print(
        "\nEBITDA Bridge"
    )

    print(
        "-" * 100
    )

    for item in analysis[
        "ebitda_bridge"
    ]:

        material_marker = (
            "MATERIAL"
            if item["material"]
            else ""
        )

        print(
            f"{driver_label(item['driver']):<22} | "
            f"{format_nok_m(item['impact_nok']):>15} | "
            f"{material_marker}"
        )

    print(
        "-" * 100
    )

    reconciliation = analysis[
        "reconciliation"
    ]

    print(
        f"Bridge total:          "
        f"{format_nok_m(reconciliation['ebitda_bridge_total_nok'])}"
    )

    print(
        f"Reported EBITDA var:   "
        f"{format_nok_m(reconciliation['reported_ebitda_variance_nok'])}"
    )

    print(
        f"Residual:              "
        f"NOK {reconciliation['residual_nok']:,.2f}"
    )

    print(
        "Reconciliation:        "
        + (
            "PASSED"
            if reconciliation[
                "passed"
            ]
            else "FAILED"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    data = load_all_data()

    cutoff = pd.Timestamp(
        ACTUAL_THROUGH
    )

    # ========================================================
    # FULL-YEAR LATEST FORECAST VS BUDGET
    # ========================================================

    forecast_analysis = build_analysis(
        comparison_name=(
            COMPARISON_FORECAST_VS_BUDGET
        ),
        base_label="Budget",
        comparison_label=(
            "Latest Forecast"
        ),
        base_sales=(
            data[
                "budget_sales"
            ]
        ),
        comparison_sales=(
            data[
                "forecast_sales"
            ]
        ),
        base_opex=(
            data[
                "budget_opex"
            ]
        ),
        comparison_opex=(
            data[
                "forecast_opex"
            ]
        ),
    )

    forecast_output_path = save_analysis(
        forecast_analysis,
        FORECAST_ANALYSIS_FILENAME,
    )

    # ========================================================
    # JAN-AUG ACTUAL VS BUDGET
    # ========================================================

    budget_sales_ytd = (
        data[
            "budget_sales"
        ][
            data[
                "budget_sales"
            ]["date"]
            <= cutoff
        ]
        .copy()
    )

    actual_sales_ytd = (
        data[
            "actual_sales"
        ][
            (
                data[
                    "actual_sales"
                ]["date"].dt.year
                == 2026
            )
            & (
                data[
                    "actual_sales"
                ]["date"]
                <= cutoff
            )
        ]
        .copy()
    )

    budget_opex_ytd = (
        data[
            "budget_opex"
        ][
            data[
                "budget_opex"
            ]["date"]
            <= cutoff
        ]
        .copy()
    )

    actual_opex_ytd = (
        data[
            "actual_opex"
        ][
            (
                data[
                    "actual_opex"
                ]["date"].dt.year
                == 2026
            )
            & (
                data[
                    "actual_opex"
                ]["date"]
                <= cutoff
            )
        ]
        .copy()
    )

    ytd_analysis = build_analysis(
        comparison_name=(
            COMPARISON_ACTUAL_YTD_VS_BUDGET
        ),
        base_label="Budget YTD",
        comparison_label="Actual YTD",
        base_sales=budget_sales_ytd,
        comparison_sales=actual_sales_ytd,
        base_opex=budget_opex_ytd,
        comparison_opex=actual_opex_ytd,
    )

    ytd_output_path = save_analysis(
        ytd_analysis,
        YTD_ANALYSIS_FILENAME,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\nNorthstar Systems AS"
    )

    print(
        "Deterministic Finance Analysis Engine"
    )

    print_analysis(
        forecast_analysis
    )

    print_analysis(
        ytd_analysis
    )

    print(
        "\nStructured analysis outputs:"
    )

    print(
        forecast_output_path
    )

    print(
        ytd_output_path
    )

    print(
        "\nFinance Analysis Engine: PASSED"
    )


if __name__ == "__main__":
    main()