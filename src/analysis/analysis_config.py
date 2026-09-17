# ============================================================
# NORTHSTAR SYSTEMS AS
# FINANCE ANALYSIS ENGINE CONFIGURATION
# ============================================================
#
# This module defines the analytical rules used by the
# deterministic Finance Analysis Engine.
#
# The LLM layer will NOT calculate financial variances.
# It will receive structured outputs produced by this engine.
#
# ============================================================


# ============================================================
# ANALYSIS PERIOD
# ============================================================

ANALYSIS_YEAR = 2026

ACTUAL_THROUGH = "2026-08-01"


# ============================================================
# COMPARISONS
# ============================================================
#
# We will initially support two core FP&A comparisons:
#
# 1. Full-Year Latest Forecast vs Budget
#
#    Question:
#       Has the full-year outlook changed?
#
# 2. Jan-Aug Actual vs Budget
#
#    Question:
#       How has the business performed YTD?
#
# ============================================================

COMPARISON_FORECAST_VS_BUDGET = (
    "latest_forecast_vs_budget"
)

COMPARISON_ACTUAL_YTD_VS_BUDGET = (
    "actual_ytd_vs_budget"
)


# ============================================================
# MATERIALITY
# ============================================================
#
# An item can be financially real without being important
# enough for management attention.
#
# The engine therefore distinguishes:
#
#   calculated variance
#
# from
#
#   material variance
#
# Materiality rules will later be used by the AI layer
# to decide which issues deserve emphasis.
# ============================================================

MATERIALITY_ABSOLUTE_NOK = 1_000_000

MATERIALITY_EBITDA_SHARE = 0.05


# ============================================================
# RANKING
# ============================================================

TOP_N_DRIVERS = 5

TOP_N_COUNTRIES = 5

TOP_N_PRODUCTS = 5

TOP_N_OPEX_ACCOUNTS = 5

TOP_N_COST_CENTRES = 5


# ============================================================
# COMMERCIAL VARIANCE BRIDGE
# ============================================================
#
# Revenue:
#
#     units
#     x list price
#     x (1 - discount rate)
#
# COGS:
#
#     units
#     x unit cost
#
# Gross Profit:
#
#     Revenue - COGS
#
#
# The bridge applies changes sequentially:
#
# 1. Volume & mix
# 2. List price
# 3. Discount
# 4. Unit cost
#
# This allows every Gross Profit variance to reconcile
# exactly to the underlying financial data.
#
# "Volume & mix" is deliberately labelled as a combined
# effect because changes occur at the detailed grain:
#
# month x country x product x segment
#
# A change in the distribution of units between products,
# markets or customer segments is therefore naturally
# embedded in the volume contribution.
# ============================================================

COMMERCIAL_BRIDGE_ORDER = [
    "volume_mix",
    "list_price",
    "discount",
    "unit_cost",
]


# ============================================================
# OPEX VARIANCE BRIDGE
# ============================================================
#
# Payroll:
#
#     headcount
#     x average employee cost
#
# Payroll variance is decomposed into:
#
# 1. Headcount
# 2. Employee cost
#
#
# Non-payroll:
#
# Variance is attributed directly to the relevant
# account / cost centre because the underlying data
# already represents the financial spending category.
# ============================================================

OPEX_BRIDGE_ORDER = [
    "headcount",
    "employee_cost",
    "non_payroll",
]


# ============================================================
# OUTPUT SETTINGS
# ============================================================

ANALYSIS_OUTPUT_DIRECTORY = (
    "outputs/analysis"
)

FORECAST_ANALYSIS_FILENAME = (
    "latest_forecast_vs_budget.json"
)

YTD_ANALYSIS_FILENAME = (
    "actual_ytd_vs_budget.json"
)


# ============================================================
# RECONCILIATION TOLERANCE
# ============================================================
#
# Financial datasets are stored to two decimal places.
#
# Variance bridges must reconcile extremely closely, but a
# small tolerance protects against insignificant floating
# point differences.
# ============================================================

RECONCILIATION_TOLERANCE_NOK = 1.00


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_analysis_configuration() -> None:
    """
    Validate analytical rules before the finance engine runs.
    """

    if MATERIALITY_ABSOLUTE_NOK <= 0:

        raise ValueError(
            "Absolute materiality must be positive."
        )

    if not 0 < MATERIALITY_EBITDA_SHARE <= 1:

        raise ValueError(
            "EBITDA-share materiality must be "
            "between 0 and 1."
        )

    ranking_values = [
        TOP_N_DRIVERS,
        TOP_N_COUNTRIES,
        TOP_N_PRODUCTS,
        TOP_N_OPEX_ACCOUNTS,
        TOP_N_COST_CENTRES,
    ]

    if any(
        value <= 0
        for value in ranking_values
    ):

        raise ValueError(
            "All ranking limits must be positive."
        )

    expected_commercial_bridge = [
        "volume_mix",
        "list_price",
        "discount",
        "unit_cost",
    ]

    if (
        COMMERCIAL_BRIDGE_ORDER
        != expected_commercial_bridge
    ):

        raise ValueError(
            "Unexpected commercial bridge order."
        )

    expected_opex_bridge = [
        "headcount",
        "employee_cost",
        "non_payroll",
    ]

    if (
        OPEX_BRIDGE_ORDER
        != expected_opex_bridge
    ):

        raise ValueError(
            "Unexpected OPEX bridge order."
        )

    if RECONCILIATION_TOLERANCE_NOK <= 0:

        raise ValueError(
            "Reconciliation tolerance must be positive."
        )


validate_analysis_configuration()