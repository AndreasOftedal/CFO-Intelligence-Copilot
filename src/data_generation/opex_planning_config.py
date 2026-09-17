from config import (
    COST_CENTRES,
    OPEX_ACCOUNTS,
)


# ============================================================
# NORTHSTAR SYSTEMS AS
# OPEX BUDGET & FORECAST ASSUMPTIONS
# ============================================================
#
# Budget:
#   Built from 2025 Actual OPEX and prepared before 2026.
#
# Latest Forecast:
#   Prepared after August 2026 close.
#
#   January-August = Final Actual OPEX
#   September-December = updated management estimate
#
# ============================================================


# ============================================================
# FORECAST CUT-OFF
# ============================================================

ACTUAL_THROUGH = "2026-08-01"

FORECAST_FUTURE_START = "2026-09-01"


# ============================================================
# 2026 BUDGET — PAYROLL
# ============================================================
#
# Headcount assumptions represent management's original
# staffing plan versus the corresponding 2025 level.
# ============================================================

BUDGET_HEADCOUNT_GROWTH = {
    "Sales": 0.07,
    "Operations": 0.04,
    "Product & R&D": 0.09,
    "Finance": 0.03,
    "People": 0.05,
    "Technology": 0.08,
    "Corporate": 0.02,
}


# Planned increase in average employee cost.
BUDGET_EMPLOYEE_COST_INCREASE = 0.045


# ============================================================
# 2026 BUDGET — NON-PAYROLL
# ============================================================
#
# Growth versus corresponding 2025 Actual month.
#
# Using the same prior-year month preserves the realistic
# seasonal pattern already present in the baseline data.
# ============================================================

BUDGET_NON_PAYROLL_GROWTH = {
    "Contractors": 0.06,
    "Marketing": 0.05,
    "Software": 0.08,
    "Travel": 0.05,
    "Facilities": 0.035,
    "Professional Services": 0.04,
    "R&D": 0.07,
    "Other OPEX": 0.04,
}


# ============================================================
# LATEST FORECAST — HEADCOUNT REVISION
# ============================================================
#
# Applies only to September-December 2026.
#
# Values represent changes versus Budget.
#
# Examples:
#   -0.03 = 3% below planned headcount
#    0.02 = 2% above planned headcount
# ============================================================

FORECAST_HEADCOUNT_REVISION = {
    "Sales": -0.03,
    "Operations": -0.01,
    "Product & R&D": 0.02,
    "Finance": 0.00,
    "People": 0.00,
    "Technology": 0.015,
    "Corporate": 0.00,
}


# Small revision to average employee cost versus Budget.
FORECAST_EMPLOYEE_COST_REVISION = 0.005


# ============================================================
# LATEST FORECAST — NON-PAYROLL REVISION
# ============================================================
#
# Applies September-December only.
#
# Contractors and R&D remain somewhat elevated.
# Travel benefits from tighter cost control.
# ============================================================

FORECAST_NON_PAYROLL_REVISION = {
    "Contractors": 0.06,
    "Marketing": 0.02,
    "Software": 0.025,
    "Travel": -0.04,
    "Facilities": 0.005,
    "Professional Services": 0.02,
    "R&D": 0.04,
    "Other OPEX": 0.00,
}


# ============================================================
# SPECIFIC FORECAST ADJUSTMENTS
# ============================================================
#
# NOTE-017 states that elevated Product & R&D contractor
# activity is expected to continue at a lower level through
# September before normalizing.
#
# We therefore carry a smaller, explicit amount into the
# September forecast.
#
# IMPORTANT:
#
# OPEX-002 Marketing is NOT propagated as a specific forecast
# adjustment because no documentary evidence explains the
# August overspend or indicates that it will continue.
# ============================================================

FORECAST_SPECIFIC_OPEX_ADJUSTMENTS = [

    {
        "adjustment_id": "OPEX-FCAST-001",
        "date": "2026-09-01",
        "cost_centre": "Product & R&D",
        "account": "Contractors",
        "amount_change": 900_000,
        "evidence_id": "NOTE-017",
        "description": (
            "September Product & R&D contractor forecast "
            "increased by NOK 0.9m to reflect continued "
            "prototype and testing activity at a lower "
            "level than August."
        ),
    },
]


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_opex_planning_configuration() -> None:
    """
    Validate OPEX Budget and Forecast assumptions.
    """

    expected_cost_centres = set(
        COST_CENTRES
    )

    expected_non_payroll_accounts = (
        set(OPEX_ACCOUNTS)
        - {"Payroll"}
    )

    # --------------------------------------------------------
    # Budget headcount coverage
    # --------------------------------------------------------

    if (
        set(BUDGET_HEADCOUNT_GROWTH)
        != expected_cost_centres
    ):

        raise ValueError(
            "BUDGET_HEADCOUNT_GROWTH must contain "
            "exactly the configured cost centres."
        )

    # --------------------------------------------------------
    # Forecast headcount coverage
    # --------------------------------------------------------

    if (
        set(FORECAST_HEADCOUNT_REVISION)
        != expected_cost_centres
    ):

        raise ValueError(
            "FORECAST_HEADCOUNT_REVISION must contain "
            "exactly the configured cost centres."
        )

    # --------------------------------------------------------
    # Non-payroll coverage
    # --------------------------------------------------------

    if (
        set(BUDGET_NON_PAYROLL_GROWTH)
        != expected_non_payroll_accounts
    ):

        raise ValueError(
            "BUDGET_NON_PAYROLL_GROWTH must contain "
            "exactly the non-payroll OPEX accounts."
        )

    if (
        set(FORECAST_NON_PAYROLL_REVISION)
        != expected_non_payroll_accounts
    ):

        raise ValueError(
            "FORECAST_NON_PAYROLL_REVISION must contain "
            "exactly the non-payroll OPEX accounts."
        )

    # --------------------------------------------------------
    # Range checks
    # --------------------------------------------------------

    for cost_centre, value in (
        BUDGET_HEADCOUNT_GROWTH.items()
    ):

        if value <= -1:

            raise ValueError(
                f"{cost_centre}: invalid Budget "
                "headcount growth."
            )

    for cost_centre, value in (
        FORECAST_HEADCOUNT_REVISION.items()
    ):

        if value <= -1:

            raise ValueError(
                f"{cost_centre}: invalid Forecast "
                "headcount revision."
            )

    if BUDGET_EMPLOYEE_COST_INCREASE <= -1:

        raise ValueError(
            "Invalid Budget employee-cost increase."
        )

    if FORECAST_EMPLOYEE_COST_REVISION <= -1:

        raise ValueError(
            "Invalid Forecast employee-cost revision."
        )

    for account, value in (
        BUDGET_NON_PAYROLL_GROWTH.items()
    ):

        if value <= -1:

            raise ValueError(
                f"{account}: invalid Budget "
                "non-payroll growth."
            )

    for account, value in (
        FORECAST_NON_PAYROLL_REVISION.items()
    ):

        if value <= -1:

            raise ValueError(
                f"{account}: invalid Forecast "
                "non-payroll revision."
            )

    # --------------------------------------------------------
    # Specific adjustments
    # --------------------------------------------------------

    adjustment_ids = [
        adjustment["adjustment_id"]
        for adjustment
        in FORECAST_SPECIFIC_OPEX_ADJUSTMENTS
    ]

    if (
        len(adjustment_ids)
        != len(set(adjustment_ids))
    ):

        raise ValueError(
            "Duplicate OPEX Forecast adjustment IDs."
        )

    for adjustment in (
        FORECAST_SPECIFIC_OPEX_ADJUSTMENTS
    ):

        if (
            adjustment["cost_centre"]
            not in expected_cost_centres
        ):

            raise ValueError(
                f"{adjustment['adjustment_id']}: "
                "unknown cost centre."
            )

        if (
            adjustment["account"]
            not in expected_non_payroll_accounts
        ):

            raise ValueError(
                f"{adjustment['adjustment_id']}: "
                "invalid adjustment account."
            )

        if adjustment["amount_change"] == 0:

            raise ValueError(
                f"{adjustment['adjustment_id']}: "
                "amount change cannot be zero."
            )

        if not adjustment["evidence_id"]:

            raise ValueError(
                f"{adjustment['adjustment_id']}: "
                "documented forecast adjustment "
                "must have evidence_id."
            )


validate_opex_planning_configuration()