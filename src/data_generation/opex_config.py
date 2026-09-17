from dataclasses import dataclass

from config import (
    COST_CENTRES,
    OPEX_ACCOUNTS,
)


# ============================================================
# NORTHSTAR SYSTEMS AS
# OPEX MODEL CONFIGURATION
# ============================================================
#
# The OPEX model is designed to create a realistic FP&A
# environment.
#
# Payroll is driver-based:
#
#     Headcount
#     x average monthly employee cost
#
# Non-payroll OPEX is generated from account / cost-centre
# assumptions with controlled growth, seasonality and
# operational volatility.
#
# ============================================================


# ============================================================
# GLOBAL OPEX ASSUMPTIONS
# ============================================================

OPEX_ACTUAL_START = "2024-01-01"
OPEX_ACTUAL_END = "2026-08-01"

OPEX_RANDOM_SEED = 84

ANNUAL_SALARY_INFLATION = 0.045

PAYROLL_VOLATILITY = 0.005

NON_PAYROLL_VOLATILITY = 0.035


# ============================================================
# HEADCOUNT MODEL
# ============================================================

@dataclass(frozen=True)
class HeadcountAssumption:
    cost_centre: str
    base_headcount_2024: int
    average_monthly_employee_cost_2024: float
    annual_headcount_growth: float


HEADCOUNT_ASSUMPTIONS = [

    HeadcountAssumption(
        cost_centre="Sales",
        base_headcount_2024=42,
        average_monthly_employee_cost_2024=95000,
        annual_headcount_growth=0.08,
    ),

    HeadcountAssumption(
        cost_centre="Operations",
        base_headcount_2024=34,
        average_monthly_employee_cost_2024=85000,
        annual_headcount_growth=0.05,
    ),

    HeadcountAssumption(
        cost_centre="Product & R&D",
        base_headcount_2024=46,
        average_monthly_employee_cost_2024=105000,
        annual_headcount_growth=0.10,
    ),

    HeadcountAssumption(
        cost_centre="Finance",
        base_headcount_2024=10,
        average_monthly_employee_cost_2024=100000,
        annual_headcount_growth=0.04,
    ),

    HeadcountAssumption(
        cost_centre="People",
        base_headcount_2024=7,
        average_monthly_employee_cost_2024=95000,
        annual_headcount_growth=0.06,
    ),

    HeadcountAssumption(
        cost_centre="Technology",
        base_headcount_2024=15,
        average_monthly_employee_cost_2024=100000,
        annual_headcount_growth=0.09,
    ),

    HeadcountAssumption(
        cost_centre="Corporate",
        base_headcount_2024=8,
        average_monthly_employee_cost_2024=125000,
        annual_headcount_growth=0.03,
    ),
]


# ============================================================
# NON-PAYROLL OPEX MODEL
# ============================================================

@dataclass(frozen=True)
class ExpenseAssumption:
    cost_centre: str
    account: str
    monthly_base_cost_2024: float
    annual_growth_rate: float
    seasonality_profile: str


NON_PAYROLL_ASSUMPTIONS = [

    # --------------------------------------------------------
    # Contractors
    # --------------------------------------------------------

    ExpenseAssumption(
        cost_centre="Operations",
        account="Contractors",
        monthly_base_cost_2024=500000,
        annual_growth_rate=0.04,
        seasonality_profile="flat",
    ),

    ExpenseAssumption(
        cost_centre="Product & R&D",
        account="Contractors",
        monthly_base_cost_2024=1600000,
        annual_growth_rate=0.07,
        seasonality_profile="project",
    ),

    ExpenseAssumption(
        cost_centre="Technology",
        account="Contractors",
        monthly_base_cost_2024=600000,
        annual_growth_rate=0.06,
        seasonality_profile="project",
    ),

    # --------------------------------------------------------
    # Marketing
    # --------------------------------------------------------

    ExpenseAssumption(
        cost_centre="Sales",
        account="Marketing",
        monthly_base_cost_2024=2600000,
        annual_growth_rate=0.06,
        seasonality_profile="commercial",
    ),

    # --------------------------------------------------------
    # Software
    # --------------------------------------------------------

    ExpenseAssumption(
        cost_centre="Technology",
        account="Software",
        monthly_base_cost_2024=900000,
        annual_growth_rate=0.08,
        seasonality_profile="flat",
    ),

    ExpenseAssumption(
        cost_centre="Product & R&D",
        account="Software",
        monthly_base_cost_2024=300000,
        annual_growth_rate=0.08,
        seasonality_profile="flat",
    ),

    ExpenseAssumption(
        cost_centre="Finance",
        account="Software",
        monthly_base_cost_2024=150000,
        annual_growth_rate=0.05,
        seasonality_profile="flat",
    ),

    # --------------------------------------------------------
    # Travel
    # --------------------------------------------------------

    ExpenseAssumption(
        cost_centre="Sales",
        account="Travel",
        monthly_base_cost_2024=550000,
        annual_growth_rate=0.05,
        seasonality_profile="travel",
    ),

    ExpenseAssumption(
        cost_centre="Operations",
        account="Travel",
        monthly_base_cost_2024=150000,
        annual_growth_rate=0.04,
        seasonality_profile="travel",
    ),

    ExpenseAssumption(
        cost_centre="Corporate",
        account="Travel",
        monthly_base_cost_2024=100000,
        annual_growth_rate=0.04,
        seasonality_profile="travel",
    ),

    # --------------------------------------------------------
    # Facilities
    # --------------------------------------------------------

    ExpenseAssumption(
        cost_centre="Corporate",
        account="Facilities",
        monthly_base_cost_2024=1250000,
        annual_growth_rate=0.035,
        seasonality_profile="flat",
    ),

    # --------------------------------------------------------
    # Professional Services
    # --------------------------------------------------------

    ExpenseAssumption(
        cost_centre="Finance",
        account="Professional Services",
        monthly_base_cost_2024=250000,
        annual_growth_rate=0.04,
        seasonality_profile="year_end",
    ),

    ExpenseAssumption(
        cost_centre="People",
        account="Professional Services",
        monthly_base_cost_2024=180000,
        annual_growth_rate=0.04,
        seasonality_profile="flat",
    ),

    ExpenseAssumption(
        cost_centre="Corporate",
        account="Professional Services",
        monthly_base_cost_2024=350000,
        annual_growth_rate=0.05,
        seasonality_profile="year_end",
    ),

    # --------------------------------------------------------
    # R&D / testing expenditure
    # --------------------------------------------------------

    ExpenseAssumption(
        cost_centre="Product & R&D",
        account="R&D",
        monthly_base_cost_2024=1400000,
        annual_growth_rate=0.08,
        seasonality_profile="project",
    ),

    # --------------------------------------------------------
    # Other OPEX
    # --------------------------------------------------------

    ExpenseAssumption(
        cost_centre="Sales",
        account="Other OPEX",
        monthly_base_cost_2024=180000,
        annual_growth_rate=0.04,
        seasonality_profile="flat",
    ),

    ExpenseAssumption(
        cost_centre="Operations",
        account="Other OPEX",
        monthly_base_cost_2024=160000,
        annual_growth_rate=0.04,
        seasonality_profile="flat",
    ),

    ExpenseAssumption(
        cost_centre="Product & R&D",
        account="Other OPEX",
        monthly_base_cost_2024=140000,
        annual_growth_rate=0.04,
        seasonality_profile="flat",
    ),

    ExpenseAssumption(
        cost_centre="Finance",
        account="Other OPEX",
        monthly_base_cost_2024=80000,
        annual_growth_rate=0.03,
        seasonality_profile="flat",
    ),

    ExpenseAssumption(
        cost_centre="People",
        account="Other OPEX",
        monthly_base_cost_2024=70000,
        annual_growth_rate=0.03,
        seasonality_profile="flat",
    ),

    ExpenseAssumption(
        cost_centre="Technology",
        account="Other OPEX",
        monthly_base_cost_2024=100000,
        annual_growth_rate=0.04,
        seasonality_profile="flat",
    ),

    ExpenseAssumption(
        cost_centre="Corporate",
        account="Other OPEX",
        monthly_base_cost_2024=120000,
        annual_growth_rate=0.03,
        seasonality_profile="flat",
    ),
]


# ============================================================
# SEASONALITY PROFILES
# ============================================================
#
# These do not represent overall business seasonality.
#
# They describe the expected timing of specific OPEX types.
# ============================================================

OPEX_SEASONALITY = {

    "flat": {
        1: 1.00,
        2: 1.00,
        3: 1.00,
        4: 1.00,
        5: 1.00,
        6: 1.00,
        7: 1.00,
        8: 1.00,
        9: 1.00,
        10: 1.00,
        11: 1.00,
        12: 1.00,
    },

    "commercial": {
        1: 0.90,
        2: 0.95,
        3: 1.10,
        4: 1.05,
        5: 1.10,
        6: 1.00,
        7: 0.65,
        8: 0.80,
        9: 1.15,
        10: 1.15,
        11: 1.10,
        12: 1.05,
    },

    "travel": {
        1: 0.85,
        2: 0.95,
        3: 1.05,
        4: 1.05,
        5: 1.10,
        6: 1.05,
        7: 0.55,
        8: 0.70,
        9: 1.15,
        10: 1.15,
        11: 1.10,
        12: 0.70,
    },

    "project": {
        1: 0.90,
        2: 0.95,
        3: 1.00,
        4: 1.05,
        5: 1.05,
        6: 1.10,
        7: 0.90,
        8: 0.95,
        9: 1.05,
        10: 1.10,
        11: 1.05,
        12: 0.90,
    },

    "year_end": {
        1: 0.90,
        2: 0.90,
        3: 1.00,
        4: 0.95,
        5: 0.95,
        6: 1.00,
        7: 0.85,
        8: 0.90,
        9: 1.00,
        10: 1.10,
        11: 1.20,
        12: 1.35,
    },
}


# ============================================================
# FUTURE BUSINESS EVENTS
# ============================================================
#
# We will later inject controlled OPEX events separately.
#
# Examples:
#
# - documented Product & R&D contractor overspend
# - unexplained Marketing overspend
#
# They are deliberately NOT hard-coded into the baseline
# OPEX assumptions.
# ============================================================


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_opex_configuration() -> None:
    """
    Validate OPEX assumptions before data generation.
    """

    valid_cost_centres = set(
        COST_CENTRES
    )

    valid_accounts = set(
        OPEX_ACCOUNTS
    )

    # --------------------------------------------------------
    # Headcount
    # --------------------------------------------------------

    headcount_cost_centres = [
        assumption.cost_centre
        for assumption in HEADCOUNT_ASSUMPTIONS
    ]

    if (
        set(headcount_cost_centres)
        != valid_cost_centres
    ):

        raise ValueError(
            "HEADCOUNT_ASSUMPTIONS must contain exactly "
            "the configured cost centres."
        )

    if (
        len(headcount_cost_centres)
        != len(set(headcount_cost_centres))
    ):

        raise ValueError(
            "Duplicate headcount cost centres detected."
        )

    for assumption in HEADCOUNT_ASSUMPTIONS:

        if assumption.base_headcount_2024 <= 0:

            raise ValueError(
                f"{assumption.cost_centre}: "
                "headcount must be positive."
            )

        if (
            assumption
            .average_monthly_employee_cost_2024
            <= 0
        ):

            raise ValueError(
                f"{assumption.cost_centre}: "
                "employee cost must be positive."
            )

        if assumption.annual_headcount_growth <= -1:

            raise ValueError(
                f"{assumption.cost_centre}: "
                "invalid headcount growth."
            )

    # --------------------------------------------------------
    # Non-payroll assumptions
    # --------------------------------------------------------

    seen_pairs = set()

    for assumption in NON_PAYROLL_ASSUMPTIONS:

        if (
            assumption.cost_centre
            not in valid_cost_centres
        ):

            raise ValueError(
                f"{assumption.cost_centre}: "
                "unknown cost centre."
            )

        if assumption.account not in valid_accounts:

            raise ValueError(
                f"{assumption.account}: "
                "unknown OPEX account."
            )

        if assumption.account == "Payroll":

            raise ValueError(
                "Payroll must be generated from "
                "the headcount model, not from "
                "NON_PAYROLL_ASSUMPTIONS."
            )

        if assumption.monthly_base_cost_2024 <= 0:

            raise ValueError(
                f"{assumption.cost_centre} / "
                f"{assumption.account}: "
                "base cost must be positive."
            )

        if assumption.annual_growth_rate <= -1:

            raise ValueError(
                f"{assumption.cost_centre} / "
                f"{assumption.account}: "
                "invalid growth rate."
            )

        if (
            assumption.seasonality_profile
            not in OPEX_SEASONALITY
        ):

            raise ValueError(
                f"{assumption.cost_centre} / "
                f"{assumption.account}: "
                "unknown seasonality profile."
            )

        pair = (
            assumption.cost_centre,
            assumption.account,
        )

        if pair in seen_pairs:

            raise ValueError(
                f"Duplicate OPEX assumption: {pair}"
            )

        seen_pairs.add(pair)

    # --------------------------------------------------------
    # Seasonality
    # --------------------------------------------------------

    for profile_name, profile in OPEX_SEASONALITY.items():

        if set(profile) != set(range(1, 13)):

            raise ValueError(
                f"{profile_name}: "
                "seasonality must contain months 1-12."
            )

        if any(
            factor <= 0
            for factor in profile.values()
        ):

            raise ValueError(
                f"{profile_name}: "
                "seasonality factors must be positive."
            )


validate_opex_configuration()