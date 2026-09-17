# ============================================================
# NORTHSTAR SYSTEMS AS
# BUDGET & FORECAST PLANNING ASSUMPTIONS
# ============================================================
#
# Budget:
#   Prepared before the start of 2026.
#   Uses 2025 Actual as the planning baseline.
#
# Latest Forecast:
#   Prepared after August 2026 close.
#
#   January-August = Actual
#   September-December = updated management estimate
#
# ============================================================


# ============================================================
# FORECAST CUT-OFF
# ============================================================

ACTUAL_THROUGH = "2026-08-01"

FORECAST_FUTURE_START = "2026-09-01"


# ============================================================
# 2026 BUDGET — GENERAL ASSUMPTIONS
# ============================================================

# Planned annual list-price increase versus 2025.
BUDGET_PRICE_INCREASE = 0.030

# Planned unit-cost increase versus 2025.
BUDGET_COST_INCREASE = 0.032


# ============================================================
# 2026 BUDGET — MARKET VOLUME GROWTH
# ============================================================
#
# These represent management's original expectations
# before the start of 2026.
# ============================================================

BUDGET_VOLUME_GROWTH = {
    "Norway": 0.07,
    "Sweden": 0.09,
    "Denmark": 0.05,
    "Germany": 0.10,
    "Netherlands": 0.10,
    "United Kingdom": 0.07,
}


# ============================================================
# 2026 BUDGET — DISCOUNT ASSUMPTIONS
# ============================================================
#
# Values are changes in percentage points versus the
# corresponding 2025 Actual discount rate.
#
# Example:
#   -0.003 = 0.3 percentage-point reduction.
#
# The budget assumes modest commercial discipline.
# ============================================================

BUDGET_DISCOUNT_CHANGE = {
    "Norway": -0.002,
    "Sweden": -0.001,
    "Denmark": -0.001,
    "Germany": -0.003,
    "Netherlands": -0.002,
    "United Kingdom": -0.001,
}


# ============================================================
# LATEST FORECAST — VOLUME REVISION
# ============================================================
#
# Applies to September-December 2026 only.
#
# Values represent revisions versus the original Budget.
#
# Germany has softened materially versus plan.
# Netherlands is outperforming.
# Other markets show smaller revisions.
# ============================================================

FORECAST_VOLUME_REVISION = {
    "Norway": -0.02,
    "Sweden": 0.00,
    "Denmark": -0.01,
    "Germany": -0.06,
    "Netherlands": 0.03,
    "United Kingdom": -0.02,
}


# ============================================================
# LATEST FORECAST — DISCOUNT REVISION
# ============================================================
#
# Percentage-point change versus Budget for
# September-December.
#
# Germany is experiencing greater commercial discount
# pressure than originally planned.
# ============================================================

FORECAST_DISCOUNT_CHANGE = {
    "Norway": 0.001,
    "Sweden": 0.001,
    "Denmark": 0.001,
    "Germany": 0.010,
    "Netherlands": 0.000,
    "United Kingdom": 0.003,
}


# ============================================================
# LATEST FORECAST — COST REVISION
# ============================================================
#
# Percentage changes versus Budget unit cost for
# September-December.
#
# These represent modest procurement inflation beyond
# the assumptions used in the original Budget.
#
# The temporary Germany EdgeHub Pro surcharge observed
# in August is NOT extended into future months because
# management expects it to normalize from September.
# ============================================================

FORECAST_COST_REVISION = {
    "Norway": 0.004,
    "Sweden": 0.005,
    "Denmark": 0.004,
    "Germany": 0.006,
    "Netherlands": 0.003,
    "United Kingdom": 0.005,
}


# ============================================================
# SPECIFIC FORECAST ADJUSTMENTS
# ============================================================
#
# These represent known timing effects identified during
# the August close.
#
# They create realistic differences between:
#
#   underlying run-rate
#   and
#   timing-related forecast changes
# ============================================================

FORECAST_SPECIFIC_ADJUSTMENTS = [

    # --------------------------------------------------------
    # UK delayed August rollout moves into September.
    # --------------------------------------------------------

    {
        "adjustment_id": "FCAST-001",
        "date": "2026-09-01",
        "country": "United Kingdom",
        "product": "Sensor X",
        "segment": "Distributor",
        "driver": "units",
        "factor": 1.22,
        "description": (
            "September volume increased to reflect "
            "the customer rollout delayed from August."
        ),
    },

    # --------------------------------------------------------
    # Norway August acceleration pulls some demand forward
    # from September.
    # --------------------------------------------------------

    {
        "adjustment_id": "FCAST-002",
        "date": "2026-09-01",
        "country": "Norway",
        "product": "Sensor X Pro",
        "segment": "Enterprise",
        "driver": "units",
        "factor": 0.80,
        "description": (
            "September volume reduced because part of "
            "the planned demand was pulled forward "
            "into August."
        ),
    },
]


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

EXPECTED_MARKETS = {
    "Norway",
    "Sweden",
    "Denmark",
    "Germany",
    "Netherlands",
    "United Kingdom",
}


def validate_planning_configuration() -> None:
    """
    Validate Budget and Forecast assumptions before
    they are used by the planning engine.
    """

    market_configs = {
        "BUDGET_VOLUME_GROWTH":
            set(BUDGET_VOLUME_GROWTH),

        "BUDGET_DISCOUNT_CHANGE":
            set(BUDGET_DISCOUNT_CHANGE),

        "FORECAST_VOLUME_REVISION":
            set(FORECAST_VOLUME_REVISION),

        "FORECAST_DISCOUNT_CHANGE":
            set(FORECAST_DISCOUNT_CHANGE),

        "FORECAST_COST_REVISION":
            set(FORECAST_COST_REVISION),
    }

    # --------------------------------------------------------
    # Market coverage
    # --------------------------------------------------------

    for config_name, markets in market_configs.items():

        if markets != EXPECTED_MARKETS:

            raise ValueError(
                f"{config_name} does not contain exactly "
                "the required markets."
            )

    # --------------------------------------------------------
    # Budget volume growth
    # --------------------------------------------------------

    for market, value in BUDGET_VOLUME_GROWTH.items():

        if value <= -1:

            raise ValueError(
                f"{market}: invalid Budget volume growth."
            )

    # --------------------------------------------------------
    # Budget discount changes
    # --------------------------------------------------------

    for market, value in BUDGET_DISCOUNT_CHANGE.items():

        if not -0.10 <= value <= 0.10:

            raise ValueError(
                f"{market}: unreasonable Budget "
                "discount assumption."
            )

    # --------------------------------------------------------
    # Forecast revisions
    # --------------------------------------------------------

    for market, value in FORECAST_VOLUME_REVISION.items():

        if value <= -1:

            raise ValueError(
                f"{market}: invalid Forecast "
                "volume revision."
            )

    for market, value in FORECAST_DISCOUNT_CHANGE.items():

        if not -0.10 <= value <= 0.10:

            raise ValueError(
                f"{market}: unreasonable Forecast "
                "discount revision."
            )

    for market, value in FORECAST_COST_REVISION.items():

        if value <= -1:

            raise ValueError(
                f"{market}: invalid Forecast "
                "cost revision."
            )

    # --------------------------------------------------------
    # Specific forecast adjustments
    # --------------------------------------------------------

    adjustment_ids = [
        adjustment["adjustment_id"]
        for adjustment in FORECAST_SPECIFIC_ADJUSTMENTS
    ]

    if len(adjustment_ids) != len(set(adjustment_ids)):

        raise ValueError(
            "Duplicate forecast adjustment IDs detected."
        )

    for adjustment in FORECAST_SPECIFIC_ADJUSTMENTS:

        if adjustment["country"] not in EXPECTED_MARKETS:

            raise ValueError(
                f"{adjustment['adjustment_id']}: "
                "unknown country."
            )

        if adjustment["driver"] != "units":

            raise ValueError(
                f"{adjustment['adjustment_id']}: "
                "unsupported forecast adjustment driver."
            )

        if adjustment["factor"] <= 0:

            raise ValueError(
                f"{adjustment['adjustment_id']}: "
                "adjustment factor must be positive."
            )


validate_planning_configuration()