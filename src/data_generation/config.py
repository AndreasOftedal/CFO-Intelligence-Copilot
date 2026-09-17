from dataclasses import dataclass


# ============================================================
# GLOBAL PROJECT SETTINGS
# ============================================================

COMPANY_NAME = "Northstar Systems AS"
REPORTING_CURRENCY = "NOK"

ACTUAL_START = "2024-01-01"
ACTUAL_END = "2026-08-01"

BUDGET_START = "2026-01-01"
BUDGET_END = "2026-12-01"

FORECAST_START = "2026-01-01"
FORECAST_END = "2026-12-01"

# Makes synthetic data reproducible.
RANDOM_SEED = 42


# ============================================================
# PRODUCTS
# ============================================================

@dataclass(frozen=True)
class Product:
    name: str
    family: str
    base_list_price: float
    base_unit_cost: float
    monthly_base_units: int


PRODUCTS = [
    Product(
        name="EdgeHub Pro",
        family="Control Systems",
        base_list_price=42000,
        base_unit_cost=23000,
        monthly_base_units=90,
    ),
    Product(
        name="EdgeHub Core",
        family="Control Systems",
        base_list_price=28000,
        base_unit_cost=14500,
        monthly_base_units=140,
    ),
    Product(
        name="Control Mini",
        family="Control Systems",
        base_list_price=14500,
        base_unit_cost=7200,
        monthly_base_units=180,
    ),
    Product(
        name="Sensor X Pro",
        family="Sensor Systems",
        base_list_price=11500,
        base_unit_cost=5200,
        monthly_base_units=220,
    ),
    Product(
        name="Sensor X",
        family="Sensor Systems",
        base_list_price=7600,
        base_unit_cost=3200,
        monthly_base_units=300,
    ),
    Product(
        name="Sensor Lite",
        family="Sensor Systems",
        base_list_price=3900,
        base_unit_cost=1550,
        monthly_base_units=400,
    ),
    Product(
        name="Service Kit",
        family="Accessories & Services",
        base_list_price=4800,
        base_unit_cost=1450,
        monthly_base_units=260,
    ),
    Product(
        name="Connectivity Pack",
        family="Accessories & Services",
        base_list_price=2900,
        base_unit_cost=850,
        monthly_base_units=330,
    ),
]


# ============================================================
# MARKETS
# ============================================================

@dataclass(frozen=True)
class Market:
    name: str
    volume_factor: float
    price_factor: float
    annual_growth_rate: float


MARKETS = [
    Market(
        name="Norway",
        volume_factor=1.00,
        price_factor=1.00,
        annual_growth_rate=0.06,
    ),
    Market(
        name="Sweden",
        volume_factor=0.75,
        price_factor=0.98,
        annual_growth_rate=0.08,
    ),
    Market(
        name="Denmark",
        volume_factor=0.55,
        price_factor=1.02,
        annual_growth_rate=0.05,
    ),
    Market(
        name="Germany",
        volume_factor=1.35,
        price_factor=0.96,
        annual_growth_rate=0.11,
    ),
    Market(
        name="Netherlands",
        volume_factor=0.65,
        price_factor=0.99,
        annual_growth_rate=0.09,
    ),
    Market(
        name="United Kingdom",
        volume_factor=0.85,
        price_factor=1.04,
        annual_growth_rate=0.07,
    ),
]


# ============================================================
# CUSTOMER SEGMENTS
# ============================================================

@dataclass(frozen=True)
class Segment:
    name: str
    base_discount_rate: float


SEGMENTS = [
    Segment(
        name="Enterprise",
        base_discount_rate=0.07,
    ),
    Segment(
        name="SME",
        base_discount_rate=0.04,
    ),
    Segment(
        name="Distributor",
        base_discount_rate=0.14,
    ),
    Segment(
        name="Systems Integrator",
        base_discount_rate=0.10,
    ),
]


# ============================================================
# MARKET-SPECIFIC CUSTOMER MIX
# ============================================================
#
# The customer mix differs by country.
#
# This creates realistic structural differences in discount rates,
# margins and channel exposure across markets.
#
# Each country's shares must sum to 1.00.
# ============================================================

MARKET_SEGMENT_MIX = {
    "Norway": {
        "Enterprise": 0.40,
        "SME": 0.25,
        "Distributor": 0.18,
        "Systems Integrator": 0.17,
    },
    "Sweden": {
        "Enterprise": 0.34,
        "SME": 0.24,
        "Distributor": 0.24,
        "Systems Integrator": 0.18,
    },
    "Denmark": {
        "Enterprise": 0.36,
        "SME": 0.24,
        "Distributor": 0.22,
        "Systems Integrator": 0.18,
    },
    "Germany": {
        "Enterprise": 0.24,
        "SME": 0.15,
        "Distributor": 0.38,
        "Systems Integrator": 0.23,
    },
    "Netherlands": {
        "Enterprise": 0.30,
        "SME": 0.20,
        "Distributor": 0.31,
        "Systems Integrator": 0.19,
    },
    "United Kingdom": {
        "Enterprise": 0.33,
        "SME": 0.21,
        "Distributor": 0.27,
        "Systems Integrator": 0.19,
    },
}


# ============================================================
# MARKET-SPECIFIC PRODUCT DEMAND
# ============================================================
#
# A factor above 1.00 means that the product is relatively more
# popular in that market.
#
# A factor below 1.00 means relatively weaker demand.
#
# This gives us realistic product-mix effects instead of assuming
# every country buys exactly the same portfolio.
# ============================================================

MARKET_PRODUCT_DEMAND = {
    "Norway": {
        "EdgeHub Pro": 1.10,
        "EdgeHub Core": 1.05,
        "Control Mini": 0.95,
        "Sensor X Pro": 1.00,
        "Sensor X": 1.00,
        "Sensor Lite": 0.90,
        "Service Kit": 1.05,
        "Connectivity Pack": 1.00,
    },
    "Sweden": {
        "EdgeHub Pro": 0.95,
        "EdgeHub Core": 1.00,
        "Control Mini": 1.05,
        "Sensor X Pro": 1.05,
        "Sensor X": 1.10,
        "Sensor Lite": 1.05,
        "Service Kit": 0.95,
        "Connectivity Pack": 1.00,
    },
    "Denmark": {
        "EdgeHub Pro": 0.90,
        "EdgeHub Core": 0.95,
        "Control Mini": 1.05,
        "Sensor X Pro": 1.00,
        "Sensor X": 1.05,
        "Sensor Lite": 1.10,
        "Service Kit": 1.00,
        "Connectivity Pack": 1.05,
    },
    "Germany": {
        "EdgeHub Pro": 1.18,
        "EdgeHub Core": 1.12,
        "Control Mini": 1.05,
        "Sensor X Pro": 1.02,
        "Sensor X": 0.95,
        "Sensor Lite": 0.88,
        "Service Kit": 0.92,
        "Connectivity Pack": 0.90,
    },
    "Netherlands": {
        "EdgeHub Pro": 0.92,
        "EdgeHub Core": 0.95,
        "Control Mini": 1.00,
        "Sensor X Pro": 1.08,
        "Sensor X": 1.12,
        "Sensor Lite": 1.08,
        "Service Kit": 1.00,
        "Connectivity Pack": 1.05,
    },
    "United Kingdom": {
        "EdgeHub Pro": 1.05,
        "EdgeHub Core": 1.08,
        "Control Mini": 1.00,
        "Sensor X Pro": 1.02,
        "Sensor X": 0.98,
        "Sensor Lite": 0.95,
        "Service Kit": 1.02,
        "Connectivity Pack": 1.00,
    },
}


# ============================================================
# SEASONALITY
# ============================================================
#
# Northstar has weaker summer activity and stronger Q4 demand.
# ============================================================

MONTHLY_SEASONALITY = {
    1: 0.86,
    2: 0.91,
    3: 1.00,
    4: 0.97,
    5: 1.03,
    6: 1.08,
    7: 0.79,
    8: 0.88,
    9: 1.06,
    10: 1.11,
    11: 1.15,
    12: 1.06,
}


# ============================================================
# OPERATING EXPENSE ACCOUNTS
# ============================================================

OPEX_ACCOUNTS = [
    "Payroll",
    "Contractors",
    "Marketing",
    "Software",
    "Travel",
    "Facilities",
    "Professional Services",
    "R&D",
    "Other OPEX",
]


# ============================================================
# COST CENTRES
# ============================================================

COST_CENTRES = [
    "Sales",
    "Operations",
    "Product & R&D",
    "Finance",
    "People",
    "Technology",
    "Corporate",
]


# ============================================================
# DATA QUALITY RULES
# ============================================================

REQUIRED_SALES_COLUMNS = [
    "date",
    "scenario",
    "country",
    "product",
    "product_family",
    "segment",
    "units",
    "list_price",
    "discount_rate",
    "gross_sales",
    "discount_value",
    "net_revenue",
    "unit_cost",
    "cogs",
    "gross_profit",
    "gross_margin_pct",
]


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_configuration() -> None:
    """
    Validate the synthetic business configuration before data generation.

    The project deliberately fails early if the commercial assumptions
    are internally inconsistent.
    """

    market_names = {
        market.name
        for market in MARKETS
    }

    product_names = {
        product.name
        for product in PRODUCTS
    }

    segment_names = {
        segment.name
        for segment in SEGMENTS
    }

    # --------------------------------------------------------
    # Basic product economics
    # --------------------------------------------------------

    for product in PRODUCTS:
        if product.base_list_price <= 0:
            raise ValueError(
                f"{product.name}: list price must be positive."
            )

        if product.base_unit_cost <= 0:
            raise ValueError(
                f"{product.name}: unit cost must be positive."
            )

        if product.base_unit_cost >= product.base_list_price:
            raise ValueError(
                f"{product.name}: unit cost must be below list price."
            )

        if product.monthly_base_units <= 0:
            raise ValueError(
                f"{product.name}: base units must be positive."
            )

    # --------------------------------------------------------
    # Market assumptions
    # --------------------------------------------------------

    for market in MARKETS:
        if market.volume_factor <= 0:
            raise ValueError(
                f"{market.name}: volume factor must be positive."
            )

        if market.price_factor <= 0:
            raise ValueError(
                f"{market.name}: price factor must be positive."
            )

        if market.annual_growth_rate <= -1:
            raise ValueError(
                f"{market.name}: invalid annual growth rate."
            )

    # --------------------------------------------------------
    # Segment discounts
    # --------------------------------------------------------

    for segment in SEGMENTS:
        if not 0 <= segment.base_discount_rate <= 0.25:
            raise ValueError(
                f"{segment.name}: invalid base discount rate."
            )

    # --------------------------------------------------------
    # Customer mix coverage
    # --------------------------------------------------------

    if set(MARKET_SEGMENT_MIX) != market_names:
        raise ValueError(
            "MARKET_SEGMENT_MIX does not contain exactly "
            "the configured markets."
        )

    for market_name, segment_mix in MARKET_SEGMENT_MIX.items():

        if set(segment_mix) != segment_names:
            raise ValueError(
                f"{market_name}: customer mix does not contain "
                "exactly the configured segments."
            )

        mix_total = sum(segment_mix.values())

        if abs(mix_total - 1.0) > 1e-9:
            raise ValueError(
                f"{market_name}: customer mix must sum to 1.00, "
                f"but sums to {mix_total:.6f}."
            )

        if any(
            share < 0
            for share in segment_mix.values()
        ):
            raise ValueError(
                f"{market_name}: negative segment share detected."
            )

    # --------------------------------------------------------
    # Product-demand coverage
    # --------------------------------------------------------

    if set(MARKET_PRODUCT_DEMAND) != market_names:
        raise ValueError(
            "MARKET_PRODUCT_DEMAND does not contain exactly "
            "the configured markets."
        )

    for market_name, product_demand in MARKET_PRODUCT_DEMAND.items():

        if set(product_demand) != product_names:
            raise ValueError(
                f"{market_name}: product demand configuration "
                "does not contain exactly the configured products."
            )

        if any(
            factor <= 0
            for factor in product_demand.values()
        ):
            raise ValueError(
                f"{market_name}: product-demand factors "
                "must be positive."
            )

    # --------------------------------------------------------
    # Seasonality
    # --------------------------------------------------------

    if set(MONTHLY_SEASONALITY) != set(range(1, 13)):
        raise ValueError(
            "MONTHLY_SEASONALITY must contain months 1 through 12."
        )

    if any(
        factor <= 0
        for factor in MONTHLY_SEASONALITY.values()
    ):
        raise ValueError(
            "Seasonality factors must be positive."
        )


# Validate configuration whenever this module is imported.
validate_configuration()