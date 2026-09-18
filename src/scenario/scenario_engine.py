from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable
import sys

import pandas as pd


# ---------------------------------------------------------------------
# Existing deterministic finance engine
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIRECTORY = PROJECT_ROOT / "src" / "analysis"

# variance_engine.py currently imports analysis_config as a local module.
# Adding the existing analysis directory to sys.path lets this scenario
# layer reuse the production finance engine without duplicating formulas.
if str(ANALYSIS_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(ANALYSIS_DIRECTORY),
    )

from src.analysis import variance_engine as finance_engine  # noqa: E402


# ---------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------

SCENARIO_COMPARISON_NAME = "scenario_vs_baseline"

SCENARIO_DRIVER_FIELDS = {
    "volume": "volume_pct",
    "price": "price_pct",
    "unit_cost": "unit_cost_pct",
    "headcount": "headcount_pct",
    "non_payroll_opex": "non_payroll_opex_pct",
}

SCENARIO_DRIVER_LABELS = {
    "volume": "Volume",
    "price": "List price",
    "unit_cost": "Unit cost",
    "headcount": "Headcount",
    "non_payroll_opex": "Non-payroll OPEX",
}


@dataclass(frozen=True)
class ScenarioInputs:
    """
    Relative scenario assumptions expressed as decimal changes.

    Examples:
        0.05  = +5%
        -0.10 = -10%

    The scenario engine intentionally changes only explicit operational
    drivers. Discount rate and average employee cost remain unchanged in
    this first version of the Scenario & Sensitivity Lab.
    """

    volume_pct: float = 0.0
    price_pct: float = 0.0
    unit_cost_pct: float = 0.0
    headcount_pct: float = 0.0
    non_payroll_opex_pct: float = 0.0

    def validate(self) -> None:
        values = asdict(self)

        for field_name, value in values.items():
            if not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"{field_name} must be numeric."
                )

            if value <= -1.0:
                raise ValueError(
                    f"{field_name} must be greater than -100%."
                )

            if value > 5.0:
                raise ValueError(
                    f"{field_name} must not exceed +500%."
                )

    def as_percentages(self) -> dict[str, float]:
        return {
            field_name: float(value) * 100
            for field_name, value
            in asdict(self).items()
        }


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> None:
    missing = sorted(
        required_columns
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: "
            + ", ".join(missing)
        )


def _safe_change_pct(
    baseline: float,
    scenario: float,
) -> float | None:
    if baseline == 0:
        return None

    return (
        scenario
        / baseline
        - 1
    )


# ---------------------------------------------------------------------
# Scenario transformation
# ---------------------------------------------------------------------

def build_sales_scenario(
    baseline_sales: pd.DataFrame,
    inputs: ScenarioInputs,
) -> pd.DataFrame:
    """
    Apply commercial assumptions to a copy of the baseline sales data.

    Revenue and gross profit are recalculated using the exact production
    finance-engine helpers, including the same rounding logic.
    """

    inputs.validate()

    _require_columns(
        baseline_sales,
        {
            "units",
            "list_price",
            "discount_rate",
            "unit_cost",
            "net_revenue",
            "gross_profit",
        },
        "baseline_sales",
    )

    scenario = baseline_sales.copy(
        deep=True
    )

    scenario["units"] = (
        scenario["units"].astype(float)
        * (1.0 + inputs.volume_pct)
    )

    scenario["list_price"] = (
        scenario["list_price"].astype(float)
        * (1.0 + inputs.price_pct)
    ).round(6)

    scenario["unit_cost"] = (
        scenario["unit_cost"].astype(float)
        * (1.0 + inputs.unit_cost_pct)
    ).round(6)

    scenario["net_revenue"] = (
        finance_engine.calculate_revenue(
            scenario["units"],
            scenario["list_price"],
            scenario["discount_rate"],
        )
    )

    scenario["gross_profit"] = (
        finance_engine.calculate_gross_profit(
            scenario["units"],
            scenario["list_price"],
            scenario["discount_rate"],
            scenario["unit_cost"],
        )
    )

    return scenario


def build_opex_scenario(
    baseline_opex: pd.DataFrame,
    inputs: ScenarioInputs,
) -> pd.DataFrame:
    """
    Apply workforce and non-payroll assumptions to a copy of baseline OPEX.

    Payroll amount is recalculated from headcount × average employee cost.
    Non-payroll accounts are scaled directly.
    """

    inputs.validate()

    _require_columns(
        baseline_opex,
        {
            "account",
            "amount",
            "headcount",
            "average_employee_cost",
        },
        "baseline_opex",
    )

    scenario = baseline_opex.copy(
        deep=True
    )

    # Real Northstar OPEX files store headcount as integers. Scenario
    # assumptions can produce fractional analytical headcount values
    # (for example +4%), so the mutable numeric columns must be promoted
    # to float before masked assignments. This also keeps amount and
    # employee-cost calculations dtype-safe across pandas versions.
    for column in (
        "amount",
        "headcount",
        "average_employee_cost",
    ):
        scenario[column] = pd.to_numeric(
            scenario[column],
            errors="raise",
        ).astype(float)

    payroll_mask = (
        scenario["account"]
        == "Payroll"
    )

    non_payroll_mask = (
        ~payroll_mask
    )

    scenario.loc[
        payroll_mask,
        "headcount",
    ] = (
        scenario.loc[
            payroll_mask,
            "headcount",
        ].astype(float)
        * (1.0 + inputs.headcount_pct)
    )

    scenario.loc[
        payroll_mask,
        "amount",
    ] = (
        scenario.loc[
            payroll_mask,
            "headcount",
        ].astype(float)
        * scenario.loc[
            payroll_mask,
            "average_employee_cost",
        ].astype(float)
    ).round(2)

    scenario.loc[
        non_payroll_mask,
        "amount",
    ] = (
        scenario.loc[
            non_payroll_mask,
            "amount",
        ].astype(float)
        * (
            1.0
            + inputs.non_payroll_opex_pct
        )
    ).round(2)

    return scenario


# ---------------------------------------------------------------------
# Scenario analysis
# ---------------------------------------------------------------------

def run_scenario(
    baseline_sales: pd.DataFrame,
    baseline_opex: pd.DataFrame,
    inputs: ScenarioInputs,
    baseline_label: str = "Latest Forecast",
    scenario_label: str = "Scenario",
) -> dict:
    """
    Run one deterministic what-if scenario against a supplied baseline.

    The existing build_analysis() function performs all bridge,
    reconciliation, EBITDA and margin calculations.
    """

    inputs.validate()

    scenario_sales = build_sales_scenario(
        baseline_sales,
        inputs,
    )

    scenario_opex = build_opex_scenario(
        baseline_opex,
        inputs,
    )

    analysis = finance_engine.build_analysis(
        comparison_name=(
            SCENARIO_COMPARISON_NAME
        ),
        base_label=baseline_label,
        comparison_label=scenario_label,
        base_sales=baseline_sales,
        comparison_sales=scenario_sales,
        base_opex=baseline_opex,
        comparison_opex=scenario_opex,
    )

    summary = analysis["summary"]

    baseline_revenue = float(
        summary["base_revenue_nok"]
    )
    scenario_revenue = float(
        summary["comparison_revenue_nok"]
    )

    baseline_gp = float(
        summary["base_gross_profit_nok"]
    )
    scenario_gp = float(
        summary["comparison_gross_profit_nok"]
    )

    baseline_opex_value = float(
        summary["base_opex_nok"]
    )
    scenario_opex_value = float(
        summary["comparison_opex_nok"]
    )

    baseline_ebitda = float(
        summary["base_ebitda_nok"]
    )
    scenario_ebitda = float(
        summary["comparison_ebitda_nok"]
    )

    baseline_margin = float(
        summary["base_ebitda_margin"]
    )
    scenario_margin = float(
        summary["comparison_ebitda_margin"]
    )

    return {
        "metadata": {
            "comparison": (
                SCENARIO_COMPARISON_NAME
            ),
            "baseline_label": baseline_label,
            "scenario_label": scenario_label,
            "finance_engine": "deterministic",
            "source_engine": (
                "src.analysis.variance_engine"
            ),
        },
        "inputs": {
            "decimal": asdict(inputs),
            "percentage_points": (
                inputs.as_percentages()
            ),
        },
        "summary": {
            "baseline_revenue_nok":
                baseline_revenue,
            "scenario_revenue_nok":
                scenario_revenue,
            "revenue_change_nok": (
                scenario_revenue
                - baseline_revenue
            ),
            "revenue_change_pct":
                _safe_change_pct(
                    baseline_revenue,
                    scenario_revenue,
                ),
            "baseline_gross_profit_nok":
                baseline_gp,
            "scenario_gross_profit_nok":
                scenario_gp,
            "gross_profit_change_nok": (
                scenario_gp
                - baseline_gp
            ),
            "gross_profit_change_pct":
                _safe_change_pct(
                    baseline_gp,
                    scenario_gp,
                ),
            "baseline_opex_nok":
                baseline_opex_value,
            "scenario_opex_nok":
                scenario_opex_value,
            "opex_change_nok": (
                scenario_opex_value
                - baseline_opex_value
            ),
            "opex_change_pct":
                _safe_change_pct(
                    baseline_opex_value,
                    scenario_opex_value,
                ),
            "baseline_ebitda_nok":
                baseline_ebitda,
            "scenario_ebitda_nok":
                scenario_ebitda,
            "ebitda_change_nok": (
                scenario_ebitda
                - baseline_ebitda
            ),
            "ebitda_change_pct":
                _safe_change_pct(
                    baseline_ebitda,
                    scenario_ebitda,
                ),
            "baseline_ebitda_margin":
                baseline_margin,
            "scenario_ebitda_margin":
                scenario_margin,
            "ebitda_margin_change_pp": (
                (
                    scenario_margin
                    - baseline_margin
                )
                * 100
            ),
        },
        "ebitda_bridge": (
            analysis["ebitda_bridge"]
        ),
        "commercial_bridge": (
            analysis["commercial_bridge"]
        ),
        "opex_bridge": (
            analysis["opex_bridge"]
        ),
        "reconciliation": (
            analysis["reconciliation"]
        ),
    }


def run_latest_forecast_scenario(
    inputs: ScenarioInputs,
    scenario_label: str = "Scenario",
) -> dict:
    """
    Convenience entry point for the Streamlit app.

    Uses Latest Forecast as the operational baseline.
    """

    data = finance_engine.load_all_data()

    return run_scenario(
        baseline_sales=(
            data["forecast_sales"]
        ),
        baseline_opex=(
            data["forecast_opex"]
        ),
        inputs=inputs,
        baseline_label="Latest Forecast",
        scenario_label=scenario_label,
    )


# ---------------------------------------------------------------------
# One-way sensitivity analysis
# ---------------------------------------------------------------------

def run_sensitivity(
    baseline_sales: pd.DataFrame,
    baseline_opex: pd.DataFrame,
    driver: str,
    values: Iterable[float],
    base_inputs: ScenarioInputs | None = None,
) -> list[dict]:
    """
    Run a one-way sensitivity table for one scenario driver.

    Each value is a decimal change, e.g. -0.10, 0.00, +0.10.
    Other assumptions remain fixed at base_inputs.
    """

    if driver not in SCENARIO_DRIVER_FIELDS:
        raise ValueError(
            "Unsupported sensitivity driver: "
            f"{driver}"
        )

    if base_inputs is None:
        base_inputs = ScenarioInputs()

    base_inputs.validate()

    field_name = (
        SCENARIO_DRIVER_FIELDS[
            driver
        ]
    )

    records: list[dict] = []

    for value in values:
        scenario_inputs = replace(
            base_inputs,
            **{
                field_name: float(value)
            },
        )

        result = run_scenario(
            baseline_sales=baseline_sales,
            baseline_opex=baseline_opex,
            inputs=scenario_inputs,
            baseline_label=(
                "Latest Forecast"
            ),
            scenario_label=(
                f"{SCENARIO_DRIVER_LABELS[driver]} "
                f"{float(value) * 100:+.1f}%"
            ),
        )

        summary = result["summary"]

        records.append(
            {
                "driver": driver,
                "driver_label": (
                    SCENARIO_DRIVER_LABELS[
                        driver
                    ]
                ),
                "input_change_pct": (
                    float(value)
                    * 100
                ),
                "revenue_change_nok": (
                    summary[
                        "revenue_change_nok"
                    ]
                ),
                "gross_profit_change_nok": (
                    summary[
                        "gross_profit_change_nok"
                    ]
                ),
                "opex_change_nok": (
                    summary[
                        "opex_change_nok"
                    ]
                ),
                "ebitda_change_nok": (
                    summary[
                        "ebitda_change_nok"
                    ]
                ),
                "scenario_ebitda_margin": (
                    summary[
                        "scenario_ebitda_margin"
                    ]
                ),
                "ebitda_margin_change_pp": (
                    summary[
                        "ebitda_margin_change_pp"
                    ]
                ),
            }
        )

    return records


def run_latest_forecast_sensitivity(
    driver: str,
    values: Iterable[float],
    base_inputs: ScenarioInputs | None = None,
) -> list[dict]:
    """
    Convenience wrapper for one-way sensitivity against Latest Forecast.
    """

    data = finance_engine.load_all_data()

    return run_sensitivity(
        baseline_sales=(
            data["forecast_sales"]
        ),
        baseline_opex=(
            data["forecast_opex"]
        ),
        driver=driver,
        values=values,
        base_inputs=base_inputs,
    )
