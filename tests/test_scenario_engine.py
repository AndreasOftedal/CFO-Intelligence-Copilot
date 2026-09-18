from __future__ import annotations

import pandas as pd
import pytest

from src.scenario.scenario_engine import (
    ScenarioInputs,
    build_opex_scenario,
    build_sales_scenario,
    run_scenario,
    run_sensitivity,
)


def make_sales() -> pd.DataFrame:
    rows = [
        {
            "date": "2026-01-01",
            "country": "Norway",
            "product": "EdgeHub Pro",
            "product_family": "EdgeHub",
            "segment": "Enterprise",
            "units": 100.0,
            "list_price": 1000.0,
            "discount_rate": 0.10,
            "unit_cost": 500.0,
        },
        {
            "date": "2026-01-01",
            "country": "Germany",
            "product": "Sensor X",
            "product_family": "Sensor",
            "segment": "Distributor",
            "units": 200.0,
            "list_price": 500.0,
            "discount_rate": 0.05,
            "unit_cost": 250.0,
        },
    ]

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(
        frame["date"]
    )

    gross_sales = (
        frame["units"]
        * frame["list_price"]
    ).round(2)

    discount_value = (
        gross_sales
        * frame["discount_rate"]
    ).round(2)

    frame["net_revenue"] = (
        gross_sales
        - discount_value
    ).round(2)

    frame["gross_profit"] = (
        frame["net_revenue"]
        - (
            frame["units"]
            * frame["unit_cost"]
        ).round(2)
    ).round(2)

    return frame


def make_opex() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "cost_centre": "Product & R&D",
                "account": "Payroll",
                "headcount": 10.0,
                "average_employee_cost": 10_000.0,
                "amount": 100_000.0,
            },
            {
                "date": "2026-01-01",
                "cost_centre": "Sales",
                "account": "Marketing",
                "headcount": 0.0,
                "average_employee_cost": 0.0,
                "amount": 50_000.0,
            },
        ]
    )

    frame["date"] = pd.to_datetime(
        frame["date"]
    )

    return frame


def test_zero_scenario_returns_zero_changes():
    result = run_scenario(
        make_sales(),
        make_opex(),
        ScenarioInputs(),
    )

    summary = result["summary"]

    assert summary[
        "revenue_change_nok"
    ] == pytest.approx(0.0)

    assert summary[
        "gross_profit_change_nok"
    ] == pytest.approx(0.0)

    assert summary[
        "opex_change_nok"
    ] == pytest.approx(0.0)

    assert summary[
        "ebitda_change_nok"
    ] == pytest.approx(0.0)

    assert result[
        "reconciliation"
    ]["passed"] is True


def test_price_increase_improves_revenue_gp_and_ebitda():
    result = run_scenario(
        make_sales(),
        make_opex(),
        ScenarioInputs(
            price_pct=0.10
        ),
    )

    summary = result["summary"]

    assert summary[
        "revenue_change_nok"
    ] > 0

    assert summary[
        "gross_profit_change_nok"
    ] > 0

    assert summary[
        "opex_change_nok"
    ] == pytest.approx(0.0)

    assert summary[
        "ebitda_change_nok"
    ] == pytest.approx(
        summary[
            "gross_profit_change_nok"
        ]
    )


def test_unit_cost_increase_reduces_gp_and_ebitda_only():
    result = run_scenario(
        make_sales(),
        make_opex(),
        ScenarioInputs(
            unit_cost_pct=0.10
        ),
    )

    summary = result["summary"]

    assert summary[
        "revenue_change_nok"
    ] == pytest.approx(0.0)

    assert summary[
        "gross_profit_change_nok"
    ] < 0

    assert summary[
        "opex_change_nok"
    ] == pytest.approx(0.0)

    assert summary[
        "ebitda_change_nok"
    ] < 0


def test_non_payroll_opex_increase_reduces_ebitda():
    result = run_scenario(
        make_sales(),
        make_opex(),
        ScenarioInputs(
            non_payroll_opex_pct=0.20
        ),
    )

    summary = result["summary"]

    assert summary[
        "revenue_change_nok"
    ] == pytest.approx(0.0)

    assert summary[
        "gross_profit_change_nok"
    ] == pytest.approx(0.0)

    assert summary[
        "opex_change_nok"
    ] > 0

    assert summary[
        "ebitda_change_nok"
    ] == pytest.approx(
        -summary[
            "opex_change_nok"
        ]
    )


def test_headcount_increase_recalculates_payroll():
    baseline = make_opex()

    scenario = build_opex_scenario(
        baseline,
        ScenarioInputs(
            headcount_pct=0.10
        ),
    )

    payroll = scenario[
        scenario["account"] == "Payroll"
    ].iloc[0]

    assert payroll[
        "headcount"
    ] == pytest.approx(11.0)

    assert payroll[
        "amount"
    ] == pytest.approx(110_000.0)

    marketing = scenario[
        scenario["account"] == "Marketing"
    ].iloc[0]

    assert marketing[
        "amount"
    ] == pytest.approx(50_000.0)


def test_volume_scenario_recalculates_financial_columns():
    baseline = make_sales()

    scenario = build_sales_scenario(
        baseline,
        ScenarioInputs(
            volume_pct=0.10
        ),
    )

    assert scenario[
        "units"
    ].sum() == pytest.approx(
        baseline["units"].sum()
        * 1.10
    )

    assert scenario[
        "net_revenue"
    ].sum() > baseline[
        "net_revenue"
    ].sum()

    assert scenario[
        "gross_profit"
    ].sum() > baseline[
        "gross_profit"
    ].sum()


def test_source_frames_are_not_mutated():
    sales = make_sales()
    opex = make_opex()

    sales_original = sales.copy(
        deep=True
    )
    opex_original = opex.copy(
        deep=True
    )

    run_scenario(
        sales,
        opex,
        ScenarioInputs(
            volume_pct=0.05,
            price_pct=0.03,
            unit_cost_pct=0.02,
            headcount_pct=0.04,
            non_payroll_opex_pct=0.06,
        ),
    )

    pd.testing.assert_frame_equal(
        sales,
        sales_original,
    )

    pd.testing.assert_frame_equal(
        opex,
        opex_original,
    )


def test_combined_scenario_reconciles():
    result = run_scenario(
        make_sales(),
        make_opex(),
        ScenarioInputs(
            volume_pct=0.08,
            price_pct=0.04,
            unit_cost_pct=0.03,
            headcount_pct=0.05,
            non_payroll_opex_pct=0.10,
        ),
    )

    assert result[
        "reconciliation"
    ]["passed"] is True

    assert abs(
        result[
            "reconciliation"
        ]["residual_nok"]
    ) < 1e-6


def test_sensitivity_returns_one_record_per_value():
    records = run_sensitivity(
        make_sales(),
        make_opex(),
        driver="price",
        values=[
            -0.10,
            0.0,
            0.10,
        ],
    )

    assert len(records) == 3

    assert [
        row["input_change_pct"]
        for row in records
    ] == pytest.approx(
        [
            -10.0,
            0.0,
            10.0,
        ]
    )

    assert records[0][
        "ebitda_change_nok"
    ] < 0

    assert records[1][
        "ebitda_change_nok"
    ] == pytest.approx(0.0)

    assert records[2][
        "ebitda_change_nok"
    ] > 0


def test_invalid_driver_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unsupported sensitivity driver",
    ):
        run_sensitivity(
            make_sales(),
            make_opex(),
            driver="magic",
            values=[0.0],
        )


def test_invalid_below_minus_100_percent_is_rejected():
    with pytest.raises(
        ValueError,
        match="greater than -100%",
    ):
        ScenarioInputs(
            volume_pct=-1.0
        ).validate()

def test_integer_opex_dtypes_support_fractional_scenario_changes():
    baseline = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "cost_centre": "Product & R&D",
                "account": "Payroll",
                "headcount": 8,
                "average_employee_cost": 10_000,
                "amount": 80_000,
            },
            {
                "date": "2026-01-01",
                "cost_centre": "Sales",
                "account": "Marketing",
                "headcount": 0,
                "average_employee_cost": 0,
                "amount": 50_000,
            },
        ]
    )

    baseline["date"] = pd.to_datetime(
        baseline["date"]
    )

    assert str(
        baseline["headcount"].dtype
    ).startswith("int")

    scenario = build_opex_scenario(
        baseline,
        ScenarioInputs(
            headcount_pct=0.04,
            non_payroll_opex_pct=0.05,
        ),
    )

    payroll = scenario[
        scenario["account"] == "Payroll"
    ].iloc[0]

    marketing = scenario[
        scenario["account"] == "Marketing"
    ].iloc[0]

    assert payroll[
        "headcount"
    ] == pytest.approx(8.32)

    assert payroll[
        "amount"
    ] == pytest.approx(83_200.0)

    assert marketing[
        "amount"
    ] == pytest.approx(52_500.0)

    assert scenario[
        "headcount"
    ].dtype.kind == "f"

    assert scenario[
        "amount"
    ].dtype.kind == "f"

