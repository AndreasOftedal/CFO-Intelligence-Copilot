from __future__ import annotations

import pytest

from src.ai.scenario_brief import (
    ScenarioBriefError,
    build_scenario_payload,
    generate_scenario_brief,
    validate_scenario_brief,
)


def make_scenario_result() -> dict:
    return {
        "metadata": {
            "baseline_label": (
                "Latest Forecast"
            ),
            "scenario_label": (
                "Management Scenario"
            ),
        },
        "inputs": {
            "percentage_points": {
                "volume_pct": 0.0,
                "price_pct": 0.0,
                "discount_rate_delta": 2.0,
                "unit_cost_pct": 0.0,
                "headcount_pct": 0.0,
                "non_payroll_opex_pct": 0.0,
            },
        },
        "summary": {
            "revenue_change_nok": -30_050_000.0,
            "gross_profit_change_nok": -30_050_000.0,
            "opex_change_nok": 0.0,
            "ebitda_change_nok": -30_050_000.0,
            "ebitda_margin_change_pp": -1.87,
        },
        "ebitda_bridge": [
            {
                "driver": "discount",
                "category": "commercial",
                "impact_nok": -30_050_000.0,
                "material": True,
            },
            {
                "driver": "volume_mix",
                "category": "commercial",
                "impact_nok": 0.0,
                "material": False,
            },
        ],
        "reconciliation": {
            "passed": True,
            "residual_nok": 0.05,
        },
    }


def test_payload_contains_only_controlled_scenario_context():
    payload = build_scenario_payload(
        make_scenario_result()
    )

    assert (
        payload["scenario_type"]
        == "deterministic_what_if"
    )

    assert (
        payload["ground_truth_access"]
        is False
    )

    assert (
        payload["evidence_access"]
        is False
    )

    assert payload[
        "scenario_assumptions"
    ][
        "discount_rate_delta"
    ] == pytest.approx(2.0)

    assert payload[
        "calculated_outcomes"
    ][
        "ebitda_change_nok"
    ] == pytest.approx(
        -30_050_000.0
    )

    assert [
        item["driver"]
        for item
        in payload[
            "active_ebitda_drivers"
        ]
    ] == [
        "discount",
    ]

    serialized = str(
        payload
    ).lower()

    assert "management evidence" not in serialized
    assert "ground_truth_events" not in serialized


def test_dry_run_does_not_send_api_request():
    result = generate_scenario_brief(
        make_scenario_result(),
        dry_run=True,
    )

    assert result[
        "brief"
    ] is None

    assert result[
        "diagnostics"
    ][
        "api_request_sent"
    ] is False

    assert result[
        "diagnostics"
    ][
        "ground_truth_access"
    ] is False

    assert result[
        "diagnostics"
    ][
        "evidence_access"
    ] is False

    assert result[
        "diagnostics"
    ][
        "active_driver_ids"
    ] == [
        "discount",
    ]


def test_valid_brief_is_accepted():
    payload = build_scenario_payload(
        make_scenario_result()
    )

    result = validate_scenario_brief(
        {
            "headline": (
                "Discount pressure reduces profitability"
            ),
            "executive_summary": (
                "Higher discount is the dominant mechanical "
                "headwind to revenue, gross profit and EBITDA, "
                "while operating expenses remain unchanged."
            ),
            "driver_ids": [
                "discount",
            ],
            "management_takeaways": [
                (
                    "Discount is the only active EBITDA "
                    "driver in this scenario."
                ),
                (
                    "The profitability effect is commercial "
                    "rather than operating-expense driven."
                ),
            ],
            "limitations": [
                (
                    "This is a deterministic what-if result, "
                    "not a probability-weighted forecast."
                ),
            ],
        },
        payload,
    )

    assert result[
        "driver_ids"
    ] == [
        "discount",
    ]


def test_mechanical_customer_pricing_language_is_allowed():
    payload = build_scenario_payload(
        make_scenario_result()
    )

    result = validate_scenario_brief(
        {
            "headline": (
                "Discount pressure reduces profitability"
            ),
            "executive_summary": (
                "The explicit discount assumption changes "
                "customer pricing mechanically and reduces "
                "revenue, gross profit and EBITDA."
            ),
            "driver_ids": [
                "discount",
            ],
            "management_takeaways": [],
            "limitations": [],
        },
        payload,
    )

    assert result[
        "driver_ids"
    ] == [
        "discount",
    ]


def test_unknown_driver_is_rejected():
    payload = build_scenario_payload(
        make_scenario_result()
    )

    with pytest.raises(
        ScenarioBriefError,
        match="unsupported driver IDs",
    ):
        validate_scenario_brief(
            {
                "headline": "Profitability pressure",
                "executive_summary": (
                    "The scenario reduces EBITDA."
                ),
                "driver_ids": [
                    "customer_churn",
                ],
                "management_takeaways": [],
                "limitations": [],
            },
            payload,
        )


def test_numeric_values_in_ai_narrative_are_rejected():
    payload = build_scenario_payload(
        make_scenario_result()
    )

    with pytest.raises(
        ScenarioBriefError,
        match="must not contain numeric values",
    ):
        validate_scenario_brief(
            {
                "headline": (
                    "EBITDA pressure"
                ),
                "executive_summary": (
                    "Revenue falls by 30 million under "
                    "the selected scenario."
                ),
                "driver_ids": [
                    "discount",
                ],
                "management_takeaways": [],
                "limitations": [],
            },
            payload,
        )


def test_unsupported_operational_story_is_rejected():
    payload = build_scenario_payload(
        make_scenario_result()
    )

    with pytest.raises(
        ScenarioBriefError,
        match="unsupported operational explanation",
    ):
        validate_scenario_brief(
            {
                "headline": (
                    "Customer demand weakens profitability"
                ),
                "executive_summary": (
                    "Higher discount is the active driver."
                ),
                "driver_ids": [
                    "discount",
                ],
                "management_takeaways": [],
                "limitations": [],
            },
            payload,
        )


def test_active_scenario_requires_at_least_one_driver_id():
    payload = build_scenario_payload(
        make_scenario_result()
    )

    with pytest.raises(
        ScenarioBriefError,
        match="omitted all active EBITDA drivers",
    ):
        validate_scenario_brief(
            {
                "headline": "Profitability changes",
                "executive_summary": (
                    "The scenario changes the financial outcome."
                ),
                "driver_ids": [],
                "management_takeaways": [],
                "limitations": [],
            },
            payload,
        )


def test_unchanged_scenario_accepts_empty_driver_list():
    scenario = make_scenario_result()

    scenario[
        "ebitda_bridge"
    ] = []

    scenario[
        "summary"
    ] = {
        "revenue_change_nok": 0.0,
        "gross_profit_change_nok": 0.0,
        "opex_change_nok": 0.0,
        "ebitda_change_nok": 0.0,
        "ebitda_margin_change_pp": 0.0,
    }

    payload = build_scenario_payload(
        scenario
    )

    result = validate_scenario_brief(
        {
            "headline": (
                "Scenario remains unchanged"
            ),
            "executive_summary": (
                "The selected assumptions leave the "
                "financial outcome unchanged from baseline."
            ),
            "driver_ids": [],
            "management_takeaways": [],
            "limitations": [
                (
                    "This is a deterministic what-if result."
                ),
            ],
        },
        payload,
    )

    assert result[
        "driver_ids"
    ] == []


def test_missing_api_key_is_rejected(
    monkeypatch,
):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    with pytest.raises(
        ScenarioBriefError,
        match="OPENAI_API_KEY",
    ):
        generate_scenario_brief(
            make_scenario_result(),
        )
