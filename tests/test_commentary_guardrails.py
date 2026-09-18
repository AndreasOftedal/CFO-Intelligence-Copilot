from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ai.cfo_commentary import (
    build_ai_payload,
    build_model_visible_payload,
    load_grounded_analysis,
    validate_commentary_against_payload,
)
from src.ai.schemas import (
    CFOCommentary,
    CalculatedObservation,
    EvidenceBackedExplanation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUNDING_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "grounding"
    / "latest_forecast_vs_budget_grounded.json"
)

AI_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "ai"
    / "latest_forecast_vs_budget_commentary.json"
)


@pytest.fixture
def grounded_data() -> dict:
    return load_grounded_analysis(
        GROUNDING_PATH
    )


@pytest.fixture
def payload(
    grounded_data: dict,
) -> dict:
    return build_ai_payload(
        grounded_data
    )


def get_material_finding(
    payload: dict,
    finding_id: str,
) -> dict:
    for finding in payload[
        "material_findings"
    ]:
        if (
            finding["finding_id"]
            == finding_id
        ):
            return finding

    raise AssertionError(
        "Finding not found in AI payload: "
        f"{finding_id}"
    )


def get_evidence_ids(
    finding: dict,
) -> set[str]:
    return {
        evidence["evidence_id"]
        for evidence
        in finding["usable_evidence"]
    }


def test_ground_truth_access_is_blocked(
    payload: dict,
) -> None:
    assert (
        payload["metadata"][
            "ground_truth_access"
        ]
        is False
    )


def test_only_directionally_approved_evidence_reaches_ai(
    payload: dict,
) -> None:
    evidence_ids = {
        evidence["evidence_id"]
        for finding
        in payload[
            "material_findings"
        ]
        for evidence
        in finding[
            "usable_evidence"
        ]
    }

    assert evidence_ids == {
        "NOTE-015",
        "NOTE-017",
    }


def test_note_014_is_excluded_from_ai_payload(
    payload: dict,
) -> None:
    evidence_ids = {
        evidence["evidence_id"]
        for finding
        in payload[
            "material_findings"
        ]
        for evidence
        in finding[
            "usable_evidence"
        ]
    }

    assert "NOTE-014" not in evidence_ids


def test_note_016_is_withheld_by_directional_guardrail(
    payload: dict,
) -> None:
    directional = payload[
        "_internal"
    ]["directional_guardrail"]

    assert (
        "NOTE-016"
        in directional[
            "withheld_evidence_ids"
        ]
    )


def test_headline_fact_ids_are_deterministic(
    payload: dict,
) -> None:
    headline_ids = {
        item["finding_id"]
        for item
        in payload[
            "headline_facts"
        ]
    }

    assert headline_ids == {
        "summary::revenue",
        "summary::gross_profit",
        "summary::opex",
        "summary::ebitda",
        "summary::ebitda_margin",
    }


def test_directional_guardrail_summary(
    payload: dict,
) -> None:
    directional = payload[
        "_internal"
    ]["directional_guardrail"]

    assert (
        directional[
            "assessment_count"
        ]
        == 5
    )

    assert (
        directional[
            "directionally_consistent_count"
        ]
        == 4
    )

    assert (
        directional[
            "offsetting_count"
        ]
        == 1
    )

    assert (
        directional[
            "direction_unknown_count"
        ]
        == 0
    )


def test_norway_acceleration_is_not_used_as_explanation(
    payload: dict,
) -> None:
    norway = get_material_finding(
        payload,
        "country::Norway",
    )

    assert (
        norway["explanation_status"]
        == "insufficient_evidence"
    )

    assert (
        norway["usable_evidence"]
        == []
    )

    assert (
        "NOTE-016"
        not in get_evidence_ids(
            norway
        )
    )


def test_uk_shipment_delay_remains_usable(
    payload: dict,
) -> None:
    uk = get_material_finding(
        payload,
        "country::United Kingdom",
    )

    assert (
        uk["explanation_status"]
        == "evidence_available"
    )

    assert (
        get_evidence_ids(uk)
        == {"NOTE-015"}
    )


def test_contractor_evidence_remains_usable(
    payload: dict,
) -> None:
    contractors = get_material_finding(
        payload,
        "opex_account::Contractors",
    )

    assert (
        contractors[
            "explanation_status"
        ]
        == "evidence_available"
    )

    assert (
        get_evidence_ids(
            contractors
        )
        == {"NOTE-017"}
    )


def test_product_r_and_d_evidence_remains_usable(
    payload: dict,
) -> None:
    product_r_and_d = (
        get_material_finding(
            payload,
            "cost_centre::Product & R&D",
        )
    )

    assert (
        product_r_and_d[
            "explanation_status"
        ]
        == "evidence_available"
    )

    assert (
        get_evidence_ids(
            product_r_and_d
        )
        == {"NOTE-017"}
    )


def test_insufficient_evidence_findings_have_no_usable_evidence(
    payload: dict,
) -> None:
    for finding in payload[
        "material_findings"
    ]:
        if (
            finding[
                "explanation_status"
            ]
            == "insufficient_evidence"
        ):
            assert (
                finding[
                    "usable_evidence"
                ]
                == []
            )


def test_internal_diagnostics_are_not_visible_to_model(
    payload: dict,
) -> None:
    model_payload = (
        build_model_visible_payload(
            payload
        )
    )

    assert "_internal" not in (
        model_payload
    )


def test_note_016_does_not_reach_model_visible_payload(
    payload: dict,
) -> None:
    model_payload = (
        build_model_visible_payload(
            payload
        )
    )

    serialized = json.dumps(
        model_payload,
        ensure_ascii=False,
    )

    assert "NOTE-016" not in serialized


def test_validator_rejects_unknown_observation_id(
    payload: dict,
) -> None:
    commentary = CFOCommentary(
        executive_summary=(
            "Test summary."
        ),
        observations=[
            CalculatedObservation(
                finding_id=(
                    "fake::finding"
                ),
                statement=(
                    "Test statement."
                ),
                source_type=(
                    "calculated_fact"
                ),
            )
        ],
        supported_explanations=[],
        unresolved_findings=[],
        management_questions=[],
    )

    with pytest.raises(
        ValueError,
        match=(
            "unauthorized finding ID"
        ),
    ):
        validate_commentary_against_payload(
            commentary,
            payload,
        )


def test_validator_rejects_explanation_for_insufficient_evidence(
    payload: dict,
) -> None:
    norway = get_material_finding(
        payload,
        "country::Norway",
    )

    assert (
        norway["explanation_status"]
        == "insufficient_evidence"
    )

    commentary = CFOCommentary(
        executive_summary=(
            "Test summary."
        ),
        observations=[],
        supported_explanations=[
            EvidenceBackedExplanation(
                finding_id=(
                    "country::Norway"
                ),
                explanation=(
                    "The shipment "
                    "acceleration explains "
                    "the Norway shortfall."
                ),
                evidence_ids=[
                    "NOTE-016"
                ],
                source_type=(
                    "evidence_supported"
                ),
            )
        ],
        unresolved_findings=[],
        management_questions=[],
    )

    with pytest.raises(
        ValueError,
        match=(
            "not authorized "
            "for explanation"
        ),
    ):
        validate_commentary_against_payload(
            commentary,
            payload,
        )


def test_validator_rejects_unauthorized_evidence_id(
    payload: dict,
) -> None:
    commentary = CFOCommentary(
        executive_summary=(
            "Test summary."
        ),
        observations=[],
        supported_explanations=[
            EvidenceBackedExplanation(
                finding_id=(
                    "country::United Kingdom"
                ),
                explanation=(
                    "Test explanation."
                ),
                evidence_ids=[
                    "NOTE-017"
                ],
                source_type=(
                    "evidence_supported"
                ),
            )
        ],
        unresolved_findings=[],
        management_questions=[],
    )

    with pytest.raises(
        ValueError,
        match="not approved",
    ):
        validate_commentary_against_payload(
            commentary,
            payload,
        )


def test_validator_accepts_approved_uk_evidence(
    payload: dict,
) -> None:
    commentary = CFOCommentary(
        executive_summary=(
            "Test summary."
        ),
        observations=[],
        supported_explanations=[
            EvidenceBackedExplanation(
                finding_id=(
                    "country::United Kingdom"
                ),
                explanation=(
                    "The negative volume/mix "
                    "finding is partially "
                    "supported by the documented "
                    "shipment delay."
                ),
                evidence_ids=[
                    "NOTE-015"
                ],
                source_type=(
                    "evidence_supported"
                ),
            )
        ],
        unresolved_findings=[],
        management_questions=[],
    )

    validate_commentary_against_payload(
        commentary,
        payload,
    )


def test_validator_accepts_approved_contractor_evidence(
    payload: dict,
) -> None:
    commentary = CFOCommentary(
        executive_summary=(
            "Test summary."
        ),
        observations=[],
        supported_explanations=[
            EvidenceBackedExplanation(
                finding_id=(
                    "opex_account::Contractors"
                ),
                explanation=(
                    "The contractor OPEX "
                    "variance is partially "
                    "supported by the documented "
                    "Product & R&D contractor "
                    "acceleration."
                ),
                evidence_ids=[
                    "NOTE-017"
                ],
                source_type=(
                    "evidence_supported"
                ),
            )
        ],
        unresolved_findings=[],
        management_questions=[],
    )

    validate_commentary_against_payload(
        commentary,
        payload,
    )


def test_saved_ai_output_passes_current_guardrails(
    payload: dict,
) -> None:
    if not AI_OUTPUT_PATH.exists():
        pytest.skip(
            "No saved AI commentary "
            "output exists yet."
        )

    saved = json.loads(
        AI_OUTPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    metadata = saved.get(
        "metadata",
        {},
    )

    if (
        metadata.get(
            "directional_guardrail_applied"
        )
        is not True
    ):
        pytest.skip(
            "Saved AI output predates "
            "the directional guardrail."
        )

    commentary = (
        CFOCommentary.model_validate(
            saved["commentary"]
        )
    )

    validate_commentary_against_payload(
        commentary,
        payload,
    )