from __future__ import annotations

import pytest

from src.ai.cfo_qa import (
    CFOQAError,
    ask_cfo,
    build_approved_context,
    build_model_payload,
    select_relevant_context,
    validate_answer,
)


@pytest.fixture
def commentary() -> dict:
    return {
        "observations": [
            {
                "finding_id": "summary::ebitda",
                "statement": (
                    "EBITDA is NOK 25.0m below budget."
                ),
            },
            {
                "finding_id": "country::UK",
                "statement": (
                    "UK revenue is below budget."
                ),
            },
            {
                "finding_id": "country::Norway",
                "statement": (
                    "Norway revenue is below budget."
                ),
            },
            {
                "finding_id": (
                    "cost_centre::Product_&_R&D"
                ),
                "statement": (
                    "Product & R&D OPEX is above budget."
                ),
            },
        ],
        "supported_explanations": [
            {
                "finding_id": "country::UK",
                "explanation": (
                    "A shipment delay reduced Sensor X "
                    "Distributor volume in the UK."
                ),
                "evidence_ids": [
                    "NOTE-015",
                ],
            },
            {
                "finding_id": (
                    "cost_centre::Product_&_R&D"
                ),
                "explanation": (
                    "Higher contractor spend contributed "
                    "to Product & R&D OPEX."
                ),
                "evidence_ids": [
                    "NOTE-017",
                ],
            },
            {
                "finding_id": "country::Norway",
                "explanation": (
                    "Sensor X Pro shipments accelerated."
                ),
                "evidence_ids": [
                    "NOTE-016",
                ],
            },
        ],
        "unresolved_findings": [
            {
                "finding_id": "country::Norway",
                "statement": (
                    "The underlying cause of the Norway "
                    "shortfall remains unresolved."
                ),
            },
        ],
    }


@pytest.fixture
def metadata() -> dict:
    return {
        "source_usable_evidence_ids": [
            "NOTE-015",
            "NOTE-017",
        ],
        "directional_guardrail_summary": {
            "withheld_evidence_ids": [
                "NOTE-016",
            ],
        },
    }


def test_approved_context_excludes_withheld_evidence(
    commentary: dict,
    metadata: dict,
) -> None:
    context = build_approved_context(
        commentary,
        metadata,
    )

    assert (
        context["approved_evidence_ids"]
        == [
            "NOTE-015",
            "NOTE-017",
        ]
    )

    all_supported_ids = {
        evidence_id
        for record in context["supported"]
        for evidence_id in record.evidence_ids
    }

    assert "NOTE-016" not in all_supported_ids


def test_uk_question_selects_uk_evidence(
    commentary: dict,
    metadata: dict,
) -> None:
    selected = select_relevant_context(
        "What explains the UK shortfall?",
        commentary,
        metadata,
    )

    assert (
        "NOTE-015"
        in selected["selected_evidence_ids"]
    )


def test_opex_question_selects_contractor_evidence(
    commentary: dict,
    metadata: dict,
) -> None:
    selected = select_relevant_context(
        "Why is Product & R&D OPEX above budget?",
        commentary,
        metadata,
    )

    assert (
        "NOTE-017"
        in selected["selected_evidence_ids"]
    )


def test_validator_rejects_unapproved_evidence(
    commentary: dict,
    metadata: dict,
) -> None:
    selected = select_relevant_context(
        "What explains the UK shortfall?",
        commentary,
        metadata,
    )

    with pytest.raises(
        CFOQAError,
        match="not approved",
    ):
        validate_answer(
            {
                "answer_type": (
                    "supported_explanation"
                ),
                "answer": "A cause was identified.",
                "finding_ids": [
                    "country::UK",
                ],
                "evidence_ids": [
                    "NOTE-999",
                ],
                "limitations": [],
            },
            selected,
        )


def test_validator_rejects_withheld_evidence_in_text(
    commentary: dict,
    metadata: dict,
) -> None:
    selected = select_relevant_context(
        "What is happening in Norway?",
        commentary,
        metadata,
    )

    with pytest.raises(
        CFOQAError,
        match="withheld",
    ):
        validate_answer(
            {
                "answer_type": (
                    "insufficient_evidence"
                ),
                "answer": (
                    "NOTE-016 explains the Norway result."
                ),
                "finding_ids": [
                    "country::Norway",
                ],
                "evidence_ids": [],
                "limitations": [],
            },
            selected,
        )


def test_validator_rejects_ground_truth_leakage(
    commentary: dict,
    metadata: dict,
) -> None:
    selected = select_relevant_context(
        "What explains the UK shortfall?",
        commentary,
        metadata,
    )

    with pytest.raises(
        CFOQAError,
        match="Ground-truth",
    ):
        validate_answer(
            {
                "answer_type": (
                    "calculated_fact"
                ),
                "answer": (
                    "The hidden event EVT-003 caused it."
                ),
                "finding_ids": [
                    "country::UK",
                ],
                "evidence_ids": [],
                "limitations": [],
            },
            selected,
        )


def test_supported_explanation_requires_evidence(
    commentary: dict,
    metadata: dict,
) -> None:
    selected = select_relevant_context(
        "What explains the UK shortfall?",
        commentary,
        metadata,
    )

    with pytest.raises(
        CFOQAError,
        match="must cite",
    ):
        validate_answer(
            {
                "answer_type": (
                    "supported_explanation"
                ),
                "answer": (
                    "A shipment delay contributed."
                ),
                "finding_ids": [
                    "country::UK",
                ],
                "evidence_ids": [],
                "limitations": [],
            },
            selected,
        )


def test_insufficient_evidence_accepts_no_causal_citation(
    commentary: dict,
    metadata: dict,
) -> None:
    selected = select_relevant_context(
        "Why is Norway below budget?",
        commentary,
        metadata,
    )

    result = validate_answer(
        {
            "answer_type": (
                "insufficient_evidence"
            ),
            "answer": (
                "Norway is below budget, but the "
                "available approved evidence does not "
                "establish the underlying cause."
            ),
            "finding_ids": [
                "country::Norway",
            ],
            "evidence_ids": [],
            "limitations": [
                (
                    "The underlying cause remains "
                    "unresolved."
                ),
            ],
        },
        selected,
    )

    assert (
        result["answer_type"]
        == "insufficient_evidence"
    )


def test_legacy_calculated_observations_key_is_supported(
    metadata: dict,
) -> None:
    legacy_commentary = {
        "calculated_observations": [
            {
                "finding_id": "summary::ebitda",
                "statement": (
                    "EBITDA is NOK 25.0m below budget."
                ),
            },
        ],
        "supported_explanations": [],
        "unresolved_findings": [],
    }

    context = build_approved_context(
        legacy_commentary,
        metadata,
    )

    assert len(context["facts"]) == 1
    assert (
        context["facts"][0].finding_id
        == "summary::ebitda"
    )


def test_norway_question_does_not_select_unrelated_evidence(
    commentary: dict,
    metadata: dict,
) -> None:
    selected = select_relevant_context(
        "Can the Norway revenue shortfall be explained?",
        commentary,
        metadata,
    )

    assert (
        selected["selected_evidence_ids"]
        == []
    )


def test_ebitda_question_surfaces_multiple_approved_driver_evidence(
    commentary: dict,
    metadata: dict,
) -> None:
    selected = select_relevant_context(
        "Why is EBITDA below budget?",
        commentary,
        metadata,
    )

    assert set(
        selected["selected_evidence_ids"]
    ) == {
        "NOTE-015",
        "NOTE-017",
    }


def test_model_payload_uses_requested_analysis_period(
    commentary: dict,
    metadata: dict,
) -> None:
    selected = select_relevant_context(
        "Why is EBITDA below budget?",
        commentary,
        metadata,
    )

    payload = build_model_payload(
        question="Why is EBITDA below budget?",
        selected_context=selected,
        analysis_period="Actual YTD vs Budget",
    )

    assert (
        payload["analysis_period"]
        == "Actual YTD vs Budget"
    )


def test_ask_cfo_dry_run_propagates_analysis_period(
    commentary: dict,
    metadata: dict,
) -> None:
    result = ask_cfo(
        question="Why is EBITDA below budget?",
        commentary=commentary,
        ai_metadata=metadata,
        analysis_period="Actual YTD vs Budget",
        dry_run=True,
    )

    assert (
        result["model_payload"]["analysis_period"]
        == "Actual YTD vs Budget"
    )
    assert (
        result["diagnostics"]["analysis_period"]
        == "Actual YTD vs Budget"
    )
    assert (
        result["diagnostics"]["api_request_sent"]
        is False
    )
