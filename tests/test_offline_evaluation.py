from __future__ import annotations

import pytest

from src.evaluation.offline_evaluator import (
    run_evaluation,
)


@pytest.fixture(scope="module")
def evaluation() -> dict:
    """
    Run the offline hidden-ground-truth evaluation once
    for the complete test module.

    No API calls are made.
    """
    return run_evaluation()


def get_event(
    evaluation: dict,
    event_id: str,
) -> dict:
    """
    Retrieve one evaluated hidden ground-truth event.
    """

    for event in evaluation["events"]:
        if event["event_id"] == event_id:
            return event

    raise AssertionError(
        f"Ground-truth event not found: {event_id}"
    )


def test_offline_evaluation_passes(
    evaluation: dict,
) -> None:
    assert (
        evaluation["summary"]["overall_passed"]
        is True
    )


def test_ground_truth_event_counts(
    evaluation: dict,
) -> None:
    summary = evaluation["summary"]

    assert summary[
        "ground_truth_event_count"
    ] == 6

    assert summary[
        "events_with_analyst_evidence"
    ] == 4

    assert summary[
        "events_without_analyst_evidence"
    ] == 2


def test_pipeline_routing_counts(
    evaluation: dict,
) -> None:
    routing = evaluation[
        "pipeline_routing"
    ]

    assert routing == {
        "not_observable_no_evidence": 2,
        "reached_ai": 2,
        "rejected_by_grounding": 1,
        "withheld_by_directional_guardrail": 1,
    }


def test_evt_001_is_rejected_by_grounding(
    evaluation: dict,
) -> None:
    event = get_event(
        evaluation,
        "EVT-001",
    )

    assert event["evidence_id"] == "NOTE-014"

    assert (
        event["pipeline_classification"]
        == "rejected_by_grounding"
    )


def test_evt_002_is_not_observable(
    evaluation: dict,
) -> None:
    event = get_event(
        evaluation,
        "EVT-002",
    )

    assert (
        event["evidence_available"]
        is False
    )

    assert event["evidence_id"] is None

    assert (
        event["pipeline_classification"]
        == "not_observable_no_evidence"
    )


def test_evt_003_reaches_ai(
    evaluation: dict,
) -> None:
    event = get_event(
        evaluation,
        "EVT-003",
    )

    assert event["evidence_id"] == "NOTE-015"

    assert (
        event["pipeline_classification"]
        == "reached_ai"
    )


def test_evt_004_is_withheld_by_directional_guardrail(
    evaluation: dict,
) -> None:
    event = get_event(
        evaluation,
        "EVT-004",
    )

    assert event["evidence_id"] == "NOTE-016"

    assert (
        event["pipeline_classification"]
        == "withheld_by_directional_guardrail"
    )


def test_opex_001_reaches_ai(
    evaluation: dict,
) -> None:
    event = get_event(
        evaluation,
        "OPEX-001",
    )

    assert event["evidence_id"] == "NOTE-017"

    assert (
        event["pipeline_classification"]
        == "reached_ai"
    )


def test_opex_002_is_not_observable(
    evaluation: dict,
) -> None:
    event = get_event(
        evaluation,
        "OPEX-002",
    )

    assert (
        event["evidence_available"]
        is False
    )

    assert event["evidence_id"] is None

    assert (
        event["pipeline_classification"]
        == "not_observable_no_evidence"
    )


def test_ai_visible_evidence_ids_are_correct(
    evaluation: dict,
) -> None:
    assert set(
        evaluation["summary"][
            "ai_visible_evidence_ids"
        ]
    ) == {
        "NOTE-015",
        "NOTE-017",
    }


def test_directionally_withheld_evidence_is_correct(
    evaluation: dict,
) -> None:
    assert set(
        evaluation["summary"][
            "directionally_withheld_evidence_ids"
        ]
    ) == {
        "NOTE-016",
    }


def test_ai_citations_are_valid(
    evaluation: dict,
) -> None:
    summary = evaluation["summary"]

    assert set(
        summary[
            "ai_cited_evidence_ids"
        ]
    ) == {
        "NOTE-015",
        "NOTE-017",
    }

    assert (
        summary[
            "invalid_ai_evidence_ids"
        ]
        == []
    )


def test_citation_precision_is_100_percent(
    evaluation: dict,
) -> None:
    assert (
        evaluation["summary"][
            "citation_precision"
        ]
        == pytest.approx(1.0)
    )


def test_ai_visible_evidence_utilization_is_100_percent(
    evaluation: dict,
) -> None:
    assert (
        evaluation["summary"][
            "ai_visible_evidence_utilization"
        ]
        == pytest.approx(1.0)
    )


def test_ground_truth_security_isolation(
    evaluation: dict,
) -> None:
    security = evaluation[
        "security_isolation"
    ]

    assert security["passed"] is True

    assert (
        security[
            "ground_truth_access_flag_false"
        ]
        is True
    )

    assert (
        security["leaked_event_ids"]
        == []
    )

    assert (
        security[
            "leaked_ground_truth_paths"
        ]
        == []
    )


def test_model_never_used_ground_truth(
    evaluation: dict,
) -> None:
    metadata = evaluation[
        "metadata"
    ]

    assert (
        metadata[
            "ground_truth_used_by_model"
        ]
        is False
    )

    assert (
        metadata[
            "api_request_sent"
        ]
        is False
    )