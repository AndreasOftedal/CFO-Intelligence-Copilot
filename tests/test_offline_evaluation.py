from __future__ import annotations

import pytest

from src.evaluation.offline_evaluator import (
    EVALUATION_CASES,
    load_ground_truth_events,
    run_evaluation,
    run_evaluation_suite,
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

@pytest.fixture(scope="module")
def ytd_evaluation() -> dict:
    """
    Run the Actual YTD vs Budget evaluation once for the
    complete test module.

    No API calls are made.
    """

    return run_evaluation(
        "actual_ytd_vs_budget"
    )


@pytest.fixture(scope="module")
def evaluation_suite() -> dict:
    """
    Run the cross-period evaluation suite once.

    No API calls are made.
    """

    return run_evaluation_suite()


def test_ytd_evaluation_passes(
    ytd_evaluation: dict,
) -> None:
    assert (
        ytd_evaluation[
            "summary"
        ][
            "overall_passed"
        ]
        is True
    )

    assert (
        ytd_evaluation[
            "metadata"
        ][
            "comparison"
        ]
        == "actual_ytd_vs_budget"
    )


def test_ytd_ground_truth_scope_is_correct(
    ytd_evaluation: dict,
) -> None:
    summary = ytd_evaluation[
        "summary"
    ]

    assert summary[
        "ground_truth_event_count"
    ] == 6

    assert summary[
        "events_with_analyst_evidence"
    ] == 4

    assert summary[
        "events_without_analyst_evidence"
    ] == 2

    assert (
        ytd_evaluation[
            "metadata"
        ][
            "ground_truth_cutoff_date"
        ]
        == "2026-08-31"
    )


def test_ground_truth_cutoff_is_temporally_correct() -> None:
    before_august = (
        load_ground_truth_events(
            "2026-07-31"
        )
    )

    through_august_first = (
        load_ground_truth_events(
            "2026-08-01"
        )
    )

    assert before_august == []

    assert len(
        through_august_first
    ) == 6


def test_ytd_ai_visible_evidence_is_correct(
    ytd_evaluation: dict,
) -> None:
    summary = ytd_evaluation[
        "summary"
    ]

    assert set(
        summary[
            "ai_visible_evidence_ids"
        ]
    ) == {
        "NOTE-017",
    }

    assert (
        summary[
            "directionally_withheld_evidence_ids"
        ]
        == []
    )


def test_ytd_ai_citations_are_valid(
    ytd_evaluation: dict,
) -> None:
    summary = ytd_evaluation[
        "summary"
    ]

    assert set(
        summary[
            "ai_cited_evidence_ids"
        ]
    ) == {
        "NOTE-017",
    }

    assert (
        summary[
            "invalid_ai_evidence_ids"
        ]
        == []
    )

    assert (
        summary[
            "citations_authorized"
        ]
        is True
    )

    assert (
        summary[
            "citation_precision"
        ]
        == pytest.approx(1.0)
    )

    assert (
        summary[
            "ai_visible_evidence_utilization"
        ]
        == pytest.approx(1.0)
    )


def test_ytd_security_isolation_passes(
    ytd_evaluation: dict,
) -> None:
    security = ytd_evaluation[
        "security_isolation"
    ]

    assert security[
        "passed"
    ] is True

    assert security[
        "leaked_event_ids"
    ] == []

    assert security[
        "leaked_ground_truth_paths"
    ] == []


def test_ytd_routing_accounts_for_all_events(
    ytd_evaluation: dict,
) -> None:
    assert sum(
        ytd_evaluation[
            "pipeline_routing"
        ].values()
    ) == 6

    assert (
        ytd_evaluation[
            "pipeline_routing"
        ].get(
            "reached_ai",
            0,
        )
        == 1
    )

    assert (
        ytd_evaluation[
            "pipeline_routing"
        ].get(
            "not_observable_no_evidence",
            0,
        )
        == 2
    )


def test_evaluation_suite_passes_both_cases(
    evaluation_suite: dict,
) -> None:
    summary = evaluation_suite[
        "summary"
    ]

    assert summary[
        "overall_passed"
    ] is True

    assert summary[
        "case_count"
    ] == 2

    assert summary[
        "cases_passed"
    ] == 2

    assert summary[
        "cases_failed"
    ] == 0

    assert summary[
        "case_pass_rate"
    ] == pytest.approx(1.0)


def test_evaluation_suite_contains_expected_cases(
    evaluation_suite: dict,
) -> None:
    assert set(
        evaluation_suite[
            "cases"
        ]
    ) == set(
        EVALUATION_CASES
    )

    assert (
        evaluation_suite[
            "cases"
        ][
            "latest_forecast_vs_budget"
        ][
            "overall_passed"
        ]
        is True
    )

    assert (
        evaluation_suite[
            "cases"
        ][
            "actual_ytd_vs_budget"
        ][
            "overall_passed"
        ]
        is True
    )


def test_evaluation_suite_security_and_routing(
    evaluation_suite: dict,
) -> None:
    summary = evaluation_suite[
        "summary"
    ]

    assert (
        summary[
            "all_security_isolation_passed"
        ]
        is True
    )

    assert (
        summary[
            "all_citations_authorized"
        ]
        is True
    )

    assert sum(
        evaluation_suite[
            "aggregate_pipeline_routing"
        ].values()
    ) == 12


def test_unknown_evaluation_case_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown evaluation case",
    ):
        run_evaluation(
            "does_not_exist"
        )
