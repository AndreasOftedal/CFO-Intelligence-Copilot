from __future__ import annotations

from copy import deepcopy

import pytest

from src.evaluation.adversarial_benchmark import (
    load_benchmark_cases,
)
from src.evaluation.llm_guardrail_comparison import (
    _chunks,
    build_model_tasks,
    score_decisions,
)


@pytest.fixture(scope="module")
def fixture() -> dict:
    return load_benchmark_cases()


def _perfect_decisions(
    fixture: dict,
) -> list[dict]:
    decisions = []

    for case in fixture[
        "cases"
    ]:
        expected_by_finding = {
            record[
                "finding_id"
            ]: record
            for record
            in case[
                "expected"
            ]
        }

        for finding in case[
            "findings"
        ]:
            expected = (
                expected_by_finding[
                    finding[
                        "finding_id"
                    ]
                ]
            )

            explained = (
                expected[
                    "expected_routing"
                ]
                == "reached_ai"
            )

            decisions.append(
                {
                    "case_id": case[
                        "case_id"
                    ],
                    "finding_id": finding[
                        "finding_id"
                    ],
                    "decision": (
                        "explained"
                        if explained
                        else "unresolved"
                    ),
                    "evidence_ids": (
                        expected.get(
                            "expected_allowed_evidence_ids",
                            [],
                        )
                        if explained
                        else []
                    ),
                    "explanation": (
                        "Supported by supplied evidence."
                        if explained
                        else "Cause remains unresolved."
                    ),
                }
            )

    return decisions


def test_same_number_of_tasks_in_both_conditions(
    fixture: dict,
) -> None:
    guarded = build_model_tasks(
        fixture,
        "guarded",
    )

    naive = build_model_tasks(
        fixture,
        "naive",
    )

    assert len(guarded) == 38
    assert len(naive) == 38


def test_naive_condition_sees_all_case_evidence(
    fixture: dict,
) -> None:
    tasks = build_model_tasks(
        fixture,
        "naive",
    )

    case_map = {
        case[
            "case_id"
        ]: case
        for case
        in fixture[
            "cases"
        ]
    }

    for task in tasks:
        expected_ids = {
            evidence[
                "evidence_id"
            ]
            for evidence
            in case_map[
                task[
                    "case_id"
                ]
            ].get(
                "evidence",
                [],
            )
        }

        actual_ids = {
            evidence[
                "evidence_id"
            ]
            for evidence
            in task[
                "management_evidence"
            ]
        }

        assert actual_ids == expected_ids


def test_guarded_condition_matches_expected_authorized_evidence(
    fixture: dict,
) -> None:
    tasks = build_model_tasks(
        fixture,
        "guarded",
    )

    expected = {
        (
            case[
                "case_id"
            ],
            record[
                "finding_id"
            ],
        ): set(
            record.get(
                "expected_allowed_evidence_ids",
                [],
            )
        )
        for case
        in fixture[
            "cases"
        ]
        for record
        in case[
            "expected"
        ]
    }

    for task in tasks:
        actual_ids = {
            evidence[
                "evidence_id"
            ]
            for evidence
            in task[
                "management_evidence"
            ]
        }

        assert actual_ids == expected[
            (
                task[
                    "case_id"
                ],
                task[
                    "finding_id"
                ],
            )
        ]


def test_model_payload_hides_parsed_grounding_metadata(
    fixture: dict,
) -> None:
    tasks = build_model_tasks(
        fixture,
        "naive",
    )

    for task in tasks:
        for evidence in task[
            "management_evidence"
        ]:
            assert "entities" not in evidence
            assert (
                "primary_evidence_driver"
                not in evidence
            )


def test_batching_is_deterministic(
    fixture: dict,
) -> None:
    tasks = build_model_tasks(
        fixture,
        "guarded",
    )

    batches = _chunks(
        tasks,
        8,
    )

    assert len(batches) == 5
    assert [
        len(batch)
        for batch in batches
    ] == [
        8,
        8,
        8,
        8,
        6,
    ]


def test_perfect_decisions_score_100_percent(
    fixture: dict,
) -> None:
    score = score_decisions(
        fixture,
        _perfect_decisions(
            fixture
        ),
    )

    summary = score[
        "summary"
    ]

    assert summary[
        "routing_accuracy"
    ] == pytest.approx(1.0)

    assert summary[
        "evidence_authorization_accuracy"
    ] == pytest.approx(1.0)

    assert summary[
        "correct_abstention_rate"
    ] == pytest.approx(1.0)

    assert summary[
        "unsupported_explanation_rate"
    ] == pytest.approx(0.0)

    assert summary[
        "supported_explanation_recall"
    ] == pytest.approx(1.0)

    assert summary[
        "citation_precision"
    ] == pytest.approx(1.0)


def test_unsupported_explanation_is_detected(
    fixture: dict,
) -> None:
    decisions = _perfect_decisions(
        fixture
    )

    target = next(
        decision
        for decision
        in decisions
        if decision[
            "case_id"
        ]
        == "B013"
    )

    target[
        "decision"
    ] = "explained"

    target[
        "evidence_ids"
    ] = [
        "BENCH-013"
    ]

    score = score_decisions(
        fixture,
        decisions,
    )

    summary = score[
        "summary"
    ]

    assert (
        summary[
            "unsupported_explanation_rate"
        ]
        > 0
    )

    assert (
        summary[
            "forbidden_evidence_use_count"
        ]
        >= 1
    )


def test_directional_conflict_explanation_is_detected(
    fixture: dict,
) -> None:
    decisions = _perfect_decisions(
        fixture
    )

    target = next(
        decision
        for decision
        in decisions
        if decision[
            "case_id"
        ]
        == "B023"
    )

    target[
        "decision"
    ] = "explained"

    target[
        "evidence_ids"
    ] = [
        "BENCH-023"
    ]

    score = score_decisions(
        fixture,
        decisions,
    )

    record = next(
        record
        for record
        in score[
            "decisions"
        ]
        if record[
            "case_id"
        ]
        == "B023"
    )

    assert record[
        "unsupported_explanation"
    ] is True

    assert record[
        "forbidden_evidence_used"
    ] == [
        "BENCH-023"
    ]


def test_wrong_driver_citation_is_detected(
    fixture: dict,
) -> None:
    decisions = _perfect_decisions(
        fixture
    )

    target = next(
        decision
        for decision
        in decisions
        if decision[
            "case_id"
        ]
        == "B018"
    )

    target[
        "decision"
    ] = "explained"

    target[
        "evidence_ids"
    ] = [
        "BENCH-018"
    ]

    score = score_decisions(
        fixture,
        decisions,
    )

    record = next(
        record
        for record
        in score[
            "decisions"
        ]
        if record[
            "case_id"
        ]
        == "B018"
    )

    assert record[
        "routing_correct"
    ] is False

    assert record[
        "evidence_authorized"
    ] is False


def test_cross_case_or_invented_evidence_id_is_detected(
    fixture: dict,
) -> None:
    decisions = _perfect_decisions(
        fixture
    )

    target = next(
        decision
        for decision
        in decisions
        if decision[
            "case_id"
        ]
        == "B001"
    )

    target[
        "evidence_ids"
    ] = [
        "DOES-NOT-EXIST"
    ]

    score = score_decisions(
        fixture,
        decisions,
    )

    record = next(
        record
        for record
        in score[
            "decisions"
        ]
        if record[
            "case_id"
        ]
        == "B001"
    )

    assert record[
        "invalid_evidence_ids"
    ] == [
        "DOES-NOT-EXIST"
    ]

    assert record[
        "evidence_authorized"
    ] is False


def test_missing_supported_explanation_reduces_recall(
    fixture: dict,
) -> None:
    decisions = _perfect_decisions(
        fixture
    )

    target = next(
        decision
        for decision
        in decisions
        if decision[
            "case_id"
        ]
        == "B001"
    )

    target[
        "decision"
    ] = "unresolved"

    target[
        "evidence_ids"
    ] = []

    score = score_decisions(
        fixture,
        decisions,
    )

    assert (
        score[
            "summary"
        ][
            "supported_explanation_recall"
        ]
        < 1.0
    )


def test_duplicate_decisions_are_rejected(
    fixture: dict,
) -> None:
    decisions = _perfect_decisions(
        fixture
    )

    decisions.append(
        deepcopy(
            decisions[
                0
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        score_decisions(
            fixture,
            decisions,
        )


def test_missing_decision_is_rejected(
    fixture: dict,
) -> None:
    decisions = _perfect_decisions(
        fixture
    )[
        :-1
    ]

    with pytest.raises(
        ValueError,
        match="Decision set",
    ):
        score_decisions(
            fixture,
            decisions,
        )
