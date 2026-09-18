from __future__ import annotations

from collections import Counter

import pytest

from src.evaluation.adversarial_benchmark import (
    load_benchmark_cases,
    run_benchmark,
)


@pytest.fixture(scope="module")
def fixture() -> dict:
    return load_benchmark_cases()


@pytest.fixture(scope="module")
def benchmark() -> dict:
    return run_benchmark()


def test_benchmark_has_32_unique_cases(
    fixture: dict,
) -> None:
    cases = fixture["cases"]

    assert len(cases) == 32

    case_ids = [
        case["case_id"]
        for case in cases
    ]

    assert len(case_ids) == len(
        set(case_ids)
    )


def test_case_category_distribution(
    fixture: dict,
) -> None:
    counts = Counter(
        case["case_type"]
        for case in fixture[
            "cases"
        ]
    )

    assert counts == {
        "valid_supported": 6,
        "no_evidence": 6,
        "wrong_entity": 5,
        "wrong_driver": 5,
        "directional_conflict": 4,
        "ambiguous_evidence": 3,
        "multi_event": 3,
    }


def test_deterministic_benchmark_passes(
    benchmark: dict,
) -> None:
    assert (
        benchmark["summary"][
            "overall_passed"
        ]
        is True
    )


def test_all_32_cases_pass(
    benchmark: dict,
) -> None:
    summary = benchmark[
        "summary"
    ]

    assert summary[
        "case_count"
    ] == 32

    assert summary[
        "cases_passed"
    ] == 32

    assert summary[
        "case_pass_rate"
    ] == pytest.approx(1.0)


def test_routing_accuracy_is_100_percent(
    benchmark: dict,
) -> None:
    assert benchmark[
        "summary"
    ][
        "routing_accuracy"
    ] == pytest.approx(1.0)


def test_evidence_authorization_accuracy_is_100_percent(
    benchmark: dict,
) -> None:
    assert benchmark[
        "summary"
    ][
        "evidence_authorization_accuracy"
    ] == pytest.approx(1.0)


def test_correct_abstention_rate_is_100_percent(
    benchmark: dict,
) -> None:
    assert benchmark[
        "summary"
    ][
        "correct_abstention_rate"
    ] == pytest.approx(1.0)


def test_unsupported_explanation_rate_is_zero(
    benchmark: dict,
) -> None:
    assert benchmark[
        "summary"
    ][
        "unsupported_explanation_rate"
    ] == pytest.approx(0.0)


def test_forbidden_evidence_is_never_used(
    benchmark: dict,
) -> None:
    assert benchmark[
        "summary"
    ][
        "forbidden_evidence_use_count"
    ] == 0


def _cases_of_type(
    benchmark: dict,
    case_type: str,
) -> list[dict]:
    return [
        case
        for case in benchmark[
            "cases"
        ]
        if case[
            "case_type"
        ]
        == case_type
    ]


def test_valid_supported_cases_reach_ai(
    benchmark: dict,
) -> None:
    cases = _cases_of_type(
        benchmark,
        "valid_supported",
    )

    assert all(
        finding["routing"]
        == "reached_ai"
        for case in cases
        for finding in case[
            "findings"
        ]
    )


def test_no_evidence_cases_abstain(
    benchmark: dict,
) -> None:
    cases = _cases_of_type(
        benchmark,
        "no_evidence",
    )

    assert all(
        finding["routing"]
        == "insufficient_evidence"
        and finding["abstained"]
        for case in cases
        for finding in case[
            "findings"
        ]
    )


def test_wrong_entity_cases_are_rejected(
    benchmark: dict,
) -> None:
    cases = _cases_of_type(
        benchmark,
        "wrong_entity",
    )

    assert all(
        finding["routing"]
        == "rejected_by_grounding"
        for case in cases
        for finding in case[
            "findings"
        ]
    )


def test_wrong_driver_cases_are_rejected(
    benchmark: dict,
) -> None:
    cases = _cases_of_type(
        benchmark,
        "wrong_driver",
    )

    assert all(
        finding["routing"]
        == "rejected_by_grounding"
        for case in cases
        for finding in case[
            "findings"
        ]
    )


def test_directional_conflicts_are_withheld(
    benchmark: dict,
) -> None:
    cases = _cases_of_type(
        benchmark,
        "directional_conflict",
    )

    assert all(
        finding["routing"]
        == "withheld_by_directional_guardrail"
        for case in cases
        for finding in case[
            "findings"
        ]
    )


def test_ambiguous_evidence_is_withheld(
    benchmark: dict,
) -> None:
    cases = _cases_of_type(
        benchmark,
        "ambiguous_evidence",
    )

    assert all(
        finding["routing"]
        == "withheld_by_directional_guardrail"
        for case in cases
        for finding in case[
            "findings"
        ]
    )


def test_multi_event_cases_mix_outcomes_safely(
    benchmark: dict,
) -> None:
    cases = _cases_of_type(
        benchmark,
        "multi_event",
    )

    routes = {
        finding[
            "routing"
        ]
        for case in cases
        for finding in case[
            "findings"
        ]
    }

    assert "reached_ai" in routes
    assert (
        "rejected_by_grounding"
        in routes
    )
    assert (
        "withheld_by_directional_guardrail"
        in routes
    )


def test_benchmark_never_calls_api(
    benchmark: dict,
) -> None:
    assert benchmark[
        "metadata"
    ][
        "api_request_sent"
    ] is False


def test_benchmark_does_not_use_ground_truth_as_model_input(
    benchmark: dict,
) -> None:
    assert benchmark[
        "metadata"
    ][
        "ground_truth_used_by_model"
    ] is False
