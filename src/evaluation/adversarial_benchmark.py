from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.evidence.direction_guardrail import (
    filter_directionally_consistent_evidence,
)
from src.evidence.grounding_engine import (
    classify_evidence_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BENCHMARK_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "benchmark"
    / "benchmark_cases.json"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "adversarial_benchmark.json"
)


def load_benchmark_cases(
    path: Path = DEFAULT_BENCHMARK_PATH,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark fixture not found: {path}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    cases = data.get(
        "cases",
        [],
    )

    if len(cases) != 32:
        raise ValueError(
            "Expected exactly 32 adversarial benchmark cases, "
            f"found {len(cases)}."
        )

    case_ids = [
        case["case_id"]
        for case in cases
    ]

    if len(case_ids) != len(
        set(case_ids)
    ):
        raise ValueError(
            "Benchmark case IDs must be unique."
        )

    return data


def _expected_by_finding(
    case: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        record["finding_id"]: record
        for record in case[
            "expected"
        ]
    }


def _evaluate_finding(
    finding: dict[str, Any],
    evidence_records: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_assessments: list[
        dict[str, Any]
    ] = []

    usable_evidence: list[
        dict[str, Any]
    ] = []

    for evidence in evidence_records:
        support_status = (
            classify_evidence_support(
                finding,
                evidence,
            )
        )

        evidence_assessments.append(
            {
                "evidence_id": evidence[
                    "evidence_id"
                ],
                "support_status": (
                    support_status
                ),
            }
        )

        if support_status in {
            "supported",
            "partial_support",
        }:
            usable_evidence.append(
                evidence
            )

    accepted, direction_assessments = (
        filter_directionally_consistent_evidence(
            finding,
            usable_evidence,
        )
    )

    accepted_ids = sorted(
        evidence["evidence_id"]
        for evidence in accepted
    )

    if accepted_ids:
        routing = "reached_ai"
    elif usable_evidence:
        routing = (
            "withheld_by_directional_guardrail"
        )
    elif evidence_records:
        routing = "rejected_by_grounding"
    else:
        routing = "insufficient_evidence"

    return {
        "finding_id": finding[
            "finding_id"
        ],
        "routing": routing,
        "accepted_evidence_ids": (
            accepted_ids
        ),
        "grounding_assessments": (
            evidence_assessments
        ),
        "direction_assessments": [
            assessment.to_dict()
            if hasattr(
                assessment,
                "to_dict",
            )
            else {
                "finding_id": getattr(
                    assessment,
                    "finding_id",
                    "",
                ),
                "evidence_id": getattr(
                    assessment,
                    "evidence_id",
                    "",
                ),
                "status": getattr(
                    assessment,
                    "status",
                    "",
                ),
                "evidence_direction": getattr(
                    assessment,
                    "evidence_direction",
                    "",
                ),
                "reason": getattr(
                    assessment,
                    "reason",
                    "",
                ),
            }
            for assessment
            in direction_assessments
        ],
        "abstained": (
            len(
                accepted_ids
            )
            == 0
        ),
    }


def evaluate_case(
    case: dict[str, Any],
) -> dict[str, Any]:
    expected = (
        _expected_by_finding(
            case
        )
    )

    findings_out: list[
        dict[str, Any]
    ] = []

    for finding in case[
        "findings"
    ]:
        finding_id = finding[
            "finding_id"
        ]

        actual = _evaluate_finding(
            finding,
            case.get(
                "evidence",
                [],
            ),
        )

        expected_record = expected[
            finding_id
        ]

        expected_allowed = sorted(
            expected_record.get(
                "expected_allowed_evidence_ids",
                [],
            )
        )

        forbidden = set(
            expected_record.get(
                "forbidden_evidence_ids",
                [],
            )
        )

        accepted_set = set(
            actual[
                "accepted_evidence_ids"
            ]
        )

        routing_correct = (
            actual["routing"]
            == expected_record[
                "expected_routing"
            ]
        )

        authorization_correct = (
            actual[
                "accepted_evidence_ids"
            ]
            == expected_allowed
        )

        forbidden_used = sorted(
            accepted_set
            & forbidden
        )

        abstention_correct = (
            actual["abstained"]
            == bool(
                expected_record[
                    "expected_abstention"
                ]
            )
        )

        finding_passed = (
            routing_correct
            and authorization_correct
            and not forbidden_used
            and abstention_correct
        )

        findings_out.append(
            {
                **actual,
                "expected_routing": (
                    expected_record[
                        "expected_routing"
                    ]
                ),
                "expected_allowed_evidence_ids": (
                    expected_allowed
                ),
                "forbidden_evidence_ids": (
                    sorted(
                        forbidden
                    )
                ),
                "expected_abstention": bool(
                    expected_record[
                        "expected_abstention"
                    ]
                ),
                "routing_correct": (
                    routing_correct
                ),
                "authorization_correct": (
                    authorization_correct
                ),
                "forbidden_evidence_used": (
                    forbidden_used
                ),
                "abstention_correct": (
                    abstention_correct
                ),
                "passed": finding_passed,
            }
        )

    return {
        "case_id": case[
            "case_id"
        ],
        "case_type": case[
            "case_type"
        ],
        "description": case.get(
            "description",
            "",
        ),
        "passed": all(
            result[
                "passed"
            ]
            for result
            in findings_out
        ),
        "findings": findings_out,
    }


def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator == 0:
        return None

    return (
        numerator
        / denominator
    )


def run_benchmark(
    path: Path = DEFAULT_BENCHMARK_PATH,
) -> dict[str, Any]:
    fixture = (
        load_benchmark_cases(
            path
        )
    )

    evaluated_cases = [
        evaluate_case(case)
        for case
        in fixture["cases"]
    ]

    finding_results = [
        finding
        for case
        in evaluated_cases
        for finding
        in case["findings"]
    ]

    total_findings = len(
        finding_results
    )

    routing_correct = sum(
        finding[
            "routing_correct"
        ]
        for finding
        in finding_results
    )

    authorization_correct = sum(
        finding[
            "authorization_correct"
        ]
        for finding
        in finding_results
    )

    abstention_opportunities = [
        finding
        for finding
        in finding_results
        if finding[
            "expected_abstention"
        ]
    ]

    correct_abstentions = sum(
        finding[
            "abstention_correct"
        ]
        for finding
        in abstention_opportunities
    )

    unsupported_explanations = sum(
        bool(
            finding[
                "accepted_evidence_ids"
            ]
        )
        for finding
        in abstention_opportunities
    )

    forbidden_evidence_uses = sum(
        len(
            finding[
                "forbidden_evidence_used"
            ]
        )
        for finding
        in finding_results
    )

    category_totals = Counter(
        case[
            "case_type"
        ]
        for case
        in evaluated_cases
    )

    category_passed = Counter(
        case[
            "case_type"
        ]
        for case
        in evaluated_cases
        if case[
            "passed"
        ]
    )

    category_summary = {
        category: {
            "cases": category_totals[
                category
            ],
            "passed": category_passed[
                category
            ],
            "pass_rate": _safe_ratio(
                category_passed[
                    category
                ],
                category_totals[
                    category
                ],
            ),
        }
        for category
        in sorted(
            category_totals
        )
    }

    cases_passed = sum(
        case["passed"]
        for case
        in evaluated_cases
    )

    summary = {
        "overall_passed": all(
            case["passed"]
            for case
            in evaluated_cases
        ),
        "case_count": len(
            evaluated_cases
        ),
        "cases_passed": (
            cases_passed
        ),
        "case_pass_rate": _safe_ratio(
            cases_passed,
            len(
                evaluated_cases
            ),
        ),
        "finding_evaluation_count": (
            total_findings
        ),
        "routing_accuracy": _safe_ratio(
            routing_correct,
            total_findings,
        ),
        "evidence_authorization_accuracy": (
            _safe_ratio(
                authorization_correct,
                total_findings,
            )
        ),
        "correct_abstention_rate": (
            _safe_ratio(
                correct_abstentions,
                len(
                    abstention_opportunities
                ),
            )
        ),
        "unsupported_explanation_rate": (
            _safe_ratio(
                unsupported_explanations,
                len(
                    abstention_opportunities
                ),
            )
        ),
        "forbidden_evidence_use_count": (
            forbidden_evidence_uses
        ),
        "category_summary": (
            category_summary
        ),
    }

    return {
        "metadata": {
            **fixture[
                "metadata"
            ],
            "benchmark_type": (
                "deterministic_adversarial_routing"
            ),
            "api_request_sent": False,
            "ground_truth_used_by_model": False,
        },
        "summary": summary,
        "cases": evaluated_cases,
    }


def save_benchmark(
    result: dict[str, Any],
    path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _format_percent(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return (
        f"{value * 100:.1f}%"
    )


def main() -> None:
    result = run_benchmark()
    save_benchmark(
        result
    )

    summary = result[
        "summary"
    ]

    print(
        "Adversarial evidence benchmark"
    )
    print(
        "=============================="
    )
    print(
        "Status: "
        + (
            "PASSED"
            if summary[
                "overall_passed"
            ]
            else "FAILED"
        )
    )
    print(
        f"Cases passed: "
        f"{summary['cases_passed']}"
        f"/{summary['case_count']}"
    )
    print(
        "Routing accuracy: "
        + _format_percent(
            summary[
                "routing_accuracy"
            ]
        )
    )
    print(
        "Evidence authorization accuracy: "
        + _format_percent(
            summary[
                "evidence_authorization_accuracy"
            ]
        )
    )
    print(
        "Correct abstention rate: "
        + _format_percent(
            summary[
                "correct_abstention_rate"
            ]
        )
    )
    print(
        "Unsupported explanation rate: "
        + _format_percent(
            summary[
                "unsupported_explanation_rate"
            ]
        )
    )
    print(
        "Forbidden evidence uses: "
        f"{summary['forbidden_evidence_use_count']}"
    )
    print()
    print(
        "Category performance"
    )
    print(
        "--------------------"
    )

    for category, metrics in (
        summary[
            "category_summary"
        ].items()
    ):
        print(
            f"{category}: "
            f"{metrics['passed']}/"
            f"{metrics['cases']} passed"
        )

    print()
    print(
        "Output saved to: "
        f"{DEFAULT_OUTPUT_PATH}"
    )
    print(
        "API request sent: NO"
    )


if __name__ == "__main__":
    main()
