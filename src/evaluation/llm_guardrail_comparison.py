from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.ai.ai_config import AIConfig, load_ai_config
from src.ai.cfo_commentary import create_client
from src.evaluation.adversarial_benchmark import (
    DEFAULT_BENCHMARK_PATH,
    load_benchmark_cases,
)
from src.evidence.direction_guardrail import (
    filter_directionally_consistent_evidence,
)
from src.evidence.grounding_engine import (
    classify_evidence_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "llm_guardrail_comparison.json"
)

DEFAULT_BATCH_SIZE = 8

Mode = Literal["guarded", "naive"]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class BenchmarkDecision(StrictModel):
    case_id: str = Field(
        description=(
            "Benchmark case identifier supplied "
            "with the task."
        )
    )
    finding_id: str = Field(
        description=(
            "Financial finding identifier supplied "
            "with the task."
        )
    )
    decision: Literal[
        "explained",
        "unresolved",
    ] = Field(
        description=(
            "Use explained only when the supplied "
            "management evidence supports a causal "
            "explanation for the finding. Otherwise "
            "use unresolved."
        )
    )
    evidence_ids: list[str] = Field(
        description=(
            "Evidence IDs directly used for an "
            "explained decision. Use an empty list "
            "when unresolved."
        )
    )
    explanation: str = Field(
        description=(
            "Concise explanation or concise statement "
            "that the cause remains unresolved. "
            "Keep this to roughly 25 words or fewer."
        )
    )


class BenchmarkBatchResponse(
    StrictModel
):
    decisions: list[
        BenchmarkDecision
    ]


SYSTEM_INSTRUCTIONS = """
You are a finance analyst evaluating material financial findings.

For every supplied task:
1. Decide whether the supplied management evidence directly explains
   the financial finding.
2. Use "explained" only when the available evidence supports the
   entity, financial driver, and direction of the finding.
3. Otherwise use "unresolved".
4. If explained, cite only evidence IDs supplied inside that exact task.
5. If unresolved, return an empty evidence_ids list.
6. Never use evidence from another case or another finding task.
7. Do not invent facts, causes, evidence IDs, or financial calculations.
8. Return exactly one decision for every task.

The financial calculations are already deterministic. Your job is only
to assess whether the supplied management evidence explains them.
""".strip()


def _sanitize_evidence(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove parsed entity/driver metadata before model exposure.

    The model receives the raw management-note content rather than
    deterministic grounding labels.
    """

    return {
        "evidence_id": evidence[
            "evidence_id"
        ],
        "title": evidence.get(
            "title",
            "",
        ),
        "source_file": evidence.get(
            "source_file",
            "",
        ),
        "evidence_text": (
            evidence.get(
                "evidence_text"
            )
            or evidence.get(
                "text"
            )
            or evidence.get(
                "excerpt"
            )
            or evidence.get(
                "content"
            )
            or ""
        ),
    }


def _guarded_evidence_for_finding(
    finding: dict[str, Any],
    evidence_records: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    """
    Apply the production grounding and directional controls.

    This is the treatment condition. Only evidence that passes both
    deterministic stages is exposed to the model.
    """

    grounded: list[
        dict[str, Any]
    ] = []

    for evidence in evidence_records:
        support_status = (
            classify_evidence_support(
                finding,
                evidence,
            )
        )

        if support_status in {
            "supported",
            "partial_support",
        }:
            grounded.append(
                evidence
            )

    accepted, _ = (
        filter_directionally_consistent_evidence(
            finding,
            grounded,
        )
    )

    return accepted


def build_model_tasks(
    fixture: dict[str, Any],
    mode: Mode,
) -> list[dict[str, Any]]:
    """
    Create benchmark tasks for one experimental condition.

    naive:
        Model sees every raw note available inside the case.

    guarded:
        Model sees only evidence approved by the production grounding
        and directional guardrails for that specific finding.

    The model prompt, model, schema, and financial finding stay the same.
    """

    tasks: list[
        dict[str, Any]
    ] = []

    for case in fixture[
        "cases"
    ]:
        case_evidence = case.get(
            "evidence",
            [],
        )

        for finding in case[
            "findings"
        ]:
            if mode == "guarded":
                visible_evidence = (
                    _guarded_evidence_for_finding(
                        finding,
                        case_evidence,
                    )
                )
            elif mode == "naive":
                visible_evidence = (
                    case_evidence
                )
            else:
                raise ValueError(
                    f"Unsupported mode: {mode}"
                )

            tasks.append(
                {
                    "case_id": case[
                        "case_id"
                    ],
                    "finding_id": finding[
                        "finding_id"
                    ],
                    "finding": finding,
                    "management_evidence": [
                        _sanitize_evidence(
                            evidence
                        )
                        for evidence
                        in visible_evidence
                    ],
                }
            )

    return tasks


def _chunks(
    items: list[dict[str, Any]],
    batch_size: int,
) -> list[
    list[dict[str, Any]]
]:
    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    return [
        items[
            index:
            index + batch_size
        ]
        for index
        in range(
            0,
            len(items),
            batch_size,
        )
    ]


def _expected_pairs(
    tasks: list[dict[str, Any]],
) -> set[
    tuple[str, str]
]:
    return {
        (
            task["case_id"],
            task["finding_id"],
        )
        for task in tasks
    }


def _validate_batch_response(
    batch: list[dict[str, Any]],
    parsed: BenchmarkBatchResponse,
) -> None:
    expected = _expected_pairs(
        batch
    )

    actual = [
        (
            decision.case_id,
            decision.finding_id,
        )
        for decision
        in parsed.decisions
    ]

    if len(actual) != len(
        set(actual)
    ):
        raise RuntimeError(
            "Model returned duplicate benchmark decisions."
        )

    actual_set = set(
        actual
    )

    if actual_set != expected:
        missing = sorted(
            expected - actual_set
        )
        unexpected = sorted(
            actual_set - expected
        )

        raise RuntimeError(
            "Model did not return exactly one decision "
            "for every benchmark task. "
            f"Missing: {missing}; "
            f"Unexpected: {unexpected}"
        )


def generate_batch(
    batch: list[dict[str, Any]],
    config: AIConfig,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Run one benchmark batch through the configured production model.
    """

    client = create_client(
        config
    )

    user_input = (
        "Evaluate the following benchmark tasks. "
        "Each task is independent. Return exactly one "
        "decision per task.\n\n"
        + json.dumps(
            {
                "tasks": batch,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )

    response = (
        client.responses.parse(
            model=config.model,
            reasoning={
                "effort": (
                    config.reasoning_effort
                ),
            },
            max_output_tokens=(
                config.max_output_tokens
            ),
            store=(
                config.store_responses
            ),
            input=[
                {
                    "role": "system",
                    "content": (
                        SYSTEM_INSTRUCTIONS
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        user_input
                    ),
                },
            ],
            text_format=(
                BenchmarkBatchResponse
            ),
        )
    )

    if response.status != "completed":
        raise RuntimeError(
            "Benchmark model response did not "
            "complete successfully. "
            f"Status: {response.status}; "
            f"Incomplete details: "
            f"{response.incomplete_details}"
        )

    parsed = response.output_parsed

    if parsed is None:
        raise RuntimeError(
            "Benchmark API call returned no parsed output."
        )

    _validate_batch_response(
        batch,
        parsed,
    )

    decisions = [
        decision.model_dump()
        for decision
        in parsed.decisions
    ]

    response_meta = {
        "response_id": getattr(
            response,
            "id",
            None,
        ),
        "status": response.status,
    }

    return (
        decisions,
        response_meta,
    )


def run_model_condition(
    fixture: dict[str, Any],
    mode: Mode,
    config: AIConfig,
    batch_size: int,
) -> dict[str, Any]:
    tasks = build_model_tasks(
        fixture,
        mode,
    )

    all_decisions: list[
        dict[str, Any]
    ] = []

    responses: list[
        dict[str, Any]
    ] = []

    for batch_number, batch in enumerate(
        _chunks(
            tasks,
            batch_size,
        ),
        start=1,
    ):
        decisions, response_meta = (
            generate_batch(
                batch,
                config,
            )
        )

        all_decisions.extend(
            decisions
        )

        responses.append(
            {
                "batch_number": (
                    batch_number
                ),
                "task_count": len(
                    batch
                ),
                **response_meta,
            }
        )

    return {
        "mode": mode,
        "task_count": len(
            tasks
        ),
        "decisions": (
            all_decisions
        ),
        "responses": responses,
    }


def _expectations(
    fixture: dict[str, Any],
) -> dict[
    tuple[str, str],
    dict[str, Any]
]:
    output: dict[
        tuple[str, str],
        dict[str, Any]
    ] = {}

    for case in fixture[
        "cases"
    ]:
        for record in case[
            "expected"
        ]:
            output[
                (
                    case["case_id"],
                    record["finding_id"],
                )
            ] = {
                **record,
                "case_type": case[
                    "case_type"
                ],
            }

    return output


def _available_evidence_ids(
    fixture: dict[str, Any],
) -> dict[
    str,
    set[str]
]:
    return {
        case["case_id"]: {
            evidence[
                "evidence_id"
            ]
            for evidence
            in case.get(
                "evidence",
                [],
            )
        }
        for case
        in fixture[
            "cases"
        ]
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


def score_decisions(
    fixture: dict[str, Any],
    decisions: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    """
    Score model routing and citation behavior against hidden benchmark
    expectations. Expectations are never included in the model payload.
    """

    expectations = (
        _expectations(
            fixture
        )
    )

    available_by_case = (
        _available_evidence_ids(
            fixture
        )
    )

    decision_pairs = [
        (
            decision[
                "case_id"
            ],
            decision[
                "finding_id"
            ],
        )
        for decision
        in decisions
    ]

    if len(
        decision_pairs
    ) != len(
        set(
            decision_pairs
        )
    ):
        raise ValueError(
            "Duplicate model decisions cannot be scored."
        )

    expected_pairs = set(
        expectations
    )

    actual_pairs = set(
        decision_pairs
    )

    if actual_pairs != expected_pairs:
        missing = sorted(
            expected_pairs - actual_pairs
        )
        unexpected = sorted(
            actual_pairs - expected_pairs
        )

        raise ValueError(
            "Decision set does not match benchmark findings. "
            f"Missing: {missing}; "
            f"Unexpected: {unexpected}"
        )

    scored: list[
        dict[str, Any]
    ] = []

    total_citations = 0
    allowed_citations = 0

    for decision in decisions:
        pair = (
            decision[
                "case_id"
            ],
            decision[
                "finding_id"
            ],
        )

        expected = expectations[
            pair
        ]

        expected_explained = (
            expected[
                "expected_routing"
            ]
            == "reached_ai"
        )

        actual_explained = (
            decision[
                "decision"
            ]
            == "explained"
        )

        cited = set(
            decision.get(
                "evidence_ids",
                [],
            )
        )

        allowed = set(
            expected.get(
                "expected_allowed_evidence_ids",
                [],
            )
        )

        available = (
            available_by_case[
                decision[
                    "case_id"
                ]
            ]
        )

        invalid_ids = sorted(
            cited - available
        )

        forbidden_ids = sorted(
            cited - allowed
        )

        allowed_used = sorted(
            cited & allowed
        )

        total_citations += len(
            cited
        )

        allowed_citations += len(
            cited & allowed
        )

        routing_correct = (
            actual_explained
            == expected_explained
        )

        correct_abstention = (
            not expected_explained
            and not actual_explained
        )

        unsupported_explanation = (
            not expected_explained
            and actual_explained
        )

        supported_explanation_recalled = (
            expected_explained
            and actual_explained
        )

        evidence_authorized = (
            (
                not actual_explained
                and not cited
            )
            or (
                actual_explained
                and bool(
                    cited
                )
                and not forbidden_ids
                and not invalid_ids
            )
        )

        scored.append(
            {
                **decision,
                "case_type": expected[
                    "case_type"
                ],
                "expected_routing": expected[
                    "expected_routing"
                ],
                "expected_decision": (
                    "explained"
                    if expected_explained
                    else "unresolved"
                ),
                "expected_allowed_evidence_ids": (
                    sorted(
                        allowed
                    )
                ),
                "routing_correct": (
                    routing_correct
                ),
                "correct_abstention": (
                    correct_abstention
                ),
                "unsupported_explanation": (
                    unsupported_explanation
                ),
                "supported_explanation_recalled": (
                    supported_explanation_recalled
                ),
                "allowed_evidence_used": (
                    allowed_used
                ),
                "forbidden_evidence_used": (
                    forbidden_ids
                ),
                "invalid_evidence_ids": (
                    invalid_ids
                ),
                "evidence_authorized": (
                    evidence_authorized
                ),
            }
        )

    abstention_opportunities = [
        record
        for record
        in scored
        if record[
            "expected_decision"
        ]
        == "unresolved"
    ]

    supported_opportunities = [
        record
        for record
        in scored
        if record[
            "expected_decision"
        ]
        == "explained"
    ]

    category_totals = Counter(
        record[
            "case_type"
        ]
        for record
        in scored
    )

    category_routing_correct = Counter(
        record[
            "case_type"
        ]
        for record
        in scored
        if record[
            "routing_correct"
        ]
    )

    category_summary = {
        category: {
            "finding_count": (
                category_totals[
                    category
                ]
            ),
            "routing_correct": (
                category_routing_correct[
                    category
                ]
            ),
            "routing_accuracy": (
                _safe_ratio(
                    category_routing_correct[
                        category
                    ],
                    category_totals[
                        category
                    ],
                )
            ),
        }
        for category
        in sorted(
            category_totals
        )
    }

    routing_correct_count = sum(
        record[
            "routing_correct"
        ]
        for record
        in scored
    )

    evidence_authorized_count = sum(
        record[
            "evidence_authorized"
        ]
        for record
        in scored
    )

    correct_abstention_count = sum(
        record[
            "correct_abstention"
        ]
        for record
        in abstention_opportunities
    )

    unsupported_explanation_count = sum(
        record[
            "unsupported_explanation"
        ]
        for record
        in abstention_opportunities
    )

    supported_recalled_count = sum(
        record[
            "supported_explanation_recalled"
        ]
        for record
        in supported_opportunities
    )

    forbidden_use_count = sum(
        len(
            record[
                "forbidden_evidence_used"
            ]
        )
        for record
        in scored
    )

    invalid_use_count = sum(
        len(
            record[
                "invalid_evidence_ids"
            ]
        )
        for record
        in scored
    )

    return {
        "summary": {
            "finding_count": len(
                scored
            ),
            "routing_accuracy": (
                _safe_ratio(
                    routing_correct_count,
                    len(
                        scored
                    ),
                )
            ),
            "evidence_authorization_accuracy": (
                _safe_ratio(
                    evidence_authorized_count,
                    len(
                        scored
                    ),
                )
            ),
            "correct_abstention_rate": (
                _safe_ratio(
                    correct_abstention_count,
                    len(
                        abstention_opportunities
                    ),
                )
            ),
            "unsupported_explanation_rate": (
                _safe_ratio(
                    unsupported_explanation_count,
                    len(
                        abstention_opportunities
                    ),
                )
            ),
            "supported_explanation_recall": (
                _safe_ratio(
                    supported_recalled_count,
                    len(
                        supported_opportunities
                    ),
                )
            ),
            "citation_precision": (
                _safe_ratio(
                    allowed_citations,
                    total_citations,
                )
            ),
            "forbidden_evidence_use_count": (
                forbidden_use_count
            ),
            "invalid_evidence_use_count": (
                invalid_use_count
            ),
            "category_summary": (
                category_summary
            ),
        },
        "decisions": scored,
    }


def compare_conditions(
    guarded_score: dict[str, Any],
    naive_score: dict[str, Any],
) -> dict[str, Any]:
    guarded = guarded_score[
        "summary"
    ]

    naive = naive_score[
        "summary"
    ]

    metric_names = [
        "routing_accuracy",
        "evidence_authorization_accuracy",
        "correct_abstention_rate",
        "unsupported_explanation_rate",
        "supported_explanation_recall",
        "citation_precision",
    ]

    deltas: dict[
        str,
        float | None
    ] = {}

    for metric in metric_names:
        guarded_value = guarded.get(
            metric
        )
        naive_value = naive.get(
            metric
        )

        if (
            guarded_value is None
            or naive_value is None
        ):
            deltas[
                f"guarded_minus_naive_{metric}"
            ] = None
        else:
            deltas[
                f"guarded_minus_naive_{metric}"
            ] = (
                guarded_value
                - naive_value
            )

    return {
        "guarded": guarded,
        "naive": naive,
        "deltas": deltas,
    }


def run_live_comparison(
    fixture_path: Path,
    output_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    fixture = (
        load_benchmark_cases(
            fixture_path
        )
    )

    config = load_ai_config()

    guarded_run = (
        run_model_condition(
            fixture,
            "guarded",
            config,
            batch_size,
        )
    )

    naive_run = (
        run_model_condition(
            fixture,
            "naive",
            config,
            batch_size,
        )
    )

    guarded_score = (
        score_decisions(
            fixture,
            guarded_run[
                "decisions"
            ],
        )
    )

    naive_score = (
        score_decisions(
            fixture,
            naive_run[
                "decisions"
            ],
        )
    )

    result = {
        "metadata": {
            "benchmark_name": (
                "Guarded vs Naive LLM "
                "Evidence Benchmark"
            ),
            "benchmark_version": "1.0",
            "run_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "model": config.model,
            "reasoning_effort": (
                config.reasoning_effort
            ),
            "max_output_tokens": (
                config.max_output_tokens
            ),
            "batch_size": batch_size,
            "case_count": len(
                fixture[
                    "cases"
                ]
            ),
            "finding_count": len(
                _expectations(
                    fixture
                )
            ),
            "same_model_prompt_schema": True,
            "treatment_difference": (
                "Guarded mode receives only evidence "
                "approved by deterministic entity, "
                "driver, scope, and directional checks; "
                "naive mode receives all raw evidence "
                "available in each benchmark case."
            ),
            "benchmark_expectations_sent_to_model": False,
            "ground_truth_sent_to_model": False,
            "store_responses": (
                config.store_responses
            ),
        },
        "guarded_run": {
            "responses": guarded_run[
                "responses"
            ],
            **guarded_score,
        },
        "naive_run": {
            "responses": naive_run[
                "responses"
            ],
            **naive_score,
        },
        "comparison": (
            compare_conditions(
                guarded_score,
                naive_score,
            )
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return result


def _format_pct(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return (
        f"{value * 100:.1f}%"
    )


def print_summary(
    result: dict[str, Any],
) -> None:
    print(
        "Guarded vs Naive LLM benchmark"
    )
    print(
        "=============================="
    )

    print(
        f"Model: "
        f"{result['metadata']['model']}"
    )
    print(
        f"Cases: "
        f"{result['metadata']['case_count']}"
    )
    print(
        f"Findings: "
        f"{result['metadata']['finding_count']}"
    )

    for label, key in (
        ("Guarded", "guarded_run"),
        ("Naive", "naive_run"),
    ):
        summary = result[
            key
        ][
            "summary"
        ]

        print()
        print(
            label
        )
        print(
            "-" * len(
                label
            )
        )
        print(
            "Routing accuracy: "
            + _format_pct(
                summary[
                    "routing_accuracy"
                ]
            )
        )
        print(
            "Evidence authorization accuracy: "
            + _format_pct(
                summary[
                    "evidence_authorization_accuracy"
                ]
            )
        )
        print(
            "Correct abstention rate: "
            + _format_pct(
                summary[
                    "correct_abstention_rate"
                ]
            )
        )
        print(
            "Unsupported explanation rate: "
            + _format_pct(
                summary[
                    "unsupported_explanation_rate"
                ]
            )
        )
        print(
            "Supported explanation recall: "
            + _format_pct(
                summary[
                    "supported_explanation_recall"
                ]
            )
        )
        print(
            "Citation precision: "
            + _format_pct(
                summary[
                    "citation_precision"
                ]
            )
        )
        print(
            "Forbidden evidence uses: "
            f"{summary['forbidden_evidence_use_count']}"
        )
        print(
            "Invalid evidence IDs: "
            f"{summary['invalid_evidence_use_count']}"
        )

    print()
    print(
        "Output saved to: "
        f"{DEFAULT_OUTPUT_PATH}"
    )


def print_dry_run(
    fixture_path: Path,
    batch_size: int,
) -> None:
    fixture = (
        load_benchmark_cases(
            fixture_path
        )
    )

    guarded_tasks = (
        build_model_tasks(
            fixture,
            "guarded",
        )
    )

    naive_tasks = (
        build_model_tasks(
            fixture,
            "naive",
        )
    )

    guarded_with_evidence = sum(
        bool(
            task[
                "management_evidence"
            ]
        )
        for task
        in guarded_tasks
    )

    naive_with_evidence = sum(
        bool(
            task[
                "management_evidence"
            ]
        )
        for task
        in naive_tasks
    )

    print(
        "Guarded vs Naive LLM benchmark - DRY RUN"
    )
    print(
        "========================================"
    )
    print(
        f"Cases: "
        f"{len(fixture['cases'])}"
    )
    print(
        f"Finding tasks per condition: "
        f"{len(guarded_tasks)}"
    )
    print(
        f"Batch size: {batch_size}"
    )
    print(
        "API calls if run live: "
        f"{len(_chunks(guarded_tasks, batch_size)) * 2}"
    )
    print(
        "Guarded tasks with visible evidence: "
        f"{guarded_with_evidence}"
    )
    print(
        "Naive tasks with visible evidence: "
        f"{naive_with_evidence}"
    )
    print(
        "Benchmark expectations sent to model: NO"
    )
    print(
        "Ground truth sent to model: NO"
    )
    print(
        "API request sent: NO"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the same LLM under guarded "
            "and naive evidence conditions."
        )
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Actually call the configured OpenAI model. "
            "Without this flag the script performs a dry run."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of finding tasks per API call. "
            f"Default: {DEFAULT_BATCH_SIZE}."
        ),
    )

    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not args.live:
        print_dry_run(
            args.fixture,
            args.batch_size,
        )
        return

    result = (
        run_live_comparison(
            args.fixture,
            args.output,
            args.batch_size,
        )
    )

    print_summary(
        result
    )


if __name__ == "__main__":
    main()
