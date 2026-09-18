from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from src.ai.cfo_commentary import (
    build_ai_payload,
    build_model_visible_payload,
    load_grounded_analysis,
)
from src.ai.schemas import CFOCommentary


EVALUATION_CASES: dict[str, dict[str, Any]] = {
    "latest_forecast_vs_budget": {
        "label": "Latest Forecast vs Budget",
        "grounding_path": Path(
            "outputs/grounding/latest_forecast_vs_budget_grounded.json"
        ),
        "ai_output_path": Path(
            "outputs/ai/latest_forecast_vs_budget_commentary.json"
        ),
        "evaluation_output_path": Path(
            "outputs/evaluation/latest_forecast_vs_budget_evaluation.json"
        ),
        "ground_truth_cutoff_date": None,
    },
    "actual_ytd_vs_budget": {
        "label": "Actual YTD vs Budget",
        "grounding_path": Path(
            "outputs/grounding/actual_ytd_vs_budget_grounded.json"
        ),
        "ai_output_path": Path(
            "outputs/ai/actual_ytd_vs_budget_commentary.json"
        ),
        "evaluation_output_path": Path(
            "outputs/evaluation/actual_ytd_vs_budget_evaluation.json"
        ),
        "ground_truth_cutoff_date": "2026-08-31",
    },
}

DEFAULT_EVALUATION_CASE = "latest_forecast_vs_budget"

# Backwards-compatible aliases for the original single-case evaluator.
GROUNDING_PATH = EVALUATION_CASES[
    DEFAULT_EVALUATION_CASE
]["grounding_path"]

AI_OUTPUT_PATH = EVALUATION_CASES[
    DEFAULT_EVALUATION_CASE
]["ai_output_path"]

SALES_GROUND_TRUTH_PATH = Path(
    "data/ground_truth/ground_truth_events.csv"
)

OPEX_GROUND_TRUTH_PATH = Path(
    "data/ground_truth/ground_truth_opex_events.csv"
)

EVALUATION_OUTPUT_PATH = EVALUATION_CASES[
    DEFAULT_EVALUATION_CASE
]["evaluation_output_path"]

EVALUATION_SUITE_OUTPUT_PATH = Path(
    "outputs/evaluation/evaluation_suite.json"
)


def parse_bool(value: str) -> bool:
    """
    Parse boolean values stored in CSV files.
    """

    normalized = value.strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Unexpected boolean value in ground truth: {value}"
    )


def parse_float(
    value: str,
) -> float | None:
    """
    Parse optional numeric CSV values.
    """

    value = value.strip()

    if not value:
        return None

    return float(value)


def load_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    """
    Load a CSV file as dictionaries.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        return list(reader)


def normalize_sales_ground_truth(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Normalize controlled sales events into one common
    evaluation structure.
    """

    normalized: list[dict[str, Any]] = []

    for row in rows:
        evidence_available = parse_bool(
            row["evidence_available"]
        )

        evidence_id = (
            row.get("evidence_id", "").strip()
            or None
        )

        if evidence_available and not evidence_id:
            raise ValueError(
                "Sales ground-truth event is marked as "
                "evidence_available but has no evidence_id: "
                f"{row['event_id']}"
            )

        driver = row["driver"].strip()

        driver_map = {
            "units": "volume_mix",
            "discount_rate": "discount",
            "unit_cost": "unit_cost",
        }

        normalized.append(
            {
                "event_id": row["event_id"],
                "domain": "sales",
                "date": row["date"],
                "country": row.get(
                    "country",
                    "",
                ),
                "product": row.get(
                    "product",
                    "",
                ),
                "segment": row.get(
                    "segment",
                    "",
                ),
                "driver_raw": driver,
                "driver_canonical": driver_map.get(
                    driver,
                    driver,
                ),
                "description": row[
                    "description"
                ],
                "financial_impact_nok": (
                    parse_float(
                        row.get(
                            "gross_profit_impact",
                            "",
                        )
                    )
                ),
                "evidence_available": (
                    evidence_available
                ),
                "evidence_id": evidence_id,
            }
        )

    return normalized


def normalize_opex_ground_truth(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Normalize controlled OPEX events into the same
    evaluation structure as sales events.
    """

    normalized: list[dict[str, Any]] = []

    for row in rows:
        evidence_available = parse_bool(
            row["evidence_available"]
        )

        evidence_id = (
            row.get("evidence_id", "").strip()
            or None
        )

        if evidence_available and not evidence_id:
            raise ValueError(
                "OPEX ground-truth event is marked as "
                "evidence_available but has no evidence_id: "
                f"{row['event_id']}"
            )

        normalized.append(
            {
                "event_id": row["event_id"],
                "domain": "opex",
                "date": row["date"],
                "cost_centre": row.get(
                    "cost_centre",
                    "",
                ),
                "account": row.get(
                    "account",
                    "",
                ),
                "description": row[
                    "description"
                ],
                "financial_impact_nok": (
                    parse_float(
                        row.get(
                            "ebitda_impact",
                            "",
                        )
                    )
                ),
                "evidence_available": (
                    evidence_available
                ),
                "evidence_id": evidence_id,
            }
        )

    return normalized


def load_ground_truth_events(
    cutoff_date: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load all hidden controlled events that are in scope for
    the requested evaluation period.

    IMPORTANT:
    This function belongs only to the offline evaluation layer.
    It must never be imported by the AI generation pipeline.

    When cutoff_date is provided, only events on or before that
    ISO date are included. This keeps period-specific evaluation
    temporally correct if future controlled events are added.
    """

    sales_rows = load_csv_rows(
        SALES_GROUND_TRUTH_PATH
    )

    opex_rows = load_csv_rows(
        OPEX_GROUND_TRUTH_PATH
    )

    events = (
        normalize_sales_ground_truth(
            sales_rows
        )
        + normalize_opex_ground_truth(
            opex_rows
        )
    )

    if cutoff_date is None:
        return events

    cutoff = date.fromisoformat(
        cutoff_date
    )

    return [
        event
        for event in events
        if date.fromisoformat(
            event["date"]
        )
        <= cutoff
    ]


def collect_grounding_candidate_statuses(
    grounded_data: dict[str, Any],
) -> dict[str, set[str]]:
    """
    Collect how each evidence ID was classified by the
    grounding engine.
    """

    statuses: dict[
        str,
        set[str],
    ] = {}

    for finding in grounded_data[
        "findings"
    ]:
        grounding = finding.get(
            "grounding",
            {},
        )

        for candidate in grounding.get(
            "evidence_candidates",
            [],
        ):
            evidence_id = candidate.get(
                "evidence_id"
            )

            if not evidence_id:
                continue

            support_status = candidate.get(
                "support_status",
                "unknown",
            )

            statuses.setdefault(
                evidence_id,
                set(),
            ).add(
                support_status
            )

    return statuses


def collect_ai_visible_evidence_ids(
    payload: dict[str, Any],
) -> set[str]:
    """
    Collect evidence IDs that survived all guardrails and
    were eligible to reach the model.
    """

    return {
        evidence["evidence_id"]
        for finding in payload[
            "material_findings"
        ]
        for evidence in finding[
            "usable_evidence"
        ]
    }


def collect_withheld_evidence_ids(
    payload: dict[str, Any],
) -> set[str]:
    """
    Collect evidence withheld by the directional guardrail.
    """

    return set(
        payload.get(
            "_internal",
            {},
        )
        .get(
            "directional_guardrail",
            {},
        )
        .get(
            "withheld_evidence_ids",
            [],
        )
    )


def classify_ground_truth_event(
    event: dict[str, Any],
    ai_visible_ids: set[str],
    withheld_ids: set[str],
    grounding_statuses: dict[
        str,
        set[str],
    ],
) -> str:
    """
    Classify how a hidden ground-truth event was handled
    by the production pipeline.

    This is descriptive evaluation, not model input.
    """

    if not event["evidence_available"]:
        return "not_observable_no_evidence"

    evidence_id = event["evidence_id"]

    if evidence_id in ai_visible_ids:
        return "reached_ai"

    if evidence_id in withheld_ids:
        return "withheld_by_directional_guardrail"

    candidate_statuses = grounding_statuses.get(
        evidence_id,
        set(),
    )

    if (
        "related_not_explanatory"
        in candidate_statuses
    ):
        return "rejected_by_grounding"

    if candidate_statuses:
        return "not_approved_by_grounding"

    return "not_connected_to_material_finding"


def load_ai_output(
    path: Path = AI_OUTPUT_PATH,
) -> dict[str, Any]:
    """
    Load a saved and already-validated AI output.
    """

    if not path.exists():
        raise FileNotFoundError(
            "Validated AI output not found: "
            f"{path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def get_evaluation_case(
    case_key: str,
) -> dict[str, Any]:
    """
    Return one configured evaluation case.
    """

    if case_key not in EVALUATION_CASES:
        raise ValueError(
            "Unknown evaluation case: "
            f"{case_key}. Expected one of: "
            + ", ".join(
                sorted(
                    EVALUATION_CASES
                )
            )
        )

    return EVALUATION_CASES[
        case_key
    ]


def validate_case_alignment(
    case_key: str,
    grounded_data: dict[str, Any],
    ai_output: dict[str, Any],
) -> None:
    """
    Ensure the selected grounding and AI artifacts belong to
    the same configured comparison before evaluation proceeds.
    """

    grounded_comparison = (
        grounded_data
        .get(
            "metadata",
            {},
        )
        .get(
            "comparison"
        )
    )

    ai_comparison = (
        ai_output
        .get(
            "metadata",
            {},
        )
        .get(
            "comparison"
        )
    )

    if (
        grounded_comparison
        != case_key
    ):
        raise ValueError(
            "Grounded analysis comparison does not match "
            f"evaluation case '{case_key}': "
            f"{grounded_comparison}"
        )

    if ai_comparison != case_key:
        raise ValueError(
            "AI commentary comparison does not match "
            f"evaluation case '{case_key}': "
            f"{ai_comparison}"
        )


def collect_ai_citations(
    ai_output: dict[str, Any],
) -> list[str]:
    """
    Collect evidence IDs cited in final AI explanations.
    """

    commentary = CFOCommentary.model_validate(
        ai_output["commentary"]
    )

    citations: list[str] = []

    for explanation in (
        commentary.supported_explanations
    ):
        citations.extend(
            explanation.evidence_ids
        )

    return citations


def evaluate_security_isolation(
    ground_truth_events: list[
        dict[str, Any]
    ],
    payload: dict[str, Any],
    ai_output: dict[str, Any],
) -> dict[str, Any]:
    """
    Verify that hidden ground-truth identifiers and paths
    did not enter the model-visible payload.

    The model is allowed to see the boolean statement
    ground_truth_access = false.
    It must not see the actual hidden event IDs or files.
    """

    model_payload = (
        build_model_visible_payload(
            payload
        )
    )

    serialized_payload = json.dumps(
        model_payload,
        ensure_ascii=False,
    )

    event_ids = {
        event["event_id"]
        for event in ground_truth_events
    }

    leaked_event_ids = sorted(
        event_id
        for event_id in event_ids
        if event_id in serialized_payload
    )

    forbidden_paths = (
        "data/ground_truth",
        "ground_truth_events.csv",
        "ground_truth_opex_events.csv",
    )

    leaked_paths = [
        path
        for path in forbidden_paths
        if path in serialized_payload
    ]

    output_metadata = ai_output.get(
        "metadata",
        {},
    )

    ground_truth_flag_ok = (
        output_metadata.get(
            "ground_truth_access"
        )
        is False
    )

    passed = (
        not leaked_event_ids
        and not leaked_paths
        and ground_truth_flag_ok
    )

    return {
        "passed": passed,
        "ground_truth_access_flag_false": (
            ground_truth_flag_ok
        ),
        "leaked_event_ids": (
            leaked_event_ids
        ),
        "leaked_ground_truth_paths": (
            leaked_paths
        ),
    }


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    """
    Compute a ratio safely.
    """

    if denominator == 0:
        return None

    return numerator / denominator


def run_evaluation(
    case_key: str = DEFAULT_EVALUATION_CASE,
) -> dict[str, Any]:
    """
    Run one complete offline hidden-ground-truth evaluation.

    No API calls are made.
    """

    case = get_evaluation_case(
        case_key
    )

    grounded_data = (
        load_grounded_analysis(
            case[
                "grounding_path"
            ]
        )
    )

    payload = build_ai_payload(
        grounded_data
    )

    ground_truth_events = (
        load_ground_truth_events(
            cutoff_date=case[
                "ground_truth_cutoff_date"
            ]
        )
    )

    ai_output = load_ai_output(
        case[
            "ai_output_path"
        ]
    )

    validate_case_alignment(
        case_key,
        grounded_data,
        ai_output,
    )

    grounding_statuses = (
        collect_grounding_candidate_statuses(
            grounded_data
        )
    )

    ai_visible_ids = (
        collect_ai_visible_evidence_ids(
            payload
        )
    )

    withheld_ids = (
        collect_withheld_evidence_ids(
            payload
        )
    )

    evaluated_events: list[
        dict[str, Any]
    ] = []

    for event in ground_truth_events:
        classification = (
            classify_ground_truth_event(
                event,
                ai_visible_ids,
                withheld_ids,
                grounding_statuses,
            )
        )

        evaluated_events.append(
            {
                **event,
                "pipeline_classification": (
                    classification
                ),
            }
        )

    routing_counts = Counter(
        event["pipeline_classification"]
        for event in evaluated_events
    )

    evidence_available_events = [
        event
        for event in evaluated_events
        if event["evidence_available"]
    ]

    no_evidence_events = [
        event
        for event in evaluated_events
        if not event["evidence_available"]
    ]

    ground_truth_evidence_ids = {
        event["evidence_id"]
        for event in evidence_available_events
        if event["evidence_id"]
    }

    citations = collect_ai_citations(
        ai_output
    )

    unique_citations = set(
        citations
    )

    valid_citations = (
        unique_citations
        & ground_truth_evidence_ids
    )

    invalid_citations = (
        unique_citations
        - ground_truth_evidence_ids
    )

    citation_precision = safe_ratio(
        len(valid_citations),
        len(unique_citations),
    )

    visible_evidence_cited = (
        unique_citations
        & ai_visible_ids
    )

    visible_evidence_utilization = (
        safe_ratio(
            len(
                visible_evidence_cited
            ),
            len(ai_visible_ids),
        )
    )

    security = (
        evaluate_security_isolation(
            ground_truth_events,
            payload,
            ai_output,
        )
    )

    citations_authorized = (
        unique_citations
        <= ai_visible_ids
    )

    overall_passed = (
        security["passed"]
        and not invalid_citations
        and citations_authorized
    )

    return {
        "metadata": {
            "evaluation_type": (
                "offline_hidden_ground_truth"
            ),
            "evaluation_case": (
                case_key
            ),
            "analysis_period": (
                case["label"]
            ),
            "comparison": (
                grounded_data[
                    "metadata"
                ]["comparison"]
            ),
            "ground_truth_cutoff_date": (
                case[
                    "ground_truth_cutoff_date"
                ]
            ),
            "api_request_sent": False,
            "ground_truth_used_by_model": False,
        },
        "summary": {
            "overall_passed": (
                overall_passed
            ),
            "ground_truth_event_count": (
                len(
                    ground_truth_events
                )
            ),
            "events_with_analyst_evidence": (
                len(
                    evidence_available_events
                )
            ),
            "events_without_analyst_evidence": (
                len(
                    no_evidence_events
                )
            ),
            "ground_truth_evidence_ids": (
                sorted(
                    ground_truth_evidence_ids
                )
            ),
            "ai_visible_evidence_ids": (
                sorted(
                    ai_visible_ids
                )
            ),
            "directionally_withheld_evidence_ids": (
                sorted(
                    withheld_ids
                )
            ),
            "ai_cited_evidence_ids": (
                sorted(
                    unique_citations
                )
            ),
            "invalid_ai_evidence_ids": (
                sorted(
                    invalid_citations
                )
            ),
            "citations_authorized": (
                citations_authorized
            ),
            "citation_precision": (
                citation_precision
            ),
            "ai_visible_evidence_utilization": (
                visible_evidence_utilization
            ),
        },
        "pipeline_routing": {
            key: routing_counts[key]
            for key in sorted(
                routing_counts
            )
        },
        "security_isolation": (
            security
        ),
        "events": evaluated_events,
    }


def save_evaluation(
    result: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """
    Save one offline evaluation artifact.
    """

    if output_path is None:
        case_key = result[
            "metadata"
        ].get(
            "evaluation_case",
            DEFAULT_EVALUATION_CASE,
        )

        output_path = (
            get_evaluation_case(
                case_key
            )[
                "evaluation_output_path"
            ]
        )

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

    return output_path


def build_evaluation_suite(
    evaluations: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    """
    Build one cross-period evaluation artifact from already
    completed case evaluations.
    """

    expected_cases = set(
        EVALUATION_CASES
    )

    supplied_cases = set(
        evaluations
    )

    if supplied_cases != expected_cases:
        raise ValueError(
            "Evaluation suite requires exactly these cases: "
            + ", ".join(
                sorted(
                    expected_cases
                )
            )
        )

    case_summaries: dict[
        str,
        dict[str, Any],
    ] = {}

    aggregate_routing = Counter()

    for case_key in EVALUATION_CASES:
        result = evaluations[
            case_key
        ]

        summary = result[
            "summary"
        ]

        security = result[
            "security_isolation"
        ]

        aggregate_routing.update(
            result[
                "pipeline_routing"
            ]
        )

        case_summaries[
            case_key
        ] = {
            "label": (
                result[
                    "metadata"
                ][
                    "analysis_period"
                ]
            ),
            "overall_passed": (
                summary[
                    "overall_passed"
                ]
            ),
            "ground_truth_event_count": (
                summary[
                    "ground_truth_event_count"
                ]
            ),
            "ai_visible_evidence_ids": (
                summary[
                    "ai_visible_evidence_ids"
                ]
            ),
            "directionally_withheld_evidence_ids": (
                summary[
                    "directionally_withheld_evidence_ids"
                ]
            ),
            "ai_cited_evidence_ids": (
                summary[
                    "ai_cited_evidence_ids"
                ]
            ),
            "invalid_ai_evidence_ids": (
                summary[
                    "invalid_ai_evidence_ids"
                ]
            ),
            "citations_authorized": (
                summary[
                    "citations_authorized"
                ]
            ),
            "citation_precision": (
                summary[
                    "citation_precision"
                ]
            ),
            "ai_visible_evidence_utilization": (
                summary[
                    "ai_visible_evidence_utilization"
                ]
            ),
            "security_isolation_passed": (
                security[
                    "passed"
                ]
            ),
            "pipeline_routing": (
                result[
                    "pipeline_routing"
                ]
            ),
        }

    case_count = len(
        case_summaries
    )

    cases_passed = sum(
        1
        for case in case_summaries.values()
        if case[
            "overall_passed"
        ]
    )

    all_security_isolation_passed = all(
        case[
            "security_isolation_passed"
        ]
        for case in case_summaries.values()
    )

    all_citations_authorized = all(
        case[
            "citations_authorized"
        ]
        for case in case_summaries.values()
    )

    overall_passed = (
        cases_passed == case_count
        and all_security_isolation_passed
        and all_citations_authorized
    )

    return {
        "metadata": {
            "evaluation_type": (
                "offline_hidden_ground_truth_suite"
            ),
            "api_request_sent": False,
            "ground_truth_used_by_model": False,
            "case_keys": list(
                EVALUATION_CASES
            ),
        },
        "summary": {
            "overall_passed": (
                overall_passed
            ),
            "case_count": (
                case_count
            ),
            "cases_passed": (
                cases_passed
            ),
            "cases_failed": (
                case_count
                - cases_passed
            ),
            "case_pass_rate": (
                safe_ratio(
                    cases_passed,
                    case_count,
                )
            ),
            "all_security_isolation_passed": (
                all_security_isolation_passed
            ),
            "all_citations_authorized": (
                all_citations_authorized
            ),
        },
        "aggregate_pipeline_routing": {
            key: aggregate_routing[
                key
            ]
            for key in sorted(
                aggregate_routing
            )
        },
        "cases": case_summaries,
    }


def run_evaluation_suite() -> dict[str, Any]:
    """
    Run every configured period evaluation and return one
    cross-period suite artifact.

    No API calls are made.
    """

    evaluations = {
        case_key: run_evaluation(
            case_key
        )
        for case_key in EVALUATION_CASES
    }

    return build_evaluation_suite(
        evaluations
    )


def save_evaluation_suite(
    suite: dict[str, Any],
    output_path: Path = (
        EVALUATION_SUITE_OUTPUT_PATH
    ),
) -> Path:
    """
    Save the cross-period evaluation-suite artifact.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            suite,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


def print_evaluation_result(
    result: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Print a concise terminal summary for one evaluation case.
    """

    summary = result["summary"]
    routing = result[
        "pipeline_routing"
    ]
    security = result[
        "security_isolation"
    ]

    print(
        "Offline hidden-ground-truth evaluation"
    )
    print(
        "======================================"
    )
    print(
        "Analysis period: "
        f"{result['metadata']['analysis_period']}"
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
    print()
    print(
        "Ground-truth events: "
        f"{summary['ground_truth_event_count']}"
    )
    print(
        "Events with analyst evidence: "
        f"{summary['events_with_analyst_evidence']}"
    )
    print(
        "Events without analyst evidence: "
        f"{summary['events_without_analyst_evidence']}"
    )
    print()
    print(
        "Evidence routing"
    )
    print(
        "----------------"
    )

    for key in sorted(routing):
        print(
            f"{key}: {routing[key]}"
        )

    print()
    print(
        "AI-visible evidence IDs: "
        + (
            ", ".join(
                summary[
                    "ai_visible_evidence_ids"
                ]
            )
            or "NONE"
        )
    )
    print(
        "Directionally withheld evidence IDs: "
        + (
            ", ".join(
                summary[
                    "directionally_withheld_evidence_ids"
                ]
            )
            or "NONE"
        )
    )
    print(
        "AI-cited evidence IDs: "
        + (
            ", ".join(
                summary[
                    "ai_cited_evidence_ids"
                ]
            )
            or "NONE"
        )
    )
    print(
        "Invalid AI evidence IDs: "
        + (
            ", ".join(
                summary[
                    "invalid_ai_evidence_ids"
                ]
            )
            or "NONE"
        )
    )
    print(
        "Citation precision: "
        f"{format_percent(summary['citation_precision'])}"
    )
    print(
        "AI-visible evidence utilization: "
        f"{format_percent(summary['ai_visible_evidence_utilization'])}"
    )
    print()
    print(
        "Security isolation"
    )
    print(
        "------------------"
    )
    print(
        "Ground-truth event IDs leaked "
        "to model payload: "
        + (
            ", ".join(
                security[
                    "leaked_event_ids"
                ]
            )
            or "NONE"
        )
    )
    print(
        "Ground-truth file paths leaked "
        "to model payload: "
        + (
            ", ".join(
                security[
                    "leaked_ground_truth_paths"
                ]
            )
            or "NONE"
        )
    )
    print(
        "Ground truth access flag: BLOCKED"
        if security[
            "ground_truth_access_flag_false"
        ]
        else "Ground truth access flag: INVALID"
    )
    print()
    print(
        "Output saved to: "
        f"{output_path}"
    )
    print(
        "API request sent: NO"
    )


def format_percent(
    value: float | None,
) -> str:
    """
    Format optional ratios for terminal output.
    """

    if value is None:
        return "N/A"

    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline hidden-ground-truth evaluation "
            "for one or all configured analysis periods."
        )
    )

    parser.add_argument(
        "--case",
        choices=[
            "all",
            *EVALUATION_CASES.keys(),
        ],
        default="all",
        help=(
            "Evaluation case to run. Defaults to all configured "
            "analysis periods."
        ),
    )

    args = parser.parse_args()

    if args.case != "all":
        result = run_evaluation(
            args.case
        )

        output_path = save_evaluation(
            result
        )

        print_evaluation_result(
            result,
            output_path,
        )

        return

    evaluations: dict[
        str,
        dict[str, Any],
    ] = {}

    for case_key in EVALUATION_CASES:
        result = run_evaluation(
            case_key
        )

        evaluations[
            case_key
        ] = result

        output_path = save_evaluation(
            result
        )

        print_evaluation_result(
            result,
            output_path,
        )

        print()

    suite = build_evaluation_suite(
        evaluations
    )

    suite_output_path = (
        save_evaluation_suite(
            suite
        )
    )

    suite_summary = suite[
        "summary"
    ]

    print(
        "Cross-period evaluation suite"
    )
    print(
        "============================="
    )
    print(
        "Status: "
        + (
            "PASSED"
            if suite_summary[
                "overall_passed"
            ]
            else "FAILED"
        )
    )
    print(
        "Cases passed: "
        f"{suite_summary['cases_passed']}"
        "/"
        f"{suite_summary['case_count']}"
    )
    print(
        "Case pass rate: "
        f"{format_percent(suite_summary['case_pass_rate'])}"
    )
    print(
        "Security isolation across all cases: "
        + (
            "PASSED"
            if suite_summary[
                "all_security_isolation_passed"
            ]
            else "FAILED"
        )
    )
    print(
        "Citation authorization across all cases: "
        + (
            "PASSED"
            if suite_summary[
                "all_citations_authorized"
            ]
            else "FAILED"
        )
    )
    print(
        "Suite output saved to: "
        f"{suite_output_path}"
    )
    print(
        "API request sent: NO"
    )


if __name__ == "__main__":
    main()