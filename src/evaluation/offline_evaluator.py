from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.ai.cfo_commentary import (
    build_ai_payload,
    build_model_visible_payload,
    load_grounded_analysis,
)
from src.ai.schemas import CFOCommentary


GROUNDING_PATH = Path(
    "outputs/grounding/latest_forecast_vs_budget_grounded.json"
)

AI_OUTPUT_PATH = Path(
    "outputs/ai/latest_forecast_vs_budget_commentary.json"
)

SALES_GROUND_TRUTH_PATH = Path(
    "data/ground_truth/ground_truth_events.csv"
)

OPEX_GROUND_TRUTH_PATH = Path(
    "data/ground_truth/ground_truth_opex_events.csv"
)

EVALUATION_OUTPUT_PATH = Path(
    "outputs/evaluation/latest_forecast_vs_budget_evaluation.json"
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


def load_ground_truth_events() -> list[dict[str, Any]]:
    """
    Load all hidden controlled events.

    IMPORTANT:
    This function belongs only to the offline evaluation layer.
    It must never be imported by the AI generation pipeline.
    """

    sales_rows = load_csv_rows(
        SALES_GROUND_TRUTH_PATH
    )

    opex_rows = load_csv_rows(
        OPEX_GROUND_TRUTH_PATH
    )

    return (
        normalize_sales_ground_truth(
            sales_rows
        )
        + normalize_opex_ground_truth(
            opex_rows
        )
    )


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


def load_ai_output() -> dict[str, Any]:
    """
    Load the latest saved and already-validated AI output.
    """

    if not AI_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            "Validated AI output not found: "
            f"{AI_OUTPUT_PATH}"
        )

    return json.loads(
        AI_OUTPUT_PATH.read_text(
            encoding="utf-8"
        )
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


def run_evaluation() -> dict[str, Any]:
    """
    Run the complete offline hidden-ground-truth evaluation.

    No API calls are made.
    """

    grounded_data = (
        load_grounded_analysis(
            GROUNDING_PATH
        )
    )

    payload = build_ai_payload(
        grounded_data
    )

    ground_truth_events = (
        load_ground_truth_events()
    )

    ai_output = load_ai_output()

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

    overall_passed = (
        security["passed"]
        and not invalid_citations
        and (
            unique_citations
            <= ai_visible_ids
        )
    )

    return {
        "metadata": {
            "evaluation_type": (
                "offline_hidden_ground_truth"
            ),
            "comparison": (
                grounded_data[
                    "metadata"
                ]["comparison"]
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
) -> None:
    """
    Save the offline evaluation artifact.
    """

    EVALUATION_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVALUATION_OUTPUT_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
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
    result = run_evaluation()

    save_evaluation(
        result
    )

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
        f"Output saved to: "
        f"{EVALUATION_OUTPUT_PATH}"
    )
    print(
        "API request sent: NO"
    )


if __name__ == "__main__":
    main()