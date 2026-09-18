from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


DEFAULT_GROUNDED_PATH = Path(
    "outputs/grounding/latest_forecast_vs_budget_grounded.json"
)

DirectionStatus = Literal[
    "directionally_consistent",
    "offsetting",
    "direction_unknown",
]

EvidenceDirection = Literal[
    "positive_ebitda",
    "negative_ebitda",
    "unknown",
]


@dataclass(frozen=True)
class DirectionAssessment:
    """
    Deterministic assessment of whether already-approved evidence is
    directionally consistent with a calculated financial finding.

    Relevance, entity matching and driver compatibility are handled
    upstream by the grounding engine.

    This layer answers only:

    Is the directional financial implication of the evidence consistent
    with the calculated EBITDA-directional impact of the finding?
    """

    finding_id: str
    evidence_id: str
    finding_driver: str
    impact_field: str | None
    finding_driver_impact_nok: float | None
    evidence_direction: EvidenceDirection
    status: DirectionStatus
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_text(value: str) -> str:
    """
    Normalize analyst-visible evidence text for conservative matching.
    """

    return (
        value.lower()
        .replace("–", "-")
        .replace("—", "-")
        .replace("’", "'")
        .strip()
    )


def _contains_any(
    text: str,
    phrases: tuple[str, ...],
) -> bool:
    return any(
        phrase in text
        for phrase in phrases
    )


def _infer_volume_mix_direction(
    text: str,
) -> EvidenceDirection:
    """
    Infer EBITDA direction for volume/mix evidence.
    """

    negative_phrases = (
        "shipment delay",
        "shipment delayed",
        "delivery delay",
        "delivery delayed",
        "postponed shipment",
        "shipment postponement",
        "order cancellation",
        "shipment cancellation",
        "volume decline",
        "shipment shortfall",
    )

    positive_phrases = (
        "shipment acceleration",
        "shipment accelerated",
        "delivery acceleration",
        "pull-forward",
        "pull forward",
        "earlier shipment",
        "shipment uplift",
    )

    negative = _contains_any(
        text,
        negative_phrases,
    )

    positive = _contains_any(
        text,
        positive_phrases,
    )

    if negative and not positive:
        return "negative_ebitda"

    if positive and not negative:
        return "positive_ebitda"

    return "unknown"


def _infer_non_payroll_direction(
    text: str,
) -> EvidenceDirection:
    """
    Infer EBITDA direction for non-payroll OPEX evidence.

    Higher external spend reduces EBITDA.
    Lower external spend increases EBITDA.
    """

    negative_phrases = (
        "contractor acceleration",
        "contractor spend increase",
        "increased contractor spend",
        "higher contractor spend",
        "additional contractor spend",
        "cost increase",
        "spend increase",
        "higher external spend",
        "additional external spend",
        "procurement surcharge",
        "supplier surcharge",
        "surcharge",
    )

    positive_phrases = (
        "contractor spend reduction",
        "reduced contractor spend",
        "lower contractor spend",
        "cost reduction",
        "spend reduction",
        "lower external spend",
        "cost saving",
        "cost savings",
    )

    negative = _contains_any(
        text,
        negative_phrases,
    )

    positive = _contains_any(
        text,
        positive_phrases,
    )

    if negative and not positive:
        return "negative_ebitda"

    if positive and not negative:
        return "positive_ebitda"

    return "unknown"


def _infer_unit_cost_direction(
    text: str,
) -> EvidenceDirection:
    """
    Infer EBITDA direction for unit-cost evidence.
    """

    negative_phrases = (
        "procurement surcharge",
        "supplier surcharge",
        "unit cost increase",
        "higher unit cost",
        "increased unit cost",
        "higher procurement cost",
        "cost inflation",
    )

    positive_phrases = (
        "unit cost reduction",
        "lower unit cost",
        "procurement saving",
        "procurement savings",
        "supplier saving",
        "supplier savings",
    )

    negative = _contains_any(
        text,
        negative_phrases,
    )

    positive = _contains_any(
        text,
        positive_phrases,
    )

    if negative and not positive:
        return "negative_ebitda"

    if positive and not negative:
        return "positive_ebitda"

    return "unknown"


def _infer_discount_direction(
    text: str,
) -> EvidenceDirection:
    """
    Infer EBITDA direction for discount evidence.
    """

    negative_phrases = (
        "higher discount",
        "increased discount",
        "discount increase",
        "additional discount",
        "deeper discount",
    )

    positive_phrases = (
        "lower discount",
        "reduced discount",
        "discount reduction",
    )

    negative = _contains_any(
        text,
        negative_phrases,
    )

    positive = _contains_any(
        text,
        positive_phrases,
    )

    if negative and not positive:
        return "negative_ebitda"

    if positive and not negative:
        return "positive_ebitda"

    return "unknown"


def _infer_list_price_direction(
    text: str,
) -> EvidenceDirection:
    """
    Infer EBITDA direction for list-price evidence.
    """

    negative_phrases = (
        "price reduction",
        "price decrease",
        "lower list price",
        "price cut",
    )

    positive_phrases = (
        "price increase",
        "higher list price",
        "list price increase",
    )

    negative = _contains_any(
        text,
        negative_phrases,
    )

    positive = _contains_any(
        text,
        positive_phrases,
    )

    if negative and not positive:
        return "negative_ebitda"

    if positive and not negative:
        return "positive_ebitda"

    return "unknown"


def _infer_people_cost_direction(
    text: str,
) -> EvidenceDirection:
    """
    Infer EBITDA direction for payroll/headcount evidence.
    """

    negative_phrases = (
        "headcount increase",
        "increased headcount",
        "additional headcount",
        "hiring increase",
        "salary increase",
        "wage increase",
        "higher employee cost",
        "higher payroll cost",
    )

    positive_phrases = (
        "headcount reduction",
        "reduced headcount",
        "salary reduction",
        "employee cost reduction",
        "payroll reduction",
        "payroll saving",
        "payroll savings",
    )

    negative = _contains_any(
        text,
        negative_phrases,
    )

    positive = _contains_any(
        text,
        positive_phrases,
    )

    if negative and not positive:
        return "negative_ebitda"

    if positive and not negative:
        return "positive_ebitda"

    return "unknown"


def infer_evidence_direction(
    driver: str,
    evidence: dict[str, Any],
) -> EvidenceDirection:
    """
    Infer the EBITDA-directional implication of approved evidence.

    Conservative rule:
    unknown is always preferred over guessing.
    """

    text_parts = [
        str(evidence.get("title", "")),
        str(evidence.get("evidence_text", "")),
        str(evidence.get("text", "")),
        str(evidence.get("excerpt", "")),
        str(evidence.get("content", "")),
        str(evidence.get("summary", "")),
    ]

    text = _normalize_text(
        " ".join(text_parts)
    )

    if driver == "volume_mix":
        return _infer_volume_mix_direction(text)

    if driver == "non_payroll":
        return _infer_non_payroll_direction(text)

    if driver == "unit_cost":
        return _infer_unit_cost_direction(text)

    if driver == "discount":
        return _infer_discount_direction(text)

    if driver == "list_price":
        return _infer_list_price_direction(text)

    if driver in {
        "headcount",
        "employee_cost",
        "payroll",
    }:
        return _infer_people_cost_direction(text)

    return "unknown"


def _get_directional_impact(
    finding: dict[str, Any],
) -> tuple[float | None, str | None]:
    """
    Determine the EBITDA-directional impact of a finding.

    Sales findings normally contain:
        primary_driver_impact_nok

    OPEX findings normally contain:
        ebitda_impact_nok

    Both use the same sign convention:
        positive = favorable EBITDA contribution
        negative = unfavorable EBITDA contribution
    """

    calculated_fact = finding.get(
        "calculated_fact",
        {},
    )

    preferred_fields = (
        "primary_driver_impact_nok",
        "ebitda_impact_nok",
    )

    for field in preferred_fields:
        value = calculated_fact.get(field)

        if isinstance(value, bool):
            continue

        if isinstance(
            value,
            (int, float),
        ):
            return float(value), field

    return None, None


def _resolve_driver(
    finding: dict[str, Any],
    evidence: dict[str, Any],
) -> tuple[str, str]:
    """
    Resolve the driver used by the directional guardrail.

    Primary preference:
        deterministic finding driver

    Fallback:
        already-approved evidence driver

    The fallback is permitted only because this function operates after
    the grounding engine has already approved the evidence for the finding.
    """

    calculated_fact = finding.get(
        "calculated_fact",
        {},
    )

    finding_driver = calculated_fact.get(
        "primary_driver"
    )

    evidence_driver = evidence.get(
        "primary_evidence_driver"
    )

    if finding_driver:
        return str(finding_driver), "finding"

    if evidence_driver:
        return str(evidence_driver), "approved_evidence"

    return "", "unknown"


def assess_evidence_direction(
    finding: dict[str, Any],
    evidence: dict[str, Any],
) -> DirectionAssessment:
    """
    Compare an approved evidence item with the EBITDA-directional
    impact of the calculated finding.
    """

    finding_id = str(
        finding.get(
            "finding_id",
            "UNKNOWN_FINDING",
        )
    )

    evidence_id = str(
        evidence.get(
            "evidence_id",
            "UNKNOWN_EVIDENCE",
        )
    )

    calculated_fact = finding.get(
        "calculated_fact",
        {},
    )

    raw_finding_driver = calculated_fact.get(
        "primary_driver"
    )

    raw_evidence_driver = evidence.get(
        "primary_evidence_driver"
    )

    if (
        raw_finding_driver
        and raw_evidence_driver
        and str(raw_finding_driver)
        != str(raw_evidence_driver)
    ):
        return DirectionAssessment(
            finding_id=finding_id,
            evidence_id=evidence_id,
            finding_driver=str(
                raw_finding_driver
            ),
            impact_field=None,
            finding_driver_impact_nok=None,
            evidence_direction="unknown",
            status="direction_unknown",
            reason=(
                "Evidence driver does not match the "
                "deterministic finding driver: "
                f"{raw_evidence_driver} vs "
                f"{raw_finding_driver}."
            ),
        )

    driver, driver_source = _resolve_driver(
        finding,
        evidence,
    )

    if not driver:
        return DirectionAssessment(
            finding_id=finding_id,
            evidence_id=evidence_id,
            finding_driver="",
            impact_field=None,
            finding_driver_impact_nok=None,
            evidence_direction="unknown",
            status="direction_unknown",
            reason=(
                "No deterministic or approved-evidence "
                "driver is available."
            ),
        )

    impact, impact_field = _get_directional_impact(
        finding
    )

    if impact is None:
        return DirectionAssessment(
            finding_id=finding_id,
            evidence_id=evidence_id,
            finding_driver=driver,
            impact_field=None,
            finding_driver_impact_nok=None,
            evidence_direction="unknown",
            status="direction_unknown",
            reason=(
                "No numeric EBITDA-directional impact "
                "is available for the finding."
            ),
        )

    if abs(impact) < 1e-9:
        return DirectionAssessment(
            finding_id=finding_id,
            evidence_id=evidence_id,
            finding_driver=driver,
            impact_field=impact_field,
            finding_driver_impact_nok=impact,
            evidence_direction="unknown",
            status="direction_unknown",
            reason=(
                "The calculated EBITDA-directional impact "
                "is zero."
            ),
        )

    evidence_direction = infer_evidence_direction(
        driver,
        evidence,
    )

    if evidence_direction == "unknown":
        return DirectionAssessment(
            finding_id=finding_id,
            evidence_id=evidence_id,
            finding_driver=driver,
            impact_field=impact_field,
            finding_driver_impact_nok=impact,
            evidence_direction="unknown",
            status="direction_unknown",
            reason=(
                "Approved evidence is relevant, but its "
                "directional financial implication cannot "
                "be determined conservatively."
            ),
        )

    finding_sign = (
        1
        if impact > 0
        else -1
    )

    evidence_sign = (
        1
        if evidence_direction == "positive_ebitda"
        else -1
    )

    if finding_sign == evidence_sign:
        return DirectionAssessment(
            finding_id=finding_id,
            evidence_id=evidence_id,
            finding_driver=driver,
            impact_field=impact_field,
            finding_driver_impact_nok=impact,
            evidence_direction=evidence_direction,
            status="directionally_consistent",
            reason=(
                "Evidence direction is consistent with the "
                "sign of the calculated EBITDA-directional "
                f"impact. Driver source: {driver_source}."
            ),
        )

    return DirectionAssessment(
        finding_id=finding_id,
        evidence_id=evidence_id,
        finding_driver=driver,
        impact_field=impact_field,
        finding_driver_impact_nok=impact,
        evidence_direction=evidence_direction,
        status="offsetting",
        reason=(
            "Evidence direction is opposite to the sign of "
            "the calculated EBITDA-directional impact and "
            "therefore represents an offsetting or "
            "countervailing factor rather than an "
            f"explanation of the variance. Driver source: "
            f"{driver_source}."
        ),
    )


def filter_directionally_consistent_evidence(
    finding: dict[str, Any],
    evidence_items: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[DirectionAssessment],
]:
    """
    Keep only evidence that is directionally consistent.

    Offsetting and direction-unknown evidence is withheld from the
    AI explanatory payload.
    """

    accepted: list[dict[str, Any]] = []

    assessments: list[
        DirectionAssessment
    ] = []

    for evidence in evidence_items:
        assessment = assess_evidence_direction(
            finding,
            evidence,
        )

        assessments.append(
            assessment
        )

        if (
            assessment.status
            == "directionally_consistent"
        ):
            accepted.append(
                evidence
            )

    return accepted, assessments


def inspect_grounded_analysis(
    path: Path = DEFAULT_GROUNDED_PATH,
) -> list[DirectionAssessment]:
    """
    Inspect all currently approved evidence in a grounded file.

    Zero API calls.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Grounded analysis not found: {path}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assessments: list[
        DirectionAssessment
    ] = []

    for finding in data.get(
        "findings",
        [],
    ):
        grounding = finding.get(
            "grounding",
            {},
        )

        usable_ids = set(
            grounding.get(
                "usable_evidence_ids",
                [],
            )
        )

        if not usable_ids:
            continue

        for candidate in grounding.get(
            "evidence_candidates",
            [],
        ):
            if (
                candidate.get("evidence_id")
                not in usable_ids
            ):
                continue

            assessments.append(
                assess_evidence_direction(
                    finding,
                    candidate,
                )
            )

    return assessments


def main() -> None:
    assessments = inspect_grounded_analysis()

    print(
        "Directional evidence guardrail inspection"
    )
    print(
        "========================================"
    )

    for assessment in assessments:
        impact_m = (
            assessment.finding_driver_impact_nok
            / 1_000_000
            if assessment.finding_driver_impact_nok
            is not None
            else None
        )

        impact_text = (
            f"{impact_m:.2f}m NOK"
            if impact_m is not None
            else "UNKNOWN"
        )

        print()
        print(
            f"Finding: {assessment.finding_id}"
        )
        print(
            f"Evidence: {assessment.evidence_id}"
        )
        print(
            f"Driver: {assessment.finding_driver}"
        )
        print(
            f"Impact field: "
            f"{assessment.impact_field or 'UNKNOWN'}"
        )
        print(
            f"Calculated EBITDA-directional impact: "
            f"{impact_text}"
        )
        print(
            "Evidence direction: "
            f"{assessment.evidence_direction}"
        )
        print(
            f"Status: {assessment.status}"
        )
        print(
            f"Reason: {assessment.reason}"
        )

    consistent = sum(
        assessment.status
        == "directionally_consistent"
        for assessment in assessments
    )

    offsetting = sum(
        assessment.status
        == "offsetting"
        for assessment in assessments
    )

    unknown = sum(
        assessment.status
        == "direction_unknown"
        for assessment in assessments
    )

    print()
    print(
        "Directional guardrail summary"
    )
    print(
        "============================="
    )
    print(
        f"Assessments: {len(assessments)}"
    )
    print(
        f"Directionally consistent: {consistent}"
    )
    print(
        f"Offsetting: {offsetting}"
    )
    print(
        f"Direction unknown: {unknown}"
    )
    print(
        "API request sent: NO"
    )


if __name__ == "__main__":
    main()