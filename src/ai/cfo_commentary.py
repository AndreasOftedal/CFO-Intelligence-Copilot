from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from src.ai.ai_config import AIConfig, load_ai_config
from src.ai.schemas import CFOCommentary
from src.evidence.direction_guardrail import (
    filter_directionally_consistent_evidence,
)


DEFAULT_INPUT_PATH = Path(
    "outputs/grounding/latest_forecast_vs_budget_grounded.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/ai/latest_forecast_vs_budget_commentary.json"
)


SYSTEM_INSTRUCTIONS = """
You are the narrative layer of a CFO Intelligence Copilot.

All financial calculations have already been performed by a deterministic
finance engine. You are NOT a calculation engine.

The evidence supplied to you has already passed:
- entity compatibility checks,
- driver compatibility checks,
- grounding checks,
- directional consistency checks.

Only evidence that passed all applicable deterministic guardrails is included
as usable evidence.

STRICT RULES:

1. Never calculate, recalculate, estimate, interpolate, or alter financial facts.
   Use only the supplied calculated facts.

2. Never invent a finding ID.
   Use finding IDs exactly as supplied in the payload.

3. Observations may reference:
   - IDs in headline_facts, or
   - IDs in material_findings.

4. Evidence-backed explanations may reference ONLY material_findings whose
   explanation_status is "evidence_available".

5. An explanation may use ONLY evidence contained in that finding's
   usable_evidence list.

6. Never transfer evidence between countries, products, accounts,
   cost centres, segments, entities, or drivers.

7. If a material finding has explanation_status = "insufficient_evidence":
   - do not infer a cause,
   - do not suggest a likely cause,
   - do not state a hypothesis as an explanation,
   - treat the underlying cause as unresolved.

8. Unresolved findings may reference ONLY material_findings whose
   explanation_status is "insufficient_evidence".

9. Every evidence-backed explanation must cite at least one evidence_id,
   and every cited evidence_id must exist in that finding's usable_evidence list.

10. Management questions may seek additional evidence, but they must not imply
    that an unsupported cause is probably true.

11. Prioritize material and decision-relevant information.
    You do not need to mention every supplied finding.

12. Headline facts are calculated facts only.
    They must never be used as evidence of an underlying cause.

13. Fields ending in "_nok_m" are NOK millions.
    Fields ending in "_pct" are percentages.
    Fields ending in "_pp" are percentage-point changes.

14. You do not have access to hidden ground truth.

15. The output must conform exactly to the CFOCommentary schema.
""".strip()


def load_grounded_analysis(
    path: Path,
) -> dict[str, Any]:
    """
    Load deterministic grounded analysis.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Grounded analysis file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    required_keys = {
        "metadata",
        "finance_summary",
        "grounding_summary",
        "findings",
    }

    missing = required_keys - set(data)

    if missing:
        raise ValueError(
            "Grounded analysis is missing required "
            f"top-level keys: {sorted(missing)}"
        )

    if (
        data["metadata"].get(
            "ground_truth_access"
        )
        is not False
    ):
        raise RuntimeError(
            "AI layer refused to continue because "
            "ground_truth_access is not explicitly false."
        )

    return data


def normalize_fact_value(
    key: str,
    value: Any,
) -> tuple[str, Any]:
    """
    Convert deterministic finance values into management-friendly
    units before they reach the model.
    """

    if isinstance(value, bool):
        return key, value

    if not isinstance(
        value,
        (int, float),
    ):
        return key, value

    if key.endswith("_nok"):
        return (
            key[:-4] + "_nok_m",
            round(float(value) / 1_000_000, 2),
        )

    if key.endswith("_margin"):
        return (
            key + "_pct",
            round(float(value) * 100, 2),
        )

    if key.endswith("_pp"):
        return (
            key,
            round(float(value), 2),
        )

    if isinstance(value, int):
        return key, value

    return (
        key,
        round(float(value), 4),
    )


def normalize_fact_dict(
    fact_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize a dictionary of deterministic finance facts.
    """

    normalized: dict[str, Any] = {}

    for key, value in fact_dict.items():
        new_key, new_value = (
            normalize_fact_value(
                key,
                value,
            )
        )

        normalized[new_key] = new_value

    return normalized


def build_headline_facts(
    finance_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Create stable IDs for headline P&L observations.
    """

    normalized = normalize_fact_dict(
        finance_summary
    )

    definitions = [
        (
            "summary::revenue",
            "Revenue",
            [
                "base_revenue_nok_m",
                "comparison_revenue_nok_m",
                "revenue_variance_nok_m",
            ],
        ),
        (
            "summary::gross_profit",
            "Gross Profit",
            [
                "base_gross_profit_nok_m",
                "comparison_gross_profit_nok_m",
                "gross_profit_variance_nok_m",
            ],
        ),
        (
            "summary::opex",
            "OPEX",
            [
                "base_opex_nok_m",
                "comparison_opex_nok_m",
                "opex_variance_nok_m",
            ],
        ),
        (
            "summary::ebitda",
            "EBITDA",
            [
                "base_ebitda_nok_m",
                "comparison_ebitda_nok_m",
                "ebitda_variance_nok_m",
            ],
        ),
        (
            "summary::ebitda_margin",
            "EBITDA Margin",
            [
                "base_ebitda_margin_pct",
                "comparison_ebitda_margin_pct",
                "ebitda_margin_variance_pp",
            ],
        ),
    ]

    headline_facts: list[
        dict[str, Any]
    ] = []

    for (
        finding_id,
        entity,
        keys,
    ) in definitions:
        calculated_fact = {
            key: normalized[key]
            for key in keys
            if key in normalized
        }

        if not calculated_fact:
            continue

        headline_facts.append(
            {
                "finding_id": finding_id,
                "finding_type": (
                    "finance_summary"
                ),
                "entity": entity,
                "calculated_fact": (
                    calculated_fact
                ),
            }
        )

    return headline_facts


def sanitize_evidence_candidate(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """
    Keep only analyst-visible evidence fields needed downstream.

    Ground-truth content is explicitly rejected.
    """

    source_file = str(
        candidate.get(
            "source_file",
            "",
        )
    )

    if (
        "ground_truth"
        in source_file.lower()
    ):
        raise RuntimeError(
            "Ground-truth evidence was detected "
            "in the AI payload. Request blocked."
        )

    sanitized: dict[str, Any] = {
        "evidence_id": candidate.get(
            "evidence_id"
        ),
        "title": candidate.get("title"),
        "source_file": candidate.get(
            "source_file"
        ),
        "support_status": candidate.get(
            "support_status"
        ),
        "primary_evidence_driver": (
            candidate.get(
                "primary_evidence_driver"
            )
        ),
        "evidence_entities": candidate.get(
            "evidence_entities",
            {},
        ),
    }

    possible_text_fields = (
        "evidence_text",
        "text",
        "excerpt",
        "content",
        "summary",
    )

    for field in possible_text_fields:
        if candidate.get(field):
            sanitized[field] = (
                candidate[field]
            )

    return sanitized


def build_ai_payload(
    grounded_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the only payload the LLM is permitted to see.

    Evidence must pass two separate deterministic stages:

    1. Grounding engine:
       entity + driver + evidence compatibility.

    2. Direction guardrail:
       evidence direction must be consistent with the calculated
       EBITDA-directional impact.

    Evidence that fails either stage is withheld from the model.
    """

    headline_facts = build_headline_facts(
        grounded_data["finance_summary"]
    )

    material_findings: list[
        dict[str, Any]
    ] = []

    direction_assessments: list[
        dict[str, Any]
    ] = []

    withheld_evidence_ids: set[str] = set()

    for finding in grounded_data["findings"]:
        if not finding.get(
            "material",
            False,
        ):
            continue

        grounding = finding.get(
            "grounding",
            {},
        )

        upstream_status = grounding.get(
            "explanation_status"
        )

        if upstream_status not in {
            "evidence_available",
            "insufficient_evidence",
        }:
            raise ValueError(
                "Unexpected explanation_status for "
                f"{finding.get('finding_id')}: "
                f"{upstream_status}"
            )

        usable_ids = set(
            grounding.get(
                "usable_evidence_ids",
                [],
            )
        )

        grounded_evidence: list[
            dict[str, Any]
        ] = []

        if (
            upstream_status
            == "evidence_available"
        ):
            for candidate in grounding.get(
                "evidence_candidates",
                [],
            ):
                evidence_id = (
                    candidate.get(
                        "evidence_id"
                    )
                )

                if evidence_id in usable_ids:
                    grounded_evidence.append(
                        sanitize_evidence_candidate(
                            candidate
                        )
                    )

            found_ids = {
                evidence["evidence_id"]
                for evidence
                in grounded_evidence
            }

            missing_ids = (
                usable_ids - found_ids
            )

            if missing_ids:
                raise RuntimeError(
                    "Grounding output lists usable "
                    "evidence IDs that were not found "
                    "among evidence candidates for "
                    f"{finding.get('finding_id')}: "
                    f"{sorted(missing_ids)}"
                )

        elif usable_ids:
            raise RuntimeError(
                f"{finding.get('finding_id')} "
                "is marked insufficient_evidence "
                "but contains usable evidence IDs."
            )

        directionally_approved: list[
            dict[str, Any]
        ] = []

        if grounded_evidence:
            (
                directionally_approved,
                assessments,
            ) = (
                filter_directionally_consistent_evidence(
                    finding,
                    grounded_evidence,
                )
            )

            for assessment in assessments:
                assessment_dict = (
                    assessment.to_dict()
                )

                direction_assessments.append(
                    assessment_dict
                )

                if (
                    assessment.status
                    != "directionally_consistent"
                ):
                    withheld_evidence_ids.add(
                        assessment.evidence_id
                    )

        if (
            upstream_status
            == "evidence_available"
            and directionally_approved
        ):
            effective_status = (
                "evidence_available"
            )

            allowed_ai_behavior = (
                "Use only the supplied usable "
                "evidence. The evidence has passed "
                "grounding and directional consistency "
                "checks. Do not generalize beyond its "
                "documented scope."
            )

        else:
            effective_status = (
                "insufficient_evidence"
            )

            directionally_approved = []

            allowed_ai_behavior = (
                "State that available approved evidence "
                "is insufficient to determine the "
                "underlying cause. Do not infer or "
                "invent a cause."
            )

        material_findings.append(
            {
                "finding_id": finding.get(
                    "finding_id"
                ),
                "finding_type": finding.get(
                    "finding_type"
                ),
                "entity": finding.get(
                    "entity"
                ),
                "material": True,
                "calculated_fact": (
                    normalize_fact_dict(
                        finding.get(
                            "calculated_fact",
                            {},
                        )
                    )
                ),
                "explanation_status": (
                    effective_status
                ),
                "usable_evidence": (
                    directionally_approved
                ),
                "allowed_ai_behavior": (
                    allowed_ai_behavior
                ),
            }
        )

    headline_ids = {
        item["finding_id"]
        for item in headline_facts
    }

    material_ids = {
        item["finding_id"]
        for item in material_findings
    }

    evidence_available_ids = {
        item["finding_id"]
        for item in material_findings
        if item["explanation_status"]
        == "evidence_available"
    }

    insufficient_ids = {
        item["finding_id"]
        for item in material_findings
        if item["explanation_status"]
        == "insufficient_evidence"
    }

    directional_consistent_count = sum(
        assessment["status"]
        == "directionally_consistent"
        for assessment
        in direction_assessments
    )

    offsetting_count = sum(
        assessment["status"]
        == "offsetting"
        for assessment
        in direction_assessments
    )

    unknown_count = sum(
        assessment["status"]
        == "direction_unknown"
        for assessment
        in direction_assessments
    )

    payload = {
        "metadata": {
            "comparison": (
                grounded_data["metadata"].get(
                    "comparison"
                )
            ),
            "grounding_engine": (
                grounded_data["metadata"].get(
                    "grounding_engine"
                )
            ),
            "grounding_version": (
                grounded_data["metadata"].get(
                    "grounding_version"
                )
            ),
            "directional_guardrail": True,
            "ground_truth_access": False,
        },
        "unit_conventions": {
            "_nok_m": "NOK millions",
            "_pct": "percent",
            "_pp": "percentage points",
        },
        "allowed_ids": {
            "observations": sorted(
                headline_ids
                | material_ids
            ),
            "supported_explanations": sorted(
                evidence_available_ids
            ),
            "unresolved_findings": sorted(
                insufficient_ids
            ),
            "management_questions": sorted(
                material_ids
            ),
        },
        "headline_facts": headline_facts,
        "material_findings": (
            material_findings
        ),
        "_internal": {
            "upstream_grounding_summary": (
                grounded_data[
                    "grounding_summary"
                ]
            ),
            "directional_guardrail": {
                "assessment_count": len(
                    direction_assessments
                ),
                "directionally_consistent_count": (
                    directional_consistent_count
                ),
                "offsetting_count": (
                    offsetting_count
                ),
                "direction_unknown_count": (
                    unknown_count
                ),
                "withheld_evidence_ids": (
                    sorted(
                        withheld_evidence_ids
                    )
                ),
                "assessments": (
                    direction_assessments
                ),
            },
        },
    }

    return payload


def build_model_visible_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove internal diagnostics before constructing the API request.

    The model sees only approved finance facts and usable evidence.
    """

    return {
        key: value
        for key, value in payload.items()
        if key != "_internal"
    }


def validate_commentary_against_payload(
    commentary: CFOCommentary,
    payload: dict[str, Any],
) -> None:
    """
    Deterministically validate semantic permissions after
    Structured Outputs has validated response shape.
    """

    headline_map = {
        item["finding_id"]: item
        for item
        in payload["headline_facts"]
    }

    material_map = {
        item["finding_id"]: item
        for item
        in payload["material_findings"]
    }

    allowed = payload["allowed_ids"]

    allowed_observation_ids = set(
        allowed["observations"]
    )

    allowed_explanation_ids = set(
        allowed[
            "supported_explanations"
        ]
    )

    allowed_unresolved_ids = set(
        allowed["unresolved_findings"]
    )

    allowed_question_ids = set(
        allowed["management_questions"]
    )

    for observation in (
        commentary.observations
    ):
        if (
            observation.finding_id
            not in allowed_observation_ids
        ):
            raise ValueError(
                "AI observation references "
                "unauthorized finding ID: "
                f"{observation.finding_id}"
            )

        if (
            observation.finding_id
            not in headline_map
            and observation.finding_id
            not in material_map
        ):
            raise ValueError(
                "AI observation references "
                "finding ID absent from payload: "
                f"{observation.finding_id}"
            )

    for explanation in (
        commentary.supported_explanations
    ):
        finding_id = (
            explanation.finding_id
        )

        if (
            finding_id
            not in allowed_explanation_ids
        ):
            raise ValueError(
                "AI attempted an explanation "
                "for a finding that was not "
                "authorized for explanation: "
                f"{finding_id}"
            )

        source_finding = (
            material_map[finding_id]
        )

        allowed_evidence_ids = {
            evidence["evidence_id"]
            for evidence
            in source_finding[
                "usable_evidence"
            ]
        }

        cited_evidence_ids = set(
            explanation.evidence_ids
        )

        if not cited_evidence_ids:
            raise ValueError(
                "Evidence-backed explanation "
                "contains no evidence IDs: "
                f"{finding_id}"
            )

        unauthorized_ids = (
            cited_evidence_ids
            - allowed_evidence_ids
        )

        if unauthorized_ids:
            raise ValueError(
                "AI cited evidence that was "
                "not approved by the grounding "
                "and directional guardrails for "
                f"{finding_id}: "
                f"{sorted(unauthorized_ids)}"
            )

    for unresolved in (
        commentary.unresolved_findings
    ):
        finding_id = (
            unresolved.finding_id
        )

        if (
            finding_id
            not in allowed_unresolved_ids
        ):
            raise ValueError(
                "AI marked a finding as "
                "unresolved that was not "
                "authorized as unresolved: "
                f"{finding_id}"
            )

    for question in (
        commentary.management_questions
    ):
        for finding_id in (
            question.related_finding_ids
        ):
            if (
                finding_id
                not in allowed_question_ids
            ):
                raise ValueError(
                    "Management question "
                    "references unauthorized "
                    "finding ID: "
                    f"{finding_id}"
                )


def create_client(
    config: AIConfig,
) -> OpenAI:
    """
    Create OpenAI client with explicit timeout and retry controls.
    """

    return OpenAI(
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


def generate_commentary(
    payload: dict[str, Any],
    config: AIConfig,
) -> tuple[CFOCommentary, Any]:
    """
    Generate Structured Output through the Responses API.
    """

    client = create_client(
        config
    )

    model_payload = (
        build_model_visible_payload(
            payload
        )
    )

    user_input = (
        "Prepare CFO management commentary "
        "from the following deterministic "
        "finance and approved evidence payload. "
        "Use only IDs explicitly listed under "
        "allowed_ids.\n\n"
        + json.dumps(
            model_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )

    try:
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
                        "content": user_input,
                    },
                ],
                text_format=CFOCommentary,
            )
        )

    except ValidationError as exc:
        raise RuntimeError(
            "Structured model output could "
            "not be parsed. The response may "
            "have been cut off by the output "
            "token limit. Do not automatically "
            "retry."
        ) from exc

    if response.status != "completed":
        raise RuntimeError(
            "OpenAI response did not complete "
            "successfully. "
            f"Status: {response.status}; "
            f"Incomplete details: "
            f"{response.incomplete_details}"
        )

    commentary = response.output_parsed

    if commentary is None:
        raise RuntimeError(
            "The API returned no parsed "
            "CFOCommentary object."
        )

    return commentary, response


def save_output(
    output_path: Path,
    commentary: CFOCommentary,
    response: Any,
    payload: dict[str, Any],
    config: AIConfig,
) -> None:
    """
    Save validated commentary and audit metadata locally.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    usage = None

    if response.usage is not None:
        if hasattr(
            response.usage,
            "model_dump",
        ):
            usage = (
                response.usage.model_dump()
            )
        else:
            usage = str(
                response.usage
            )

    internal = payload.get(
        "_internal",
        {},
    )

    directional_metadata = (
        internal.get(
            "directional_guardrail",
            {},
        )
    )

    source_usable_evidence_ids = sorted(
        {
            evidence["evidence_id"]
            for finding
            in payload[
                "material_findings"
            ]
            for evidence
            in finding[
                "usable_evidence"
            ]
        }
    )

    output_document = {
        "metadata": {
            "comparison": (
                payload["metadata"][
                    "comparison"
                ]
            ),
            "model": config.model,
            "reasoning_effort": (
                config.reasoning_effort
            ),
            "response_id": response.id,
            "response_status": (
                response.status
            ),
            "store_response": (
                config.store_responses
            ),
            "usage": usage,
            "validated_against_grounding": (
                True
            ),
            "directional_guardrail_applied": (
                True
            ),
            "directional_guardrail_summary": {
                "assessment_count": (
                    directional_metadata.get(
                        "assessment_count",
                        0,
                    )
                ),
                "directionally_consistent_count": (
                    directional_metadata.get(
                        "directionally_consistent_count",
                        0,
                    )
                ),
                "offsetting_count": (
                    directional_metadata.get(
                        "offsetting_count",
                        0,
                    )
                ),
                "direction_unknown_count": (
                    directional_metadata.get(
                        "direction_unknown_count",
                        0,
                    )
                ),
                "withheld_evidence_ids": (
                    directional_metadata.get(
                        "withheld_evidence_ids",
                        [],
                    )
                ),
            },
            "ground_truth_access": False,
            "source_headline_fact_ids": [
                item["finding_id"]
                for item
                in payload[
                    "headline_facts"
                ]
            ],
            "source_material_finding_ids": [
                item["finding_id"]
                for item
                in payload[
                    "material_findings"
                ]
            ],
            "source_usable_evidence_ids": (
                source_usable_evidence_ids
            ),
        },
        "commentary": (
            commentary.model_dump()
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_document,
            file,
            ensure_ascii=False,
            indent=2,
        )


def print_dry_run_summary(
    payload: dict[str, Any],
    config: AIConfig,
) -> None:
    """
    Show exactly what would reach the model without making an API call.
    """

    findings = payload[
        "material_findings"
    ]

    evidence_available = [
        finding
        for finding in findings
        if (
            finding[
                "explanation_status"
            ]
            == "evidence_available"
        )
    ]

    insufficient = [
        finding
        for finding in findings
        if (
            finding[
                "explanation_status"
            ]
            == "insufficient_evidence"
        )
    ]

    evidence_ids = sorted(
        {
            evidence["evidence_id"]
            for finding in findings
            for evidence
            in finding[
                "usable_evidence"
            ]
        }
    )

    internal = payload.get(
        "_internal",
        {},
    )

    directional = internal.get(
        "directional_guardrail",
        {},
    )

    print(
        "CFO Commentary AI dry run PASSED"
    )
    print(
        f"Model: {config.model}"
    )
    print(
        "Comparison: "
        f"{payload['metadata']['comparison']}"
    )
    print(
        "Headline facts supplied: "
        f"{len(payload['headline_facts'])}"
    )
    print(
        "Material findings supplied: "
        f"{len(findings)}"
    )
    print(
        "Findings with usable evidence "
        "after directional guardrail: "
        f"{len(evidence_available)}"
    )
    print(
        "Findings with insufficient "
        "evidence after directional guardrail: "
        f"{len(insufficient)}"
    )
    print(
        "Usable evidence IDs supplied to AI: "
        + (
            ", ".join(evidence_ids)
            if evidence_ids
            else "NONE"
        )
    )
    print(
        "Directional assessments: "
        f"{directional.get('assessment_count', 0)}"
    )
    print(
        "Directionally consistent: "
        f"{directional.get('directionally_consistent_count', 0)}"
    )
    print(
        "Offsetting: "
        f"{directional.get('offsetting_count', 0)}"
    )
    print(
        "Direction unknown: "
        f"{directional.get('direction_unknown_count', 0)}"
    )

    withheld = directional.get(
        "withheld_evidence_ids",
        [],
    )

    print(
        "Evidence withheld from AI by "
        "directional guardrail: "
        + (
            ", ".join(withheld)
            if withheld
            else "NONE"
        )
    )

    print(
        "Allowed headline observation IDs: "
        + ", ".join(
            item["finding_id"]
            for item
            in payload[
                "headline_facts"
            ]
        )
    )
    print(
        "Ground truth access: BLOCKED"
    )
    print(
        "API request sent: NO"
    )


def parse_arguments() -> (
    argparse.Namespace
):
    parser = argparse.ArgumentParser(
        description=(
            "Generate evidence-grounded "
            "CFO management commentary."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Path to grounded analysis JSON."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Path for validated AI "
            "commentary JSON."
        ),
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help=(
            "Send one API request. "
            "Without --run, perform "
            "a zero-cost dry run."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config = load_ai_config()

    grounded_data = (
        load_grounded_analysis(
            args.input
        )
    )

    payload = build_ai_payload(
        grounded_data
    )

    if not args.run:
        print_dry_run_summary(
            payload,
            config,
        )
        return

    print(
        "Sending one structured CFO "
        "commentary request to OpenAI..."
    )

    commentary, response = (
        generate_commentary(
            payload,
            config,
        )
    )

    validate_commentary_against_payload(
        commentary,
        payload,
    )

    save_output(
        args.output,
        commentary,
        response,
        payload,
        config,
    )

    print(
        "CFO Commentary generation PASSED"
    )
    print(
        f"Output saved to: {args.output}"
    )

    if response.usage is not None:
        print(
            "Input tokens: "
            f"{response.usage.input_tokens}"
        )
        print(
            "Output tokens: "
            f"{response.usage.output_tokens}"
        )
        print(
            "Total tokens: "
            f"{response.usage.total_tokens}"
        )

        output_details = getattr(
            response.usage,
            "output_tokens_details",
            None,
        )

        if output_details is not None:
            reasoning_tokens = getattr(
                output_details,
                "reasoning_tokens",
                None,
            )

            if reasoning_tokens is not None:
                print(
                    "Reasoning tokens: "
                    f"{reasoning_tokens}"
                )


if __name__ == "__main__":
    main()