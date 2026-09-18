from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "gpt-5.6-luna"

SCENARIO_BRIDGE_LABELS = {
    "volume_mix": "Volume & mix",
    "list_price": "List price",
    "discount": "Discount",
    "unit_cost": "Unit cost",
    "headcount": "Headcount",
    "employee_cost": "Employee cost",
    "non_payroll": "Non-payroll OPEX",
}

UNSUPPORTED_OPERATIONAL_PATTERNS = (
    (
        re.compile(
            r"\bcustomer\s+(?:demand|behaviou?r|churn|loss|losses|orders?)\b",
            re.IGNORECASE,
        ),
        "customer demand or behaviour",
    ),
    (
        re.compile(
            r"\bdemand\s+(?:weakness|decline|drop|softness|growth|recovery)\b",
            re.IGNORECASE,
        ),
        "demand conditions",
    ),
    (
        re.compile(
            r"\bsuppliers?\b|\bsupply\s+(?:disruption|shortage|constraint)\b",
            re.IGNORECASE,
        ),
        "supplier or supply event",
    ),
    (
        re.compile(
            r"\bshipments?\b|\bshipping\s+delay\b",
            re.IGNORECASE,
        ),
        "shipment event",
    ),
    (
        re.compile(
            r"\bcompetitors?\b|\bcompetitive\s+pressure\b",
            re.IGNORECASE,
        ),
        "competitive explanation",
    ),
    (
        re.compile(
            r"\bmarket\s+conditions\b|\bmacroeconomic\b|\bmacro\s+conditions\b",
            re.IGNORECASE,
        ),
        "market or macro explanation",
    ),
    (
        re.compile(
            r"\boutages?\b|\bdelays?\b|\bdelayed\b|\blaunch(?:es|ed)?\b|\bchurn\b",
            re.IGNORECASE,
        ),
        "operational event",
    ),
    (
        re.compile(
            r"\brecovery\s+timing\b|\bmanagement\s+actions?\b",
            re.IGNORECASE,
        ),
        "unsupported management or timing explanation",
    ),
)


class ScenarioBriefError(RuntimeError):
    """Raised when the controlled scenario-brief layer rejects a response."""


def get_openai_api_key() -> str | None:
    value = os.getenv(
        "OPENAI_API_KEY"
    )

    if not value:
        return None

    return value.strip() or None


def _safe_float(
    value: Any,
) -> float:
    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ScenarioBriefError(
            "Scenario result contains a non-numeric value."
        ) from exc


def _active_bridge_rows(
    scenario_result: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for item in scenario_result.get(
        "ebitda_bridge",
        [],
    ):
        driver = str(
            item.get(
                "driver",
                "",
            )
        ).strip()

        if not driver:
            continue

        impact_nok = _safe_float(
            item.get(
                "impact_nok",
                0.0,
            )
        )

        if abs(impact_nok) < 0.005:
            continue

        rows.append(
            {
                "driver": driver,
                "driver_label": (
                    SCENARIO_BRIDGE_LABELS.get(
                        driver,
                        driver.replace(
                            "_",
                            " ",
                        ).title(),
                    )
                ),
                "impact_nok": impact_nok,
                "material": bool(
                    item.get(
                        "material",
                        False,
                    )
                ),
            }
        )

    return rows


def build_scenario_payload(
    scenario_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the only context that can reach the scenario AI layer.

    The model receives deterministic scenario assumptions, calculated
    financial outcomes and the reconciled EBITDA bridge. It receives no
    management evidence, source documents or hidden ground truth.
    """

    summary = scenario_result.get(
        "summary",
        {},
    )

    inputs = scenario_result.get(
        "inputs",
        {},
    ).get(
        "percentage_points",
        {},
    )

    reconciliation = scenario_result.get(
        "reconciliation",
        {},
    )

    active_bridge = _active_bridge_rows(
        scenario_result
    )

    return {
        "scenario_type": (
            "deterministic_what_if"
        ),
        "baseline": (
            scenario_result.get(
                "metadata",
                {},
            ).get(
                "baseline_label",
                "Latest Forecast",
            )
        ),
        "scenario_assumptions": {
            str(key): _safe_float(value)
            for key, value
            in inputs.items()
        },
        "calculated_outcomes": {
            "revenue_change_nok": _safe_float(
                summary.get(
                    "revenue_change_nok",
                    0.0,
                )
            ),
            "gross_profit_change_nok": _safe_float(
                summary.get(
                    "gross_profit_change_nok",
                    0.0,
                )
            ),
            "opex_change_nok": _safe_float(
                summary.get(
                    "opex_change_nok",
                    0.0,
                )
            ),
            "ebitda_change_nok": _safe_float(
                summary.get(
                    "ebitda_change_nok",
                    0.0,
                )
            ),
            "ebitda_margin_change_pp": _safe_float(
                summary.get(
                    "ebitda_margin_change_pp",
                    0.0,
                )
            ),
        },
        "active_ebitda_drivers": (
            active_bridge
        ),
        "reconciliation": {
            "passed": bool(
                reconciliation.get(
                    "passed",
                    False,
                )
            ),
            "residual_nok": _safe_float(
                reconciliation.get(
                    "residual_nok",
                    0.0,
                )
            ),
        },
        "evidence_access": False,
        "ground_truth_access": False,
    }


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "headline",
            "executive_summary",
            "driver_ids",
            "management_takeaways",
            "limitations",
        ],
        "properties": {
            "headline": {
                "type": "string",
            },
            "executive_summary": {
                "type": "string",
            },
            "driver_ids": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "management_takeaways": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "string",
                },
            },
            "limitations": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "string",
                },
            },
        },
    }


SYSTEM_INSTRUCTIONS = """
You are the controlled Scenario Brief layer for a CFO finance application.

The supplied context contains only deterministic what-if calculations.

Rules:
1. Treat every supplied financial value as already calculated and final.
2. Do not calculate, estimate or alter any financial value.
3. Do not invent business events, causes, customer behaviour, market conditions,
   management actions, recovery timing or operational explanations.
4. You may explain the mechanical relationship between an explicit scenario
   assumption and the resulting financial direction.
5. Do not infer why management selected an assumption. For example, a higher
   discount may be described as a higher discount assumption, but never as a
   response to customer demand, competition or market conditions.
6. Use only driver IDs present in active_ebitda_drivers.
7. Do not mention evidence, hidden files, source documents or ground truth.
8. Do not include numeric values in headline, executive_summary,
   management_takeaways or limitations. The UI displays deterministic numbers
   separately.
9. Focus on direction, relative importance and mechanical trade-offs.
10. If no active EBITDA driver exists, say the scenario is unchanged from the
    baseline and return an empty driver_ids list.
11. Keep the brief concise and management-oriented.

Return structured JSON only.
""".strip()


def _contains_numeric_text(
    value: str,
) -> bool:
    return bool(
        re.search(
            r"\d",
            value,
        )
    )


def _contains_speculation(
    value: str,
) -> str | None:
    for pattern, label in UNSUPPORTED_OPERATIONAL_PATTERNS:
        if pattern.search(
            value
        ):
            return label

    return None


def validate_scenario_brief(
    brief: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    headline = str(
        brief.get(
            "headline",
            "",
        )
    ).strip()

    executive_summary = str(
        brief.get(
            "executive_summary",
            "",
        )
    ).strip()

    driver_ids = [
        str(value).strip()
        for value
        in brief.get(
            "driver_ids",
            [],
        )
        if str(value).strip()
    ]

    management_takeaways = [
        str(value).strip()
        for value
        in brief.get(
            "management_takeaways",
            [],
        )
        if str(value).strip()
    ]

    limitations = [
        str(value).strip()
        for value
        in brief.get(
            "limitations",
            [],
        )
        if str(value).strip()
    ]

    if not headline:
        raise ScenarioBriefError(
            "Scenario brief headline is empty."
        )

    if not executive_summary:
        raise ScenarioBriefError(
            "Scenario brief executive summary is empty."
        )

    if len(
        management_takeaways
    ) > 3:
        raise ScenarioBriefError(
            "Scenario brief contains too many management takeaways."
        )

    if len(
        limitations
    ) > 3:
        raise ScenarioBriefError(
            "Scenario brief contains too many limitations."
        )

    allowed_driver_ids = {
        str(item["driver"])
        for item
        in payload.get(
            "active_ebitda_drivers",
            [],
        )
    }

    invalid_driver_ids = sorted(
        set(driver_ids)
        - allowed_driver_ids
    )

    if invalid_driver_ids:
        raise ScenarioBriefError(
            "Scenario brief cited unsupported driver IDs: "
            + ", ".join(
                invalid_driver_ids
            )
        )

    if (
        allowed_driver_ids
        and not driver_ids
    ):
        raise ScenarioBriefError(
            "Scenario brief omitted all active EBITDA drivers."
        )

    if (
        not allowed_driver_ids
        and driver_ids
    ):
        raise ScenarioBriefError(
            "Scenario brief cited drivers for an unchanged scenario."
        )

    narrative_fields = [
        headline,
        executive_summary,
        *management_takeaways,
        *limitations,
    ]

    for text in narrative_fields:
        if _contains_numeric_text(
            text
        ):
            raise ScenarioBriefError(
                "Scenario brief narrative must not contain numeric values."
            )

        speculative_term = (
            _contains_speculation(
                text
            )
        )

        if speculative_term:
            raise ScenarioBriefError(
                "Scenario brief introduced an unsupported "
                f"operational explanation: {speculative_term}"
            )

    return {
        "headline": headline,
        "executive_summary": (
            executive_summary
        ),
        "driver_ids": driver_ids,
        "management_takeaways": (
            management_takeaways
        ),
        "limitations": limitations,
    }


def generate_scenario_brief(
    scenario_result: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
) -> dict[str, Any]:
    payload = build_scenario_payload(
        scenario_result
    )

    diagnostics = {
        "model": model,
        "ground_truth_access": False,
        "evidence_access": False,
        "api_request_sent": False,
        "active_driver_ids": [
            item["driver"]
            for item
            in payload[
                "active_ebitda_drivers"
            ]
        ],
    }

    if dry_run:
        return {
            "brief": None,
            "diagnostics": diagnostics,
            "model_payload": payload,
        }

    resolved_api_key = (
        api_key
        or get_openai_api_key()
    )

    if not resolved_api_key:
        raise ScenarioBriefError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=resolved_api_key
    )

    response = client.responses.create(
        model=model,
        store=False,
        instructions=SYSTEM_INSTRUCTIONS,
        input=json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        max_output_tokens=650,
        text={
            "format": {
                "type": "json_schema",
                "name": "scenario_brief",
                "strict": True,
                "schema": _response_schema(),
            }
        },
    )

    diagnostics[
        "api_request_sent"
    ] = True

    try:
        parsed = json.loads(
            response.output_text
        )
    except (
        json.JSONDecodeError,
        TypeError,
    ) as exc:
        raise ScenarioBriefError(
            "Model response was not valid JSON."
        ) from exc

    validated = validate_scenario_brief(
        brief=parsed,
        payload=payload,
    )

    usage = getattr(
        response,
        "usage",
        None,
    )

    if usage is not None:
        diagnostics["usage"] = {
            "input_tokens": getattr(
                usage,
                "input_tokens",
                None,
            ),
            "output_tokens": getattr(
                usage,
                "output_tokens",
                None,
            ),
            "total_tokens": getattr(
                usage,
                "total_tokens",
                None,
            ),
        }

    return {
        "brief": validated,
        "diagnostics": diagnostics,
        "model_payload": payload,
    }
