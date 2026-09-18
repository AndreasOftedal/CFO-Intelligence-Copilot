from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from openai import OpenAI


DEFAULT_MODEL = "gpt-5.6-luna"

ANSWER_TYPES = {
    "supported_explanation",
    "calculated_fact",
    "insufficient_evidence",
}

GROUND_TRUTH_PATTERNS = (
    re.compile(r"\bEVT-\d+\b", re.IGNORECASE),
    re.compile(r"\bOPEX-\d+\b", re.IGNORECASE),
    re.compile(r"data[/\\]ground_truth", re.IGNORECASE),
    re.compile(r"ground[_ -]?truth[_ -]?events", re.IGNORECASE),
)

STOP_WORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "explain",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "me",
    "more",
    "most",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "own",
    "same",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}

ENTITY_GROUPS = (
    ("norway",),
    ("sweden",),
    ("denmark",),
    ("germany",),
    ("netherlands",),
    ("united kingdom", "uk"),
    ("edgehub pro",),
    ("edgehub core",),
    ("control mini",),
    ("sensor x pro",),
    ("sensor x",),
    ("sensor lite",),
    ("service kit",),
    ("connectivity pack",),
    ("product & r&d", "product r&d"),
)


DOMAIN_EXPANSIONS = {
    "ebitda": {
        "ebitda",
        "revenue",
        "gross",
        "profit",
        "opex",
        "margin",
        "budget",
        "forecast",
    },
    "revenue": {
        "revenue",
        "sales",
        "volume",
        "mix",
        "price",
        "discount",
        "budget",
        "forecast",
    },
    "sales": {
        "revenue",
        "sales",
        "volume",
        "mix",
        "price",
        "discount",
    },
    "margin": {
        "margin",
        "gross",
        "profit",
        "price",
        "discount",
        "cost",
        "unit",
    },
    "opex": {
        "opex",
        "cost",
        "cost centre",
        "account",
        "contractors",
        "marketing",
        "payroll",
        "employee",
    },
    "cost": {
        "cost",
        "unit",
        "opex",
        "contractors",
        "payroll",
        "employee",
        "supplier",
    },
    "uk": {
        "uk",
        "sensor",
        "shipment",
        "delay",
        "volume",
        "revenue",
    },
    "germany": {
        "germany",
        "edgehub",
        "discount",
        "cost",
        "revenue",
        "gross",
        "profit",
    },
    "norway": {
        "norway",
        "sensor",
        "acceleration",
        "revenue",
        "gross",
        "profit",
    },
    "product": {
        "product",
        "edgehub",
        "sensor",
        "service",
        "connectivity",
    },
    "country": {
        "country",
        "norway",
        "sweden",
        "denmark",
        "germany",
        "netherlands",
        "uk",
    },
}


@dataclass(frozen=True)
class ContextRecord:
    finding_id: str
    text: str
    evidence_ids: tuple[str, ...] = ()


class CFOQAError(RuntimeError):
    """Raised when the CFO Q&A layer cannot safely return an answer."""


def get_openai_api_key() -> str | None:
    """
    Read the API key from the process environment.

    Streamlit should copy st.secrets["OPENAI_API_KEY"] into the
    environment or pass the key explicitly to ask_cfo().
    """
    value = os.getenv("OPENAI_API_KEY")
    if value:
        return value.strip()

    return None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def _normalise_finding_id(value: Any) -> str:
    return _safe_text(value)


def _normalise_evidence_ids(
    value: Any,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        return ()

    output: list[str] = []

    for item in value:
        text = _safe_text(item)
        if text and text not in output:
            output.append(text)

    return tuple(output)


def _first_non_empty(
    mapping: dict[str, Any],
    keys: Iterable[str],
) -> str:
    for key in keys:
        value = _safe_text(mapping.get(key))
        if value:
            return value

    return ""


def build_approved_context(
    commentary: dict[str, Any],
    ai_metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the only context the interactive CFO assistant is allowed to see.

    The function intentionally relies on the already-controlled production
    outputs:
      - deterministic calculated observations
      - evidence-supported explanations
      - unresolved findings
      - approved evidence IDs from the directional guardrail layer

    Hidden ground truth is never accepted as input.
    """

    approved_ids = set(
        _normalise_evidence_ids(
            ai_metadata.get(
                "source_usable_evidence_ids",
                [],
            )
        )
    )

    directional = ai_metadata.get(
        "directional_guardrail_summary",
        {},
    )

    withheld_ids = set(
        _normalise_evidence_ids(
            directional.get(
                "withheld_evidence_ids",
                [],
            )
        )
    )

    approved_ids -= withheld_ids

    facts: list[ContextRecord] = []

    # The production commentary schema stores deterministic finance
    # observations under "observations". "calculated_observations" is
    # retained as a backwards-compatible fallback for older test fixtures
    # or saved outputs.
    raw_observations = commentary.get(
        "observations"
    )

    if raw_observations is None:
        raw_observations = commentary.get(
            "calculated_observations",
            [],
        )

    for item in raw_observations:
        if not isinstance(item, dict):
            continue

        finding_id = _normalise_finding_id(
            item.get("finding_id")
        )
        statement = _first_non_empty(
            item,
            (
                "statement",
                "observation",
                "text",
            ),
        )

        if finding_id and statement:
            facts.append(
                ContextRecord(
                    finding_id=finding_id,
                    text=statement,
                )
            )

    supported: list[ContextRecord] = []

    for item in commentary.get(
        "supported_explanations",
        [],
    ):
        if not isinstance(item, dict):
            continue

        finding_id = _normalise_finding_id(
            item.get("finding_id")
        )
        explanation = _first_non_empty(
            item,
            (
                "explanation",
                "statement",
                "text",
            ),
        )

        evidence_ids = tuple(
            evidence_id
            for evidence_id in _normalise_evidence_ids(
                item.get("evidence_ids", [])
            )
            if evidence_id in approved_ids
        )

        if (
            finding_id
            and explanation
            and evidence_ids
        ):
            supported.append(
                ContextRecord(
                    finding_id=finding_id,
                    text=explanation,
                    evidence_ids=evidence_ids,
                )
            )

    unresolved: list[ContextRecord] = []

    for item in commentary.get(
        "unresolved_findings",
        [],
    ):
        if not isinstance(item, dict):
            continue

        finding_id = _normalise_finding_id(
            item.get("finding_id")
        )
        statement = _first_non_empty(
            item,
            (
                "statement",
                "finding",
                "observation",
                "text",
                "reason",
            ),
        )

        if finding_id and statement:
            unresolved.append(
                ContextRecord(
                    finding_id=finding_id,
                    text=statement,
                )
            )

    return {
        "facts": facts,
        "supported": supported,
        "unresolved": unresolved,
        "approved_evidence_ids": sorted(
            approved_ids
        ),
        "withheld_evidence_ids": sorted(
            withheld_ids
        ),
    }


def _tokenise(value: str) -> set[str]:
    words = re.findall(
        r"[a-z0-9]+",
        value.lower(),
    )

    return {
        word
        for word in words
        if (
            len(word) >= 2
            and word not in STOP_WORDS
        )
    }


def _expanded_query_terms(
    question: str,
) -> set[str]:
    lowered = question.lower()
    terms = _tokenise(question)

    for trigger, additions in (
        DOMAIN_EXPANSIONS.items()
    ):
        if trigger in lowered:
            for addition in additions:
                terms.update(
                    _tokenise(addition)
                )

    return terms


def _record_score(
    question: str,
    query_terms: set[str],
    record: ContextRecord,
) -> float:
    combined = (
        f"{record.finding_id} {record.text}"
    ).lower()

    record_terms = _tokenise(combined)

    overlap = len(
        query_terms.intersection(
            record_terms
        )
    )

    score = float(overlap)

    question_lower = question.lower()

    finding_entity = (
        record.finding_id.split(
            "::",
            1,
        )[-1]
        .replace("_", " ")
        .lower()
    )

    if (
        finding_entity
        and finding_entity in question_lower
    ):
        score += 8.0

    finding_type = (
        record.finding_id.split(
            "::",
            1,
        )[0]
        .replace("_", " ")
        .lower()
    )

    if (
        finding_type
        and finding_type in question_lower
    ):
        score += 2.0

    for phrase in (
        "ebitda",
        "gross profit",
        "revenue",
        "opex",
        "margin",
        "volume",
        "discount",
        "unit cost",
        "contractor",
        "marketing",
        "payroll",
        "forecast",
        "budget",
    ):
        if (
            phrase in question_lower
            and phrase in combined
        ):
            score += 2.0

    return score


def _question_entity_groups(
    question: str,
) -> list[tuple[str, ...]]:
    lowered = question.lower()

    return [
        group
        for group in ENTITY_GROUPS
        if any(
            entity in lowered
            for entity in group
        )
    ]


def _record_matches_entity_groups(
    record: ContextRecord,
    entity_groups: list[
        tuple[str, ...]
    ],
) -> bool:
    if not entity_groups:
        return True

    combined = (
        f"{record.finding_id} {record.text}"
    ).lower()

    return all(
        any(
            entity in combined
            for entity in group
        )
        for group in entity_groups
    )


def _rank_records(
    question: str,
    records: list[ContextRecord],
    limit: int,
    *,
    require_entity_match: bool = False,
    fallback_if_no_match: bool = True,
    fill_with_remaining: bool = False,
) -> list[ContextRecord]:
    if not records:
        return []

    candidate_records = records

    if require_entity_match:
        entity_groups = (
            _question_entity_groups(
                question
            )
        )

        if entity_groups:
            candidate_records = [
                record
                for record in records
                if _record_matches_entity_groups(
                    record,
                    entity_groups,
                )
            ]

            if not candidate_records:
                return []

    query_terms = _expanded_query_terms(
        question
    )

    scored = [
        (
            _record_score(
                question,
                query_terms,
                record,
            ),
            index,
            record,
        )
        for index, record in enumerate(
            candidate_records
        )
    ]

    scored.sort(
        key=lambda item: (
            item[0],
            -item[1],
        ),
        reverse=True,
    )

    positive = [
        record
        for score, _, record in scored
        if score > 0
    ]

    if positive:
        selected = positive[:limit]

        if (
            fill_with_remaining
            and len(selected) < limit
        ):
            selected_ids = {
                id(record)
                for record in selected
            }

            for _, _, record in scored:
                if id(record) in selected_ids:
                    continue

                selected.append(record)
                selected_ids.add(
                    id(record)
                )

                if len(selected) >= limit:
                    break

        return selected

    if not fallback_if_no_match:
        return []

    return [
        record
        for _, _, record in scored[:limit]
    ]


def _is_broad_company_question(
    question: str,
) -> bool:
    lowered = question.lower()

    if _question_entity_groups(
        question
    ):
        return False

    broad_terms = (
        "ebitda",
        "overall",
        "company",
        "business",
        "total",
        "group",
        "forecast performance",
        "financial performance",
    )

    return any(
        term in lowered
        for term in broad_terms
    )


def select_relevant_context(
    question: str,
    commentary: dict[str, Any],
    ai_metadata: dict[str, Any],
    *,
    max_facts: int = 8,
    max_supported: int = 6,
    max_unresolved: int = 6,
) -> dict[str, Any]:
    approved = build_approved_context(
        commentary=commentary,
        ai_metadata=ai_metadata,
    )

    selected_facts = _rank_records(
        question,
        approved["facts"],
        max_facts,
    )

    broad_company_question = (
        _is_broad_company_question(
            question
        )
    )

    selected_supported = _rank_records(
        question,
        approved["supported"],
        max_supported,
        require_entity_match=True,
        fallback_if_no_match=(
            broad_company_question
        ),
        fill_with_remaining=(
            broad_company_question
        ),
    )

    selected_unresolved = _rank_records(
        question,
        approved["unresolved"],
        max_unresolved,
    )

    selected_evidence_ids = sorted(
        {
            evidence_id
            for record in selected_supported
            for evidence_id in (
                record.evidence_ids
            )
        }
    )

    return {
        "facts": selected_facts,
        "supported": selected_supported,
        "unresolved": selected_unresolved,
        "selected_evidence_ids": (
            selected_evidence_ids
        ),
        "approved_evidence_ids": (
            approved[
                "approved_evidence_ids"
            ]
        ),
        "withheld_evidence_ids": (
            approved[
                "withheld_evidence_ids"
            ]
        ),
    }


def _record_to_payload(
    record: ContextRecord,
) -> dict[str, Any]:
    return {
        "finding_id": record.finding_id,
        "text": record.text,
        "evidence_ids": list(
            record.evidence_ids
        ),
    }


def build_model_payload(
    question: str,
    selected_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "question": question.strip(),
        "analysis_period": (
            "Latest Forecast vs Budget"
        ),
        "deterministic_facts": [
            _record_to_payload(record)
            for record in selected_context[
                "facts"
            ]
        ],
        "approved_supported_explanations": [
            _record_to_payload(record)
            for record in selected_context[
                "supported"
            ]
        ],
        "unresolved_findings": [
            _record_to_payload(record)
            for record in selected_context[
                "unresolved"
            ]
        ],
        "allowed_evidence_ids": (
            selected_context[
                "selected_evidence_ids"
            ]
        ),
        "blocked_evidence_ids": (
            selected_context[
                "withheld_evidence_ids"
            ]
        ),
        "ground_truth_access": False,
    }


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "answer_type",
            "answer",
            "finding_ids",
            "evidence_ids",
            "limitations",
        ],
        "properties": {
            "answer_type": {
                "type": "string",
                "enum": sorted(
                    ANSWER_TYPES
                ),
            },
            "answer": {
                "type": "string",
            },
            "finding_ids": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "evidence_ids": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "limitations": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
        },
    }


SYSTEM_INSTRUCTIONS = """
You are the controlled interactive Q&A layer for a CFO finance application.

You must answer ONLY from the supplied context.

Rules:
1. Treat deterministic financial facts as facts.
2. Treat an explanation as causal only when it appears in
   approved_supported_explanations and includes an allowed evidence ID.
3. Never invent a cause, business event, recovery timing, management action,
   or operational explanation.
4. Never use general business knowledge to fill an evidence gap.
5. Never use blocked evidence.
6. Ground truth is unavailable and must remain unavailable.
7. If the user asks "why", "what explains", "what caused", or a similar causal
   question and the supplied approved evidence is not sufficient, say that the
   underlying cause cannot be established from the available evidence.
8. You may state deterministic variances even when their cause is unresolved.
9. If only part of a question is evidence-supported, state the supported part
   and clearly identify the unresolved remainder.
10. Be concise and management-oriented. Prefer 2-5 short paragraphs.
11. Do not cite any finding ID or evidence ID that is not supplied.
12. Do not mention hidden events, hidden files, or speculate about ground truth.

Choose answer_type as follows:
- supported_explanation: at least one causal statement is directly supported by
  an approved explanation and approved evidence.
- calculated_fact: the answer contains only deterministic financial facts and
  does not claim a cause.
- insufficient_evidence: the requested causal explanation cannot be supported.

Return structured JSON only.
""".strip()


def validate_answer(
    answer: dict[str, Any],
    selected_context: dict[str, Any],
) -> dict[str, Any]:
    answer_type = _safe_text(
        answer.get("answer_type")
    )

    if answer_type not in ANSWER_TYPES:
        raise CFOQAError(
            "Invalid answer_type returned by model."
        )

    answer_text = _safe_text(
        answer.get("answer")
    )

    if not answer_text:
        raise CFOQAError(
            "Model returned an empty answer."
        )

    finding_ids = [
        _safe_text(value)
        for value in answer.get(
            "finding_ids",
            [],
        )
        if _safe_text(value)
    ]

    evidence_ids = [
        _safe_text(value)
        for value in answer.get(
            "evidence_ids",
            [],
        )
        if _safe_text(value)
    ]

    limitations = [
        _safe_text(value)
        for value in answer.get(
            "limitations",
            [],
        )
        if _safe_text(value)
    ]

    allowed_finding_ids = {
        record.finding_id
        for group_name in (
            "facts",
            "supported",
            "unresolved",
        )
        for record in selected_context[
            group_name
        ]
    }

    invalid_findings = sorted(
        set(finding_ids)
        - allowed_finding_ids
    )

    if invalid_findings:
        raise CFOQAError(
            "Model cited unknown finding IDs: "
            + ", ".join(
                invalid_findings
            )
        )

    selected_evidence_ids = set(
        selected_context[
            "selected_evidence_ids"
        ]
    )

    invalid_evidence = sorted(
        set(evidence_ids)
        - selected_evidence_ids
    )

    if invalid_evidence:
        raise CFOQAError(
            "Model cited evidence that was not "
            "approved for this question: "
            + ", ".join(
                invalid_evidence
            )
        )

    withheld_ids = set(
        selected_context[
            "withheld_evidence_ids"
        ]
    )

    leaked_withheld = sorted(
        evidence_id
        for evidence_id in withheld_ids
        if (
            evidence_id in evidence_ids
            or evidence_id.lower()
            in answer_text.lower()
        )
    )

    if leaked_withheld:
        raise CFOQAError(
            "Model attempted to use withheld "
            "evidence: "
            + ", ".join(
                leaked_withheld
            )
        )

    combined_output = " ".join(
        [
            answer_text,
            *finding_ids,
            *evidence_ids,
            *limitations,
        ]
    )

    for pattern in GROUND_TRUTH_PATTERNS:
        if pattern.search(
            combined_output
        ):
            raise CFOQAError(
                "Ground-truth identifier or path "
                "appeared in model output."
            )

    if (
        answer_type
        == "supported_explanation"
        and not evidence_ids
    ):
        raise CFOQAError(
            "A supported explanation must cite "
            "at least one approved evidence ID."
        )

    if (
        answer_type
        in {
            "calculated_fact",
            "insufficient_evidence",
        }
        and evidence_ids
    ):
        raise CFOQAError(
            f"{answer_type} must not cite "
            "causal evidence."
        )

    return {
        "answer_type": answer_type,
        "answer": answer_text,
        "finding_ids": finding_ids,
        "evidence_ids": evidence_ids,
        "limitations": limitations,
    }


def ask_cfo(
    question: str,
    commentary: dict[str, Any],
    ai_metadata: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
) -> dict[str, Any]:
    question = question.strip()

    if not question:
        raise CFOQAError(
            "Question cannot be empty."
        )

    selected = select_relevant_context(
        question=question,
        commentary=commentary,
        ai_metadata=ai_metadata,
    )

    payload = build_model_payload(
        question=question,
        selected_context=selected,
    )

    diagnostics = {
        "model": model,
        "selected_fact_count": len(
            selected["facts"]
        ),
        "selected_supported_count": len(
            selected["supported"]
        ),
        "selected_unresolved_count": len(
            selected["unresolved"]
        ),
        "selected_evidence_ids": (
            selected[
                "selected_evidence_ids"
            ]
        ),
        "withheld_evidence_ids": (
            selected[
                "withheld_evidence_ids"
            ]
        ),
        "ground_truth_access": False,
        "api_request_sent": False,
    }

    if dry_run:
        return {
            "answer": None,
            "diagnostics": diagnostics,
            "model_payload": payload,
        }

    resolved_api_key = (
        api_key
        or get_openai_api_key()
    )

    if not resolved_api_key:
        raise CFOQAError(
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
        max_output_tokens=900,
        text={
            "format": {
                "type": "json_schema",
                "name": "cfo_qa_answer",
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
        raise CFOQAError(
            "Model response was not valid JSON."
        ) from exc

    validated = validate_answer(
        answer=parsed,
        selected_context=selected,
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
        "answer": validated,
        "diagnostics": diagnostics,
        "model_payload": payload,
    }
