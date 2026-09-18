from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """
    Base model for all structured AI outputs.

    Extra fields are rejected so the AI cannot silently add
    unsupported categories or information outside the schema.
    """

    model_config = ConfigDict(extra="forbid")


class CalculatedObservation(StrictModel):
    """
    A factual observation based on deterministic finance calculations.

    The AI is allowed to communicate the finding, but it is not
    responsible for calculating it.
    """

    finding_id: str = Field(
        description="Identifier of the deterministic financial finding."
    )
    statement: str = Field(
        description=(
            "Concise factual management statement based only on the "
            "supplied deterministic financial finding."
        )
    )
    source_type: Literal["calculated_fact"] = Field(
        description="Identifies the statement as a deterministic calculated fact."
    )


class EvidenceBackedExplanation(StrictModel):
    """
    An explanation that is explicitly supported by analyst-visible evidence.
    """

    finding_id: str = Field(
        description="Identifier of the financial finding being explained."
    )
    explanation: str = Field(
        description=(
            "Explanation of the finding using only the supplied usable evidence."
        )
    )
    evidence_ids: list[str] = Field(
        description=(
            "Evidence IDs that directly support the explanation. "
            "Every ID must exist in the supplied usable evidence."
        )
    )
    source_type: Literal["evidence_supported"] = Field(
        description="Identifies the explanation as evidence-supported."
    )


class UnresolvedFinding(StrictModel):
    """
    A material finding for which the supplied evidence does not support
    an underlying explanation.
    """

    finding_id: str = Field(
        description="Identifier of the unresolved financial finding."
    )
    statement: str = Field(
        description=(
            "Concise statement explaining that the underlying cause "
            "cannot be determined from the available evidence."
        )
    )
    reason: Literal["insufficient_evidence"] = Field(
        description="The only permitted reason for an unresolved finding."
    )


class ManagementQuestion(StrictModel):
    """
    A management follow-up question.

    Questions are allowed because they identify where additional evidence
    would be useful without presenting an unsupported hypothesis as fact.
    """

    related_finding_ids: list[str] = Field(
        description="Financial finding IDs related to the question."
    )
    question: str = Field(
        description=(
            "A concise question management could investigate to obtain "
            "additional evidence."
        )
    )


class CFOCommentary(StrictModel):
    """
    Complete structured output from the CFO Intelligence Copilot.
    """

    executive_summary: str = Field(
        description=(
            "Short CFO-level summary of the most material financial developments. "
            "It must distinguish facts from explanations and must not invent causes."
        )
    )

    observations: list[CalculatedObservation] = Field(
        description=(
            "Material factual observations derived from deterministic calculations."
        )
    )

    supported_explanations: list[EvidenceBackedExplanation] = Field(
        description=(
            "Only explanations that are directly supported by supplied evidence."
        )
    )

    unresolved_findings: list[UnresolvedFinding] = Field(
        description=(
            "Material findings where the available evidence is insufficient "
            "to determine the underlying cause."
        )
    )

    management_questions: list[ManagementQuestion] = Field(
        description=(
            "Evidence-seeking follow-up questions for unresolved or important findings."
        )
    )


if __name__ == "__main__":
    print("AI output schemas loaded successfully")
    print("Top-level schema: CFOCommentary")
    print(
        "Sections: executive_summary, observations, supported_explanations, "
        "unresolved_findings, management_questions"
    )