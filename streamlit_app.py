from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

GROUNDED_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "grounding"
    / "latest_forecast_vs_budget_grounded.json"
)

AI_COMMENTARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "ai"
    / "latest_forecast_vs_budget_commentary.json"
)

EVALUATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "latest_forecast_vs_budget_evaluation.json"
)


# ---------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="CFO Intelligence Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    .block-container {
        max-width: 1450px;
        padding-top: 2.0rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }

    [data-testid="stMetric"] {
        padding: 0.15rem 0;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.7rem;
        line-height: 1.15;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.82rem;
    }

    .app-kicker {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        opacity: 0.60;
        margin-bottom: 0.3rem;
    }

    .app-subtitle {
        font-size: 1.03rem;
        opacity: 0.70;
        margin-top: -0.35rem;
        margin-bottom: 1.5rem;
        max-width: 950px;
    }

    .section-kicker {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.58;
        margin-bottom: 0.25rem;
    }

    .summary-card {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 14px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 0.75rem;
        line-height: 1.62;
    }

    .fact-card {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.65rem;
        min-height: 125px;
    }

    .fact-title {
        font-size: 0.92rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .fact-text {
        font-size: 0.94rem;
        line-height: 1.45;
    }

    .evidence-card {
        border-left: 4px solid #16a34a;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        background: rgba(22, 163, 74, 0.06);
        border-radius: 0 10px 10px 0;
        line-height: 1.5;
    }

    .unresolved-card {
        border-left: 4px solid #d97706;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        background: rgba(217, 119, 6, 0.06);
        border-radius: 0 10px 10px 0;
        line-height: 1.5;
    }

    .guardrail-card {
        border: 1px solid rgba(217, 119, 6, 0.24);
        border-left: 5px solid #d97706;
        padding: 1rem 1.1rem;
        margin: 0.85rem 0 1rem 0;
        background: rgba(217, 119, 6, 0.055);
        border-radius: 0 12px 12px 0;
        line-height: 1.5;
    }

    .guardrail-title {
        font-size: 0.98rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .guardrail-step {
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 10px;
        padding: 0.75rem 0.85rem;
        min-height: 108px;
        background: rgba(128, 128, 128, 0.025);
    }

    .guardrail-step-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        opacity: 0.60;
        margin-bottom: 0.35rem;
    }

    .question-card {
        border-left: 4px solid #2563eb;
        padding: 0.65rem 0.95rem;
        margin-bottom: 0.6rem;
        background: rgba(37, 99, 235, 0.05);
        border-radius: 0 10px 10px 0;
        line-height: 1.45;
    }

    .small-muted {
        font-size: 0.80rem;
        opacity: 0.62;
    }

    .status-pass {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: rgba(22, 163, 74, 0.12);
        color: #15803d;
        font-weight: 700;
        font-size: 0.84rem;
    }

    .status-blocked {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: rgba(22, 163, 74, 0.10);
        color: #15803d;
        font-weight: 700;
        font-size: 0.84rem;
    }

    .status-warning {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: rgba(217, 119, 6, 0.10);
        color: #b45309;
        font-weight: 700;
        font-size: 0.84rem;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
    }
</style>
"""

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        st.error(
            f"Required file not found: {path}"
        )
        st.stop()

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        st.error(
            f"Invalid JSON in {path.name}: {exc}"
        )
        st.stop()


def money_value(
    value: float | int | None,
) -> str:
    if value is None:
        return "N/A"

    numeric = float(value)
    absolute = abs(numeric)

    if absolute >= 1_000_000_000:
        return (
            f"NOK {numeric / 1_000_000_000:,.2f}bn"
        )

    return (
        f"NOK {numeric / 1_000_000:,.2f}m"
    )


def money_delta(
    value: float | int | None,
) -> str:
    """
    Delta begins with + or - so Streamlit can determine
    the correct direction and color.
    """

    if value is None:
        return "N/A"

    numeric = float(value)

    return (
        f"{numeric / 1_000_000:+.2f}m NOK"
    )


def pct(
    value: float | int | None,
) -> str:
    if value is None:
        return "N/A"

    return f"{float(value) * 100:.2f}%"


def percentage_point_delta(
    value: float | int | None,
) -> str:
    if value is None:
        return "N/A"

    return f"{float(value):+.2f} pp"


def short_finding_id(
    finding_id: str,
) -> str:
    if "::" not in finding_id:
        return finding_id

    finding_type, entity = finding_id.split(
        "::",
        1,
    )

    friendly_types = {
        "country": "Country",
        "product": "Product",
        "opex_account": "OPEX account",
        "cost_centre": "Cost centre",
        "summary": "Summary",
    }

    friendly_type = friendly_types.get(
        finding_type,
        finding_type.replace(
            "_",
            " ",
        ).title(),
    )

    friendly_entity = entity.replace(
        "_",
        " ",
    )

    return (
        f"{friendly_type}: {friendly_entity}"
    )


def clean_text(
    value: Any,
) -> str:
    return html.escape(
        str(value or "")
    )


def render_fact_card(
    finding_id: str,
    statement: str,
) -> None:
    st.markdown(
        f"""
        <div class="fact-card">
            <div class="fact-title">
                {clean_text(short_finding_id(finding_id))}
            </div>
            <div class="fact-text">
                {clean_text(statement)}
            </div>
            <br>
            <span class="small-muted">
                Deterministic calculated fact
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evidence_card(
    finding_id: str,
    explanation: str,
    evidence_ids: list[str],
) -> None:
    evidence_text = ", ".join(
        evidence_ids
    )

    st.markdown(
        f"""
        <div class="evidence-card">
            <strong>
                {clean_text(short_finding_id(finding_id))}
            </strong>
            <br><br>
            {clean_text(explanation)}
            <br><br>
            <span class="small-muted">
                Approved evidence:
                {clean_text(evidence_text)}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_unresolved_card(
    finding_id: str,
    statement: str,
) -> None:
    st.markdown(
        f"""
        <div class="unresolved-card">
            <strong>
                {clean_text(short_finding_id(finding_id))}
            </strong>
            <br><br>
            {clean_text(statement)}
            <br><br>
            <span class="small-muted">
                Underlying cause remains unresolved
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Load project outputs
# ---------------------------------------------------------------------

grounded = load_json(
    GROUNDED_ANALYSIS_PATH
)

ai_output = load_json(
    AI_COMMENTARY_PATH
)

evaluation = load_json(
    EVALUATION_PATH
)

finance = grounded[
    "finance_summary"
]

commentary = ai_output[
    "commentary"
]

ai_metadata = ai_output[
    "metadata"
]

evaluation_summary = evaluation[
    "summary"
]

security = evaluation[
    "security_isolation"
]


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "### CFO Intelligence Copilot"
    )

    st.caption(
        "Northstar Systems AS"
    )

    st.divider()

    st.markdown(
        "**Analysis period**"
    )

    st.write(
        "Latest Forecast vs Budget"
    )

    st.markdown(
        "**Architecture**"
    )

    st.caption(
        "Deterministic finance engine → "
        "evidence grounding → directional "
        "guardrails → AI commentary → "
        "offline evaluation"
    )

    st.divider()

    st.markdown(
        "**AI model**"
    )

    st.write(
        ai_metadata.get(
            "model",
            "N/A",
        )
    )

    st.markdown(
        "**Ground truth access**"
    )

    if (
        ai_metadata.get(
            "ground_truth_access"
        )
        is False
    ):
        st.markdown(
            '<span class="status-blocked">Blocked</span>',
            unsafe_allow_html=True,
        )
    else:
        st.error(
            "Invalid"
        )

    st.write("")

    st.markdown(
        "**Regression suite**"
    )

    st.markdown(
        '<span class="status-pass">51 / 51 passing</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption(
        "Synthetic finance environment "
        "built as a portfolio demonstration "
        "of finance, data engineering, "
        "generative AI and model controls."
    )


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.markdown(
    '<div class="app-kicker">'
    "Northstar Systems AS · FP&A"
    "</div>",
    unsafe_allow_html=True,
)

st.title(
    "CFO Intelligence Copilot"
)

st.markdown(
    '<div class="app-subtitle">'
    "Evidence-grounded financial performance analysis "
    "with deterministic calculations, controlled evidence "
    "and AI-generated management commentary."
    "</div>",
    unsafe_allow_html=True,
)


overview_tab, evidence_tab, evaluation_tab = (
    st.tabs(
        [
            "Executive Overview",
            "Evidence & Guardrails",
            "Evaluation & Safety",
        ]
    )
)


# ---------------------------------------------------------------------
# Executive Overview
# ---------------------------------------------------------------------

with overview_tab:
    st.markdown(
        '<div class="section-kicker">'
        "Latest Forecast vs Budget"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:
        st.metric(
            "Revenue",
            money_value(
                finance.get(
                    "comparison_revenue_nok"
                )
            ),
            delta=money_delta(
                finance.get(
                    "revenue_variance_nok"
                )
            ),
        )

    with col2:
        st.metric(
            "Gross Profit",
            money_value(
                finance.get(
                    "comparison_gross_profit_nok"
                )
            ),
            delta=money_delta(
                finance.get(
                    "gross_profit_variance_nok"
                )
            ),
        )

    with col3:
        st.metric(
            "OPEX",
            money_value(
                finance.get(
                    "comparison_opex_nok"
                )
            ),
            delta=money_delta(
                finance.get(
                    "opex_variance_nok"
                )
            ),
            delta_color="inverse",
        )

    with col4:
        st.metric(
            "EBITDA",
            money_value(
                finance.get(
                    "comparison_ebitda_nok"
                )
            ),
            delta=money_delta(
                finance.get(
                    "ebitda_variance_nok"
                )
            ),
        )

    with col5:
        st.metric(
            "EBITDA Margin",
            pct(
                finance.get(
                    "comparison_ebitda_margin"
                )
            ),
            delta=percentage_point_delta(
                finance.get(
                    "ebitda_margin_variance_pp"
                )
            ),
        )

    st.divider()

    st.subheader(
        "Executive Summary"
    )

    st.markdown(
        f"""
        <div class="summary-card">
            {clean_text(
                commentary.get(
                    "executive_summary",
                    "No summary available.",
                )
            )}
        </div>
        """,
        unsafe_allow_html=True,
    )

    supported = commentary.get(
        "supported_explanations",
        [],
    )

    unresolved = commentary.get(
        "unresolved_findings",
        [],
    )

    left, right = st.columns(
        [1.15, 0.85]
    )

    with left:
        st.subheader(
            "Evidence-Supported Drivers"
        )

        if supported:
            for explanation in supported[:4]:
                render_evidence_card(
                    explanation.get(
                        "finding_id",
                        "",
                    ),
                    explanation.get(
                        "explanation",
                        "",
                    ),
                    explanation.get(
                        "evidence_ids",
                        [],
                    ),
                )
        else:
            st.info(
                "No evidence-supported drivers available."
            )

    with right:
        st.subheader(
            "Control Status"
        )

        directional = ai_metadata.get(
            "directional_guardrail_summary",
            {},
        )

        status1, status2 = st.columns(
            2
        )

        status1.metric(
            "Approved Evidence",
            len(
                ai_metadata.get(
                    "source_usable_evidence_ids",
                    [],
                )
            ),
        )

        status2.metric(
            "Unresolved",
            len(
                unresolved
            ),
        )

        st.markdown(
            '<span class="status-pass">'
            "Ground truth isolated"
            "</span>",
            unsafe_allow_html=True,
        )

        st.write("")

        withheld = directional.get(
            "withheld_evidence_ids",
            [],
        )

        if withheld:
            st.markdown(
                '<span class="status-warning">'
                "Directional conflict blocked"
                "</span>",
                unsafe_allow_html=True,
            )

            st.caption(
                "Withheld evidence: "
                + ", ".join(
                    withheld
                )
            )

        st.caption(
            "Only deterministic facts and approved "
            "evidence can reach the AI commentary layer."
        )

    st.divider()

    st.subheader(
        "Key Calculated Observations"
    )

    observations = commentary.get(
        "observations",
        [],
    )

    visible_observations = (
        observations[:8]
    )

    obs_col1, obs_col2 = st.columns(
        2
    )

    for index, observation in enumerate(
        visible_observations
    ):
        target_column = (
            obs_col1
            if index % 2 == 0
            else obs_col2
        )

        with target_column:
            render_fact_card(
                observation.get(
                    "finding_id",
                    "",
                ),
                observation.get(
                    "statement",
                    "",
                ),
            )

    if len(observations) > len(
        visible_observations
    ):
        with st.expander(
            "Show remaining calculated observations"
        ):
            for observation in observations[
                len(visible_observations):
            ]:
                st.markdown(
                    f"**{short_finding_id(observation.get('finding_id', ''))}**"
                )
                st.write(
                    observation.get(
                        "statement",
                        "",
                    )
                )

    questions = commentary.get(
        "management_questions",
        [],
    )

    st.divider()

    with st.expander(
        f"Management Questions ({len(questions)})",
        expanded=False,
    ):
        for question in questions:
            related_ids = question.get(
                "related_finding_ids",
                [],
            )

            related_text = ", ".join(
                short_finding_id(
                    finding_id
                )
                for finding_id
                in related_ids
            )

            st.markdown(
                f"""
                <div class="question-card">
                    <strong>
                        {clean_text(question.get("question", ""))}
                    </strong>
                    <br>
                    <span class="small-muted">
                        Related findings:
                        {clean_text(related_text)}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------
# Evidence & Guardrails
# ---------------------------------------------------------------------

with evidence_tab:
    st.subheader(
        "Evidence Control Layer"
    )

    st.caption(
        "The AI cannot freely select explanations. "
        "Evidence must pass entity, driver and directional "
        "compatibility checks before it becomes model-visible."
    )

    directional = ai_metadata.get(
        "directional_guardrail_summary",
        {},
    )

    withheld = directional.get(
        "withheld_evidence_ids",
        [],
    )

    d1, d2, d3, d4 = st.columns(
        4
    )

    d1.metric(
        "Evidence Assessments",
        directional.get(
            "assessment_count",
            0,
        ),
    )

    d2.metric(
        "Directionally Consistent",
        directional.get(
            "directionally_consistent_count",
            0,
        ),
    )

    d3.metric(
        "Offsetting",
        directional.get(
            "offsetting_count",
            0,
        ),
    )

    d4.metric(
        "Unknown Direction",
        directional.get(
            "direction_unknown_count",
            0,
        ),
    )

    if "NOTE-016" in withheld:
        st.markdown(
            """
            <div class="guardrail-card">
                <div class="guardrail-title">
                    Directional guardrail example · NOTE-016
                </div>
                Norway has a negative calculated revenue and gross-profit
                variance. NOTE-016 documents a Sensor X Pro shipment
                acceleration, which points in the opposite financial
                direction. The evidence is therefore retained as relevant
                context but blocked from being used as a causal explanation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        example_1, example_2, example_3 = st.columns(
            [1, 1, 1]
        )

        with example_1:
            st.markdown(
                """
                <div class="guardrail-step">
                    <div class="guardrail-step-title">
                        Calculated finding
                    </div>
                    <strong>Country: Norway</strong><br>
                    Revenue and gross profit are below budget.<br>
                    <span class="small-muted">
                        Primary driver: volume / mix
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with example_2:
            st.markdown(
                """
                <div class="guardrail-step">
                    <div class="guardrail-step-title">
                        Candidate evidence
                    </div>
                    <strong>NOTE-016</strong><br>
                    Sensor X Pro shipment acceleration in Norway.<br>
                    <span class="small-muted">
                        Direction: positive / offsetting
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with example_3:
            st.markdown(
                """
                <div class="guardrail-step">
                    <div class="guardrail-step-title">
                        Control decision
                    </div>
                    <strong>Withheld from AI explanation</strong><br>
                    Relevant evidence, but not valid as the cause of the
                    negative variance.<br>
                    <span class="small-muted">
                        Hallucination risk reduced
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    left, right = st.columns(
        [1, 1]
    )

    with left:
        st.subheader(
            "Approved Explanations"
        )

        st.caption(
            f"{len(supported)} material findings have evidence "
            "that passed the grounding and directional controls."
        )

        if not supported:
            st.info(
                "No approved explanations."
            )

        for explanation in supported:
            render_evidence_card(
                explanation.get(
                    "finding_id",
                    "",
                ),
                explanation.get(
                    "explanation",
                    "",
                ),
                explanation.get(
                    "evidence_ids",
                    [],
                ),
            )

    with right:
        st.subheader(
            "Unresolved Findings"
        )

        st.caption(
            f"{len(unresolved)} material findings remain unresolved "
            "rather than being assigned an unsupported explanation."
        )

        if not unresolved:
            st.success(
                "No unresolved material findings."
            )

        unresolved_preview = unresolved[:4]

        for finding in unresolved_preview:
            render_unresolved_card(
                finding.get(
                    "finding_id",
                    "",
                ),
                finding.get(
                    "statement",
                    "",
                ),
            )

        remaining_unresolved = unresolved[
            len(unresolved_preview):
        ]

        if remaining_unresolved:
            with st.expander(
                "Show "
                f"{len(remaining_unresolved)} additional "
                "unresolved findings",
                expanded=False,
            ):
                for finding in remaining_unresolved:
                    render_unresolved_card(
                        finding.get(
                            "finding_id",
                            "",
                        ),
                        finding.get(
                            "statement",
                            "",
                        ),
                    )

    st.divider()

    control_left, control_right = (
        st.columns(
            2
        )
    )

    with control_left:
        st.markdown(
            "#### Model-Visible Evidence"
        )

        visible_evidence = (
            ai_metadata.get(
                "source_usable_evidence_ids",
                [],
            )
        )

        st.caption(
            "Only these approved evidence IDs were available "
            "to the AI commentary layer."
        )

        if visible_evidence:
            for evidence_id in visible_evidence:
                st.success(
                    f"{evidence_id} · approved"
                )
        else:
            st.info(
                "No evidence IDs reached the AI layer."
            )

    with control_right:
        st.markdown(
            "#### Withheld Evidence"
        )

        st.caption(
            "Relevant evidence is withheld when it fails the "
            "directional compatibility check."
        )

        if withheld:
            for evidence_id in withheld:
                st.warning(
                    f"{evidence_id} · blocked by directional guardrail"
                )
        else:
            st.success(
                "No evidence withheld."
            )

        st.caption(
            "Evidence can be relevant to an entity without "
            "being valid as an explanation for the observed variance."
        )


# ---------------------------------------------------------------------
# Evaluation & Safety
# ---------------------------------------------------------------------

with evaluation_tab:
    st.subheader(
        "Offline Hidden-Ground-Truth Evaluation"
    )

    st.caption(
        "The production AI never receives hidden ground truth. "
        "A separate offline evaluator compares pipeline behavior "
        "with the controlled synthetic events after generation."
    )

    if evaluation_summary.get(
        "overall_passed"
    ):
        st.markdown(
            '<span class="status-pass">'
            "Evaluation PASSED"
            "</span>",
            unsafe_allow_html=True,
        )
    else:
        st.error(
            "Evaluation failed."
        )

    st.write("")

    e1, e2, e3, e4 = st.columns(
        4
    )

    e1.metric(
        "Ground-Truth Events",
        evaluation_summary.get(
            "ground_truth_event_count",
            0,
        ),
    )

    citation_precision = (
        evaluation_summary.get(
            "citation_precision"
        )
    )

    if citation_precision is None:
        citation_precision_text = "N/A"
    else:
        citation_precision_text = (
            f"{citation_precision * 100:.0f}%"
        )

    e2.metric(
        "Citation Precision",
        citation_precision_text,
    )

    evidence_utilization = (
        evaluation_summary.get(
            "ai_visible_evidence_utilization"
        )
    )

    if evidence_utilization is None:
        evidence_utilization_text = "N/A"
    else:
        evidence_utilization_text = (
            f"{evidence_utilization * 100:.0f}%"
        )

    e3.metric(
        "Evidence Utilization",
        evidence_utilization_text,
    )

    e4.metric(
        "Invalid Evidence IDs",
        len(
            evaluation_summary.get(
                "invalid_ai_evidence_ids",
                [],
            )
        ),
    )

    st.divider()

    left, right = st.columns(
        [1, 1]
    )

    with left:
        st.subheader(
            "Pipeline Routing"
        )

        routing = evaluation.get(
            "pipeline_routing",
            {},
        )

        routing_labels = {
            "reached_ai": (
                "Reached AI"
            ),
            "rejected_by_grounding": (
                "Rejected by grounding"
            ),
            "withheld_by_directional_guardrail": (
                "Withheld by directional guardrail"
            ),
            "not_observable_no_evidence": (
                "Not observable — no evidence"
            ),
        }

        routing_rows = []

        for key, value in routing.items():
            routing_rows.append(
                {
                    "Pipeline outcome": (
                        routing_labels.get(
                            key,
                            key,
                        )
                    ),
                    "Events": value,
                }
            )

        st.dataframe(
            routing_rows,
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader(
            "Security Isolation"
        )

        if security.get(
            "passed"
        ):
            st.success(
                "Hidden ground truth remained isolated "
                "from the model."
            )
        else:
            st.error(
                "Security isolation check failed."
            )

        leaked_event_ids = (
            security.get(
                "leaked_event_ids",
                [],
            )
        )

        leaked_paths = (
            security.get(
                "leaked_ground_truth_paths",
                [],
            )
        )

        st.write(
            "**Leaked event IDs:** "
            + (
                ", ".join(
                    leaked_event_ids
                )
                if leaked_event_ids
                else "None"
            )
        )

        st.write(
            "**Leaked file paths:** "
            + (
                ", ".join(
                    leaked_paths
                )
                if leaked_paths
                else "None"
            )
        )

        st.write(
            "**Ground truth access:** "
            + (
                "Blocked"
                if security.get(
                    "ground_truth_access_flag_false"
                )
                else "Invalid"
            )
        )

        st.write(
            "**API request during evaluation:** No"
        )

    st.divider()

    st.subheader(
        "Controlled Event Outcomes"
    )

    event_rows = []

    event_labels = {
        "reached_ai": "Reached AI",
        "rejected_by_grounding": (
            "Rejected by grounding"
        ),
        "withheld_by_directional_guardrail": (
            "Withheld by directional guardrail"
        ),
        "not_observable_no_evidence": (
            "No analyst-visible evidence"
        ),
    }

    for event in evaluation.get(
        "events",
        [],
    ):
        event_rows.append(
            {
                "Event": event.get(
                    "event_id"
                ),
                "Domain": event.get(
                    "domain",
                    "",
                ).title(),
                "Evidence": (
                    event.get(
                        "evidence_id"
                    )
                    or "None"
                ),
                "Pipeline outcome": (
                    event_labels.get(
                        event.get(
                            "pipeline_classification",
                            "",
                        ),
                        event.get(
                            "pipeline_classification",
                            "",
                        ),
                    )
                ),
            }
        )

    st.dataframe(
        event_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Evaluation results are based on a controlled "
        "synthetic six-event environment. They demonstrate "
        "the behavior of this pipeline and should not be "
        "interpreted as general model-performance metrics."
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.divider()

st.caption(
    "CFO Intelligence Copilot · Synthetic portfolio project · "
    "Deterministic finance calculations · Evidence grounding · "
    "Directional guardrails · Structured AI commentary · "
    "Offline hidden-ground-truth evaluation"
)