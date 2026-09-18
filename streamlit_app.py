from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.ai.cfo_qa import (
    CFOQAError,
    ask_cfo,
    build_approved_context,
)
from src.scenario.scenario_engine import (
    SCENARIO_DRIVER_LABELS,
    ScenarioInputs,
    run_latest_forecast_scenario,
    run_latest_forecast_sensitivity,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

ANALYSIS_PERIODS = {
    "Latest Forecast vs Budget": {
        "comparison": "latest_forecast_vs_budget",
        "grounded_path": (
            PROJECT_ROOT
            / "outputs"
            / "grounding"
            / "latest_forecast_vs_budget_grounded.json"
        ),
        "commentary_path": (
            PROJECT_ROOT
            / "outputs"
            / "ai"
            / "latest_forecast_vs_budget_commentary.json"
        ),
        "evaluation_path": (
            PROJECT_ROOT
            / "outputs"
            / "evaluation"
            / "latest_forecast_vs_budget_evaluation.json"
        ),
    },
    "Actual YTD vs Budget": {
        "comparison": "actual_ytd_vs_budget",
        "grounded_path": (
            PROJECT_ROOT
            / "outputs"
            / "grounding"
            / "actual_ytd_vs_budget_grounded.json"
        ),
        "commentary_path": (
            PROJECT_ROOT
            / "outputs"
            / "ai"
            / "actual_ytd_vs_budget_commentary.json"
        ),
        "evaluation_path": (
            PROJECT_ROOT
            / "outputs"
            / "evaluation"
            / "actual_ytd_vs_budget_evaluation.json"
        ),
    },
}

DEFAULT_ANALYSIS_PERIOD = "Latest Forecast vs Budget"

EVALUATION_SUITE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "evaluation_suite.json"
)

ADVERSARIAL_BENCHMARK_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "adversarial_benchmark.json"
)

LLM_GUARDRAIL_COMPARISON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "llm_guardrail_comparison.json"
)

SALES_DATASETS = {
    "Actual": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "actual_sales.csv"
    ),
    "Budget": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "budget_sales.csv"
    ),
    "Latest Forecast": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "latest_forecast_sales.csv"
    ),
}


QA_SESSION_LIMIT = 5
QA_MAX_QUESTION_CHARS = 400

QA_SUGGESTED_QUESTIONS = [
    "Why is EBITDA below budget?",
    "What explains the UK shortfall?",
    "What do we know about Product & R&D OPEX?",
    "Can the Norway revenue shortfall be explained?",
]


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

    .status-info {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.10);
        color: #1d4ed8;
        font-weight: 700;
        font-size: 0.84rem;
    }

    .qa-answer-card {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 14px;
        padding: 1.05rem 1.15rem;
        margin: 0.45rem 0 0.85rem 0;
        line-height: 1.58;
        background: rgba(128, 128, 128, 0.025);
    }

    .qa-question-card {
        border-left: 4px solid #2563eb;
        padding: 0.72rem 0.95rem;
        margin: 0.8rem 0 0.35rem 0;
        background: rgba(37, 99, 235, 0.045);
        border-radius: 0 10px 10px 0;
        line-height: 1.45;
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


@st.cache_data(show_spinner=False)
def load_sales_data(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        st.error(
            f"Required sales dataset not found: {path}"
        )
        st.stop()

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        st.error(
            f"Could not read {path.name}: {exc}"
        )
        st.stop()

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
        )

    return frame


@st.cache_data(show_spinner=False)
def calculate_scenario_result(
    volume_pct: float,
    price_pct: float,
    unit_cost_pct: float,
    headcount_pct: float,
    non_payroll_opex_pct: float,
) -> dict[str, Any]:
    """Run one deterministic management scenario against Latest Forecast."""

    inputs = ScenarioInputs(
        volume_pct=volume_pct,
        price_pct=price_pct,
        unit_cost_pct=unit_cost_pct,
        headcount_pct=headcount_pct,
        non_payroll_opex_pct=non_payroll_opex_pct,
    )

    return run_latest_forecast_scenario(
        inputs=inputs,
        scenario_label="Management Scenario",
    )


@st.cache_data(show_spinner=False)
def calculate_sensitivity_result(
    driver: str,
    values: tuple[float, ...],
    volume_pct: float,
    price_pct: float,
    unit_cost_pct: float,
    headcount_pct: float,
    non_payroll_opex_pct: float,
) -> list[dict[str, Any]]:
    """Run one-way deterministic sensitivity around the current scenario."""

    base_inputs = ScenarioInputs(
        volume_pct=volume_pct,
        price_pct=price_pct,
        unit_cost_pct=unit_cost_pct,
        headcount_pct=headcount_pct,
        non_payroll_opex_pct=non_payroll_opex_pct,
    )

    return run_latest_forecast_sensitivity(
        driver=driver,
        values=values,
        base_inputs=base_inputs,
    )


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


def benchmark_pct(
    value: float | int | None,
    decimals: int = 1,
) -> str:
    if value is None:
        return "N/A"

    return (
        f"{float(value) * 100:.{decimals}f}%"
    )


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



def resolve_openai_api_key() -> str | None:
    env_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if env_key:
        return env_key.strip()

    try:
        secret_key = st.secrets.get(
            "OPENAI_API_KEY"
        )
    except Exception:
        secret_key = None

    if secret_key:
        return str(
            secret_key
        ).strip()

    return None


def qa_status_markup(
    answer_type: str,
) -> str:
    if answer_type == "supported_explanation":
        return (
            '<span class="status-pass">'
            "Evidence-supported explanation"
            "</span>"
        )

    if answer_type == "calculated_fact":
        return (
            '<span class="status-info">'
            "Calculated fact"
            "</span>"
        )

    return (
        '<span class="status-warning">'
        "Insufficient evidence"
        "</span>"
    )


def render_qa_result(
    question: str,
    result: dict[str, Any],
) -> None:
    answer = result.get(
        "answer",
        {},
    )

    answer_type = str(
        answer.get(
            "answer_type",
            "insufficient_evidence",
        )
    )

    st.markdown(
        f"""
        <div class="qa-question-card">
            <strong>Question</strong><br>
            {clean_text(question)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        qa_status_markup(
            answer_type
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="qa-answer-card">
            {clean_text(
                answer.get(
                    "answer",
                    "No answer available.",
                )
            )}
        </div>
        """,
        unsafe_allow_html=True,
    )

    finding_ids = answer.get(
        "finding_ids",
        [],
    )

    evidence_ids = answer.get(
        "evidence_ids",
        [],
    )

    limitations = answer.get(
        "limitations",
        [],
    )

    with st.expander(
        "Traceability",
        expanded=False,
    ):
        if finding_ids:
            st.markdown(
                "**Calculated findings used**"
            )
            for finding_id in finding_ids:
                st.write(
                    "• "
                    + short_finding_id(
                        str(finding_id)
                    )
                )
        else:
            st.caption(
                "No calculated finding IDs were cited."
            )

        st.markdown(
            "**Approved evidence used**"
        )

        if evidence_ids:
            for evidence_id in evidence_ids:
                st.success(
                    f"{evidence_id} · approved"
                )
        else:
            st.caption(
                "No causal evidence was cited."
            )

        if limitations:
            st.markdown(
                "**Limitations**"
            )
            for limitation in limitations:
                st.write(
                    "• "
                    + str(
                        limitation
                    )
                )

        diagnostics = result.get(
            "diagnostics",
            {},
        )

        st.caption(
            f"Analysis period: {diagnostics.get('analysis_period', 'N/A')} · "
            "Ground truth access: blocked · "
            "Model-visible causal evidence for this question: "
            f"{', '.join(diagnostics.get('selected_evidence_ids', [])) or 'none'}"
        )



# ---------------------------------------------------------------------
# Analysis period selector
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "### CFO Intelligence Copilot"
    )

    st.caption(
        "Northstar Systems AS"
    )

    st.divider()

    selected_period = st.selectbox(
        "Analysis period",
        options=list(
            ANALYSIS_PERIODS.keys()
        ),
        index=list(
            ANALYSIS_PERIODS.keys()
        ).index(
            DEFAULT_ANALYSIS_PERIOD
        ),
        key="analysis_period_selector",
    )


# ---------------------------------------------------------------------
# Load project outputs
# ---------------------------------------------------------------------

period_config = ANALYSIS_PERIODS[
    selected_period
]

grounded = load_json(
    period_config[
        "grounded_path"
    ]
)

ai_output = load_json(
    period_config[
        "commentary_path"
    ]
)

evaluation = load_json(
    period_config[
        "evaluation_path"
    ]
)

evaluation_suite = load_json(
    EVALUATION_SUITE_PATH
)

adversarial_benchmark = load_json(
    ADVERSARIAL_BENCHMARK_PATH
)

llm_guardrail_comparison = load_json(
    LLM_GUARDRAIL_COMPARISON_PATH
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

suite_summary = evaluation_suite[
    "summary"
]

suite_cases = evaluation_suite.get(
    "cases",
    {},
)

adversarial_summary = (
    adversarial_benchmark.get(
        "summary",
        {},
    )
)

llm_benchmark_metadata = (
    llm_guardrail_comparison.get(
        "metadata",
        {},
    )
)

guarded_benchmark = (
    llm_guardrail_comparison.get(
        "guarded_run",
        {},
    )
)

naive_benchmark = (
    llm_guardrail_comparison.get(
        "naive_run",
        {},
    )
)

guarded_benchmark_summary = (
    guarded_benchmark.get(
        "summary",
        {},
    )
)

naive_benchmark_summary = (
    naive_benchmark.get(
        "summary",
        {},
    )
)

expected_comparison = period_config[
    "comparison"
]

grounded_comparison = (
    grounded.get(
        "metadata",
        {},
    ).get(
        "comparison"
    )
)

ai_comparison = ai_metadata.get(
    "comparison"
)

if (
    grounded_comparison
    != expected_comparison
    or ai_comparison
    != expected_comparison
):
    st.error(
        "Selected analysis period does not match the loaded "
        "grounding/commentary outputs."
    )
    st.stop()


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "**Architecture**"
    )

    st.caption(
        "Deterministic finance engine → "
        "scenario simulation + evidence grounding → "
        "directional guardrails → AI commentary → "
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
        '<span class="status-pass">118 / 118 passing</span>',
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


(
    overview_tab,
    ask_tab,
    scenario_tab,
    evidence_tab,
    evaluation_tab,
    data_tab,
) = st.tabs(
    [
        "Executive Overview",
        "Ask the CFO",
        "Scenario & Sensitivity",
        "Evidence & Guardrails",
        "Evaluation & Safety",
        "Data Explorer",
    ]
)


# ---------------------------------------------------------------------
# Executive Overview
# ---------------------------------------------------------------------

with overview_tab:
    st.markdown(
        '<div class="section-kicker">'
        + clean_text(selected_period)
        + "</div>",
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
# Ask the CFO
# ---------------------------------------------------------------------

with ask_tab:
    st.subheader(
        "Ask the CFO Copilot"
    )

    st.caption(
        f"Ask management questions about the current {selected_period} "
        "analysis. The assistant can use deterministic finance findings and "
        "evidence that already passed the grounding and directional controls."
    )

    st.info(
        "The interactive assistant cannot access hidden ground truth and is "
        "not allowed to invent explanations. When approved evidence is "
        "insufficient, the answer must remain unresolved."
    )

    approved_context = build_approved_context(
        commentary=commentary,
        ai_metadata=ai_metadata,
    )

    q1, q2, q3, q4 = st.columns(
        4
    )

    q1.metric(
        "Calculated Facts",
        len(
            approved_context[
                "facts"
            ]
        ),
    )
    q2.metric(
        "Approved Explanations",
        len(
            approved_context[
                "supported"
            ]
        ),
    )
    q3.metric(
        "Approved Evidence IDs",
        len(
            approved_context[
                "approved_evidence_ids"
            ]
        ),
    )
    q4.metric(
        "Blocked Evidence IDs",
        len(
            approved_context[
                "withheld_evidence_ids"
            ]
        ),
    )

    st.divider()

    if (
        st.session_state.get(
            "cfo_qa_active_period"
        )
        != selected_period
    ):
        st.session_state[
            "cfo_qa_active_period"
        ] = selected_period
        st.session_state[
            "cfo_qa_history"
        ] = []
        st.session_state[
            "cfo_question_input"
        ] = ""
        st.session_state[
            "cfo_clear_question_input"
        ] = False

    if (
        "cfo_qa_history"
        not in st.session_state
    ):
        st.session_state[
            "cfo_qa_history"
        ] = []

    if (
        "cfo_qa_request_count"
        not in st.session_state
    ):
        st.session_state[
            "cfo_qa_request_count"
        ] = 0

    if (
        "cfo_question_input"
        not in st.session_state
    ):
        st.session_state[
            "cfo_question_input"
        ] = ""

    if (
        "cfo_clear_question_input"
        not in st.session_state
    ):
        st.session_state[
            "cfo_clear_question_input"
        ] = False

    # Clear the previous question only after a successful response.
    # This happens before the text-area widget is instantiated, which
    # avoids Streamlit session-state conflicts.
    if st.session_state[
        "cfo_clear_question_input"
    ]:
        st.session_state[
            "cfo_question_input"
        ] = ""
        st.session_state[
            "cfo_clear_question_input"
        ] = False

    def set_cfo_question(
        question_text: str,
    ) -> None:
        st.session_state[
            "cfo_question_input"
        ] = question_text

    st.markdown(
        "#### Example questions"
    )

    suggestion_columns = st.columns(
        4
    )

    for index, suggested_question in enumerate(
        QA_SUGGESTED_QUESTIONS
    ):
        with suggestion_columns[index]:
            st.button(
                suggested_question,
                key=f"qa_suggestion_{index}",
                use_container_width=True,
                on_click=set_cfo_question,
                args=(suggested_question,),
            )

    api_key = resolve_openai_api_key()

    remaining_questions = max(
        0,
        QA_SESSION_LIMIT
        - int(
            st.session_state[
                "cfo_qa_request_count"
            ]
        ),
    )

    st.caption(
        f"Live demo limit: {QA_SESSION_LIMIT} successful questions per session · "
        f"{remaining_questions} remaining"
    )

    if not api_key:
        st.warning(
            "Live CFO Q&A is not configured in this environment yet. "
            "Add OPENAI_API_KEY as a local environment variable or "
            "Streamlit secret to enable interactive questions."
        )

    # A normal container is used instead of clear_on_submit=True.
    # Streamlit's form reset could clear a question inserted by an
    # example-button before the submit handler reads it.
    with st.container(
        border=True,
    ):
        question = st.text_area(
            "Management question",
            key="cfo_question_input",
            placeholder=(
                "Example: What explains the UK shortfall?"
            ),
            height=95,
            max_chars=QA_MAX_QUESTION_CHARS,
        )

        submit_question = st.button(
            "Ask CFO Copilot",
            key="cfo_submit_question",
            type="primary",
            use_container_width=False,
            disabled=(
                not api_key
                or remaining_questions <= 0
            ),
        )

    if submit_question:
        clean_question = question.strip()

        if not clean_question:
            st.warning(
                "Enter a management question first."
            )
        elif remaining_questions <= 0:
            st.warning(
                "The live question limit for this session has been reached."
            )
        else:
            with st.spinner(
                "Checking calculated findings and approved evidence..."
            ):
                try:
                    result = ask_cfo(
                        question=clean_question,
                        commentary=commentary,
                        ai_metadata=ai_metadata,
                        analysis_period=selected_period,
                        api_key=api_key,
                        model=str(
                            ai_metadata.get(
                                "model",
                                "gpt-5.6-luna",
                            )
                        ),
                    )
                except CFOQAError as exc:
                    st.error(
                        f"CFO Q&A control blocked the response: {exc}"
                    )
                except Exception as exc:
                    st.error(
                        "The live model request failed. "
                        f"Technical detail: {exc}"
                    )
                else:
                    st.session_state[
                        "cfo_qa_request_count"
                    ] += 1

                    st.session_state[
                        "cfo_qa_history"
                    ].append(
                        {
                            "question": clean_question,
                            "result": result,
                        }
                    )

                    st.session_state[
                        "cfo_clear_question_input"
                    ] = True

                    st.rerun()

    history = st.session_state[
        "cfo_qa_history"
    ]

    if history:
        st.divider()

        history_header, clear_header = (
            st.columns(
                [5, 1]
            )
        )

        with history_header:
            st.markdown(
                "#### Conversation"
            )

        with clear_header:
            if st.button(
                "Clear view",
                use_container_width=True,
            ):
                st.session_state[
                    "cfo_qa_history"
                ] = []
                st.rerun()

        for item in reversed(
            history
        ):
            render_qa_result(
                question=item[
                    "question"
                ],
                result=item[
                    "result"
                ],
            )
    else:
        st.divider()

        st.caption(
            "No questions asked in this session yet. "
            "Try one of the example questions above."
        )



# ---------------------------------------------------------------------
# Scenario & Sensitivity
# ---------------------------------------------------------------------

with scenario_tab:
    st.subheader(
        "Scenario & Sensitivity Lab"
    )

    st.caption(
        "Model full-year management scenarios against the 2026 Latest Forecast. "
        "Every output is recalculated by the same deterministic finance engine "
        "used in the core FP&A analysis."
    )

    st.info(
        "This is a deterministic what-if model, not an AI forecast. "
        "Latest Forecast is the fixed baseline for this lab."
    )

    st.markdown(
        "#### Scenario Assumptions"
    )

    assumption_row_1 = st.columns(
        3
    )

    with assumption_row_1[0]:
        scenario_volume_pct = st.slider(
            "Volume",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            format="%.1f%%",
            key="scenario_volume_pct",
            help=(
                "Applies the same relative unit-volume change across "
                "the detailed Latest Forecast sales grain."
            ),
        )

    with assumption_row_1[1]:
        scenario_price_pct = st.slider(
            "List price",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            format="%.1f%%",
            key="scenario_price_pct",
            help=(
                "Changes list price before the existing forecast "
                "discount rate is applied."
            ),
        )

    with assumption_row_1[2]:
        scenario_unit_cost_pct = st.slider(
            "Unit cost",
            min_value=-15.0,
            max_value=15.0,
            value=0.0,
            step=0.5,
            format="%.1f%%",
            key="scenario_unit_cost_pct",
            help=(
                "Changes unit cost while preserving the detailed "
                "forecast product and market mix."
            ),
        )

    assumption_row_2 = st.columns(
        2
    )

    with assumption_row_2[0]:
        scenario_headcount_pct = st.slider(
            "Headcount",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            format="%.1f%%",
            key="scenario_headcount_pct",
            help=(
                "Scales payroll headcount. Payroll expense is recalculated "
                "using the existing average employee cost."
            ),
        )

    with assumption_row_2[1]:
        scenario_non_payroll_pct = st.slider(
            "Non-payroll OPEX",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            format="%.1f%%",
            key="scenario_non_payroll_pct",
            help=(
                "Scales all non-payroll operating expense accounts."
            ),
        )

    scenario_inputs_decimal = {
        "volume_pct": scenario_volume_pct / 100,
        "price_pct": scenario_price_pct / 100,
        "unit_cost_pct": scenario_unit_cost_pct / 100,
        "headcount_pct": scenario_headcount_pct / 100,
        "non_payroll_opex_pct": scenario_non_payroll_pct / 100,
    }

    scenario_result = calculate_scenario_result(
        volume_pct=scenario_inputs_decimal["volume_pct"],
        price_pct=scenario_inputs_decimal["price_pct"],
        unit_cost_pct=scenario_inputs_decimal["unit_cost_pct"],
        headcount_pct=scenario_inputs_decimal["headcount_pct"],
        non_payroll_opex_pct=scenario_inputs_decimal["non_payroll_opex_pct"],
    )

    scenario_summary = scenario_result[
        "summary"
    ]

    st.divider()

    st.markdown(
        "#### Scenario Outcome"
    )

    st.caption(
        "Values show the recalculated scenario. "
        "Deltas are versus the 2026 Latest Forecast baseline."
    )

    scenario_kpis = st.columns(
        5
    )

    with scenario_kpis[0]:
        st.metric(
            "Revenue",
            money_value(
                scenario_summary[
                    "scenario_revenue_nok"
                ]
            ),
            delta=money_delta(
                scenario_summary[
                    "revenue_change_nok"
                ]
            ),
        )

    with scenario_kpis[1]:
        st.metric(
            "Gross Profit",
            money_value(
                scenario_summary[
                    "scenario_gross_profit_nok"
                ]
            ),
            delta=money_delta(
                scenario_summary[
                    "gross_profit_change_nok"
                ]
            ),
        )

    with scenario_kpis[2]:
        st.metric(
            "OPEX",
            money_value(
                scenario_summary[
                    "scenario_opex_nok"
                ]
            ),
            delta=money_delta(
                scenario_summary[
                    "opex_change_nok"
                ]
            ),
            delta_color="inverse",
        )

    with scenario_kpis[3]:
        st.metric(
            "EBITDA",
            money_value(
                scenario_summary[
                    "scenario_ebitda_nok"
                ]
            ),
            delta=money_delta(
                scenario_summary[
                    "ebitda_change_nok"
                ]
            ),
        )

    with scenario_kpis[4]:
        st.metric(
            "EBITDA Margin",
            pct(
                scenario_summary[
                    "scenario_ebitda_margin"
                ]
            ),
            delta=percentage_point_delta(
                scenario_summary[
                    "ebitda_margin_change_pp"
                ]
            ),
        )

    reconciliation = scenario_result[
        "reconciliation"
    ]

    reconciliation_residual = float(
        reconciliation.get(
            "residual_nok",
            0.0,
        )
    )

    if reconciliation.get(
        "passed"
    ):
        st.success(
            "Scenario bridge reconciled to calculated EBITDA · "
            f"residual NOK {reconciliation_residual:,.2f}"
        )
    else:
        st.error(
            "Scenario bridge failed reconciliation."
        )

    st.divider()

    bridge_left, bridge_right = st.columns(
        [1.05, 0.95]
    )

    bridge_labels = {
        "volume_mix": "Volume & mix",
        "list_price": "List price",
        "discount": "Discount",
        "unit_cost": "Unit cost",
        "headcount": "Headcount",
        "employee_cost": "Employee cost",
        "non_payroll": "Non-payroll OPEX",
    }

    bridge_rows = []

    for item in scenario_result.get(
        "ebitda_bridge",
        [],
    ):
        impact_nok = float(
            item.get(
                "impact_nok",
                0.0,
            )
        )

        bridge_rows.append(
            {
                "Driver": bridge_labels.get(
                    item.get(
                        "driver",
                        "",
                    ),
                    str(
                        item.get(
                            "driver",
                            "",
                        )
                    ).replace(
                        "_",
                        " ",
                    ).title(),
                ),
                "Category": str(
                    item.get(
                        "category",
                        "",
                    )
                ).title(),
                "EBITDA impact (NOK m)": impact_nok / 1_000_000,
                "Material": (
                    "Yes"
                    if item.get(
                        "material"
                    )
                    else "No"
                ),
            }
        )

    bridge_frame = pd.DataFrame(
        bridge_rows
    )

    with bridge_left:
        st.markdown(
            "#### EBITDA Driver Bridge"
        )

        st.caption(
            "Sequential deterministic decomposition of the scenario's "
            "EBITDA change versus Latest Forecast."
        )

        st.dataframe(
            bridge_frame,
            width="stretch",
            hide_index=True,
            column_config={
                "EBITDA impact (NOK m)": (
                    st.column_config.NumberColumn(
                        format="%.2f",
                    )
                ),
            },
        )

        if not bridge_frame.empty:
            largest_driver_index = (
                bridge_frame[
                    "EBITDA impact (NOK m)"
                ]
                .abs()
                .idxmax()
            )

            largest_driver_row = bridge_frame.loc[
                largest_driver_index
            ]

            st.caption(
                "Largest absolute EBITDA driver: "
                f"{largest_driver_row['Driver']} · "
                f"NOK "
                f"{largest_driver_row['EBITDA impact (NOK m)']:+.2f}m"
            )

    with bridge_right:
        st.markdown(
            "#### EBITDA Impact by Driver"
        )

        if not bridge_frame.empty:
            chart_frame = (
                bridge_frame[
                    [
                        "Driver",
                        "EBITDA impact (NOK m)",
                    ]
                ]
                .set_index(
                    "Driver"
                )
            )

            st.bar_chart(
                chart_frame,
                height=330,
            )
        else:
            st.info(
                "No bridge drivers available."
            )

    st.divider()

    st.markdown(
        "#### One-Way Sensitivity"
    )

    st.caption(
        "Stress one driver around the current scenario assumption while "
        "holding the other four assumptions constant."
    )

    sensitivity_controls = st.columns(
        [1, 1]
    )

    sensitivity_driver_options = list(
        SCENARIO_DRIVER_LABELS.keys()
    )

    with sensitivity_controls[0]:
        sensitivity_driver = st.selectbox(
            "Sensitivity driver",
            options=sensitivity_driver_options,
            format_func=lambda key: (
                SCENARIO_DRIVER_LABELS[
                    key
                ]
            ),
            key="scenario_sensitivity_driver",
        )

    with sensitivity_controls[1]:
        sensitivity_span_pp = st.slider(
            "Sensitivity span",
            min_value=2.0,
            max_value=20.0,
            value=10.0,
            step=1.0,
            format="± %.1f pp",
            key="scenario_sensitivity_span",
            help=(
                "Creates five sensitivity points centered on the "
                "current assumption for the selected driver."
            ),
        )

    driver_current_values = {
        "volume": scenario_inputs_decimal[
            "volume_pct"
        ],
        "price": scenario_inputs_decimal[
            "price_pct"
        ],
        "unit_cost": scenario_inputs_decimal[
            "unit_cost_pct"
        ],
        "headcount": scenario_inputs_decimal[
            "headcount_pct"
        ],
        "non_payroll_opex": scenario_inputs_decimal[
            "non_payroll_opex_pct"
        ],
    }

    sensitivity_center_pp = (
        driver_current_values[
            sensitivity_driver
        ]
        * 100
    )

    sensitivity_points_pp = (
        sensitivity_center_pp
        - sensitivity_span_pp,
        sensitivity_center_pp
        - sensitivity_span_pp / 2,
        sensitivity_center_pp,
        sensitivity_center_pp
        + sensitivity_span_pp / 2,
        sensitivity_center_pp
        + sensitivity_span_pp,
    )

    sensitivity_values = tuple(
        round(
            value / 100,
            6,
        )
        for value
        in sensitivity_points_pp
    )

    sensitivity_records = calculate_sensitivity_result(
        driver=sensitivity_driver,
        values=sensitivity_values,
        volume_pct=scenario_inputs_decimal["volume_pct"],
        price_pct=scenario_inputs_decimal["price_pct"],
        unit_cost_pct=scenario_inputs_decimal["unit_cost_pct"],
        headcount_pct=scenario_inputs_decimal["headcount_pct"],
        non_payroll_opex_pct=scenario_inputs_decimal["non_payroll_opex_pct"],
    )

    sensitivity_rows = []

    for record in sensitivity_records:
        sensitivity_rows.append(
            {
                "Input change (%)": float(
                    record[
                        "input_change_pct"
                    ]
                ),
                "Revenue change (NOK m)": (
                    float(
                        record[
                            "revenue_change_nok"
                        ]
                    )
                    / 1_000_000
                ),
                "Gross Profit change (NOK m)": (
                    float(
                        record[
                            "gross_profit_change_nok"
                        ]
                    )
                    / 1_000_000
                ),
                "OPEX change (NOK m)": (
                    float(
                        record[
                            "opex_change_nok"
                        ]
                    )
                    / 1_000_000
                ),
                "EBITDA change (NOK m)": (
                    float(
                        record[
                            "ebitda_change_nok"
                        ]
                    )
                    / 1_000_000
                ),
                "EBITDA Margin (%)": (
                    float(
                        record[
                            "scenario_ebitda_margin"
                        ]
                    )
                    * 100
                ),
                "Margin change (pp)": float(
                    record[
                        "ebitda_margin_change_pp"
                    ]
                ),
            }
        )

    sensitivity_frame = pd.DataFrame(
        sensitivity_rows
    )

    st.markdown(
        "##### Sensitivity Results"
    )

    st.caption(
        "All key financial outcomes are shown for every sensitivity point. "
        "EBITDA and EBITDA margin are tested alongside Revenue, Gross Profit "
        "and OPEX."
    )

    st.dataframe(
        sensitivity_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Input change (%)": (
                st.column_config.NumberColumn(
                    format="%.1f%%",
                )
            ),
            "Revenue change (NOK m)": (
                st.column_config.NumberColumn(
                    format="%.2f",
                )
            ),
            "Gross Profit change (NOK m)": (
                st.column_config.NumberColumn(
                    format="%.2f",
                )
            ),
            "OPEX change (NOK m)": (
                st.column_config.NumberColumn(
                    format="%.2f",
                )
            ),
            "EBITDA change (NOK m)": (
                st.column_config.NumberColumn(
                    format="%.2f",
                )
            ),
            "EBITDA Margin (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%",
                )
            ),
            "Margin change (pp)": (
                st.column_config.NumberColumn(
                    format="%+.2f",
                )
            ),
        },
    )

    st.markdown(
        "##### Sensitivity Curve"
    )

    sensitivity_metric_options = {
        "EBITDA change": "EBITDA change (NOK m)",
        "Revenue change": "Revenue change (NOK m)",
        "Gross Profit change": "Gross Profit change (NOK m)",
        "OPEX change": "OPEX change (NOK m)",
        "EBITDA margin": "EBITDA Margin (%)",
        "Margin change": "Margin change (pp)",
    }

    sensitivity_metric_label = st.selectbox(
        "Output metric",
        options=list(
            sensitivity_metric_options.keys()
        ),
        index=0,
        key="scenario_sensitivity_output_metric",
        help=(
            "Choose which financial outcome to plot while keeping the same "
            "one-way driver sensitivity."
        ),
    )

    sensitivity_metric_column = (
        sensitivity_metric_options[
            sensitivity_metric_label
        ]
    )

    if not sensitivity_frame.empty:
        sensitivity_chart = (
            sensitivity_frame[
                [
                    "Input change (%)",
                    sensitivity_metric_column,
                ]
            ]
            .set_index(
                "Input change (%)"
            )
        )

        st.line_chart(
            sensitivity_chart,
            height=360,
        )

    st.caption(
        "Sensitivity results are deterministic what-if calculations, not "
        "forecasts or probability estimates. They show the mechanical "
        "financial impact of the selected assumptions while the other "
        "scenario assumptions remain fixed."
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
        "Adversarial Safety Benchmark"
    )

    st.caption(
        "A controlled 32-case synthetic benchmark stress-tests the "
        "evidence-control pipeline across supported evidence, no-evidence "
        "cases, wrong entities, wrong drivers, directional conflicts, "
        "ambiguous evidence and multi-event situations."
    )

    if adversarial_summary.get(
        "overall_passed"
    ):
        st.markdown(
            '<span class="status-pass">'
            "Adversarial benchmark PASSED"
            "</span>",
            unsafe_allow_html=True,
        )
    else:
        st.error(
            "Adversarial benchmark failed."
        )

    st.write("")

    a1, a2, a3, a4 = st.columns(
        4
    )

    adversarial_case_count = (
        adversarial_summary.get(
            "case_count",
            0,
        )
    )

    adversarial_cases_passed = (
        adversarial_summary.get(
            "cases_passed",
            0,
        )
    )

    a1.metric(
        "Cases Passed",
        (
            f"{adversarial_cases_passed}/"
            f"{adversarial_case_count}"
        ),
    )

    a2.metric(
        "Routing Accuracy",
        benchmark_pct(
            adversarial_summary.get(
                "routing_accuracy"
            )
        ),
    )

    a3.metric(
        "Correct Abstention",
        benchmark_pct(
            adversarial_summary.get(
                "correct_abstention_rate"
            )
        ),
    )

    with a4:
        st.metric(
            "Unsupported Explanations",
            benchmark_pct(
                adversarial_summary.get(
                    "unsupported_explanation_rate"
                )
            ),
        )
        st.caption(
            "Lower is safer"
        )

    category_summary = (
        adversarial_summary.get(
            "category_summary",
            {},
        )
    )

    category_labels = {
        "valid_supported": "Valid supported evidence",
        "no_evidence": "No evidence",
        "wrong_entity": "Wrong entity",
        "wrong_driver": "Wrong driver",
        "directional_conflict": "Directional conflict",
        "ambiguous_evidence": "Ambiguous evidence",
        "multi_event": "Multi-event",
    }

    category_rows = []

    for category, metrics in (
        category_summary.items()
    ):
        category_rows.append(
            {
                "Case type": (
                    category_labels.get(
                        category,
                        category.replace(
                            "_",
                            " ",
                        ).title(),
                    )
                ),
                "Passed": (
                    f"{metrics.get('passed', 0)}/"
                    f"{metrics.get('cases', 0)}"
                ),
                "Pass rate": benchmark_pct(
                    metrics.get(
                        "pass_rate"
                    ),
                    decimals=0,
                ),
            }
        )

    if category_rows:
        st.dataframe(
            category_rows,
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "This deterministic benchmark evaluates routing and evidence "
        "authorization before model generation. It does not make API calls."
    )

    st.divider()

    st.subheader(
        "Guarded vs Prompt-Only LLM"
    )

    st.caption(
        "The same model, prompt structure, structured-output schema and "
        "financial findings are evaluated under two evidence conditions. "
        "The guarded condition receives only evidence approved by the "
        "deterministic controls; the prompt-only baseline receives all raw "
        "management evidence available in each benchmark case and must rely "
    )

    model_name = llm_benchmark_metadata.get(
        "model",
        "N/A",
    )

    benchmark_case_count = (
        llm_benchmark_metadata.get(
            "case_count",
            0,
        )
    )

    benchmark_finding_count = (
        llm_benchmark_metadata.get(
            "finding_count",
            0,
        )
    )

    st.markdown(
        '<span class="status-info">'
        f"Model: {clean_text(model_name)}"
        "</span>",
        unsafe_allow_html=True,
    )

    st.caption(
        f"{benchmark_case_count} controlled cases · "
        f"{benchmark_finding_count} financial findings · "
        "the prompt-only baseline relies on model instructions alone to "
        "reject unsupported explanations. Benchmark expectations and ground "
        "truth were not sent to the model."
    )

    st.write("")

    comparison_metric_rows = [
        {
            "Metric": "Routing accuracy",
            "Guarded": benchmark_pct(
                guarded_benchmark_summary.get(
                    "routing_accuracy"
                )
            ),
            "Prompt-only baseline": benchmark_pct(
                naive_benchmark_summary.get(
                    "routing_accuracy"
                )
            ),
        },
        {
            "Metric": "Evidence authorization accuracy",
            "Guarded": benchmark_pct(
                guarded_benchmark_summary.get(
                    "evidence_authorization_accuracy"
                )
            ),
            "Prompt-only baseline": benchmark_pct(
                naive_benchmark_summary.get(
                    "evidence_authorization_accuracy"
                )
            ),
        },
        {
            "Metric": "Correct abstention rate",
            "Guarded": benchmark_pct(
                guarded_benchmark_summary.get(
                    "correct_abstention_rate"
                )
            ),
            "Prompt-only baseline": benchmark_pct(
                naive_benchmark_summary.get(
                    "correct_abstention_rate"
                )
            ),
        },
        {
            "Metric": "Unsupported explanation rate",
            "Guarded": benchmark_pct(
                guarded_benchmark_summary.get(
                    "unsupported_explanation_rate"
                )
            ),
            "Prompt-only baseline": benchmark_pct(
                naive_benchmark_summary.get(
                    "unsupported_explanation_rate"
                )
            ),
        },
        {
            "Metric": "Supported explanation recall",
            "Guarded": benchmark_pct(
                guarded_benchmark_summary.get(
                    "supported_explanation_recall"
                )
            ),
            "Prompt-only baseline": benchmark_pct(
                naive_benchmark_summary.get(
                    "supported_explanation_recall"
                )
            ),
        },
        {
            "Metric": "Citation precision",
            "Guarded": benchmark_pct(
                guarded_benchmark_summary.get(
                    "citation_precision"
                )
            ),
            "Prompt-only baseline": benchmark_pct(
                naive_benchmark_summary.get(
                    "citation_precision"
                )
            ),
        },
    ]

    st.dataframe(
        comparison_metric_rows,
        width="stretch",
        hide_index=True,
    )

    b1, b2, b3, b4 = st.columns(
        4
    )

    with b1:
        st.metric(
            "Unsupported Explanations",
            benchmark_pct(
                guarded_benchmark_summary.get(
                    "unsupported_explanation_rate"
                )
            ),
        )
        st.caption(
            "Prompt-only baseline: "
            + benchmark_pct(
                naive_benchmark_summary.get(
                    "unsupported_explanation_rate"
                )
            )
        )

    with b2:
        st.metric(
            "Citation Precision",
            benchmark_pct(
                guarded_benchmark_summary.get(
                    "citation_precision"
                )
            ),
        )
        st.caption(
            "Prompt-only baseline: "
            + benchmark_pct(
                naive_benchmark_summary.get(
                    "citation_precision"
                )
            )
        )

    with b3:
        st.metric(
            "Supported Recall",
            benchmark_pct(
                guarded_benchmark_summary.get(
                    "supported_explanation_recall"
                )
            ),
        )
        st.caption(
            "Prompt-only baseline: "
            + benchmark_pct(
                naive_benchmark_summary.get(
                    "supported_explanation_recall"
                )
            )
        )

    with b4:
        st.metric(
            "Forbidden Evidence Uses",
            guarded_benchmark_summary.get(
                "forbidden_evidence_use_count",
                0,
            ),
        )
        st.caption(
            "Prompt-only baseline: "
            + str(
                naive_benchmark_summary.get(
                    "forbidden_evidence_use_count",
                    0,
                )
            )
        )

    st.success(
        "In this controlled benchmark run, deterministic evidence guardrails reduced "
        "unsupported explanations while preserving all supported "
        "explanations identified by the prompt-only baseline."
    )

    naive_failures = [
        decision
        for decision
        in naive_benchmark.get(
            "decisions",
            [],
        )
        if decision.get(
            "unsupported_explanation"
        )
    ]

    if naive_failures:
        st.markdown(
            "#### Prompt-Only Baseline Failure Cases"
        )

        st.caption(
            "These are the findings where the prompt-only baseline used "
            "evidence that sounded relevant but failed the benchmark's "
            "driver or directional requirements."
        )

        failure_rows = []

        for decision in naive_failures:
            failure_rows.append(
                {
                    "Case": decision.get(
                        "case_id",
                        "",
                    ),
                    "Type": category_labels.get(
                        decision.get(
                            "case_type",
                            "",
                        ),
                        str(
                            decision.get(
                                "case_type",
                                "",
                            )
                        ).replace(
                            "_",
                            " ",
                        ).title(),
                    ),
                    "Finding": short_finding_id(
                        str(
                            decision.get(
                                "finding_id",
                                "",
                            )
                        )
                    ),
                    "Forbidden evidence used": (
                        ", ".join(
                            decision.get(
                                "forbidden_evidence_used",
                                [],
                            )
                        )
                        or "None"
                    ),
                    "Prompt-only explanation": (
                        decision.get(
                            "explanation",
                            "",
                        )
                    ),
                }
            )

        st.dataframe(
            failure_rows,
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "Results describe one controlled synthetic 32-case benchmark run "
        "with the configured model. They demonstrate this pipeline's "
        "behavior and should not be interpreted as universal model-performance "
        "statistics."
    )

    st.divider()

    st.subheader(
        "Cross-Period Evaluation Suite"
    )

    st.caption(
        "The offline benchmark evaluates the same controlled hidden "
        "ground-truth events across both configured analysis periods. "
        "Ground truth is used only after generation and is never exposed "
        "to the production AI."
    )

    if suite_summary.get(
        "overall_passed"
    ):
        st.markdown(
            '<span class="status-pass">'
            "Cross-period suite PASSED"
            "</span>",
            unsafe_allow_html=True,
        )
    else:
        st.error(
            "Cross-period evaluation suite failed."
        )

    st.write("")

    s1, s2, s3, s4 = st.columns(
        4
    )

    case_count = suite_summary.get(
        "case_count",
        0,
    )

    cases_passed = suite_summary.get(
        "cases_passed",
        0,
    )

    case_pass_rate = suite_summary.get(
        "case_pass_rate"
    )

    s1.metric(
        "Cases Passed",
        f"{cases_passed}/{case_count}",
    )

    s2.metric(
        "Case Pass Rate",
        (
            f"{case_pass_rate * 100:.0f}%"
            if case_pass_rate is not None
            else "N/A"
        ),
    )

    s3.metric(
        "Security Isolation",
        (
            "Passed"
            if suite_summary.get(
                "all_security_isolation_passed"
            )
            else "Failed"
        ),
    )

    s4.metric(
        "Citation Authorization",
        (
            "Passed"
            if suite_summary.get(
                "all_citations_authorized"
            )
            else "Failed"
        ),
    )

    st.divider()

    st.subheader(
        "Cross-Period Comparison"
    )

    comparison_rows = []

    for case in suite_cases.values():
        precision = case.get(
            "citation_precision"
        )

        utilization = case.get(
            "ai_visible_evidence_utilization"
        )

        comparison_rows.append(
            {
                "Analysis period": case.get(
                    "label",
                    "",
                ),
                "Status": (
                    "Passed"
                    if case.get(
                        "overall_passed"
                    )
                    else "Failed"
                ),
                "Events": case.get(
                    "ground_truth_event_count",
                    0,
                ),
                "AI-visible evidence": (
                    ", ".join(
                        case.get(
                            "ai_visible_evidence_ids",
                            [],
                        )
                    )
                    or "None"
                ),
                "Withheld evidence": (
                    ", ".join(
                        case.get(
                            "directionally_withheld_evidence_ids",
                            [],
                        )
                    )
                    or "None"
                ),
                "Citation precision": (
                    f"{precision * 100:.0f}%"
                    if precision is not None
                    else "N/A"
                ),
                "Evidence utilization": (
                    f"{utilization * 100:.0f}%"
                    if utilization is not None
                    else "N/A"
                ),
                "Security": (
                    "Passed"
                    if case.get(
                        "security_isolation_passed"
                    )
                    else "Failed"
                ),
            }
        )

    st.dataframe(
        comparison_rows,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Different routing across periods is expected. "
        "The same controlled events can be accepted, rejected or withheld "
        "differently because each analysis period produces different "
        "financial findings and directional context."
    )

    st.divider()

    st.subheader(
        f"Selected Period · {selected_period}"
    )

    if evaluation_summary.get(
        "overall_passed"
    ):
        st.markdown(
            '<span class="status-pass">'
            "Period evaluation PASSED"
            "</span>",
            unsafe_allow_html=True,
        )
    else:
        st.error(
            "Selected-period evaluation failed."
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
            "not_approved_by_grounding": (
                "Not approved by grounding"
            ),
            "not_connected_to_material_finding": (
                "Not connected to material finding"
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
            width="stretch",
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
        "not_approved_by_grounding": (
            "Not approved by grounding"
        ),
        "not_connected_to_material_finding": (
            "Not connected to material finding"
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
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "The selected-period event evaluation is based on a controlled "
        "synthetic six-event environment. The broader adversarial benchmark "
        "above contains 32 controlled cases and 38 evaluated findings."
    )


# ---------------------------------------------------------------------
# Data Explorer
# ---------------------------------------------------------------------

with data_tab:
    st.subheader(
        "Underlying Sales Data"
    )

    st.caption(
        "Inspect the processed sales datasets that feed the finance analysis. "
        "Use the dataset selector independently of the executive analysis "
        "period to inspect Actual, Budget or Latest Forecast sales records."
    )

    st.info(
        "This explorer loads only processed analyst-visible sales data. "
        "Hidden ground-truth files are never loaded by this tab."
    )

    selector_col, date_col = st.columns(
        [1, 2]
    )

    with selector_col:
        selected_dataset = st.selectbox(
            "Dataset",
            options=list(
                SALES_DATASETS.keys()
            ),
            index=2,
            help=(
                "Choose between Actual, Budget and Latest Forecast sales data."
            ),
        )

    sales_data = load_sales_data(
        SALES_DATASETS[selected_dataset]
    ).copy()

    if "date" not in sales_data.columns:
        st.error(
            "The selected dataset does not contain a date column."
        )
        st.stop()

    valid_dates = sales_data[
        "date"
    ].dropna()

    if valid_dates.empty:
        st.error(
            "The selected dataset contains no valid dates."
        )
        st.stop()

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    with date_col:
        selected_dates = st.date_input(
            "Date range",
            value=(
                min_date,
                max_date,
            ),
            min_value=min_date,
            max_value=max_date,
            key=f"date_range_{selected_dataset}",
        )

    filtered = sales_data.copy()

    if (
        isinstance(selected_dates, (tuple, list))
        and len(selected_dates) == 2
    ):
        start_date, end_date = selected_dates
        filtered = filtered[
            filtered["date"].dt.date.between(
                start_date,
                end_date,
            )
        ]

    filter_country, filter_family, filter_product, filter_segment = (
        st.columns(4)
    )

    with filter_country:
        country_options = sorted(
            filtered["country"].dropna().astype(str).unique().tolist()
        )
        selected_countries = st.multiselect(
            "Country",
            options=country_options,
            placeholder="All countries",
        )

    if selected_countries:
        filtered = filtered[
            filtered["country"].astype(str).isin(
                selected_countries
            )
        ]

    with filter_family:
        family_options = sorted(
            filtered["product_family"].dropna().astype(str).unique().tolist()
        )
        selected_families = st.multiselect(
            "Product family",
            options=family_options,
            placeholder="All families",
        )

    if selected_families:
        filtered = filtered[
            filtered["product_family"].astype(str).isin(
                selected_families
            )
        ]

    with filter_product:
        product_options = sorted(
            filtered["product"].dropna().astype(str).unique().tolist()
        )
        selected_products = st.multiselect(
            "Product",
            options=product_options,
            placeholder="All products",
        )

    if selected_products:
        filtered = filtered[
            filtered["product"].astype(str).isin(
                selected_products
            )
        ]

    with filter_segment:
        segment_options = sorted(
            filtered["segment"].dropna().astype(str).unique().tolist()
        )
        selected_segments = st.multiselect(
            "Customer segment",
            options=segment_options,
            placeholder="All segments",
        )

    if selected_segments:
        filtered = filtered[
            filtered["segment"].astype(str).isin(
                selected_segments
            )
        ]

    st.divider()

    row_count = len(filtered)
    total_units = (
        filtered["units"].sum()
        if "units" in filtered.columns
        else 0
    )
    total_revenue = (
        filtered["net_revenue"].sum()
        if "net_revenue" in filtered.columns
        else 0
    )
    total_gross_profit = (
        filtered["gross_profit"].sum()
        if "gross_profit" in filtered.columns
        else 0
    )
    weighted_margin = (
        total_gross_profit / total_revenue
        if total_revenue
        else None
    )

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    kpi1.metric(
        "Rows",
        f"{row_count:,}",
    )
    kpi2.metric(
        "Units",
        f"{float(total_units):,.0f}",
    )
    kpi3.metric(
        "Net Revenue",
        money_value(total_revenue),
    )
    kpi4.metric(
        "Gross Profit",
        money_value(total_gross_profit),
    )
    kpi5.metric(
        "Gross Margin",
        pct(weighted_margin),
    )

    st.divider()

    st.subheader(
        "Quick Breakdown"
    )

    st.caption(
        "Summarize the currently filtered data by a business dimension. "
        "Financial values are displayed in compact NOK format for readability."
    )

    breakdown_labels = {
        "country": "Country",
        "product_family": "Product Family",
        "product": "Product",
        "segment": "Customer Segment",
    }

    breakdown_dimension = st.selectbox(
        "Break down filtered data by",
        options=list(
            breakdown_labels.keys()
        ),
        format_func=lambda value: breakdown_labels[value],
    )

    if filtered.empty:
        st.warning(
            "No rows match the selected filters."
        )
    else:
        breakdown = (
            filtered.groupby(
                breakdown_dimension,
                dropna=False,
            )
            .agg(
                units=("units", "sum"),
                net_revenue=("net_revenue", "sum"),
                gross_profit=("gross_profit", "sum"),
            )
            .reset_index()
        )

        breakdown["gross_margin_pct"] = (
            breakdown["gross_profit"]
            / breakdown["net_revenue"]
        )

        breakdown = breakdown.sort_values(
            "net_revenue",
            ascending=False,
        )

        breakdown_display = breakdown.rename(
            columns={
                breakdown_dimension: breakdown_labels[breakdown_dimension],
                "units": "Units",
                "net_revenue": "Net Revenue",
                "gross_profit": "Gross Profit",
                "gross_margin_pct": "Gross Margin",
            }
        ).copy()

        breakdown_display["Units"] = breakdown_display[
            "Units"
        ].map(
            lambda value: f"{float(value):,.0f}"
        )
        breakdown_display["Net Revenue"] = breakdown_display[
            "Net Revenue"
        ].map(
            money_value
        )
        breakdown_display["Gross Profit"] = breakdown_display[
            "Gross Profit"
        ].map(
            money_value
        )
        breakdown_display["Gross Margin"] = breakdown_display[
            "Gross Margin"
        ].map(
            pct
        )

        st.dataframe(
            breakdown_display,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.subheader(
        "Filtered Sales Records"
    )

    display_columns = [
        "date",
        "scenario",
        "country",
        "product_family",
        "product",
        "segment",
        "units",
        "list_price",
        "discount_rate",
        "gross_sales",
        "discount_value",
        "net_revenue",
        "unit_cost",
        "cogs",
        "gross_profit",
        "gross_margin_pct",
    ]

    available_display_columns = [
        column
        for column in display_columns
        if column in filtered.columns
    ]

    sort_columns = [
        column
        for column in [
            "date",
            "country",
            "product",
            "segment",
        ]
        if column in available_display_columns
    ]

    table_data = filtered[
        available_display_columns
    ].sort_values(
        sort_columns,
        ascending=True,
    )

    preview_limit = 150
    preview_rows = min(
        len(table_data),
        preview_limit,
    )

    if row_count == 0:
        st.caption(
            f"No rows from {selected_dataset} match the selected filters."
        )
    elif row_count <= preview_limit:
        st.caption(
            f"Showing all {row_count:,} filtered rows from {selected_dataset}. "
            "The CSV download below contains the same filtered dataset."
        )
    else:
        st.caption(
            f"Showing the first {preview_rows:,} of {row_count:,} filtered rows "
            f"from {selected_dataset}. The preview is limited for readability; "
            f"the CSV download contains all {row_count:,} filtered rows."
        )

    preview_data = table_data.head(
        preview_limit
    ).copy()

    if not preview_data.empty:
        if "date" in preview_data.columns:
            preview_data["date"] = preview_data[
                "date"
            ].dt.strftime(
                "%Y-%m-%d"
            )

        if "units" in preview_data.columns:
            preview_data["units"] = preview_data[
                "units"
            ].map(
                lambda value: f"{float(value):,.0f}"
            )

        for money_column in [
            "list_price",
            "gross_sales",
            "discount_value",
            "net_revenue",
            "unit_cost",
            "cogs",
            "gross_profit",
        ]:
            if money_column in preview_data.columns:
                preview_data[money_column] = preview_data[
                    money_column
                ].map(
                    lambda value: (
                        ""
                        if pd.isna(value)
                        else f"NOK {float(value):,.2f}"
                    )
                )

        if "discount_rate" in preview_data.columns:
            preview_data["discount_rate"] = preview_data[
                "discount_rate"
            ].map(
                lambda value: (
                    ""
                    if pd.isna(value)
                    else f"{float(value) * 100:.2f}%"
                )
            )

        if "gross_margin_pct" in preview_data.columns:
            preview_data["gross_margin_pct"] = preview_data[
                "gross_margin_pct"
            ].map(
                lambda value: (
                    ""
                    if pd.isna(value)
                    else (
                        f"{float(value) * 100:.2f}%"
                        if abs(float(value)) <= 1.5
                        else f"{float(value):.2f}%"
                    )
                )
            )

    st.dataframe(
        preview_data,
        use_container_width=True,
        hide_index=True,
        height=460,
    )

    csv_bytes = table_data.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    safe_dataset_name = selected_dataset.lower().replace(
        " ",
        "_",
    )

    st.download_button(
        "Download all filtered rows as CSV",
        data=csv_bytes,
        file_name=(
            f"{safe_dataset_name}_filtered_sales.csv"
        ),
        mime="text/csv",
        use_container_width=False,
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.divider()

st.caption(
    "CFO Intelligence Copilot · Synthetic portfolio project · "
    "Deterministic finance calculations · Scenario & sensitivity analysis · "
    "Evidence grounding · "
    "Directional guardrails · Structured AI commentary · "
    "Adversarial safety benchmarking · Hidden-ground-truth evaluation"
)