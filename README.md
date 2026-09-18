# CFO Intelligence Copilot

**Live demo:** https://cfo-intelligence-copilot.streamlit.app

An evidence-grounded financial analysis system that combines deterministic finance calculations with controlled AI-generated management commentary.

The project is built around a simple principle:

> AI should explain financial performance only when the available evidence actually supports the explanation.

Rather than giving a language model raw financial data and asking it to "analyze the business", the system separates calculation, evidence retrieval, validation and AI interpretation into distinct layers.

---

## Overview

CFO Intelligence Copilot analyzes financial performance for a synthetic company, **Northstar Systems AS**.

The system compares financial performance against budget and produces:

- Revenue variance analysis
- Gross profit variance analysis
- OPEX variance analysis
- EBITDA and margin analysis
- Country-level performance drivers
- Product-level performance drivers
- Cost-centre and account-level OPEX drivers
- Evidence-supported management explanations
- Explicit unresolved findings when evidence is insufficient
- Management questions for further investigation

The AI model is **not responsible for calculating the financial results**.

All core financial calculations are produced deterministically in Python before information reaches the AI layer.

---

## Why This Project Exists

Large language models are good at explaining information, but financial analysis requires more than fluent text.

A useful finance AI system should be able to distinguish between:

1. **Calculated financial facts**
2. **Available supporting evidence**
3. **AI-generated interpretation**
4. **Unsupported hypotheses**

CFO Intelligence Copilot was designed around this separation.

If the system cannot support an explanation with approved evidence, it explicitly leaves the finding unresolved instead of allowing the AI to invent a plausible cause.

---

## Architecture

```mermaid
flowchart LR
    A[Financial Data] --> B[Deterministic Finance Engine]
    B --> C[Variance & Driver Analysis]
    C --> D[Evidence Grounding]
    D --> E[Directional Guardrails]
    E --> F[AI Commentary Layer]
    F --> G[Management Output]

    H[Hidden Ground Truth] --> I[Offline Evaluation]
    G --> I

    H -. Never visible to production AI .-> F