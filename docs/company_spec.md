# Northstar Systems AS — Company & Financial Model Specification

## 1. Company overview

Northstar Systems AS is a fictional Norwegian B2B industrial technology company headquartered in Stavanger.

The company develops and sells industrial control, sensor and connectivity products to customers across Northern and Western Europe.

Reporting currency: NOK

Management reporting frequency: Monthly

Primary management users:
- CFO
- FP&A / Business Controller
- Commercial Director
- Country Managers

---

## 2. Geographic markets

Northstar operates in six markets:

- Norway
- Sweden
- Denmark
- Germany
- Netherlands
- United Kingdom

Each market has different:
- sales volumes
- pricing levels
- discounts
- product mix
- cost levels
- growth rates

---

## 3. Product portfolio

### Control Systems
- EdgeHub Pro
- EdgeHub Core
- Control Mini

### Sensor Systems
- Sensor X Pro
- Sensor X
- Sensor Lite

### Accessories & Services
- Service Kit
- Connectivity Pack

Products differ in:
- list price
- standard unit cost
- gross margin
- demand
- price sensitivity

---

## 4. Customer segments

- Enterprise
- SME
- Distributor
- Systems Integrator

Segments differ in:
- average order size
- discount level
- product mix
- sales volume

Distributors normally receive higher discounts than direct enterprise customers.

---

## 5. Reporting periods

### Actual
January 2024 – August 2026

### Budget
January 2026 – December 2026

### Latest Forecast
January 2026 – December 2026

For the Latest Forecast:
- January–August 2026 should equal actual results.
- September–December 2026 should represent the latest management estimate.

This creates a realistic FP&A reporting environment in September 2026.

---

## 6. Revenue model

Revenue should be calculated deterministically:

Gross Sales =
Units Sold × List Price

Discount Value =
Gross Sales × Discount Rate

Net Revenue =
Gross Sales − Discount Value

The dataset must therefore preserve the underlying commercial drivers:

- Units
- List Price
- Discount Rate
- Net Selling Price
- Net Revenue

This allows later price-volume-mix and commercial variance analysis.

---

## 7. Cost of goods sold

Variable product cost should be based on:

COGS =
Units Sold × Unit Cost

Unit cost varies by:
- product
- time
- selected operational events

Gross Profit =
Net Revenue − COGS

Gross Margin % =
Gross Profit / Net Revenue

---

## 8. Operating expenses

Operating expenses should include:

- Payroll
- Contractors
- Marketing
- Software
- Travel
- Facilities
- Professional Services
- R&D
- Other OPEX

OPEX should be allocated by:
- month
- country or corporate function
- cost centre
- account

---

## 9. Core financial KPIs

The finance engine should ultimately calculate:

- Net Revenue
- Revenue Growth %
- Gross Profit
- Gross Margin %
- OPEX
- EBITDA
- EBITDA Margin %
- Budget Variance
- Forecast Variance
- Prior-Year Variance
- Price Impact
- Volume Impact
- Mix Impact
- Cost Impact

The LLM must not calculate these numbers independently when deterministic calculations are available.

---

## 10. Controlled business events

The synthetic dataset must contain deliberately designed business events.

Examples:

### Event with evidence
Germany — August 2026

EdgeHub Pro supplier cost increases by approximately 8%.

Financial impact:
Negative gross margin impact.

A management note will explicitly mention the temporary supplier surcharge.

The AI system should therefore be able to connect:
financial variance → documented business explanation.

### Event without evidence
Germany — August 2026

Average discount rate increases materially.

Financial impact:
Negative revenue / margin impact.

No management note will explain why.

The AI system must identify the financial effect but state that no documented explanation is available.

It must NOT invent a reason.

---

## 11. Ground truth

A separate hidden evaluation dataset will contain the real synthetic events used to create the financial data.

Example fields:

- event_id
- month
- country
- product
- metric
- event_description
- expected_financial_effect
- evidence_available
- evidence_id

This dataset must never be available to the AI agent during normal analysis.

It exists only for evaluation and testing.

---

## 12. Design principle

The project must clearly separate:

1. Calculated facts
2. Retrieved evidence
3. AI interpretation
4. Unsupported hypotheses

The system should prefer:

"Insufficient evidence is available to determine the underlying cause."

over inventing a plausible explanation.

Reliability and auditability are more important than producing an answer to every question.