import json
import re
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_INDEX_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evidence"
    / "evidence_index.json"
)

FORECAST_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "latest_forecast_vs_budget.json"
)

YTD_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "analysis"
    / "actual_ytd_vs_budget.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "grounding"
)

FORECAST_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "latest_forecast_vs_budget_grounded.json"
)

YTD_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "actual_ytd_vs_budget_grounded.json"
)


# ============================================================
# DESIGN PRINCIPLES
# ============================================================
#
# The Grounding Engine separates:
#
# 1. CALCULATED FACT
#    Produced by the deterministic finance engine.
#
# 2. SUPPORTED / PARTIALLY SUPPORTED EXPLANATION
#    Documentary evidence supports the calculated driver.
#
# 3. RELATED BUT NOT EXPLANATORY
#    Evidence concerns the same entity but supports another
#    financial driver.
#
# 4. INSUFFICIENT EVIDENCE
#    No analyst-visible evidence supports an explanation.
#
#
# IMPORTANT:
#
# Ground truth is NEVER read by this module.
#
# Evidence matching is intentionally conservative.
# ============================================================


# ============================================================
# ENTITY ALIASES
# ============================================================

COUNTRY_ALIASES = {
    "Germany": [
        "germany",
    ],
    "Norway": [
        "norway",
    ],
    "Sweden": [
        "sweden",
    ],
    "Denmark": [
        "denmark",
    ],
    "Netherlands": [
        "netherlands",
    ],
    "United Kingdom": [
        "united kingdom",
        "uk",
        "u.k.",
    ],
}


PRODUCT_ALIASES = {
    "EdgeHub Pro": [
        "edgehub pro",
    ],
    "EdgeHub Core": [
        "edgehub core",
    ],
    "Control Mini": [
        "control mini",
    ],
    "Sensor X Pro": [
        "sensor x pro",
    ],
    "Sensor X": [
        "sensor x",
    ],
    "Sensor Lite": [
        "sensor lite",
    ],
    "Service Kit": [
        "service kit",
    ],
    "Connectivity Pack": [
        "connectivity pack",
    ],
}


COST_CENTRE_ALIASES = {
    "Sales": [
        "sales",
    ],
    "Operations": [
        "operations",
    ],
    "Product & R&D": [
        "product & r&d",
        "product and r&d",
        "product and research",
    ],
    "Finance": [
        "finance",
    ],
    "People": [
        "people",
    ],
    "Technology": [
        "technology",
    ],
    "Corporate": [
        "corporate",
    ],
}


ACCOUNT_ALIASES = {
    "Payroll": [
        "payroll",
        "salary",
        "salaries",
    ],
    "Contractors": [
        "contractor",
        "contractors",
        "external engineering",
        "external testing",
    ],
    "Marketing": [
        "marketing",
    ],
    "Software": [
        "software",
    ],
    "Travel": [
        "travel",
    ],
    "Facilities": [
        "facilities",
    ],
    "Professional Services": [
        "professional services",
    ],
    "R&D": [
        "r&d spend",
        "research and development spend",
    ],
    "Other OPEX": [
        "other opex",
    ],
}


SEGMENT_ALIASES = {
    "Enterprise": [
        "enterprise",
    ],
    "SME": [
        "sme",
    ],
    "Distributor": [
        "distributor",
    ],
    "Systems Integrator": [
        "systems integrator",
    ],
}


# ============================================================
# DRIVER LANGUAGE
# ============================================================
#
# Driver detection is deliberately conservative.
#
# The title of a management note receives substantially more
# weight than incidental language appearing in its body.
#
# This prevents a note titled:
#
#     Germany EdgeHub Pro procurement surcharge
#
# from becoming volume/mix evidence merely because the body
# happens to mention shipments or deliveries.
#
# ============================================================

DRIVER_KEYWORDS = {

    "volume_mix": [
        "shipment delay",
        "shipment acceleration",
        "delivery delay",
        "delivery acceleration",
        "shipping delay",
        "demand decline",
        "demand increase",
        "demand weakness",
        "demand growth",
        "order delay",
        "order acceleration",
        "volume decline",
        "volume increase",
        "volume growth",
        "volume weakness",
        "units sold",
        "unit volumes",
        "sales volume",
    ],

    "list_price": [
        "list price",
        "pricing adjustment",
        "pricing increase",
        "pricing decrease",
        "price realization",
        "selling price",
        "price increase",
        "price decrease",
    ],

    "discount": [
        "discount",
        "discounting",
        "rebate",
        "commercial terms",
        "promotional discount",
    ],

    "unit_cost": [
        "procurement surcharge",
        "supplier surcharge",
        "supplier cost",
        "procurement cost",
        "unit cost",
        "component cost",
        "material cost",
        "input cost",
        "purchase cost",
    ],

    "headcount": [
        "headcount",
        "hiring",
        "new hires",
        "staffing",
        "employee additions",
        "employee reduction",
    ],

    "employee_cost": [
        "employee cost",
        "salary increase",
        "salary inflation",
        "wage increase",
        "wage inflation",
        "compensation increase",
    ],

    "non_payroll": [
        "contractor acceleration",
        "contractor",
        "contractors",
        "external engineering",
        "external testing",
        "marketing spend",
        "software spend",
        "travel spend",
        "professional services",
    ],
}


# ============================================================
# LOADERS
# ============================================================

def load_json(path: Path) -> dict:
    """
    Load JSON and fail clearly if the file does not exist.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize free text for deterministic matching.
    """

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(
    text: str,
    alias_map: dict,
) -> list[str]:
    """
    Extract canonical entities using longest-match-first
    resolution.

    Example:

        Sensor X Pro

    resolves to:

        Sensor X Pro

    and NOT simultaneously to:

        Sensor X
    """

    normalized = normalize_text(
        text
    )

    candidate_matches = []

    for canonical, aliases in alias_map.items():

        for alias in aliases:

            normalized_alias = (
                normalize_text(
                    alias
                )
            )

            pattern = (
                r"(?<!\w)"
                + re.escape(
                    normalized_alias
                )
                + r"(?!\w)"
            )

            for match in re.finditer(
                pattern,
                normalized,
            ):

                candidate_matches.append(
                    {
                        "canonical":
                            canonical,
                        "alias":
                            normalized_alias,
                        "start":
                            match.start(),
                        "end":
                            match.end(),
                        "length":
                            (
                                match.end()
                                - match.start()
                            ),
                    }
                )

    candidate_matches.sort(
        key=lambda item: (
            -item["length"],
            item["start"],
        )
    )

    accepted_matches = []

    for candidate in candidate_matches:

        overlaps_existing = any(
            not (
                candidate["end"]
                <= accepted["start"]
                or
                candidate["start"]
                >= accepted["end"]
            )
            for accepted
            in accepted_matches
        )

        if overlaps_existing:
            continue

        accepted_matches.append(
            candidate
        )

    canonical_entities = []

    for match in accepted_matches:

        canonical = match[
            "canonical"
        ]

        if canonical not in canonical_entities:

            canonical_entities.append(
                canonical
            )

    return canonical_entities


# ============================================================
# DRIVER MATCH HELPERS
# ============================================================

def find_driver_matches(
    text: str,
) -> dict[str, list[str]]:
    """
    Return all matched phrases by driver.
    """

    normalized = normalize_text(
        text
    )

    result = {}

    for driver, keywords in DRIVER_KEYWORDS.items():

        matches = [
            keyword
            for keyword in keywords
            if keyword in normalized
        ]

        if matches:

            result[
                driver
            ] = matches

    return result


def driver_score(
    title_matches: list[str],
    body_matches: list[str],
) -> int:
    """
    Calculate evidence-driver strength.

    Title evidence receives much higher weight than body
    evidence because note titles represent the intentionally
    documented business event.

    Longer phrases also receive a modest specificity bonus.
    """

    score = 0

    for phrase in title_matches:

        score += 100

        score += len(
            phrase
        )

    for phrase in body_matches:

        score += 10

        score += min(
            len(phrase),
            30,
        )

    return score


def resolve_primary_evidence_driver(
    title: str,
    body: str,
) -> tuple[str | None, dict]:
    """
    Resolve the primary financial driver documented by one
    management note.

    The note title dominates body text.

    Returns:
        primary_driver
        driver_match_details
    """

    title_matches = (
        find_driver_matches(
            title
        )
    )

    body_matches = (
        find_driver_matches(
            body
        )
    )

    all_drivers = set(
        title_matches.keys()
    ) | set(
        body_matches.keys()
    )

    if not all_drivers:

        return None, {}

    scores = {}

    details = {}

    for driver in all_drivers:

        driver_title_matches = (
            title_matches.get(
                driver,
                [],
            )
        )

        driver_body_matches = (
            body_matches.get(
                driver,
                [],
            )
        )

        score = driver_score(
            driver_title_matches,
            driver_body_matches,
        )

        scores[
            driver
        ] = score

        details[
            driver
        ] = {
            "score":
                score,

            "title_matches":
                driver_title_matches,

            "body_matches":
                driver_body_matches,
        }

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    primary_driver = (
        ranked[0][0]
    )

    return (
        primary_driver,
        details,
    )


# ============================================================
# EVIDENCE ENRICHMENT
# ============================================================

def enrich_evidence(
    evidence_records: list[dict],
) -> list[dict]:
    """
    Add deterministic entity and driver metadata.
    """

    enriched = []

    for record in evidence_records:

        title = record[
            "title"
        ]

        body = record[
            "text"
        ]

        combined_text = (
            f"{title} {body}"
        )

        countries = extract_entities(
            combined_text,
            COUNTRY_ALIASES,
        )

        products = extract_entities(
            combined_text,
            PRODUCT_ALIASES,
        )

        cost_centres = extract_entities(
            combined_text,
            COST_CENTRE_ALIASES,
        )

        accounts = extract_entities(
            combined_text,
            ACCOUNT_ALIASES,
        )

        segments = extract_entities(
            combined_text,
            SEGMENT_ALIASES,
        )

        (
            primary_evidence_driver,
            driver_match_details,
        ) = resolve_primary_evidence_driver(
            title,
            body,
        )

        enriched_record = {
            **record,

            "entities": {
                "countries":
                    countries,

                "products":
                    products,

                "cost_centres":
                    cost_centres,

                "accounts":
                    accounts,

                "segments":
                    segments,
            },

            "primary_evidence_driver":
                primary_evidence_driver,

            "driver_match_details":
                driver_match_details,
        }

        enriched.append(
            enriched_record
        )

    return enriched


# ============================================================
# BUILD FINANCIAL FINDINGS
# ============================================================

def build_findings(
    analysis: dict,
) -> list[dict]:
    """
    Convert deterministic finance rankings into findings.
    """

    rankings = analysis[
        "rankings"
    ]

    findings = []

    # --------------------------------------------------------
    # Countries
    # --------------------------------------------------------

    for record in rankings[
        "countries"
    ]:

        findings.append(
            {
                "finding_id":
                    (
                        "country::"
                        + record[
                            "country"
                        ]
                    ),

                "finding_type":
                    "country",

                "entity":
                    record[
                        "country"
                    ],

                "calculated_fact": {
                    "revenue_variance_nok":
                        record[
                            "revenue_variance_nok"
                        ],

                    "gross_profit_variance_nok":
                        record[
                            "gross_profit_variance_nok"
                        ],

                    "primary_driver":
                        record[
                            "largest_gp_driver"
                        ],

                    "primary_driver_impact_nok":
                        record[
                            "largest_gp_driver_impact_nok"
                        ],
                },

                "material":
                    record[
                        "material"
                    ],
            }
        )

    # --------------------------------------------------------
    # Products
    # --------------------------------------------------------

    for record in rankings[
        "products"
    ]:

        findings.append(
            {
                "finding_id":
                    (
                        "product::"
                        + record[
                            "product"
                        ]
                    ),

                "finding_type":
                    "product",

                "entity":
                    record[
                        "product"
                    ],

                "calculated_fact": {
                    "revenue_variance_nok":
                        record[
                            "revenue_variance_nok"
                        ],

                    "gross_profit_variance_nok":
                        record[
                            "gross_profit_variance_nok"
                        ],

                    "primary_driver":
                        record[
                            "largest_gp_driver"
                        ],

                    "primary_driver_impact_nok":
                        record[
                            "largest_gp_driver_impact_nok"
                        ],
                },

                "material":
                    record[
                        "material"
                    ],
            }
        )

    # --------------------------------------------------------
    # OPEX accounts
    # --------------------------------------------------------

    for record in rankings[
        "opex_accounts"
    ]:

        account = record[
            "account"
        ]

        findings.append(
            {
                "finding_id":
                    (
                        "opex_account::"
                        + account
                    ),

                "finding_type":
                    "opex_account",

                "entity":
                    account,

                "calculated_fact": {
                    "opex_variance_nok":
                        record[
                            "opex_variance_nok"
                        ],

                    "ebitda_impact_nok":
                        record[
                            "ebitda_impact_nok"
                        ],

                    "primary_driver":
                        (
                            "payroll"
                            if account
                            == "Payroll"
                            else
                            "non_payroll"
                        ),
                },

                "material":
                    record[
                        "material"
                    ],
            }
        )

    # --------------------------------------------------------
    # Cost centres
    # --------------------------------------------------------

    for record in rankings[
        "cost_centres"
    ]:

        findings.append(
            {
                "finding_id":
                    (
                        "cost_centre::"
                        + record[
                            "cost_centre"
                        ]
                    ),

                "finding_type":
                    "cost_centre",

                "entity":
                    record[
                        "cost_centre"
                    ],

                "calculated_fact": {
                    "opex_variance_nok":
                        record[
                            "opex_variance_nok"
                        ],

                    "ebitda_impact_nok":
                        record[
                            "ebitda_impact_nok"
                        ],

                    "primary_driver":
                        None,
                },

                "material":
                    record[
                        "material"
                    ],
            }
        )

    return findings


# ============================================================
# ENTITY MATCHING
# ============================================================

def entity_matches_finding(
    finding: dict,
    evidence: dict,
) -> bool:
    """
    Determine whether an evidence record concerns the entity
    represented by the financial finding.
    """

    finding_type = finding[
        "finding_type"
    ]

    entity = finding[
        "entity"
    ]

    entities = evidence[
        "entities"
    ]

    if finding_type == "country":

        return (
            entity
            in entities[
                "countries"
            ]
        )

    if finding_type == "product":

        return (
            entity
            in entities[
                "products"
            ]
        )

    if finding_type == "opex_account":

        return (
            entity
            in entities[
                "accounts"
            ]
        )

    if finding_type == "cost_centre":

        return (
            entity
            in entities[
                "cost_centres"
            ]
        )

    return False


# ============================================================
# DRIVER MATCHING
# ============================================================

def driver_matches_finding(
    finding: dict,
    evidence: dict,
) -> bool:
    """
    Determine whether the evidence's PRIMARY documented
    driver supports the calculated financial driver.
    """

    finding_driver = (
        finding[
            "calculated_fact"
        ].get(
            "primary_driver"
        )
    )

    evidence_driver = (
        evidence.get(
            "primary_evidence_driver"
        )
    )

    # --------------------------------------------------------
    # Cost-centre findings are aggregate findings without one
    # deterministic primary driver.
    # --------------------------------------------------------

    if finding_driver is None:

        return True

    # --------------------------------------------------------
    # Payroll may be driven by either headcount or employee
    # cost.
    # --------------------------------------------------------

    if finding_driver == "payroll":

        return (
            evidence_driver
            in {
                "headcount",
                "employee_cost",
            }
        )

    return (
        finding_driver
        == evidence_driver
    )


# ============================================================
# EVIDENCE SCOPE
# ============================================================

def count_specific_scope_dimensions(
    evidence: dict,
) -> int:
    """
    Count explicit business dimensions in an evidence note.

    Example:

        Germany + EdgeHub Pro

    is narrower than:

        Germany
    """

    entities = evidence[
        "entities"
    ]

    dimension_count = 0

    for key in [
        "countries",
        "products",
        "cost_centres",
        "accounts",
        "segments",
    ]:

        if entities[
            key
        ]:

            dimension_count += 1

    return dimension_count


def finding_scope_dimensions(
    finding: dict,
) -> int:
    """
    Current finance findings each represent one analytical
    dimension.
    """

    return 1


# ============================================================
# SUPPORT CLASSIFICATION
# ============================================================

def classify_evidence_support(
    finding: dict,
    evidence: dict,
) -> str:
    """
    Classify evidence against a calculated financial finding.

    Possible statuses:

        supported
        partial_support
        related_not_explanatory
        not_relevant
    """

    if not entity_matches_finding(
        finding,
        evidence,
    ):

        return "not_relevant"

    if not driver_matches_finding(
        finding,
        evidence,
    ):

        return (
            "related_not_explanatory"
        )

    evidence_scope = (
        count_specific_scope_dimensions(
            evidence
        )
    )

    finding_scope = (
        finding_scope_dimensions(
            finding
        )
    )

    if evidence_scope > finding_scope:

        return "partial_support"

    return "supported"


# ============================================================
# GROUND ONE FINDING
# ============================================================

def ground_finding(
    finding: dict,
    evidence_records: list[dict],
) -> dict:
    """
    Match analyst-visible evidence against one calculated
    financial finding.
    """

    candidates = []

    for evidence in evidence_records:

        status = (
            classify_evidence_support(
                finding,
                evidence,
            )
        )

        if status == "not_relevant":

            continue

        candidates.append(
            {
                "evidence_id":
                    evidence[
                        "evidence_id"
                    ],

                "title":
                    evidence[
                        "title"
                    ],

                "source_file":
                    evidence[
                        "source_file"
                    ],

                "support_status":
                    status,

                "primary_evidence_driver":
                    evidence[
                        "primary_evidence_driver"
                    ],

                "evidence_entities":
                    evidence[
                        "entities"
                    ],
            }
        )

    priority = {
        "supported": 3,
        "partial_support": 2,
        "related_not_explanatory": 1,
    }

    candidates.sort(
        key=lambda candidate:
            priority[
                candidate[
                    "support_status"
                ]
            ],
        reverse=True,
    )

    usable_candidates = [
        candidate
        for candidate in candidates
        if candidate[
            "support_status"
        ]
        in {
            "supported",
            "partial_support",
        }
    ]

    if usable_candidates:

        explanation_status = (
            "evidence_available"
        )

        usable_evidence_ids = [
            candidate[
                "evidence_id"
            ]
            for candidate
            in usable_candidates
        ]

        allowed_ai_behavior = (
            "Use only supported or partial-support "
            "evidence. Do not generalize evidence beyond "
            "its documented entity and driver scope."
        )

    else:

        explanation_status = (
            "insufficient_evidence"
        )

        usable_evidence_ids = []

        allowed_ai_behavior = (
            "State that available evidence is insufficient "
            "to determine the underlying cause. Do not "
            "infer or invent a cause."
        )

    return {
        **finding,

        "grounding": {
            "explanation_status":
                explanation_status,

            "usable_evidence_ids":
                usable_evidence_ids,

            "evidence_candidates":
                candidates,

            "allowed_ai_behavior":
                allowed_ai_behavior,
        },
    }


# ============================================================
# BUILD GROUNDED ANALYSIS
# ============================================================

def build_grounded_analysis(
    analysis: dict,
    evidence_records: list[dict],
) -> dict:
    """
    Build grounded analysis for one finance comparison.
    """

    findings = build_findings(
        analysis
    )

    grounded_findings = [
        ground_finding(
            finding,
            evidence_records,
        )
        for finding
        in findings
    ]

    evidence_available_count = sum(
        1
        for finding
        in grounded_findings
        if finding[
            "grounding"
        ][
            "explanation_status"
        ]
        == "evidence_available"
    )

    insufficient_count = (
        len(
            grounded_findings
        )
        - evidence_available_count
    )

    return {
        "metadata": {
            "comparison":
                analysis[
                    "metadata"
                ][
                    "comparison"
                ],

            "grounding_engine":
                "deterministic",

            "grounding_version":
                "1.1",

            "ground_truth_access":
                False,
        },

        "finance_summary":
            analysis[
                "summary"
            ],

        "grounding_summary": {
            "finding_count":
                len(
                    grounded_findings
                ),

            "evidence_available_count":
                evidence_available_count,

            "insufficient_evidence_count":
                insufficient_count,
        },

        "findings":
            grounded_findings,
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_grounded_analysis(
    grounded: dict,
    valid_evidence_ids: set[str],
) -> None:
    """
    Validate grounding integrity.
    """

    if grounded[
        "metadata"
    ][
        "ground_truth_access"
    ]:

        raise ValueError(
            "Ground truth leakage detected."
        )

    for finding in grounded[
        "findings"
    ]:

        grounding = finding[
            "grounding"
        ]

        usable_ids = set(
            grounding[
                "usable_evidence_ids"
            ]
        )

        for evidence_id in usable_ids:

            if (
                evidence_id
                not in valid_evidence_ids
            ):

                raise ValueError(
                    "Unknown evidence ID linked "
                    f"to finding: {evidence_id}"
                )

        if (
            grounding[
                "explanation_status"
            ]
            == "insufficient_evidence"
            and usable_ids
        ):

            raise ValueError(
                "Insufficient-evidence finding "
                "cannot contain usable evidence."
            )

        for candidate in grounding[
            "evidence_candidates"
        ]:

            if (
                candidate[
                    "support_status"
                ]
                == "related_not_explanatory"
                and candidate[
                    "evidence_id"
                ]
                in usable_ids
            ):

                raise ValueError(
                    "Related-but-not-explanatory evidence "
                    "was incorrectly marked as usable."
                )


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_grounded_analysis(
    payload: dict,
    output_path: Path,
) -> None:
    """
    Save grounded analysis JSON.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# CONSOLE HELPERS
# ============================================================

def format_nok_m(
    value: float | None,
) -> str:
    """
    Format NOK amount in millions.
    """

    if value is None:

        return "N/A"

    return (
        f"NOK "
        f"{value / 1_000_000:,.1f}m"
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_grounded_analysis(
    grounded: dict,
) -> None:
    """
    Print material findings and grounding status.
    """

    print(
        "\n"
        + "=" * 120
    )

    print(
        grounded[
            "metadata"
        ][
            "comparison"
        ]
    )

    print(
        "=" * 120
    )

    for finding in grounded[
        "findings"
    ]:

        if not finding[
            "material"
        ]:

            continue

        finding_type = finding[
            "finding_type"
        ]

        entity = finding[
            "entity"
        ]

        fact = finding[
            "calculated_fact"
        ]

        grounding = finding[
            "grounding"
        ]

        if (
            "gross_profit_variance_nok"
            in fact
        ):

            impact = fact[
                "gross_profit_variance_nok"
            ]

        else:

            impact = fact[
                "ebitda_impact_nok"
            ]

        primary_driver = (
            fact.get(
                "primary_driver"
            )
        )

        evidence_text = (
            ", ".join(
                grounding[
                    "usable_evidence_ids"
                ]
            )
            if grounding[
                "usable_evidence_ids"
            ]
            else "NONE"
        )

        print(
            f"{finding_type:<14} | "
            f"{entity:<24} | "
            f"{format_nok_m(impact):>14} | "
            f"Driver: "
            f"{str(primary_driver):<12} | "
            f"Evidence: "
            f"{evidence_text:<20} | "
            f"{grounding['explanation_status']}"
        )

    print(
        "-" * 120
    )

    summary = grounded[
        "grounding_summary"
    ]

    print(
        f"Findings:              "
        f"{summary['finding_count']}"
    )

    print(
        f"Evidence available:    "
        f"{summary['evidence_available_count']}"
    )

    print(
        f"Insufficient evidence: "
        f"{summary['insufficient_evidence_count']}"
    )

    print(
        "Ground truth access:   "
        "BLOCKED"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run deterministic grounding.
    """

    evidence_payload = load_json(
        EVIDENCE_INDEX_PATH
    )

    forecast_analysis = load_json(
        FORECAST_ANALYSIS_PATH
    )

    ytd_analysis = load_json(
        YTD_ANALYSIS_PATH
    )

    evidence_records = (
        enrich_evidence(
            evidence_payload[
                "evidence"
            ]
        )
    )

    valid_evidence_ids = {
        record[
            "evidence_id"
        ]
        for record
        in evidence_records
    }

    # --------------------------------------------------------
    # Latest Forecast vs Budget
    # --------------------------------------------------------

    grounded_forecast = (
        build_grounded_analysis(
            forecast_analysis,
            evidence_records,
        )
    )

    validate_grounded_analysis(
        grounded_forecast,
        valid_evidence_ids,
    )

    save_grounded_analysis(
        grounded_forecast,
        FORECAST_OUTPUT_PATH,
    )

    # --------------------------------------------------------
    # Actual YTD vs Budget
    # --------------------------------------------------------

    grounded_ytd = (
        build_grounded_analysis(
            ytd_analysis,
            evidence_records,
        )
    )

    validate_grounded_analysis(
        grounded_ytd,
        valid_evidence_ids,
    )

    save_grounded_analysis(
        grounded_ytd,
        YTD_OUTPUT_PATH,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print(
        "\nNorthstar Systems AS"
    )

    print(
        "CFO Intelligence Copilot — Grounding Engine"
    )

    print_grounded_analysis(
        grounded_forecast
    )

    print_grounded_analysis(
        grounded_ytd
    )

    print(
        "\nGrounded outputs:"
    )

    print(
        FORECAST_OUTPUT_PATH
    )

    print(
        YTD_OUTPUT_PATH
    )

    print(
        "\nGrounding Engine: PASSED"
    )


if __name__ == "__main__":
    main()