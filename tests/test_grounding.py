import json
from pathlib import Path

from src.evidence.grounding_engine import (
    EVIDENCE_INDEX_PATH,
    FORECAST_ANALYSIS_PATH,
    YTD_ANALYSIS_PATH,
    build_grounded_analysis,
    enrich_evidence,
    load_json,
    resolve_primary_evidence_driver,
)


# ============================================================
# TEST HELPERS
# ============================================================

def load_test_data():
    """
    Load the same analyst-visible evidence and deterministic
    finance outputs used by the Grounding Engine.
    """

    evidence_payload = load_json(
        EVIDENCE_INDEX_PATH
    )

    evidence_records = enrich_evidence(
        evidence_payload[
            "evidence"
        ]
    )

    forecast_analysis = load_json(
        FORECAST_ANALYSIS_PATH
    )

    ytd_analysis = load_json(
        YTD_ANALYSIS_PATH
    )

    grounded_forecast = (
        build_grounded_analysis(
            forecast_analysis,
            evidence_records,
        )
    )

    grounded_ytd = (
        build_grounded_analysis(
            ytd_analysis,
            evidence_records,
        )
    )

    return {
        "evidence":
            evidence_records,

        "forecast":
            grounded_forecast,

        "ytd":
            grounded_ytd,
    }


def find_evidence(
    evidence_records,
    evidence_id,
):
    """
    Retrieve one evidence record by ID.
    """

    for record in evidence_records:

        if (
            record[
                "evidence_id"
            ]
            == evidence_id
        ):

            return record

    raise AssertionError(
        f"Evidence not found: {evidence_id}"
    )


def find_finding(
    grounded_analysis,
    finding_type,
    entity,
):
    """
    Retrieve one grounded finding.
    """

    for finding in grounded_analysis[
        "findings"
    ]:

        if (
            finding[
                "finding_type"
            ]
            == finding_type
            and
            finding[
                "entity"
            ]
            == entity
        ):

            return finding

    raise AssertionError(
        "Finding not found: "
        f"{finding_type} / {entity}"
    )


def candidate_by_id(
    finding,
    evidence_id,
):
    """
    Retrieve one evidence candidate attached to a finding.
    """

    for candidate in finding[
        "grounding"
    ][
        "evidence_candidates"
    ]:

        if (
            candidate[
                "evidence_id"
            ]
            == evidence_id
        ):

            return candidate

    return None


# ============================================================
# DRIVER RESOLUTION TESTS
# ============================================================

def test_note_014_resolves_to_unit_cost():
    """
    Procurement surcharge must be classified as unit cost,
    not volume/mix.
    """

    data = load_test_data()

    note = find_evidence(
        data["evidence"],
        "NOTE-014",
    )

    assert (
        note[
            "primary_evidence_driver"
        ]
        == "unit_cost"
    )


def test_note_015_resolves_to_volume_mix():
    """
    UK shipment delay should support volume/mix.
    """

    data = load_test_data()

    note = find_evidence(
        data["evidence"],
        "NOTE-015",
    )

    assert (
        note[
            "primary_evidence_driver"
        ]
        == "volume_mix"
    )


def test_note_016_resolves_to_volume_mix():
    """
    Norway shipment acceleration should support volume/mix.
    """

    data = load_test_data()

    note = find_evidence(
        data["evidence"],
        "NOTE-016",
    )

    assert (
        note[
            "primary_evidence_driver"
        ]
        == "volume_mix"
    )


def test_note_017_resolves_to_non_payroll():
    """
    Contractor acceleration should support non-payroll OPEX.
    """

    data = load_test_data()

    note = find_evidence(
        data["evidence"],
        "NOTE-017",
    )

    assert (
        note[
            "primary_evidence_driver"
        ]
        == "non_payroll"
    )


# ============================================================
# FALSE CAUSAL MATCH TESTS
# ============================================================

def test_germany_volume_mix_not_explained_by_note_014():
    """
    NOTE-014 concerns Germany but documents unit-cost
    pressure.

    It must NOT explain Germany's calculated volume/mix
    driver.
    """

    data = load_test_data()

    finding = find_finding(
        data["forecast"],
        "country",
        "Germany",
    )

    assert (
        finding[
            "calculated_fact"
        ][
            "primary_driver"
        ]
        == "volume_mix"
    )

    assert (
        finding[
            "grounding"
        ][
            "usable_evidence_ids"
        ]
        == []
    )

    assert (
        finding[
            "grounding"
        ][
            "explanation_status"
        ]
        == "insufficient_evidence"
    )

    candidate = candidate_by_id(
        finding,
        "NOTE-014",
    )

    assert candidate is not None

    assert (
        candidate[
            "support_status"
        ]
        == "related_not_explanatory"
    )


def test_edgehub_pro_volume_mix_not_explained_by_note_014():
    """
    NOTE-014 concerns EdgeHub Pro but documents unit cost.

    It must NOT explain EdgeHub Pro's calculated volume/mix
    driver.
    """

    data = load_test_data()

    finding = find_finding(
        data["forecast"],
        "product",
        "EdgeHub Pro",
    )

    assert (
        finding[
            "calculated_fact"
        ][
            "primary_driver"
        ]
        == "volume_mix"
    )

    assert (
        finding[
            "grounding"
        ][
            "usable_evidence_ids"
        ]
        == []
    )

    candidate = candidate_by_id(
        finding,
        "NOTE-014",
    )

    assert candidate is not None

    assert (
        candidate[
            "support_status"
        ]
        == "related_not_explanatory"
    )


# ============================================================
# ENTITY RESOLUTION TESTS
# ============================================================

def test_sensor_x_does_not_receive_sensor_x_pro_note():
    """
    NOTE-016 is about Sensor X Pro.

    It must never be attached to Sensor X.
    """

    data = load_test_data()

    finding = find_finding(
        data["forecast"],
        "product",
        "Sensor X",
    )

    usable = finding[
        "grounding"
    ][
        "usable_evidence_ids"
    ]

    assert "NOTE-015" in usable

    assert "NOTE-016" not in usable


def test_sensor_x_pro_note_does_not_create_sensor_x_entity():
    """
    Longest-match entity resolution must prevent
    'Sensor X Pro' from also becoming 'Sensor X'.
    """

    data = load_test_data()

    note = find_evidence(
        data["evidence"],
        "NOTE-016",
    )

    products = note[
        "entities"
    ][
        "products"
    ]

    assert "Sensor X Pro" in products

    assert "Sensor X" not in products


# ============================================================
# POSITIVE GROUNDING TESTS
# ============================================================

def test_uk_volume_mix_uses_note_015():
    """
    UK shipment-delay evidence should support the calculated
    volume/mix finding.
    """

    data = load_test_data()

    finding = find_finding(
        data["forecast"],
        "country",
        "United Kingdom",
    )

    assert (
        "NOTE-015"
        in finding[
            "grounding"
        ][
            "usable_evidence_ids"
        ]
    )

    assert (
        finding[
            "grounding"
        ][
            "explanation_status"
        ]
        == "evidence_available"
    )


def test_norway_volume_mix_uses_note_016():
    """
    Norway shipment-acceleration evidence should support the
    calculated volume/mix finding.
    """

    data = load_test_data()

    finding = find_finding(
        data["forecast"],
        "country",
        "Norway",
    )

    assert (
        "NOTE-016"
        in finding[
            "grounding"
        ][
            "usable_evidence_ids"
        ]
    )


def test_contractors_use_note_017():
    """
    Contractor-management evidence should support the
    Contractors OPEX finding.
    """

    data = load_test_data()

    finding = find_finding(
        data["forecast"],
        "opex_account",
        "Contractors",
    )

    assert (
        "NOTE-017"
        in finding[
            "grounding"
        ][
            "usable_evidence_ids"
        ]
    )


def test_product_rnd_uses_note_017():
    """
    Product & R&D evidence should be available at cost-centre
    level.
    """

    data = load_test_data()

    finding = find_finding(
        data["forecast"],
        "cost_centre",
        "Product & R&D",
    )

    assert (
        "NOTE-017"
        in finding[
            "grounding"
        ][
            "usable_evidence_ids"
        ]
    )


# ============================================================
# INSUFFICIENT-EVIDENCE TESTS
# ============================================================

def test_marketing_must_remain_unexplained():
    """
    No analyst-visible management note supports the
    Marketing OPEX variance.

    The engine must refuse to invent a cause.
    """

    data = load_test_data()

    finding = find_finding(
        data["forecast"],
        "opex_account",
        "Marketing",
    )

    assert (
        finding[
            "grounding"
        ][
            "usable_evidence_ids"
        ]
        == []
    )

    assert (
        finding[
            "grounding"
        ][
            "explanation_status"
        ]
        == "insufficient_evidence"
    )


def test_ytd_germany_list_price_remains_unexplained():
    """
    NOTE-014 documents unit-cost pressure, not list-price
    pressure.

    Therefore YTD Germany must remain unexplained.
    """

    data = load_test_data()

    finding = find_finding(
        data["ytd"],
        "country",
        "Germany",
    )

    assert (
        finding[
            "calculated_fact"
        ][
            "primary_driver"
        ]
        == "list_price"
    )

    assert (
        finding[
            "grounding"
        ][
            "usable_evidence_ids"
        ]
        == []
    )


# ============================================================
# SECURITY TESTS
# ============================================================

def test_ground_truth_access_is_disabled():
    """
    Grounded outputs must explicitly state that hidden ground
    truth was not used.
    """

    data = load_test_data()

    assert (
        data["forecast"][
            "metadata"
        ][
            "ground_truth_access"
        ]
        is False
    )

    assert (
        data["ytd"][
            "metadata"
        ][
            "ground_truth_access"
        ]
        is False
    )


def test_no_ground_truth_paths_in_evidence_index():
    """
    Analyst-visible evidence must never originate from the
    hidden ground-truth directory.
    """

    data = load_test_data()

    for record in data[
        "evidence"
    ]:

        source_file = (
            record[
                "source_file"
            ].lower()
        )

        assert (
            "ground_truth"
            not in source_file
        )