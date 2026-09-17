from dataclasses import dataclass
from typing import Literal


# ============================================================
# BUSINESS EVENT MODEL
# ============================================================

DriverType = Literal[
    "unit_cost",
    "discount_rate",
    "units",
]

OperationType = Literal[
    "multiply",
    "add",
]


@dataclass(frozen=True)
class BusinessEvent:
    event_id: str
    date: str

    country: str

    product: str | None
    segment: str | None

    driver: DriverType
    operation: OperationType
    magnitude: float

    description: str

    evidence_available: bool
    evidence_id: str | None


# ============================================================
# CONTROLLED BUSINESS EVENTS
# ============================================================
#
# These events are deliberately injected into the synthetic
# financial data.
#
# They are NOT random noise.
#
# They create a known "truth" against which the future AI
# analyst can be evaluated.
# ============================================================

BUSINESS_EVENTS = [

    # --------------------------------------------------------
    # EVT-001
    #
    # Documented procurement issue.
    # The AI should be able to retrieve NOTE-014 and connect
    # the margin deterioration to the supplier surcharge.
    # --------------------------------------------------------

    BusinessEvent(
        event_id="EVT-001",
        date="2026-08-01",
        country="Germany",
        product="EdgeHub Pro",
        segment=None,
        driver="unit_cost",
        operation="multiply",
        magnitude=1.08,
        description=(
            "Temporary 8% supplier surcharge increased "
            "EdgeHub Pro unit cost in Germany during August."
        ),
        evidence_available=True,
        evidence_id="NOTE-014",
    ),

    # --------------------------------------------------------
    # EVT-002
    #
    # Undocumented commercial deterioration.
    #
    # The financial effect is real, but no management note
    # explains the reason.
    #
    # The future AI system must NOT invent a cause.
    # --------------------------------------------------------

    BusinessEvent(
        event_id="EVT-002",
        date="2026-08-01",
        country="Germany",
        product=None,
        segment="Distributor",
        driver="discount_rate",
        operation="add",
        magnitude=0.03,
        description=(
            "Distributor discount rates in Germany increased "
            "by 3 percentage points during August."
        ),
        evidence_available=False,
        evidence_id=None,
    ),

    # --------------------------------------------------------
    # EVT-003
    #
    # Documented negative volume event.
    # --------------------------------------------------------

    BusinessEvent(
        event_id="EVT-003",
        date="2026-08-01",
        country="United Kingdom",
        product="Sensor X",
        segment="Distributor",
        driver="units",
        operation="multiply",
        magnitude=0.78,
        description=(
            "UK Sensor X distributor volume fell by 22% "
            "during August due to a delayed customer rollout."
        ),
        evidence_available=True,
        evidence_id="NOTE-015",
    ),

    # --------------------------------------------------------
    # EVT-004
    #
    # Documented positive volume event.
    # --------------------------------------------------------

    BusinessEvent(
        event_id="EVT-004",
        date="2026-08-01",
        country="Norway",
        product="Sensor X Pro",
        segment="Enterprise",
        driver="units",
        operation="multiply",
        magnitude=1.25,
        description=(
            "Norway Sensor X Pro enterprise volume increased "
            "by 25% in August following an accelerated "
            "customer rollout."
        ),
        evidence_available=True,
        evidence_id="NOTE-016",
    ),
]


# ============================================================
# MANAGEMENT EVIDENCE
# ============================================================
#
# Only documented business explanations appear here.
#
# Notice that EVT-002 deliberately has NO management note.
# ============================================================

MANAGEMENT_NOTES = {

    "NOTE-014": {
        "title": (
            "Germany EdgeHub Pro procurement surcharge"
        ),
        "text": (
            "A temporary supplier surcharge remained in "
            "effect for EdgeHub Pro units sold in Germany "
            "during August. Procurement used secondary "
            "sourcing capacity to protect customer delivery "
            "commitments. The surcharge increased unit "
            "purchase cost by approximately 8%. Management "
            "expects sourcing conditions to normalize from "
            "September."
        ),
    },

    "NOTE-015": {
        "title": (
            "UK Sensor X distributor shipment delay"
        ),
        "text": (
            "Sensor X shipments through the UK distributor "
            "channel were lower than expected in August. "
            "A major end-customer implementation was moved "
            "from August into September, delaying part of "
            "the planned distributor demand."
        ),
    },

    "NOTE-016": {
        "title": (
            "Norway Sensor X Pro shipment acceleration"
        ),
        "text": (
            "Enterprise shipments of Sensor X Pro in Norway "
            "were accelerated into August after customer "
            "acceptance occurred earlier than planned. "
            "Management expects part of the increase to "
            "represent demand pulled forward from September."
        ),
    },
}


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_event_configuration() -> None:
    """
    Validate the controlled-event configuration.
    """

    event_ids = [
        event.event_id
        for event in BUSINESS_EVENTS
    ]

    if len(event_ids) != len(set(event_ids)):
        raise ValueError(
            "Duplicate event IDs detected."
        )

    for event in BUSINESS_EVENTS:

        if event.driver == "discount_rate":

            if event.operation != "add":
                raise ValueError(
                    f"{event.event_id}: discount events "
                    "must use operation='add'."
                )

        if event.driver in {
            "unit_cost",
            "units",
        }:

            if event.operation != "multiply":
                raise ValueError(
                    f"{event.event_id}: {event.driver} "
                    "events must use operation='multiply'."
                )

        if event.evidence_available:

            if event.evidence_id is None:
                raise ValueError(
                    f"{event.event_id}: evidence is marked "
                    "available but no evidence_id exists."
                )

            if event.evidence_id not in MANAGEMENT_NOTES:
                raise ValueError(
                    f"{event.event_id}: evidence_id "
                    f"{event.evidence_id} does not exist "
                    "in MANAGEMENT_NOTES."
                )

        else:

            if event.evidence_id is not None:
                raise ValueError(
                    f"{event.event_id}: undocumented event "
                    "should not have an evidence_id."
                )


validate_event_configuration()