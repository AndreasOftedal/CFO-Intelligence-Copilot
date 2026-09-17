from dataclasses import dataclass


# ============================================================
# OPEX BUSINESS EVENT MODEL
# ============================================================

@dataclass(frozen=True)
class OpexEvent:
    event_id: str
    date: str

    cost_centre: str
    account: str

    amount_change: float

    description: str

    evidence_available: bool
    evidence_id: str | None


# ============================================================
# CONTROLLED OPEX EVENTS
# ============================================================

OPEX_EVENTS = [

    # --------------------------------------------------------
    # OPEX-001
    #
    # Documented Product & R&D contractor overspend.
    # --------------------------------------------------------

    OpexEvent(
        event_id="OPEX-001",
        date="2026-08-01",
        cost_centre="Product & R&D",
        account="Contractors",
        amount_change=1_800_000,
        description=(
            "Product & R&D contractor expenditure increased "
            "by NOK 1.8m in August due to accelerated "
            "prototype and testing activity."
        ),
        evidence_available=True,
        evidence_id="NOTE-017",
    ),

    # --------------------------------------------------------
    # OPEX-002
    #
    # Real overspend, but deliberately undocumented.
    # --------------------------------------------------------

    OpexEvent(
        event_id="OPEX-002",
        date="2026-08-01",
        cost_centre="Sales",
        account="Marketing",
        amount_change=1_100_000,
        description=(
            "Sales marketing expenditure increased by "
            "NOK 1.1m in August."
        ),
        evidence_available=False,
        evidence_id=None,
    ),
]


# ============================================================
# MANAGEMENT EVIDENCE
# ============================================================

OPEX_MANAGEMENT_NOTES = {

    "NOTE-017": {
        "title": (
            "Product & R&D contractor acceleration"
        ),
        "text": (
            "External engineering and testing resources "
            "were increased during August to accelerate "
            "prototype validation for the next-generation "
            "control platform. The additional contractor "
            "spend was approximately NOK 1.8m above the "
            "normal monthly run-rate. Management expects "
            "the elevated activity to continue at a lower "
            "level through September before normalizing."
        ),
    },
}


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_opex_event_configuration() -> None:

    event_ids = [
        event.event_id
        for event in OPEX_EVENTS
    ]

    if len(event_ids) != len(set(event_ids)):

        raise ValueError(
            "Duplicate OPEX event IDs detected."
        )

    for event in OPEX_EVENTS:

        if event.amount_change == 0:

            raise ValueError(
                f"{event.event_id}: "
                "event amount cannot be zero."
            )

        if event.evidence_available:

            if event.evidence_id is None:

                raise ValueError(
                    f"{event.event_id}: "
                    "evidence marked available but "
                    "evidence_id is missing."
                )

            if (
                event.evidence_id
                not in OPEX_MANAGEMENT_NOTES
            ):

                raise ValueError(
                    f"{event.event_id}: "
                    "evidence note does not exist."
                )

        else:

            if event.evidence_id is not None:

                raise ValueError(
                    f"{event.event_id}: "
                    "undocumented event should not "
                    "have evidence_id."
                )


validate_opex_event_configuration()