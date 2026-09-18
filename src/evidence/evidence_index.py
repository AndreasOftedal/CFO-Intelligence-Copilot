import hashlib
import json
import re
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "documents"
)

GROUND_TRUTH_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "ground_truth"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "evidence"
)

OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "evidence_index.json"
)


# ============================================================
# SECURITY / INFORMATION BOUNDARY
# ============================================================
#
# CRITICAL DESIGN PRINCIPLE
#
# The CFO Intelligence Copilot may only use information that
# would realistically be available to the analyst.
#
# Allowed:
#
#     data/documents/
#
# Forbidden:
#
#     data/ground_truth/
#
# Ground truth is reserved exclusively for evaluation.
# Giving the AI access to ground truth would create
# evaluation leakage and invalidate hallucination testing.
# ============================================================

ALLOWED_SOURCE_ROOT = DOCUMENT_DIRECTORY.resolve()

FORBIDDEN_PATH_PARTS = {
    "ground_truth",
}


# ============================================================
# NOTE FORMAT
# ============================================================
#
# Expected Markdown format:
#
# ### NOTE-014 — Supplier cost pressure
#
# Supporting evidence text...
#
# ============================================================

NOTE_HEADER_PATTERN = re.compile(
    r"^###\s+"
    r"(NOTE-\d+)"
    r"\s+[—-]\s+"
    r"(.+?)\s*$"
)


# ============================================================
# SECURITY VALIDATION
# ============================================================

def validate_source_path(
    path: Path,
) -> None:
    """
    Ensure the evidence engine only accesses files inside
    the approved analyst-visible document directory.
    """

    resolved_path = path.resolve()

    path_parts = {
        part.lower()
        for part in resolved_path.parts
    }

    forbidden_overlap = (
        path_parts
        & FORBIDDEN_PATH_PARTS
    )

    if forbidden_overlap:

        raise ValueError(
            "SECURITY ERROR: evidence engine attempted "
            f"to access forbidden path: {resolved_path}"
        )

    if not resolved_path.is_relative_to(
        ALLOWED_SOURCE_ROOT
    ):

        raise ValueError(
            "SECURITY ERROR: evidence source is outside "
            f"approved directory: {resolved_path}"
        )


# ============================================================
# HASHING
# ============================================================

def calculate_text_hash(
    text: str,
) -> str:
    """
    Create a deterministic SHA-256 hash of evidence text.

    This gives us a simple audit mechanism showing exactly
    which evidence content was indexed.
    """

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# DOCUMENT PARSER
# ============================================================

def parse_document(
    path: Path,
) -> list[dict]:
    """
    Parse NOTE sections from one Markdown document.
    """

    validate_source_path(
        path
    )

    text = path.read_text(
        encoding="utf-8"
    )

    lines = text.splitlines()

    records = []

    current_note = None
    current_body = []

    def finalize_current_note():
        """
        Convert the currently buffered note into
        one structured evidence record.
        """

        nonlocal current_note
        nonlocal current_body

        if current_note is None:
            return

        body = (
            "\n".join(
                current_body
            )
            .strip()
        )

        current_note[
            "text"
        ] = body

        current_note[
            "character_count"
        ] = len(body)

        current_note[
            "content_sha256"
        ] = calculate_text_hash(
            body
        )

        records.append(
            current_note
        )

        current_note = None
        current_body = []

    for line in lines:

        stripped = line.strip()

        header_match = (
            NOTE_HEADER_PATTERN.match(
                stripped
            )
        )

        if header_match:

            finalize_current_note()

            evidence_id = (
                header_match.group(1)
            )

            title = (
                header_match.group(2)
                .strip()
            )

            current_note = {
                "evidence_id":
                    evidence_id,
                "title":
                    title,
                "source_file":
                    path.name,
                "source_type":
                    "management_note",
                "text":
                    "",
                "character_count":
                    0,
                "content_sha256":
                    "",
            }

            continue

        if current_note is not None:

            current_body.append(
                line
            )

    finalize_current_note()

    return records


# ============================================================
# INDEX BUILDER
# ============================================================

def build_evidence_index() -> list[dict]:
    """
    Build the catalogue of all analyst-visible evidence.

    Only Markdown files in data/documents are indexed.
    """

    if not DOCUMENT_DIRECTORY.exists():

        raise FileNotFoundError(
            "Document directory does not exist: "
            f"{DOCUMENT_DIRECTORY}"
        )

    document_paths = sorted(
        DOCUMENT_DIRECTORY.glob(
            "*.md"
        )
    )

    if not document_paths:

        raise ValueError(
            "No analyst-visible management "
            "documents were found."
        )

    evidence_records = []

    for path in document_paths:

        validate_source_path(
            path
        )

        records = parse_document(
            path
        )

        evidence_records.extend(
            records
        )

    evidence_records.sort(
        key=lambda record:
            record["evidence_id"]
    )

    return evidence_records


# ============================================================
# INDEX VALIDATION
# ============================================================

def validate_evidence_index(
    records: list[dict],
) -> None:
    """
    Validate catalogue integrity and security boundaries.
    """

    if not records:

        raise ValueError(
            "Evidence index is empty."
        )

    evidence_ids = [
        record["evidence_id"]
        for record in records
    ]

    # --------------------------------------------------------
    # Duplicate IDs
    # --------------------------------------------------------

    if (
        len(evidence_ids)
        != len(set(evidence_ids))
    ):

        duplicates = sorted(
            {
                evidence_id
                for evidence_id
                in evidence_ids
                if evidence_ids.count(
                    evidence_id
                ) > 1
            }
        )

        raise ValueError(
            "Duplicate evidence IDs detected: "
            f"{duplicates}"
        )

    # --------------------------------------------------------
    # Record integrity
    # --------------------------------------------------------

    for record in records:

        required_fields = [
            "evidence_id",
            "title",
            "source_file",
            "source_type",
            "text",
            "character_count",
            "content_sha256",
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in record
        ]

        if missing_fields:

            raise ValueError(
                f"{record.get('evidence_id', 'UNKNOWN')}: "
                f"missing fields {missing_fields}"
            )

        if not record[
            "evidence_id"
        ]:

            raise ValueError(
                "Evidence record missing ID."
            )

        if not record[
            "title"
        ]:

            raise ValueError(
                f"{record['evidence_id']}: "
                "missing title."
            )

        if not record[
            "text"
        ]:

            raise ValueError(
                f"{record['evidence_id']}: "
                "evidence text is empty."
            )

        if (
            record["character_count"]
            != len(
                record["text"]
            )
        ):

            raise ValueError(
                f"{record['evidence_id']}: "
                "character-count mismatch."
            )

        expected_hash = (
            calculate_text_hash(
                record["text"]
            )
        )

        if (
            record["content_sha256"]
            != expected_hash
        ):

            raise ValueError(
                f"{record['evidence_id']}: "
                "content hash mismatch."
            )

        source_path = (
            DOCUMENT_DIRECTORY
            / record["source_file"]
        )

        validate_source_path(
            source_path
        )


# ============================================================
# SECURITY ASSERTION
# ============================================================

def assert_ground_truth_isolation() -> None:
    """
    Explicitly verify that the configured evidence source
    does not overlap with the hidden evaluation directory.
    """

    document_root = (
        DOCUMENT_DIRECTORY.resolve()
    )

    ground_truth_root = (
        GROUND_TRUTH_DIRECTORY.resolve()
    )

    if (
        document_root
        == ground_truth_root
    ):

        raise ValueError(
            "SECURITY ERROR: document and ground-truth "
            "directories cannot be identical."
        )

    if document_root.is_relative_to(
        ground_truth_root
    ):

        raise ValueError(
            "SECURITY ERROR: evidence directory is "
            "inside ground-truth directory."
        )

    if ground_truth_root.is_relative_to(
        document_root
    ):

        raise ValueError(
            "SECURITY ERROR: ground-truth directory is "
            "inside analyst-visible evidence directory."
        )


# ============================================================
# SAVE INDEX
# ============================================================

def save_evidence_index(
    records: list[dict],
) -> Path:
    """
    Save evidence catalogue as structured JSON.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "metadata": {
            "index_version":
                "1.0",
            "source_directory":
                "data/documents",
            "ground_truth_access":
                False,
            "evidence_count":
                len(records),
        },
        "evidence":
            records,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return OUTPUT_PATH


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_index(
    records: list[dict],
) -> None:

    print(
        "\nNorthstar Systems AS"
    )

    print(
        "CFO Intelligence Copilot — Evidence Index"
    )

    print(
        "=" * 110
    )

    for record in records:

        print(
            f"{record['evidence_id']:<10} | "
            f"{record['source_file']:<42} | "
            f"{record['title']}"
        )

    print(
        "-" * 110
    )

    print(
        f"Evidence records:     "
        f"{len(records)}"
    )

    print(
        "Evidence source:      "
        "data/documents/"
    )

    print(
        "Ground truth access:  "
        "BLOCKED"
    )

    print(
        "Index validation:     "
        "PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # Security check before any indexing.
    assert_ground_truth_isolation()

    records = (
        build_evidence_index()
    )

    validate_evidence_index(
        records
    )

    output_path = (
        save_evidence_index(
            records
        )
    )

    print_index(
        records
    )

    print(
        "\nEvidence index saved to:"
    )

    print(
        output_path
    )

    print(
        "\nEvidence Index Engine: PASSED"
    )


if __name__ == "__main__":
    main()