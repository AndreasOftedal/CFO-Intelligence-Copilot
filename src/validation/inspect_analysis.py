import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


# ============================================================
# LOAD
# ============================================================

def load_json(path: Path) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"Analysis file not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# HELPERS
# ============================================================

def format_nok_m(value: float) -> str:

    return (
        f"NOK "
        f"{value / 1_000_000:,.1f}m"
    )


def print_commercial_ranking(
    title: str,
    records: list[dict],
    label_field: str,
) -> None:

    print(f"\n{title}")
    print("=" * 105)

    for record in records:

        marker = (
            "MATERIAL"
            if record["material"]
            else ""
        )

        print(
            f"{record[label_field]:<24} | "
            f"Revenue Var "
            f"{format_nok_m(record['revenue_variance_nok']):>14} | "
            f"GP Var "
            f"{format_nok_m(record['gross_profit_variance_nok']):>14} | "
            f"Main driver "
            f"{record['largest_gp_driver']:<12} | "
            f"{format_nok_m(record['largest_gp_driver_impact_nok']):>14} | "
            f"{marker}"
        )


def print_opex_ranking(
    title: str,
    records: list[dict],
    label_field: str,
) -> None:

    print(f"\n{title}")
    print("=" * 90)

    for record in records:

        marker = (
            "MATERIAL"
            if record["material"]
            else ""
        )

        print(
            f"{record[label_field]:<24} | "
            f"OPEX Var "
            f"{format_nok_m(record['opex_variance_nok']):>14} | "
            f"EBITDA impact "
            f"{format_nok_m(record['ebitda_impact_nok']):>14} | "
            f"{marker}"
        )


def print_analysis(
    analysis: dict,
) -> None:

    name = (
        analysis[
            "metadata"
        ][
            "comparison"
        ]
    )

    print(
        "\n\n"
        + "#" * 110
    )

    print(name)

    print(
        "#" * 110
    )

    rankings = analysis[
        "rankings"
    ]

    print_commercial_ranking(
        "Top Countries by Gross Profit Variance",
        rankings["countries"],
        "country",
    )

    print_commercial_ranking(
        "Top Products by Gross Profit Variance",
        rankings["products"],
        "product",
    )

    print_opex_ranking(
        "Top OPEX Accounts",
        rankings["opex_accounts"],
        "account",
    )

    print_opex_ranking(
        "Top Cost Centres",
        rankings["cost_centres"],
        "cost_centre",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    forecast_analysis = load_json(
        FORECAST_ANALYSIS_PATH
    )

    ytd_analysis = load_json(
        YTD_ANALYSIS_PATH
    )

    print(
        "\nNorthstar Systems AS"
    )

    print(
        "Finance Analysis Engine — Ranking Sanity Check"
    )

    print_analysis(
        forecast_analysis
    )

    print_analysis(
        ytd_analysis
    )


if __name__ == "__main__":
    main()