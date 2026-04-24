"""Create lightweight ASCII charts for business-facing summaries.

Run:
    python modules/02-data-collection-cleaning-visualization/examples/ex_04_business_visualization_ascii.py
"""

from __future__ import annotations


def ascii_bar(value: int, max_value: int, width: int = 20) -> str:
    """Build a fixed-width ASCII bar."""
    if max_value == 0:
        return "-" * width
    filled = int(round((value / max_value) * width))
    return "#" * filled + "." * (width - filled)


def main() -> None:
    """Print sorted category summary with proportional bars."""
    sales_by_region = {
        "North": 120,
        "West": 80,
        "South": 150,
        "East": 95,
    }

    sorted_items = sorted(sales_by_region.items(), key=lambda item: item[1], reverse=True)
    max_value = max(sales_by_region.values())

    print("Business visualization (ASCII)")
    for region, value in sorted_items:
        print(f"{region:>5} | {ascii_bar(value, max_value)} | {value}")


if __name__ == "__main__":
    main()
