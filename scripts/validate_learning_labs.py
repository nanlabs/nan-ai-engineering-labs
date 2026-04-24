#!/usr/bin/env python3
"""Validate learning unit structure and README heading contracts.

Core modules are strict by default for structure checks.
Extras are warning-only by default until migration is complete.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "modules"
EXTRAS_ROOT = ROOT / "trends-extras"

CORE_REQUIRED_PATHS = (
    "README.md",
    "STATUS.md",
    "theory/README.md",
    "examples/README.md",
    "practices/README.md",
    "mini-project/README.md",
    "evaluation/README.md",
    "notes/README.md",
)

EXTRAS_TARGET_PATHS = CORE_REQUIRED_PATHS

REQUIRED_HEADINGS = {
    "README.md": [
        "module objective",
        "what you will achieve",
        "internal structure",
        "level path (l1-l4)",
        "recommended plan (by progress, not by weeks)",
        "module completion criteria",
    ],
    "theory/README.md": [
        "why this module matters",
        "final comprehension checklist",
    ],
    "examples/README.md": [
        "available examples",
        "how to use these examples",
        "next steps",
    ],
    "practices/README.md": [
        "practices",
        "approval criteria",
    ],
    "mini-project/README.md": [
        "project",
        "objective",
        "scope",
        "deliverables",
        "acceptance criteria",
    ],
    "evaluation/README.md": [
        "weighting",
        "rubric",
        "final evaluation",
        "approval criteria",
    ],
    "notes/README.md": [
        "how to use this folder",
    ],
    "STATUS.md": [
        "current progress",
        "checklist",
        "blockers",
        "next steps",
    ],
}


@dataclass
class UnitResult:
    """
    Result of validating a learning unit against
    structure and heading contracts.
    """

    unit: str
    kind: str
    missing_paths: list[str] = field(default_factory=list)
    heading_gaps: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        """Check if the unit has no missing paths or heading gaps."""
        return not self.missing_paths and not self.heading_gaps


def normalize(text: str) -> str:
    """
    Normalize text for consistent comparison.

    This function performs the following normalization steps:
    1. Strips leading and trailing whitespace.
    2. Converts text to lowercase.
    3. Removes diacritical marks (accents).
    4. Removes non-alphanumeric characters except for parentheses,
       hyphens, and spaces.
    5. Collapses multiple spaces into a single space.

    Args:
        text: The text to normalize.

    Returns:
        The normalized text.
    """
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9()\-\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def heading_set(markdown_text: str) -> set[str]:
    """Extract normalized headings from markdown text.

    Args:
        markdown_text: The markdown text to extract headings from.

    Returns:
        A set of normalized headings.
    """
    headings: set[str] = set()
    for line in markdown_text.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                headings.add(normalize(heading))
    return headings


def list_units(root: Path) -> list[Path]:
    """
    List all unit directories within the given root directory.

    Args:
        root: Path to the root directory containing unit directories.

    Returns:
        List of Paths representing the unit directories.
    """
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def validate_unit(
    unit_dir: Path, *, kind: str, strict_headings: bool, target_paths: tuple[str, ...]
) -> UnitResult:
    """
    Validate a single learning unit.

    Args:
        unit_dir: Path to the unit directory.
        kind: Type of the unit ("core" or "extra").
        strict_headings: Whether to enforce heading checks.
        target_paths: Paths that must exist within the unit.

    Returns:
        UnitResult: The result of the validation.
    """
    result = UnitResult(unit=unit_dir.name, kind=kind)

    for rel in target_paths:
        if not (unit_dir / rel).exists():
            result.missing_paths.append(rel)

    if strict_headings:
        for rel, required in REQUIRED_HEADINGS.items():
            file_path = unit_dir / rel
            if not file_path.exists():
                continue
            headings = heading_set(file_path.read_text(encoding="utf-8"))
            missing = [h for h in required if normalize(h) not in headings]
            if missing:
                result.heading_gaps[rel] = missing

    return result


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the validation script.
    """
    parser = argparse.ArgumentParser(
        description="Validate nan-ai-engineering-labs learning lab contracts"
    )
    parser.add_argument("--strict-core", action="store_true", help="Fail on core structure gaps")
    parser.add_argument(
        "--strict-extras", action="store_true", help="Fail on extras structure gaps"
    )
    parser.add_argument(
        "--strict-headings", action="store_true", help="Enable heading contract checks"
    )
    parser.add_argument("--report-json", action="store_true", help="Print JSON report")
    return parser.parse_args()


def main() -> int:
    """Main function to execute the validation logic."""
    args = parse_args()

    core_units = list_units(CORE_ROOT)
    extras_units = [p for p in list_units(EXTRAS_ROOT) if p.name not in {"init-path"}]

    core_results = [
        validate_unit(
            unit,
            kind="core",
            strict_headings=args.strict_headings,
            target_paths=CORE_REQUIRED_PATHS,
        )
        for unit in core_units
    ]
    extras_results = [
        validate_unit(
            unit,
            kind="extra",
            strict_headings=args.strict_headings,
            target_paths=EXTRAS_TARGET_PATHS,
        )
        for unit in extras_units
    ]

    core_missing = sum(1 for r in core_results if r.missing_paths)
    extra_missing = sum(1 for r in extras_results if r.missing_paths)
    core_heading_gaps = sum(1 for r in core_results if r.heading_gaps)
    extra_heading_gaps = sum(1 for r in extras_results if r.heading_gaps)

    lines = [
        "Learning lab contract validation",
        f"- Core units scanned: {len(core_results)}",
        f"- Extra units scanned: {len(extras_results)}",
        f"- Core units with missing paths: {core_missing}",
        f"- Extra units with missing paths: {extra_missing}",
        f"- Core units with heading gaps: {core_heading_gaps}",
        f"- Extra units with heading gaps: {extra_heading_gaps}",
    ]

    for label, results in (("core", core_results), ("extra", extras_results)):
        for result in results:
            if not result.is_clean:
                lines.append(f"\n[{label}] {result.unit}")
                if result.missing_paths:
                    lines.append("  missing paths:")
                    lines.extend(f"    - {path}" for path in result.missing_paths)
                if result.heading_gaps:
                    lines.append("  missing headings:")
                    for rel, missing in result.heading_gaps.items():
                        lines.append(f"    - {rel}")
                        lines.extend(f"      - {item}" for item in missing)

    print("\n".join(lines))

    if args.report_json:
        report = {
            "core": [r.__dict__ for r in core_results],
            "extras": [r.__dict__ for r in extras_results],
        }
        print(json.dumps(report, indent=2))

    core_failed = args.strict_core and any(not r.is_clean for r in core_results)
    extras_failed = args.strict_extras and any(not r.is_clean for r in extras_results)

    return 1 if core_failed or extras_failed else 0


if __name__ == "__main__":
    sys.exit(main())
