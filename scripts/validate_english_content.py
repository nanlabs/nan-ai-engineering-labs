#!/usr/bin/env python3
"""Validate English-only text in governance files.

This checker is intentionally strict for AGENTS and skill/agent customization files,
where repository policies are defined.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ACCENTED_PATTERN = re.compile(r"[áéíóúÁÉÍÓÚñÑ¿¡]")
SPANISH_MARKERS = [
    "ejemplo",
    "ejemplos",
    "objetivo",
    "proximo",
    "proximos",
    "como usar",
    "bloqueos",
    "evaluacion",
    "rubrica",
    "practicas",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate English-only content in text files")
    parser.add_argument("files", nargs="*", help="Files to validate")
    return parser.parse_args()


def find_violations(path: Path) -> list[str]:
    """Return human-readable violation messages for a file."""
    violations: list[str] = []
    content = path.read_text(encoding="utf-8")

    for index, line in enumerate(content.splitlines(), start=1):
        if ACCENTED_PATTERN.search(line):
            violations.append(f"{path}:{index}: contains accented/Spanish punctuation")
            continue

        lowered = line.lower()
        for marker in SPANISH_MARKERS:
            if marker in lowered:
                violations.append(f"{path}:{index}: contains Spanish marker '{marker}'")
                break

    return violations


def main() -> int:
    """Run validator and return POSIX exit code."""
    args = parse_args()
    files = [Path(file_path) for file_path in args.files if Path(file_path).is_file()]

    all_violations: list[str] = []
    for file_path in files:
        if file_path.suffix.lower() not in {".md", ".txt", ".py", ".yml", ".yaml", ".toml"}:
            continue
        all_violations.extend(find_violations(file_path))

    if all_violations:
        print("English content validation failed:")
        for violation in all_violations:
            print(f"- {violation}")
        return 1

    print("English content validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
