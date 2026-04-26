#!/usr/bin/env python3
"""Bulk-translate Markdown content using project dictionary mappings."""

from __future__ import annotations

from pathlib import Path

from convert_to_english import translate_markdown_file

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = [
    ROOT / "modules",
    ROOT / "trends-extras",
    ROOT / "templates",
    ROOT / "docs",
    ROOT / ".github",
]
TARGET_FILES = [
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "ENGLISH_CONVERSION_AUDIT.md",
]


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for directory in TARGET_DIRS:
        if directory.exists():
            files.extend(sorted(directory.rglob("*.md")))
    for file_path in TARGET_FILES:
        if file_path.exists():
            files.append(file_path)

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for file_path in files:
        resolved = file_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(file_path)
    return unique


def main() -> int:
    files = iter_markdown_files()
    changed = 0

    for file_path in files:
        try:
            original = file_path.read_text(encoding="utf-8")
            translated = translate_markdown_file(file_path)
            if translated != original:
                file_path.write_text(translated, encoding="utf-8")
                changed += 1
                print(f"CHANGED {file_path.relative_to(ROOT)}")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            print(f"ERROR {file_path.relative_to(ROOT)}: {exc}")

    print(f"TOTAL_FILES {len(files)}")
    print(f"TOTAL_CHANGED {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
