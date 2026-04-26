#!/usr/bin/env python3
"""Translate remaining Spanish Markdown lines to English using deep_translator.

Scope: modules/, trends-extras/, templates/, docs/, .github/, README.md
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from deep_translator import GoogleTranslator

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = [
    ROOT / "modules",
    ROOT / "trends-extras",
    ROOT / "templates",
    ROOT / "docs",
    ROOT / ".github",
]
TARGET_FILES = [ROOT / "README.md"]

SPANISH_HINT = re.compile(
    r"[áéíóúñÁÉÍÓÚÑ¿¡]|\b(el|la|los|las|de|del|y|en|para|con|por|que|una|un|es|son)\b",
    re.IGNORECASE,
)
NON_TEXT_LINE = re.compile(
    r"^\s*(?:[`~]{3,}|[-*]{3,}|===+|___+|>!|!\[|https?://|\[!.*\])"
)


def iter_md_files() -> list[Path]:
    files: list[Path] = []
    for directory in TARGET_DIRS:
        if directory.exists():
            files.extend(sorted(directory.rglob("*.md")))
    for file_path in TARGET_FILES:
        if file_path.exists():
            files.append(file_path)

    seen: set[Path] = set()
    unique: list[Path] = []
    for file_path in files:
        resolved = file_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(file_path)
    return unique


def should_translate(line: str, in_code: bool) -> bool:
    if in_code:
        return False
    if not line.strip():
        return False
    if NON_TEXT_LINE.search(line):
        return False
    if line.strip().startswith("|") and line.count("|") >= 2:
        return True
    return bool(SPANISH_HINT.search(line))


def split_prefix(line: str) -> tuple[str, str]:
    # Preserve markdown markers/indentation during translation.
    prefix_match = re.match(r"^(\s*(?:[-*+]\s+|\d+\.\s+|#+\s+|>\s+)*)", line)
    if not prefix_match:
        return "", line
    prefix = prefix_match.group(1)
    return prefix, line[len(prefix):]


def main() -> int:
    translator = GoogleTranslator(source="es", target="en")
    cache: dict[str, str] = {}

    changed_files = 0
    changed_lines = 0

    for file_path in iter_md_files():
        try:
            original = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        lines = original.splitlines()
        new_lines: list[str] = []
        in_code = False
        file_changed = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code = not in_code
                new_lines.append(line)
                continue

            if not should_translate(line, in_code):
                new_lines.append(line)
                continue

            prefix, content = split_prefix(line)
            if not content.strip():
                new_lines.append(line)
                continue

            key = content
            if key not in cache:
                try:
                    translated = translator.translate(content)
                    cache[key] = translated if translated else content
                    time.sleep(0.03)
                except Exception:
                    cache[key] = content

            translated_line = prefix + cache[key]
            if translated_line != line:
                file_changed = True
                changed_lines += 1
            new_lines.append(translated_line)

        if file_changed:
            new_text = "\n".join(new_lines)
            if original.endswith("\n"):
                new_text += "\n"
            file_path.write_text(new_text, encoding="utf-8")
            changed_files += 1
            print(f"CHANGED {file_path.relative_to(ROOT)}")

    print(f"TOTAL_CHANGED_FILES {changed_files}")
    print(f"TOTAL_CHANGED_LINES {changed_lines}")
    print(f"CACHE_SIZE {len(cache)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
