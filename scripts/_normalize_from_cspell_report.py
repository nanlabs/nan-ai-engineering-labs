#!/usr/bin/env python3
"""Apply bulk replacements from a cspell report.

Priority:
1) Project Spanish->English dictionary
2) cspell suggested fix when available
3) Manual fallback map for common Spanish terms
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path

from spanish_english_dictionary import SPANISH_ENGLISH_DICT

ROOT = Path(__file__).resolve().parents[1]

COMMON_FALLBACKS = {
    "producto": "product",
    "productos": "products",
    "usuario": "user",
    "usuarios": "users",
    "similitud": "similarity",
    "coseno": "cosine",
    "mediana": "median",
    "desviacion": "deviation",
    "estándar": "standard",
    "tiempo": "time",
    "respuesta": "response",
    "respuestas": "responses",
    "resumen": "summary",
    "rango": "range",
    "filtrar": "filter",
    "detectados": "detected",
    "normales": "normal",
    "básicas": "basic",
    "descriptivas": "descriptive",
    "límites": "limits",
    "limite": "limit",
    "vectores": "vectors",
    "normalizada": "normalized",
    "análisis": "analysis",
    "distribución": "distribution",
    "clásico": "classic",
    "código": "code",
    "cuándo": "when",
    "cuánto": "how much",
    "cómo": "how",
    "dónde": "where",
}

LINE_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+) - Unknown word \((?P<word>[^)]+)\)"
    r"(?: fix: \((?P<fix>[^)]+)\))?$"
)


def load_report_entries(report_path: Path) -> tuple[set[Path], dict[str, str]]:
    files: set[Path] = set()
    word_to_fix: dict[str, str] = {}

    if not report_path.exists():
        return files, word_to_fix

    for raw in report_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LINE_RE.match(raw.strip())
        if not match:
            continue
        file_path = ROOT / match.group("file")
        if file_path.exists():
            files.add(file_path)

        word = match.group("word")
        fix = match.group("fix")
        if fix and fix.strip():
            clean_fix = fix.strip()
            if clean_fix.lower() != word.lower() and len(clean_fix) > 1:
                word_to_fix[word] = clean_fix

    return files, word_to_fix


def capitalize_like(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper() and source[1:].islower():
        return target[:1].upper() + target[1:]
    return target


def replace_word_boundary(text: str, source: str, target: str) -> str:
    pattern = re.compile(rf"(?<![\w-]){re.escape(source)}(?![\w-])")

    def repl(match: re.Match[str]) -> str:
        return capitalize_like(match.group(0), target)

    return pattern.sub(repl, text)


def build_mapping(word_to_fix: dict[str, str]) -> dict[str, str]:
    mapping: dict[str, str] = {}

    # 1) Project dictionary first
    for source, target in SPANISH_ENGLISH_DICT.items():
        mapping[source] = target

    # 2) cspell suggested fixes
    for source, target in word_to_fix.items():
        if source not in mapping:
            mapping[source] = target

    # 3) common fallback terms
    for source, target in COMMON_FALLBACKS.items():
        if source not in mapping:
            mapping[source] = target

    return mapping


def apply_mapping(files: set[Path], mapping: dict[str, str]) -> tuple[int, int]:
    changed_files = 0
    changed_replacements = 0

    ordered = sorted(mapping.items(), key=lambda pair: len(pair[0]), reverse=True)

    for file_path in sorted(files):
        if file_path.suffix.lower() not in {".md", ".py", ".txt", ".yml", ".yaml", ".toml"}:
            continue
        try:
            original = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        updated = original
        before = updated

        for source, target in ordered:
            updated2 = replace_word_boundary(updated, source, target)
            if updated2 != updated:
                changed_replacements += 1
                updated = updated2

        if updated != before:
            file_path.write_text(updated, encoding="utf-8")
            changed_files += 1
            print(f"CHANGED {file_path.relative_to(ROOT)}")

    return changed_files, changed_replacements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="/tmp/nan_ai_cspell_after_terms.txt",
        help="Path to cspell report file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files, word_to_fix = load_report_entries(Path(args.report))
    mapping = build_mapping(word_to_fix)
    changed_files, changed_replacements = apply_mapping(files, mapping)

    print(f"FILES_IN_REPORT {len(files)}")
    print(f"WORD_FIXES_FROM_CSPELL {len(word_to_fix)}")
    print(f"MAPPING_SIZE {len(mapping)}")
    print(f"CHANGED_FILES {changed_files}")
    print(f"CHANGED_REPLACEMENTS {changed_replacements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
