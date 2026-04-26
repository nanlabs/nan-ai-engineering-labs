#!/usr/bin/env python3
"""Translate unresolved unknown words from cspell report and replace in files."""

from __future__ import annotations

import argparse
import re
import signal
import unicodedata
from pathlib import Path

from deep_translator import GoogleTranslator

ROOT = Path(__file__).resolve().parents[1]
LINE_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+) - Unknown word \((?P<word>[^)]+)\)"
    r"(?: fix: \((?P<fix>[^)]+)\))?$"
)
SPANISH_CHARS = re.compile(r"[áéíóúñÁÉÍÓÚÑ¿¡]")


class Timeout(Exception):
    pass


def on_timeout(signum, frame):
    raise Timeout("translation timeout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--timeout", type=int, default=4)
    return parser.parse_args()


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def load_entries(report: Path) -> tuple[set[Path], dict[str, str | None]]:
    files: set[Path] = set()
    words: dict[str, str | None] = {}

    for raw in report.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LINE_RE.match(raw.strip())
        if not match:
            continue
        fp = ROOT / match.group("file")
        if fp.exists() and fp.suffix.lower() in {".md", ".py"}:
            files.add(fp)

        word = match.group("word")
        fix = match.group("fix")
        words[word] = fix

    return files, words


def choose_candidates(words: dict[str, str | None]) -> list[str]:
    candidates: list[str] = []
    for word, fix in words.items():
        if fix:
            continue
        # Focus on likely Spanish words, avoid technical code tokens.
        if not SPANISH_CHARS.search(word):
            continue
        if any(ch.isdigit() for ch in word):
            continue
        if len(word) <= 2:
            continue
        candidates.append(word)
    return sorted(set(candidates), key=len, reverse=True)


def translate_candidates(candidates: list[str], timeout_s: int) -> dict[str, str]:
    translator = GoogleTranslator(source="es", target="en")
    mapping: dict[str, str] = {}

    signal.signal(signal.SIGALRM, on_timeout)

    for word in candidates:
        try:
            signal.alarm(timeout_s)
            translated = translator.translate(word)
            signal.alarm(0)
        except Exception:
            signal.alarm(0)
            translated = None

        if not translated:
            translated = strip_accents(word)

        translated = translated.strip()
        if not translated:
            continue
        if translated.lower() == word.lower():
            continue

        mapping[word] = translated

    return mapping


def capitalize_like(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper() and source[1:].islower():
        return target[:1].upper() + target[1:]
    return target


def replace_word(text: str, source: str, target: str) -> str:
    pattern = re.compile(rf"(?<![\w-]){re.escape(source)}(?![\w-])")

    def repl(match: re.Match[str]) -> str:
        return capitalize_like(match.group(0), target)

    return pattern.sub(repl, text)


def apply_mapping(files: set[Path], mapping: dict[str, str]) -> tuple[int, int]:
    changed_files = 0
    changed_tokens = 0

    ordered = sorted(mapping.items(), key=lambda pair: len(pair[0]), reverse=True)

    for file_path in sorted(files):
        try:
            original = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        updated = original
        for src, dst in ordered:
            newer = replace_word(updated, src, dst)
            if newer != updated:
                changed_tokens += 1
                updated = newer

        if updated != original:
            file_path.write_text(updated, encoding="utf-8")
            changed_files += 1
            print(f"CHANGED {file_path.relative_to(ROOT)}")

    return changed_files, changed_tokens


def main() -> int:
    args = parse_args()
    report = Path(args.report)
    if not report.exists():
        print("REPORT_NOT_FOUND")
        return 1

    files, words = load_entries(report)
    candidates = choose_candidates(words)
    mapping = translate_candidates(candidates, args.timeout)
    changed_files, changed_tokens = apply_mapping(files, mapping)

    print(f"FILES_IN_REPORT {len(files)}")
    print(f"UNKNOWN_WORDS {len(words)}")
    print(f"TRANSLATION_CANDIDATES {len(candidates)}")
    print(f"MAPPING_SIZE {len(mapping)}")
    print(f"CHANGED_FILES {changed_files}")
    print(f"CHANGED_TOKENS {changed_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
