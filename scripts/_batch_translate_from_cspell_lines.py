#!/usr/bin/env python3
"""Translate specific lines from cspell report for a selected file batch."""

from __future__ import annotations

import argparse
import re
import signal
import time
from pathlib import Path

from deep_translator import GoogleTranslator

ROOT = Path(__file__).resolve().parents[1]
LINE_RE = re.compile(r"^(?P<file>.+?):(?P<line>\d+):\d+ - Unknown word \([^)]+\)(?: fix: \([^)]+\))?$")


class TimeoutErrorTranslate(Exception):
    pass


def on_timeout(signum, frame):
    raise TimeoutErrorTranslate("Translation timeout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--files-list", required=True)
    parser.add_argument("--timeout", type=int, default=4)
    return parser.parse_args()


def load_target_files(files_list: Path) -> set[Path]:
    targets: set[Path] = set()
    for raw in files_list.read_text(encoding="utf-8", errors="ignore").splitlines():
        rel = raw.strip()
        if not rel:
            continue
        path = ROOT / rel
        if path.exists():
            targets.add(path.resolve())
    return targets


def load_report_lines(report: Path, targets: set[Path]) -> dict[Path, set[int]]:
    file_lines: dict[Path, set[int]] = {}
    for raw in report.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LINE_RE.match(raw.strip())
        if not match:
            continue
        rel_file = match.group("file")
        path = (ROOT / rel_file).resolve()
        if path not in targets:
            continue
        line_number = int(match.group("line"))
        file_lines.setdefault(path, set()).add(line_number)
    return file_lines


def split_prefix(line: str) -> tuple[str, str]:
    prefix_match = re.match(r"^(\s*(?:[-*+]\s+|\d+\.\s+|#+\s+|>\s+)*)", line)
    if not prefix_match:
        return "", line
    prefix = prefix_match.group(1)
    return prefix, line[len(prefix):]


def is_translatable(line: str, in_code: bool) -> bool:
    if in_code:
        return False
    if not line.strip():
        return False
    if line.strip().startswith("|") and line.count("|") >= 2:
        return True
    if re.match(r"^\s*([`~]{3,}|[-*]{3,}|===+|___+)", line):
        return False
    return True


def translate_files(file_lines: dict[Path, set[int]], timeout_s: int) -> tuple[int, int]:
    translator = GoogleTranslator(source="es", target="en")
    cache: dict[str, str] = {}
    changed_files = 0
    changed_lines = 0

    signal.signal(signal.SIGALRM, on_timeout)

    for path, target_lines in sorted(file_lines.items()):
        original = path.read_text(encoding="utf-8")
        lines = original.splitlines()
        new_lines = lines[:]

        in_code = False
        file_changed = False

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code = not in_code

            if idx not in target_lines:
                continue
            if not is_translatable(line, in_code):
                continue

            prefix, content = split_prefix(line)
            if not content.strip():
                continue

            key = content
            if key not in cache:
                try:
                    signal.alarm(timeout_s)
                    translated = translator.translate(content)
                    signal.alarm(0)
                except Exception:
                    signal.alarm(0)
                    translated = content
                cache[key] = translated if translated else content
                time.sleep(0.02)

            translated_line = prefix + cache[key]
            if translated_line != line:
                new_lines[idx - 1] = translated_line
                changed_lines += 1
                file_changed = True

        if file_changed:
            updated = "\n".join(new_lines)
            if original.endswith("\n"):
                updated += "\n"
            path.write_text(updated, encoding="utf-8")
            changed_files += 1
            print(f"CHANGED {path.relative_to(ROOT)}")

    print(f"CHANGED_FILES {changed_files}")
    print(f"CHANGED_LINES {changed_lines}")
    print(f"CACHE_SIZE {len(cache)}")
    return changed_files, changed_lines


def main() -> int:
    args = parse_args()
    report = Path(args.report)
    files_list = Path(args.files_list)

    targets = load_target_files(files_list)
    file_lines = load_report_lines(report, targets)

    print(f"TARGET_FILES {len(targets)}")
    print(f"FILES_WITH_ISSUES {len(file_lines)}")

    translate_files(file_lines, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
