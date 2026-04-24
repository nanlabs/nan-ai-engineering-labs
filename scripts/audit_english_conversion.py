#!/usr/bin/env python3
"""
Audit English conversion status across all nan-ai modules.
Generates a comprehensive report showing language status for each file.
"""

import re
from collections import defaultdict
from pathlib import Path

# Common Spanish words and phrases to detect
SPANISH_INDICATORS = [
    # Common Spanish words
    r"\b(el|la|los|las|de|del|en|con|por|para|es|son|está|están|tiene|tienen)\b",
    r"\b(módulo|módulos|leccion|lecciones|práctica|prácticas|teoría|teorías|evaluación)\b",
    r"\b(introducción|descripción|objetivo|objetivos|contenido|contenidos)\b",
    r"\b(ejercicio|ejercicios|ejemplo|ejemplos|problema|problemas)\b",
    r"\b(instrucciones|notas|sección|tema|temas|concepto)\b",
    # Spanish technical terms
    r"\b(aprendizaje|algoritmo|algoritmos|función|funciones)\b",
    r"\b(variable|variables|datos|tipo|tipos|estructura)\b",
]

ENGLISH_INDICATORS = [
    r"\b(the|a|an|is|are|have|has|for|with|and|or|in|on|at)\b",
    r"\b(module|modules|lesson|lessons|practice|practices|theory|theories|evaluation)\b",
    r"\b(introduction|description|objective|objectives|content|contents)\b",
    r"\b(exercise|exercises|example|examples|problem|problems)\b",
    r"\b(instructions|notes|section|topic|topics|concept)\b",
    r"\b(learning|algorithm|algorithms|function|functions)\b",
    r"\b(variable|variables|data|type|types|structure)\b",
]


def count_language_indicators(text: str) -> tuple[int, int]:
    """Count Spanish and English language indicators in text."""
    spanish_count = sum(
        len(re.findall(pattern, text, re.IGNORECASE)) for pattern in SPANISH_INDICATORS
    )
    english_count = sum(
        len(re.findall(pattern, text, re.IGNORECASE)) for pattern in ENGLISH_INDICATORS
    )
    return spanish_count, english_count


def detect_language(file_path: Path) -> str:
    """
    Detect language of a file. Returns 'English', 'Spanish', or 'Mixed'.
    """
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Skip very small files
        if len(content) < 50:
            return "Unknown"

        spanish_count, english_count = count_language_indicators(content)

        if spanish_count == 0 and english_count == 0:
            return "Unknown"

        if spanish_count > english_count * 1.5:
            return "Spanish"
        elif english_count > spanish_count * 1.5:
            return "English"
        else:
            return "Mixed"
    except Exception as e:
        return f"Error: {e!s}"


def get_module_name(path: Path) -> str:
    """Extract module name from path."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "modules" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def audit_all_modules() -> dict[str, list[tuple[str, str, str]]]:
    """
    Audit all modules and return language status for each file.
    Returns: {module_name: [(relative_path, file_type, language_status)]}
    """
    base_path = Path(".")
    modules_path = base_path / "modules"

    results = defaultdict(list)

    if not modules_path.exists():
        print("Error: modules directory not found")
        return results

    # Find all Python and Markdown files
    for file_path in sorted(modules_path.rglob("*")):
        if file_path.suffix in [".py", ".md"]:
            try:
                module_name = get_module_name(file_path)
                relative_path = file_path.relative_to(modules_path)
                language = detect_language(file_path)
                file_type = "Python" if file_path.suffix == ".py" else "Markdown"

                results[module_name].append((str(relative_path), file_type, language))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return results


def generate_report(results: dict[str, list[tuple[str, str, str]]]) -> str:
    """Generate a comprehensive report from audit results."""
    report = []
    report.append("# English Conversion Audit Report - nan-ai-engineering-labs\n")
    report.append(f"Total modules scanned: {len(results)}\n")

    # Summary statistics
    total_files = sum(len(files) for files in results.values())
    total_english = 0
    total_spanish = 0
    total_mixed = 0
    total_unknown = 0

    for module_files in results.values():
        for _, _, lang in module_files:
            if lang == "English":
                total_english += 1
            elif lang == "Spanish":
                total_spanish += 1
            elif lang == "Mixed":
                total_mixed += 1
            else:
                total_unknown += 1

    report.append("\n## Summary Statistics\n")
    report.append(f"- **Total files**: {total_files}")
    report.append(f"- **English**: {total_english} ({100*total_english/total_files:.1f}%)")
    report.append(f"- **Spanish**: {total_spanish} ({100*total_spanish/total_files:.1f}%)")
    report.append(f"- **Mixed**: {total_mixed} ({100*total_mixed/total_files:.1f}%)")
    report.append(f"- **Unknown**: {total_unknown} ({100*total_unknown/total_files:.1f}%)\n")

    # Detailed module breakdown
    report.append("## Module Breakdown\n")

    for module_name in sorted(results.keys()):
        module_files = results[module_name]

        # Count by language
        eng_count = sum(1 for _, _, lang in module_files if lang == "English")
        spa_count = sum(1 for _, _, lang in module_files if lang == "Spanish")
        mix_count = sum(1 for _, _, lang in module_files if lang == "Mixed")
        unk_count = sum(1 for _, _, lang in module_files if lang == "Unknown")

        report.append(f"### {module_name}\n")
        report.append(
            f"**Status**: {eng_count} English | {spa_count} Spanish | {mix_count} Mixed | {unk_count} Unknown\n"
        )
        report.append("\n| File | Type | Language |\n")
        report.append("|------|------|----------|\n")

        for file_path, file_type, language in sorted(module_files):
            # Shorten path for readability
            display_path = file_path.replace(f"{module_name}/", "")
            report.append(f"| {display_path} | {file_type} | {language} |\n")

        report.append("")

    # Files requiring conversion
    report.append("\n## Files Requiring English Conversion\n\n")

    conversion_needed = []
    for module_name in sorted(results.keys()):
        for file_path, _file_type, language in sorted(results[module_name]):
            if language in ["Spanish", "Mixed"]:
                conversion_needed.append((module_name, file_path, language))

    report.append(f"**Total files needing conversion**: {len(conversion_needed)}\n\n")

    if conversion_needed:
        report.append("| Module | File | Current Language |\n")
        report.append("|--------|------|------------------|\n")
        for module_name, file_path, language in conversion_needed:
            display_path = file_path.replace(f"{module_name}/", "")
            report.append(f"| {module_name} | {display_path} | {language} |\n")

    return "\n".join(report)


def main():
    print("Auditing English conversion status...")
    results = audit_all_modules()
    report = generate_report(results)

    # Save report
    report_path = Path("ENGLISH_CONVERSION_AUDIT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Audit complete! Report saved to {report_path}")
    print(report)


if __name__ == "__main__":
    main()
