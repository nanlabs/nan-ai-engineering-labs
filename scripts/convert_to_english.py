#!/usr/bin/env python3
"""
Convert all Spanish content to English across nan-ai modules.
Uses spanish_english_dictionary for consistent terminology.
"""

import re
import sys
from pathlib import Path

# Import dictionary
sys.path.insert(0, str(Path(__file__).parent))
from spanish_english_dictionary import (
    translate_filename,
    translate_text,
)


def preserve_code_blocks(text: str) -> tuple[str, dict]:
    """
    Extract code blocks and replace with placeholders to prevent translation.
    Returns: (text_with_placeholders, code_blocks_dict)
    """
    code_blocks = {}
    pattern = r"```[\s\S]*?```"

    def replace_code(match):
        placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
        code_blocks[placeholder] = match.group(0)
        return placeholder

    text_with_placeholders = re.sub(pattern, replace_code, text)
    return text_with_placeholders, code_blocks


def restore_code_blocks(text: str, code_blocks: dict) -> str:
    """Restore code blocks from placeholders."""
    for placeholder, code in code_blocks.items():
        text = text.replace(placeholder, code)
    return text


def preserve_links(text: str) -> tuple[str, dict]:
    """
    Extract markdown links to prevent URL translation.
    Returns: (text_with_placeholders, links_dict)
    """
    links = {}
    # Match [text](url) or [text](url "title")
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"

    def replace_link(match):
        placeholder = f"__LINK_{len(links)}__"
        links[placeholder] = match.group(0)
        return placeholder

    text_with_placeholders = re.sub(pattern, replace_link, text)
    return text_with_placeholders, links


def restore_links(text: str, links: dict) -> str:
    """Restore links from placeholders."""
    for placeholder, link in links.items():
        text = text.replace(placeholder, link)
    return text


def translate_markdown_file(file_path: Path) -> str:
    """
    Translate a markdown file from Spanish to English.
    Preserves code blocks, links, and markdown formatting.
    """
    with open(file_path, encoding="utf-8") as f:
        original_text = f.read()

    text = original_text

    # Preserve code blocks
    text, code_blocks = preserve_code_blocks(text)

    # Preserve links
    text, links = preserve_links(text)

    # Translate content
    text = translate_text(text)

    # Restore links
    text = restore_links(text, links)

    # Restore code blocks
    text = restore_code_blocks(text, code_blocks)

    return text


def translate_python_file(file_path: Path) -> str:
    """
    Translate comments in a Python file from Spanish to English.
    Preserves code structure and strings.
    """
    with open(file_path, encoding="utf-8") as f:
        original_text = f.read()

    text = original_text

    # Translate comments (lines starting with # after optional whitespace)
    lines = text.split("\n")
    translated_lines = []

    for line in lines:
        # Check if line contains a comment
        if "#" in line:
            # Split by first # to preserve inline comments
            before_comment, comment = line.split("#", 1)
            # Translate the comment part
            translated_comment = translate_text("#" + comment)
            translated_lines.append(before_comment + translated_comment)
        else:
            translated_lines.append(line)

    return "\n".join(translated_lines)


def get_files_to_convert() -> list[Path]:
    """Get list of Spanish/Mixed files that need conversion."""
    base_path = Path(".")
    modules_path = base_path / "modules"

    files_to_convert = []

    if not modules_path.exists():
        print("Error: modules directory not found")
        return files_to_convert

    for file_path in sorted(modules_path.rglob("*")):
        if file_path.is_file() and file_path.suffix in [".py", ".md"]:
            # Skip already converted examples README
            if "examples/README.md" in str(file_path):
                continue

            # Detect language (simple heuristic)
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read(500)  # Read first 500 chars

                spanish_indicators = sum(
                    1
                    for pattern in [
                        r"\b(el|la|los|las)\b",
                        r"\b(de|del|en|con|por|para)\b",
                        r"á|é|í|ó|ú|ñ",
                    ]
                    for _ in [re.search(pattern, content, re.IGNORECASE)]
                    if _
                )

                # Only convert if has Spanish content
                if spanish_indicators > 2:
                    files_to_convert.append(file_path)
            except (OSError, UnicodeDecodeError, ValueError):
                pass

    return files_to_convert


def convert_all_files():
    """Convert all Spanish/Mixed files to English."""
    files = get_files_to_convert()

    if not files:
        print("✅ No files needing conversion found!")
        return

    print(f"📝 Found {len(files)} files to convert\n")

    converted = 0
    failed = []

    for file_path in files:
        try:
            if file_path.suffix == ".md":
                translated_content = translate_markdown_file(file_path)
            else:  # .py
                translated_content = translate_python_file(file_path)

            # Write back
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(translated_content)

            converted += 1
            relative_path = file_path.relative_to(".")
            print(f"✅ {relative_path}")

        except Exception as e:
            failed.append((file_path, str(e)))
            print(f"❌ {file_path.relative_to('.')}: {e}")

    print("\n📊 Conversion Summary:")
    print(f"   ✅ Successfully converted: {converted} files")
    print(f"   ❌ Failed: {len(failed)} files")

    if failed:
        print("\n⚠️  Failed conversions:")
        for file_path, error in failed:
            print(f"   - {file_path.relative_to('.')}: {error}")


def rename_spanish_files():
    """Rename files with Spanish names to English equivalents."""
    base_path = Path(".")
    modules_path = base_path / "modules"

    renamed = 0

    for file_path in sorted(modules_path.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix in [".md", ".py"]:
            filename = file_path.name
            new_filename = translate_filename(filename)

            # Skip if name didn't change
            if new_filename == filename:
                continue

            # Skip practice files - they have Spanish names by design in some cases
            if "practice-" in filename and "01-" in filename:
                continue

            new_path = file_path.parent / new_filename

            try:
                file_path.rename(new_path)
                renamed += 1
                print(f"📝 Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"❌ Failed to rename {filename}: {e}")

    print(f"\n✅ Renamed {renamed} files")


if __name__ == "__main__":
    print("🚀 Starting English conversion...\n")

    # First convert content
    print("=" * 60)
    print("STEP 1: Converting file contents to English")
    print("=" * 60)
    convert_all_files()

    print("\n" + "=" * 60)
    print("STEP 2: Renaming Spanish filenames to English")
    print("=" * 60)
    rename_spanish_files()

    print("\n✅ Conversion complete!")
