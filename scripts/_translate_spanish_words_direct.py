#!/usr/bin/env python3
"""Direct search and translate common Spanish words in selected files."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from deep_translator import GoogleTranslator


SPANISH_WORDS = {
    "Calidad",
    "Calcula",
    "Comentarios",
    "Completa",
    "Completar",
    "Compara",
    "Comunicación",
    "Comunicacion",
    "Codigo",
    "Consigna",
    "Contexto",
    "Convertir",
    "Crea",
    "Cumple",
    "Cumplimiento",
    "Detecta",
    "Documenta",
    "Ejecutar",
    "Ejecucion",
    "Evidencia",
    "Excluye",
    "Explico",
    "Fuente",
    "Guiados",
    "Herramientas",
    "Implementa",
    "Incluye",
    "Integrar",
    "Interpretar",
    "Medir",
    "Mejora",
    "Nivel",
    "Ninguno",
    "Pendiente",
    "Pistas",
    "Practica",
    "Priorizar",
    "Proyecto",
    "Proponer",
    "Puntaje",
    "Relaciono",
    "Recomendaciones",
    "Respuesta",
    "Riesgos",
    "Semana",
    "Teoria",
    "Usar",
    "Usuario",
    "agentes",
    "alcance",
    "accionables",
    "actividades",
    "aplicar",
    "aprueba",
    "aporta",
    "avanzadas",
    "basicas",
    "calculadas",
    "calculado",
    "contexto",
    "criterio",
    "debiasing",
    "definidos",
    "documentadas",
    "dudas",
    "ejecucion",
    "enfocada",
    "entregado",
    "entregables",
    "entrenado",
    "etapa",
    "evaluar",
    "explicando",
    "fuente",
    "funcionamiento",
    "fundamentos",
    "futuras",
    "inicial",
    "interpretadas",
    "justificar",
    "nivel",
    "minimos",
    "minutos",
    "optimizaciones",
    "pasos",
    "positivas",
    "proceso",
    "preguntas",
    "progreso",
    "propuestas",
    "pruebas",
    "reproducibles",
    "reporte",
    "reportadas",
    "recomendaciones",
    "requerimientos",
    "solucion",
    "stopwords",
    "suficiente",
    "tecnica",
    "tecnicas",
    "teoria",
    "texto",
    "transpuesta",
    "usuario",
    "utiles",
    "validado",
    "visualizaciones",
}

FALLBACK_TRANSLATIONS = {
    "Ninguno": "None",
    "Nivel": "Level",
    "validado": "validated",
    "Practica": "Practice",
    "progreso": "progress",
    "utiles": "useful",
    "teoria": "theory",
    "tecnicas": "techniques",
    "Relaciono": "I relate",
    "Priorizar": "Prioritize",
    "Pendiente": "Pending",
    "justificar": "justify",
    "Integrar": "Integrate",
    "Explico": "I explain",
    "entregables": "deliverables",
    "dudas": "questions",
    "Documenta": "Document",
    "cumplimiento": "compliance",
    "Comunicacion": "Communication",
    "Completar": "Complete",
    "aprueba": "approve",
    "aporta": "contributes",
    "aplicar": "apply",
    "alcance": "scope",
    "actividades": "activities",
    "reproducibles": "reproducible",
    "proceso": "process",
    "pasos": "steps",
    "nivel": "level",
}


def translate_word(word: str) -> str | None:
    """Translate one Spanish token to English."""
    if word in FALLBACK_TRANSLATIONS:
        return FALLBACK_TRANSLATIONS[word]

    try:
        translator = GoogleTranslator(source="es", target="en")
        result = translator.translate(word)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    return result or None


def load_files_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def build_regex() -> re.Pattern[str]:
    pattern = r"\b(" + "|".join(sorted(map(re.escape, SPANISH_WORDS))) + r")\b"
    return re.compile(pattern)


def count_matches(regex: re.Pattern[str], file_paths: list[str]) -> int:
    total = 0
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        total += sum(1 for _ in regex.finditer(content))
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files-list", required=True)
    args = parser.parse_args()

    files_to_process = load_files_list(Path(args.files_list))
    regex = build_regex()

    cache: dict[str, str | None] = {}
    changed_files: set[str] = set()
    changed_lines = 0

    for file_path in files_to_process:
        path = Path(file_path)
        if not path.exists():
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        lines = content.split("\n")
        file_changed = False

        for index, line in enumerate(lines):
            updated_line = line
            matches = list(regex.finditer(line))

            for match in reversed(matches):
                word = match.group(1)
                if word not in cache:
                    cache[word] = translate_word(word)
                    time.sleep(0.05)

                translated = cache[word]
                if not translated:
                    continue

                start, end = match.span()
                updated_line = (
                    updated_line[:start] + translated + updated_line[end:]
                )

            if updated_line != line:
                lines[index] = updated_line
                file_changed = True
                changed_lines += 1

        if not file_changed:
            continue

        try:
            path.write_text("\n".join(lines), encoding="utf-8")
        except OSError:
            continue
        changed_files.add(file_path)

    print(f"TARGET_FILES {len(files_to_process)}")
    print(f"SPANISH_WORDS_FOUND {count_matches(regex, files_to_process)}")
    print(f"CHANGED_FILES {len(changed_files)}")
    print(f"CHANGED_LINES {changed_lines}")
    print(f"CACHE_SIZE {len(cache)}")


if __name__ == "__main__":
    main()
