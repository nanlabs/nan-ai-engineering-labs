#!/usr/bin/env python3
"""Bulk replace frequent Spanish terms in Markdown files."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = [
    ROOT / "modules",
    ROOT / "trends-extras",
    ROOT / "templates",
    ROOT / "docs",
    ROOT / ".github",
]
TARGET_FILES = [ROOT / "README.md"]

REPLACEMENTS = {
    "Enunciado": "Statement",
    "Salida": "Output",
    "Entrenar": "Train",
    "completar": "complete",
    "cada": "each",
    "recomendados": "recommended",
    "Implementar": "Implement",
    "calidad": "quality",
    "Detectar": "Detect",
    "múltiples": "multiple",
    "Pendiente": "Pending",
    "completado": "completed",
    "usar": "use",
    "Evaluar": "Evaluate",
    "Definir": "Define",
    "Aplicar": "Apply",
    "producción": "production",
    "decisiones": "decisions",
    "apartado": "section",
    "Crear": "Create",
    "contexto": "context",
    "Visualizar": "Visualize",
    "texto": "text",
    "completo": "complete",
    "valores": "values",
    "desde": "from",
    "Calcular": "Calculate",
    "mejora": "improvement",
    "Comparar": "Compare",
    "respuesta": "response",
    "reales": "real",
    "Estado": "Status",
    "comunes": "common",
    "código": "code",
    "escritos": "written",
    "Distribución": "Distribution",
    "distribución": "distribution",
    "Generar": "Generate",
    "esperada": "expected",
    "usuarios": "users",
    "usuario": "user",
    "usando": "using",
    "ruido": "noise",
    "modelo": "model",
    "Identificar": "Identify",
    "Documentar": "Document",
    "sobre": "about",
    "Resumen": "Summary",
    "Construir": "Build",
    "Validar": "Validate",
    "similitud": "similarity",
    "siguiente": "next",
    "mínimo": "minimum",
    "correctamente": "correctly",
    "parámetros": "parameters",
    "cuando": "when",
    "Cargar": "Load",
    "Solución": "Solution",
    "Guardar": "Save",
    "diferentes": "different",
    "salida": "output",
    "menos": "less",
    "Detección": "Detection",
    "producto": "product",
    "pero": "but",
    "Matriz": "Matrix",
    "matriz": "matrix",
    "todas": "all",
    "similares": "similar",
    "información": "information",
    "datos": "data",
    "según": "according to",
    "Predecir": "Predict",
    "formato": "format",
    "bajo": "low",
    "técnicas": "techniques",
    "técnica": "technique",
    "punto": "point",
    "promedio": "average",
    "completa": "complete",
    "casos": "cases",
    "básico": "basic",
    "tiempo": "time",
    "requiere": "requires",
    "palabras": "words",
    "impacto": "impact",
    "Ejemplo": "Example",
    "clases": "classes",
    "Ventaja": "Advantage",
    "sistemas": "systems",
    "respuestas": "responses",
    "rápido": "fast",
    "mejor": "better",
    "grupo": "group",
    "regresión": "regression",
    "patrones": "patterns",
    "esta": "this",
    "arquitectura": "architecture",
    "ahora": "now",
    "Agregar": "Add",
    "seguridad": "security",
    "riesgos": "risks",
    "riesgo": "risk",
    "relevantes": "relevant",
    "privacidad": "privacy",
    "practicas": "practices",
    "nulos": "nulls",
    "grandes": "large",
    "Explicar": "Explain",
    "distribuciones": "distributions",
    "detectar": "detect",
    "después": "after",
    "aprendizajes": "learnings",
    "aprendidas": "learned",
    "recomendación": "recommendation",
    "proyecto": "project",
    "negocio": "business",
    "mejoras": "improvements",
    "imagen": "image",
    "Generación": "Generation",
    "Fase": "Phase",
    "Entrena": "Train",
    "Cuándo": "When",
    "alertas": "alerts",
    "validar": "validate",
    "transformaciones": "transformations",
    "semana": "week",
    "salidas": "outputs",
    "Recomendar": "Recommend",
    "mismo": "same",
    "Métricas": "Metrics",
    "flujo": "flow",
    "filas": "rows",
    "entrada": "input",
    "documentos": "documents",
    "debe": "must",
    "curvas": "curves",
    "columnas": "columns",
    "aplicado": "applied",
    "puede": "can",
    "probabilidad": "probability",
    "Implemento": "I implement",
    "interpreto": "I interpret",
    "Calculo": "I calculate",
    "describo": "I describe",
    "varianza": "variance",
    "utilidad": "utility",
    "gradiente": "gradient",
    "descendente": "descent",
    "Documento": "I document",
    "limitaciones": "limitations",
    "Vectores": "Vectors",
    "previos": "previous",
    "Importar": "Import",
    "Crear": "Create",
    "representa": "represents",
    "características": "features",
    "precio": "price",
    "popularidad": "popularity",
    "Similitud": "Similarity",
    "productos": "products",
    "coseno": "cosine",
    "cercano": "close",
    "indica": "indicates",
    "Definir": "Define",
    "Normas": "Norms",
    "propuesto": "proposed",
    "acción": "action",
    "comedia": "comedy",
}


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for directory in TARGET_DIRS:
        if directory.exists():
            files.extend(sorted(directory.rglob("*.md")))
    for file_path in TARGET_FILES:
        if file_path.exists():
            files.append(file_path)
    return files


def replace_words(text: str) -> str:
    out = text
    for source, target in REPLACEMENTS.items():
        pattern = re.compile(rf"(?<![\w-]){re.escape(source)}(?![\w-])")
        out = pattern.sub(target, out)
    return out


def main() -> int:
    changed = 0
    files = iter_markdown_files()
    for path in files:
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        updated = replace_words(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
            print(f"CHANGED {path.relative_to(ROOT)}")
    print(f"TOTAL_FILES {len(files)}")
    print(f"TOTAL_CHANGED {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
