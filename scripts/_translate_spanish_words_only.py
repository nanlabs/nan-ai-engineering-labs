#!/usr/bin/env python3
"""
Translate only Spanish words from cspell report, leave technical terms alone.
Usage: python scripts/_translate_spanish_words_only.py --report <report_file> --spanish-words <words_file>
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from deep_translator import GoogleTranslator
import time

# Spanish word patterns detected in the top unknown words
SPANISH_WORD_PATTERNS = {
    'Implementa', 'Compara', 'Usar', 'Propuestos', 'Guiados', 'Practica', 
    'Calidad', 'validado', 'progreso', 'Recomendaciones', 'Nivel', 'Convertir',
    'Mejora', 'Ejecutar', 'Crea', 'Calcula', 'Respuesta', 'entrenado', 'aprueba',
    'Usuario', 'ENTRENAMIENTO', 'Completar', 'Visualiza', 'usuario', 'texto',
    'Interpretabilidad', 'Ingeniería', 'Relevancia', 'Autorización', 'Visualización',
    'Configuración', 'Explicabilidad', 'Características', 'Clasificación',
    'Inteligencia', 'Aprendizaje', 'Datos', 'Modelo', 'Entrenamiento',
    'Prueba', 'Validación', 'Métrica', 'Precisión', 'Recall', 'Evaluación'
}

def load_report_lines(report_path):
    """Parse cspell report and extract lines with unknown words, grouped by file."""
    file_issues = defaultdict(list)
    if not Path(report_path).exists():
        print(f"Report not found: {report_path}", file=sys.stderr)
        return file_issues
    
    with open(report_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip()
            if ' - Unknown word' not in line:
                continue
            # Format: path/file.ext:line:col - Unknown word (word)
            match = re.match(r'^([^:]+):(\d+):(\d+) - Unknown word \(([^)]+)\)', line)
            if match:
                filepath, line_num_str, col, word = match.groups()
                file_issues[filepath].append({
                    'line': int(line_num_str),
                    'col': int(col),
                    'word': word,
                    'full_line': line
                })
    return file_issues

def is_spanish_word(word):
    """Check if word matches Spanish patterns or contains Spanish characters."""
    if word in SPANISH_WORD_PATTERNS:
        return True
    if re.search(r'[áéíóúñÁÉÍÓÚÑ¿¡]', word):
        return True
    return False

def translate_word(word, timeout=5):
    """Translate a single word using deep_translator with timeout."""
    try:
        translator = GoogleTranslator(source='es', target='en')
        result = translator.translate(word)
        return result if result else None
    except Exception as e:
        return None

def translate_files(files_list, report_issues, translation_cache=None):
    """Translate only Spanish words in specified files."""
    if translation_cache is None:
        translation_cache = {}
    
    changed_files = set()
    changed_lines = 0
    
    for filepath in files_list:
        if filepath not in report_issues:
            continue
        
        issues = report_issues[filepath]
        spanish_issues = [i for i in issues if is_spanish_word(i['word'])]
        
        if not spanish_issues:
            continue
        
        try:
            path_obj = Path(filepath)
            if not path_obj.exists():
                continue
            
            content = path_obj.read_text(encoding='utf-8', errors='replace')
            lines = content.split('\n')
            file_changed = False
            
            # Group issues by line number for efficient processing
            by_line = defaultdict(list)
            for issue in spanish_issues:
                by_line[issue['line'] - 1].append(issue)
            
            # Process each line with Spanish words
            for line_idx in sorted(by_line.keys()):
                if line_idx >= len(lines):
                    continue
                
                line = lines[line_idx]
                for issue in by_line[line_idx]:
                    word = issue['word']
                    if word not in translation_cache:
                        translated = translate_word(word)
                        translation_cache[word] = translated
                    
                    if translation_cache[word]:
                        old_word = word
                        new_word = translation_cache[word]
                        # Use word boundaries to avoid partial replacements
                        pattern = r'\b' + re.escape(old_word) + r'\b'
                        new_line = re.sub(pattern, new_word, line)
                        if new_line != line:
                            lines[line_idx] = new_line
                            line = new_line
                            file_changed = True
                            changed_lines += 1
            
            if file_changed:
                path_obj.write_text('\n'.join(lines), encoding='utf-8')
                changed_files.add(filepath)
        
        except Exception as e:
            pass
    
    return changed_files, changed_lines, translation_cache

def main():
    parser = argparse.ArgumentParser(description='Translate only Spanish words from cspell report')
    parser.add_argument('--report', required=True, help='cspell report file')
    parser.add_argument('--files-list', required=True, help='List of files to process')
    args = parser.parse_args()
    
    # Load Spanish words to target
    report_issues = load_report_lines(args.report)
    
    # Load file list
    files_to_process = []
    if Path(args.files_list).exists():
        with open(args.files_list, 'r') as f:
            files_to_process = [line.strip() for line in f if line.strip()]
    
    # Translate
    changed_files, changed_lines, cache = translate_files(files_to_process, report_issues)
    
    # Report metrics
    print(f"TARGET_FILES {len(files_to_process)}")
    print(f"SPANISH_WORDS_FOUND {len([i for issues in report_issues.values() for i in issues if is_spanish_word(i['word'])])}")
    print(f"CHANGED_FILES {len(changed_files)}")
    print(f"CHANGED_LINES {changed_lines}")
    print(f"CACHE_SIZE {len(cache)}")

if __name__ == '__main__':
    main()
