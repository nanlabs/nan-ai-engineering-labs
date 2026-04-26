from pathlib import Path
import hashlib

root = Path('.')
files_list = root / '.cspell_reports' / 'all_md_files.txt'
if not files_list.exists():
    md_files = sorted(str(p.relative_to(root)) for p in root.rglob('*.md'))
    files_list.parent.mkdir(parents=True, exist_ok=True)
    files_list.write_text('\n'.join(md_files) + '\n', encoding='utf-8')

mapping = {
    'Pendiente':'Pending',
    'reportadas':'reported',
    'limpio':'clean',
    'Limitaciones':'Limitations',
    'intentos':'attempts',
    'iniciales':'initial',
    'hallazgos':'findings',
    'formato':'format',
    'entrenados':'trained',
    'ejecuta':'execute',
    'ejecucion':'execution',
    'duplicados':'duplicates',
    'documentada':'documented',
    'Demostrar':'Demonstrate',
    'Combina':'Combine',
    'Codigo':'Code',
    'Claridad':'Clarity',
    'avanzado':'advanced',
    'Mide':'Measure',
}

changed = 0
for rel in files_list.read_text(encoding='utf-8').splitlines():
    if not rel.strip():
        continue
    p = root / rel.strip()
    if not p.is_file():
        continue
    before = p.read_text(encoding='utf-8', errors='replace')
    after = before
    for src, dst in mapping.items():
        after = after.replace(src, dst)
    if after != before:
        p.write_text(after, encoding='utf-8')
        changed += 1

print(f'CHANGED_FILES={changed}')
