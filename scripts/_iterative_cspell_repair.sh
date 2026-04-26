#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/nquiroga/SSDedo/Documents/projects/NanLabs/labs/nan-ai-engineering-labs"
PY="/media/nquiroga/SSDedo/Documents/projects/NanLabs/labs/.venv/bin/python"
FILE_LIST="/tmp/nan_ai_spell_files_abs.txt"

cd "$ROOT"

if [[ ! -f "$FILE_LIST" ]]; then
  find modules trends-extras templates docs .github -type f \( -name "*.md" -o -name "*.py" \) -exec realpath {} \; > "$FILE_LIST"
fi

report_prev="/tmp/nan_ai_cspell_iter_0.txt"
npx --yes cspell --no-progress --no-summary --file-list "$FILE_LIST" > "$report_prev" 2>&1 || true
prev_total=$(wc -l < "$report_prev")
echo "ITER 0 TOTAL $prev_total"

for i in 1 2 3 4 5; do
  "$PY" scripts/_normalize_from_cspell_report.py --report "$report_prev" > "/tmp/nan_norm_iter_${i}.txt"

  report_next="/tmp/nan_ai_cspell_iter_${i}.txt"
  npx --yes cspell --no-progress --no-summary --file-list "$FILE_LIST" > "$report_next" 2>&1 || true
  total=$(wc -l < "$report_next")
  changed_files=$(grep -E '^CHANGED_FILES ' "/tmp/nan_norm_iter_${i}.txt" | awk '{print $2}')
  echo "ITER $i CHANGED_FILES ${changed_files:-0} TOTAL $total"

  if [[ "$total" -ge "$prev_total" ]]; then
    echo "NO_IMPROVEMENT_STOP at ITER $i"
    break
  fi

  report_prev="$report_next"
  prev_total="$total"
done

echo "FINAL_REPORT $report_prev"
echo "FINAL_TOTAL $prev_total"
