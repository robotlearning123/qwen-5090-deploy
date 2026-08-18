#!/usr/bin/env bash
# The single verification gate. Run after ANY change; CI runs the same.
# Exit 0 = safe to commit. CPU-safe; needs network only if pytest is missing
# (throwaway venv fallback).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
fails=0
step() { printf '== %s\n' "$1"; }

step "python syntax"
python3 -m py_compile scripts/*.py || fails=$((fails+1))

step "shell syntax"
for s in scripts/*.sh; do bash -n "$s" || fails=$((fails+1)); done

step "python env (throwaway venv if pytest/pillow missing)"
PY=python3
if ! python3 -c 'import pytest' 2>/dev/null; then
  python3 -m venv /tmp/q5090-verify-venv 2>/dev/null && /tmp/q5090-verify-venv/bin/pip install -q pytest pillow 2>/dev/null \
    && PY=/tmp/q5090-verify-venv/bin/python
fi

step "entrypoint --help smoke"
for s in chat bench_speed bench_quality duel vision_test humaneval_run aime_run; do
  "$PY" "scripts/$s.py" --help >/dev/null 2>&1 || { echo "FAIL --help: $s"; fails=$((fails+1)); }
done

step "unit tests (creates throwaway venv if pytest missing)"
"$PY" -m pytest tests/ -q || fails=$((fails+1))

step "machine manifests parse"
python3 - <<'PY' || fails=$((fails+1))
import json, sys
json.load(open('scripts/profiles.json'))
json.load(open('repo.json'))
print('manifests OK')
PY

step "internal-reference leak grep (public hygiene)"
# patterns are string-split so this file itself cannot match them
PAT1="44-local"; PAT2="-llm"; PAT3="149\\.165\\."; PAT4="/home/"; PAT5="robot"
PAT6="/mnt/"; PAT7="data"; PAT8="ccz"; PAT9="ccq-"; PAT10="CIS250"; PAT11="wanman"; PAT12="~/workspace"
hits=$(grep -rnc "${PAT1}${PAT2}\|${PAT3}\|${PAT4}${PAT5}\|${PAT6}${PAT7}\|${PAT8}\b\|${PAT9}\|${PAT10}\|\b${PAT11}\b\|${PAT12}" \
  --include='*.md' --include='*.json' --include='*.tsv' --include='*.py' --include='*.yml' --include='*.env' \
  --include='*.sh' --include='*.txt' --include='*.jinja' . \
  --exclude=verify.sh --exclude-dir=.git 2>/dev/null \
  | grep -v ':0' | wc -l)
gz=$(for z in $(git ls-files '*.jsonl.gz'); do zcat "$z" 2>/dev/null | grep -cE "${PAT1}${PAT2}|${PAT4}${PAT5}|${PAT6}${PAT7}|${PAT10}|\b${PAT11}\b"; done | grep -v '^0$' | wc -l)
hits=$((hits + gz))
[ "$hits" = "0" ] || { echo "LEAK HITS: $hits (see grep above)"; fails=$((fails+1)); }

step "doc path references resolve"
python3 - <<'PY' || fails=$((fails+1))
import re, pathlib, sys, subprocess
tracked = set()
import subprocess
tracked = set(subprocess.run(['git','ls-files'],capture_output=True,text=True).stdout.split())
bad = []
for md in [p for p in tracked if p.endswith(('.md','.json'))]:
    text = pathlib.Path(md).read_text(errors='ignore')
    for m in re.finditer(r'(?:^|\]\()(?:\./)?((?:\.\./)?(?:docs|scripts|tests|benchmarks|cc-profile)/[A-Za-z0-9_./-]+)', text):
        pass
    for m in re.finditer(r'((?:\.\./)?(?:docs|scripts|tests|benchmarks|cc-profile)/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)', text):
        ref = m.group(1).rstrip('.')
        if ref not in tracked and not any(t.startswith(ref.rstrip('/')+'/') for t in tracked):
            bad.append(f'{md} -> {ref}')
print('\n'.join(bad) if bad else 'doc refs OK')
sys.exit(1 if bad else 0)
PY

if [ "$fails" = 0 ]; then echo "== VERIFY PASS (all gates green)"; else echo "== VERIFY FAIL ($fails gate(s))"; fi
exit $fails
