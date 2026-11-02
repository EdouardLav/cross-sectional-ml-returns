cd "$(dirname "$0")/.."
PY="${PY:-$PWD/venv/bin/python3}"
R="$PWD/survivorship/runs"
j() {
  n=$1; m=$2; shift 2
  if [ -f "$R/$n/meta.json" ]; then echo "skip $n (already done)"; return; fi
  s=$(date +%s)
  if [ -n "$m" ]; then MODELS=$m "$PY" survivorship/run_crsp.py --out "$R/$n" "$@" > "$R/$n.log" 2>&1
  else env -u MODELS "$PY" survivorship/run_crsp.py --out "$R/$n" "$@" > "$R/$n.log" 2>&1; fi
  if [ $? -eq 0 ]; then echo "$n: ok in $(( ($(date +%s) - s) / 60 )) min"; else echo "$n: FAILED"; tail -20 "$R/$n.log"; fi
}
j D_lin Ridge,Lasso
j D_n50 Ridge,Lasso --n 50
j D_w3 Ridge,Lasso --window 3
j D_n50_w3 Ridge,Lasso --n 50 --window 3
j D ""
"$PY" survivorship/run_all.py summary
