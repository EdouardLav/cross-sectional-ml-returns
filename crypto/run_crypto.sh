cd "$(dirname "$0")/.."
PY="$HOME/Desktop/Pro/cross-sectional-ml-returns/venv/bin/python3"
for n in 50 30 100; do
  "$PY" crypto/research.py crypto/data crypto/out_n$n $n > crypto/out_n$n.log 2>&1 && "$PY" crypto/construct.py crypto/data crypto/out_n$n >> crypto/out_n$n.log 2>&1
  echo "top $n done"; grep -v -i warning crypto/out_n$n.log | tail -16
done
