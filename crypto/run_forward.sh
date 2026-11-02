set -eo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$PWD/venv/bin/python3}"
END="${1:?usage: bash crypto/run_forward.sh YYYY-MM-DD (a Sunday)}"
f() { grep -v -i warn | grep -v "result = func" || true; }
"$PY" crypto/update_forward.py --src crypto/data --dst crypto/fwd_data --end "$END" 2>&1 | f
"$PY" crypto/make_crypto_only.py --src crypto/fwd_data --dst crypto/fwd_data_crypto 2>&1 | f
"$PY" crypto/validate.py --data crypto/fwd_data --out crypto/fwd_frozen --start 2022-01-01 --end "$END" --fee_bps 5 --trade_rate 0.25 --sizes 50 > crypto/fwd_frozen.log 2>&1
echo "########## 1. FROZEN SPEC, all USDT perpetuals (as pre-registered)"
"$PY" crypto/live_book.py --data crypto/fwd_data --out crypto/live/frozen --ref crypto/fwd_frozen/book_n50.csv,crypto/val2_perp_taker/book_n50.csv 2>&1 | f
echo "########## 2. CRYPTO ONLY (Addendum 3, main specification)"
"$PY" crypto/live_book.py --data crypto/fwd_data_crypto --out crypto/live/crypto_only --ref crypto/val3_perp_crypto/book_n50.csv 2>&1 | f
