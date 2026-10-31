cd "$(dirname "$0")/.."
PY="$HOME/Desktop/Pro/cross-sectional-ml-returns/venv/bin/python3"
"$PY" crypto/download_spot.py || exit 1
echo "== VALIDATION: Binance spot, Oct 2017 to May 2020 (never seen)"
"$PY" crypto/validate.py --data crypto/spot_data --out crypto/val_spot --start 2017-10-01 --end 2020-05-31 --fee_bps 10 > crypto/val_spot.log 2>&1
grep -v -i warn crypto/val_spot.log | grep -v "result = func"
echo "== COST REALISM: Binance perps, Jan 2022 to Sept 2026"
"$PY" crypto/validate.py --data crypto/data --out crypto/val_perp --start 2022-01-01 --end 2026-09-30 --fee_bps 5 > crypto/val_perp.log 2>&1
grep -v -i warn crypto/val_perp.log | grep -v "result = func"
