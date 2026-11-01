cd "$(dirname "$0")/.."
PY="$HOME/Desktop/Pro/cross-sectional-ml-returns/venv/bin/python3"
f() { grep -v -i warn | grep -v "result = func"; }
"$PY" crypto/make_crypto_only.py --src crypto/data --dst crypto/data_crypto 2>&1 | f
echo "########## CRYPTO ONLY, perps 2022 to Sept. 2026, taker 5 bps, quarter-step"
"$PY" crypto/validate.py --data crypto/data_crypto --out crypto/val3_perp_crypto --start 2022-01-01 --end 2026-09-30 --fee_bps 5 --trade_rate 0.25 2>&1 | f | sed -n '/== top/,$p'
T="crypto/out_n*/summary.csv,crypto/out_n*/riskbook_summary.csv,crypto/val_perp/summary.csv,crypto/val2_perp_taker/summary.csv,crypto/val2_perp_maker/summary.csv,crypto/val3_perp_crypto/summary.csv"
echo "########## ATTRIBUTION, crypto only"
"$PY" crypto/attribution.py --data crypto/data_crypto --book crypto/val3_perp_crypto/book_n50.csv --start 2022-01-01 --end 2026-09-30 --trials "$T" 2>&1 | f
