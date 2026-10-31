cd "$(dirname "$0")/.."
PY="$HOME/Desktop/Pro/cross-sectional-ml-returns/venv/bin/python3"
run() { "$PY" crypto/validate.py "$@" 2>&1 | grep -v -i warn | grep -v "result = func" | sed -n '/== BTC-hedged/,$p'; }
echo "== P, perps 2022-2026, taker 5 bps"; run --data crypto/data --out crypto/val2_perp_taker --start 2022-01-01 --end 2026-09-30 --fee_bps 5 --trade_rate 0.25
echo "== P, perps 2022-2026, maker 2 bps"; run --data crypto/data --out crypto/val2_perp_maker --start 2022-01-01 --end 2026-09-30 --fee_bps 2 --trade_rate 0.25
echo "== P, spot 2017-2020 (paper), 10 bps"; run --data crypto/spot_data --out crypto/val2_spot --start 2017-10-01 --end 2020-05-31 --fee_bps 10 --trade_rate 0.25
