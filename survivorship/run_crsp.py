import argparse, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_universe import run, top_by_dollar_volume

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--data", default=str(HERE / "data_crsp"))
ap.add_argument("--french", default=str(HERE.parent / "data" / "french_factors.parquet"))
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--window", type=int, default=6)
ap.add_argument("--cap", type=int, default=500)
a = ap.parse_args()
D = Path(a.data)
adj, close, vol, cap = (pd.read_parquet(D / f"{f}.parquet") for f in ["adj_close", "close", "volume", "mktcap"])
french = pd.read_parquet(a.french)
french.index = pd.to_datetime(french.index).to_period("M").to_timestamp("M")
cand = cap.rank(axis=1, ascending=False, method="first") <= a.cap
mask = top_by_dollar_volume(cand, close, vol, adj, n=a.n, window=a.window)
run("D", mask, adj, vol, french, str(Path(a.out).resolve()))
