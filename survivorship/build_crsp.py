import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data_crsp"))
ap.add_argument("--start", default="2003-01-01")
a = ap.parse_args()
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_parquet(a.src, columns=["permno", "date", "ret", "prc", "vol", "mktcap", "shrcd", "exchcd"])
d = d[(d.date >= a.start) & d.shrcd.isin([10, 11]) & d.exchcd.isin([1, 2, 3])].copy()
d["date"] = d.date.dt.to_period("M").dt.to_timestamp("M")
d["permno"] = d.permno.astype(str)
w = lambda c: d.pivot_table(index="date", columns="permno", values=c, aggfunc="last").sort_index()
ret = w("ret")
al = lambda x: x.reindex(index=ret.index, columns=ret.columns)
(1 + ret).cumprod().where(ret.notna()).to_parquet(OUT / "adj_close.parquet")
al(w("prc").abs()).to_parquet(OUT / "close.parquet")
al(w("vol")).to_parquet(OUT / "volume.parquet")
al(w("mktcap")).to_parquet(OUT / "mktcap.parquet")
print(f"{ret.shape[1]} stocks, {ret.index[0].date()} to {ret.index[-1].date()}; "
      f"delisted before the end: {int((ret.apply(lambda s: s.last_valid_index()) < ret.index[-1]).sum())}")
