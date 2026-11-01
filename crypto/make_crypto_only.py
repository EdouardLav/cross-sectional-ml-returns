import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--list", default=str(Path(__file__).resolve().parent / "tradfi.txt"))
a = ap.parse_args()
S, D = Path(a.src), Path(a.dst)
D.mkdir(parents=True, exist_ok=True)

c = pd.read_parquet(S / "close.parquet")
c.index = pd.to_datetime(c.index)
q = pd.read_parquet(S / "quote_volume.parquet").reindex_like(c)
c = c.where(q > 0)
r = c.pct_change(fill_method=None)
we = r.index.dayofweek >= 5
ratio = r[we].abs().mean() / r[~we].abs().mean()
nwe = r[we].notna().sum() / 2
listed = set(Path(a.list).read_text().split())
test = set(ratio.index[(ratio < 0.46) & (nwe >= 8)])
excl = listed | test
for f in ["close", "quote_volume", "funding"]:
    x = pd.read_parquet(S / f"{f}.parquet")
    x.drop(columns=[s for s in x if s in excl]).to_parquet(D / f"{f}.parquet")

adv = q.rolling(30, min_periods=20).median().iloc[-1]
top = adv[(c.notna().sum() >= 60) & c.iloc[-1].notna()].nlargest(100).index
print(f"{len(excl & set(c.columns))} non-crypto perpetuals removed ({len(listed)} in tradfi.txt, {len(test)} by the weekend test, "
      f"{len(test - listed)} by the test only); {c.shape[1] - len(excl & set(c.columns))} kept")
print(f"today's top 100 removed by the test only: {sorted(s[:-4] for s in set(top) & (test - listed))}")
print(f"today's top 100 kept with weekend ratio < 0.55 (check by name): "
      f"{ {s[:-4]: round(ratio[s], 2) for s in top if s not in excl and ratio[s] < 0.55} }")
