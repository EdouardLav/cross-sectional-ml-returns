import argparse, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_binance import s3_list, get

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--end", required=True)
a = ap.parse_args()
S, D = Path(a.src), Path(a.dst)
RK, RF = D / "raw_klines", D / "raw_funding"
RK.mkdir(parents=True, exist_ok=True)
RF.mkdir(parents=True, exist_ok=True)

src = {f: pd.read_parquet(S / f"{f}.parquet") for f in ["close", "quote_volume", "funding"]}
for x in src.values():
    x.index = pd.to_datetime(x.index)
s_end = src["close"].index[-1]
days = pd.date_range(s_end + pd.Timedelta(days=1), a.end)
months = sorted({d.strftime("%Y-%m") for d in days})
qv = src["quote_volume"]
lastq = qv.where(qv > 0).apply(lambda x: x.last_valid_index())
active = set(lastq.index[lastq >= s_end - pd.Timedelta(days=7)])
inactive = set(lastq.index[lastq < s_end - pd.Timedelta(days=7)])
_, pref = s3_list("data/futures/um/daily/klines/")
listed = {p.rstrip("/").split("/")[-1] for p in pref}
listed = {s for s in listed if s.endswith("USDT") and "_" not in s}
new = listed - set(qv.columns)
syms = sorted(active | (listed - inactive))
print(f"{len(active)} active symbols, {len(new)} new listings, {len(syms)} to update; {len(days)} days to fetch "
      f"({days[0].date()} to {days[-1].date()})", flush=True)


def ts(x):
    x = pd.to_numeric(x, errors="coerce")
    return pd.to_datetime(x, unit="us" if x.iloc[0] > 1e14 else "ms").dt.normalize()


def job(sym):
    out = RK / f"{sym}.parquet"
    old = pd.read_parquet(out) if out.exists() else None
    have = set(old.index) if old is not None else set()
    parts = [get(f"data/futures/um/daily/klines/{sym}/1d/{sym}-1d-{d:%Y-%m-%d}.zip") for d in days if d not in have]
    parts = [p for p in parts if p is not None]
    if parts:
        k = pd.concat(parts).iloc[:, [0, 4, 7]]
        k = pd.DataFrame({"close": pd.to_numeric(k.iloc[:, 1], errors="coerce").values,
                          "quote_volume": pd.to_numeric(k.iloc[:, 2], errors="coerce").values}, index=ts(k.iloc[:, 0]).values)
        k = pd.concat([old, k]) if old is not None else k
        k[~k.index.duplicated()].sort_index().to_parquet(out)
    for m in months:
        fp = RF / f"{sym}_{m}.parquet"
        if fp.exists():
            continue
        f = get(f"data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip")
        if f is None:
            continue
        f = pd.Series(pd.to_numeric(f.iloc[:, -1], errors="coerce").values, index=ts(f.iloc[:, 0]).values)
        f.groupby(level=0).sum().rename("funding").to_frame().to_parquet(fp)
    return sym


with ThreadPoolExecutor(max_workers=8) as ex:
    for i, fu in enumerate(as_completed([ex.submit(job, s) for s in syms]), 1):
        fu.result()
        if i % 100 == 0:
            print(f"{i}/{len(syms)}", flush=True)

kl = {p.stem: pd.read_parquet(p) for p in RK.glob("*.parquet")}
fr = {}
for p in RF.glob("*.parquet"):
    fr.setdefault(p.stem.rsplit("_", 1)[0], []).append(pd.read_parquet(p).funding)
add = {"close": pd.DataFrame({s: k.close for s, k in kl.items()}),
       "quote_volume": pd.DataFrame({s: k.quote_volume for s, k in kl.items()}),
       "funding": pd.DataFrame({s: pd.concat(v).groupby(level=0).sum() for s, v in fr.items()})}
end = add["close"]["BTCUSDT"].loc[:a.end].dropna().index.max()
idx = pd.date_range(src["close"].index[0], end)
D.mkdir(parents=True, exist_ok=True)
for f, x in src.items():
    y = pd.concat([x, add[f].reindex(days).loc[:end]]).reindex(idx)
    y.index = y.index.astype("datetime64[ms]")
    y.to_parquet(D / f"{f}.parquet")
c = add["close"].reindex(days)
print(f"data through {end.date()}" + ("" if end == pd.Timestamp(a.end) else f"  (asked {a.end}: later days not published yet)"))
print(f"symbols with a close on {end.date()}: {int(c.loc[end].notna().sum())} (active at {s_end.date()}: {len(active)})")
fe = add["funding"].dropna(how="all").index.max() if len(add["funding"]) else None
print(f"funding through {fe.date() if fe is not None else 'none'}; later days count as zero funding")
