import argparse
from pathlib import Path
import numpy as np
import pandas as pd

STABLE = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "PAX", "DAI", "USDE", "EUR", "GBP", "AEUR", "USD1", "XUSD", "BFUSD",
          "RLUSD", "USTC", "USDS", "USDSB", "SUSD", "UST", "BKRW", "IDRT", "BIDR", "TRY", "BRL", "RUB", "UAH", "NGN", "ZAR"}

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ref", default="")
ap.add_argument("--start", default="2022-01-01")
ap.add_argument("--fwd_start", default="2026-09-27")
ap.add_argument("--fee_bps", type=float, default=5.0)
ap.add_argument("--impact_k", type=float, default=1.0)
ap.add_argument("--band", type=float, default=0.4)
ap.add_argument("--trade_rate", type=float, default=0.25)
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--aum", type=float, default=1e7)
a = ap.parse_args()
D, OUT = Path(a.data), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
FEE, A = a.fee_bps / 1e4, a.aum

close = pd.read_parquet(D / "close.parquet")
close.index = pd.to_datetime(close.index)
qv = pd.read_parquet(D / "quote_volume.parquet").reindex_like(close)
fund = pd.read_parquet(D / "funding.parquet").reindex_like(close).fillna(0.0)
base = lambda s: s[:-4] if s.endswith("USDT") else s
close = close[[c for c in close if base(c) not in STABLE]]
qv = qv[close.columns]
close = close.where(qv > 0)
adv0 = qv.rolling(30, min_periods=20).median()
spread = (2 * 2e-4 * np.sqrt(1e8 / adv0)).clip(lower=1e-4, upper=5e-3)
fund = fund[close.columns]
ret = close.pct_change(fill_method=None)
ret = ret.where(ret.abs() < 3)
btc = "BTCUSDT"
vol = ret.rolling(30, min_periods=20).std()
mx = ret.rolling(30, min_periods=20).max()
beta = ret.rolling(90, min_periods=60).cov(ret[btc]).div(ret[btc].rolling(90, min_periods=60).var(), axis=0)
adv = qv.rolling(30, min_periods=20).median()
age = close.notna().cumsum()
last = close.apply(lambda s: s.last_valid_index())
sundays = close.index[(close.index.dayofweek == 6) & (close.index >= a.start)]
hist = sundays[sundays <= close.index[-8]]
t_live = sundays[-1]


def fwd(s, t):
    stop = min(t + pd.Timedelta(days=7), last[s])
    seg = close[s].loc[t:stop].dropna()
    if len(seg) < 2:
        return np.nan
    return seg.iloc[-1] / seg.iloc[0] - 1 - fund[s].loc[t + pd.Timedelta(days=1):stop].sum()


def universe(t, live):
    elig = (age.loc[t] >= 60) & close.loc[t].notna() & adv.loc[t].notna() & vol.loc[t].notna()
    syms = adv.loc[t][elig].nlargest(a.n).index
    g = pd.DataFrame({"vol": vol.loc[t, syms], "max": mx.loc[t, syms]})
    g["fwd"] = np.nan if live else [fwd(s, t) for s in syms]
    g = g.dropna(subset=["vol", "max"] if live else ["fwd", "vol", "max"])
    rk = lambda x: (x.rank() - x.rank().mean()) / x.rank().std()
    g["score"] = -(rk(g.vol) + rk(g["max"])) / 2
    return g


rows, W = [], {}
prev, prev_a, pL, pS = pd.Series(dtype=float), pd.Series(dtype=float), set(), set()
for t in list(hist) + [t_live]:
    live = t == t_live
    g = universe(t, live)
    if len(g) < 20:
        continue
    pct = g.score.rank(pct=True)
    L = set(pct.index[pct > 0.8]) | {x for x in pL if x in pct.index and pct[x] > 1 - a.band}
    S = set(pct.index[pct <= 0.2]) | {x for x in pS if x in pct.index and pct[x] <= a.band}
    pL, pS = L, S
    L, S = np.array(sorted(L)), np.array(sorted(S))
    iv = 1 / vol.loc[t]
    wt = pd.concat([iv[L] / iv[L].sum(), -iv[S] / iv[S].sum()]).groupby(level=0).sum()
    u = wt.index.union(prev_a.index)
    act = prev_a.reindex(u).fillna(0) + a.trade_rate * (wt.reindex(u).fillna(0) - prev_a.reindex(u).fillna(0))
    act = act[act.abs() >= 1e-3]
    if live:
        act = act[close.loc[t].reindex(act.index).notna()]
    else:
        fa = g.fwd.reindex(act.index)
        for s in fa.index[fa.isna()]:
            fa[s] = fwd(s, t)
        act = act[fa.notna()]
    prev_a = act.copy()
    w = act.copy()
    w[btc] = w.get(btc, 0.0) - (w * beta.loc[t].reindex(w.index).fillna(1.0)).sum()
    dw = (w.reindex(w.index.union(prev.index)).fillna(0) - prev.reindex(w.index.union(prev.index)).fillna(0)).abs()
    sp = spread.loc[t].reindex(dw.index).fillna(spread.loc[t].median())
    sg = vol.loc[t].reindex(dw.index).fillna(vol.loc[t].median())
    ad = adv.loc[t].reindex(dw.index).fillna(adv.loc[t].median())
    cost = dw * (FEE + sp / 2 + a.impact_k * sg * np.sqrt(dw * A / ad))
    if not live:
        f = fa.reindex(w.index)
        if pd.isna(f.get(btc)):
            f[btc] = fwd(btc, t)
        rows.append({"date": t, "gross": (w * f).sum(), "cost": cost.sum(), "net": (w * f).sum() - cost.sum(),
                     "btc": f[btc], "turnover": dw.sum(), "n": len(g)})
    W[t] = (w, prev, cost)
    prev = w

b = pd.DataFrame(rows).set_index("date")
pd.DataFrame({t: v[0] for t, v in W.items()}).T.to_parquet(OUT / "positions.parquet")
for r in [x for x in a.ref.split(",") if x]:
    ref = pd.read_csv(r, index_col=0, parse_dates=True)
    common = b.index.intersection(ref.index)
    d = max((b.gross[common] - ref.gross[common]).abs().max(), (b.net[common] - ref[f"net_{int(A / 1e6)}m"][common]).abs().max())
    print(f"check vs {r}: {len(common)} common weeks, max abs difference {d:.1e} ({'identical' if d < 1e-10 else 'MISMATCH'})")

fw = b[b.index >= a.fwd_start].copy()
fw["week"] = [f"{t:%d %b} to {t + pd.Timedelta(days=7):%d %b %Y}" for t in fw.index]
fw.to_csv(OUT / "forward_weeks.csv")
print(f"\n== Forward test, frozen spec, weeks after the freeze (net at ${A / 1e6:.0f}M, taker {a.fee_bps:.0f} bps)")
if len(fw):
    print(fw[["week", "gross", "net", "btc", "turnover", "n"]].to_string(
        formatters={"gross": "{:+.2%}".format, "net": "{:+.2%}".format, "btc": "{:+.2%}".format, "turnover": "{:.0%}".format}))
    print(f"cumulative net {(1 + fw.net).prod() - 1:+.2%} over {len(fw)} weeks; funding data through "
          f"{fund.index[(fund != 0).any(axis=1)][-1].date()}")
gap = close.index[-1] != t_live
w, pw, cost = W[t_live]
bk = pd.DataFrame({"weight": w, "usd": w * A}).reindex(w.index.union(pw.index)).fillna(0)
bk["prev_usd"] = pw.reindex(bk.index).fillna(0) * A
bk["trade_usd"] = bk.usd - bk.prev_usd
bk["side"] = np.where(bk.index == btc, "hedge", np.where(bk.weight > 0, "long", np.where(bk.weight < 0, "short", "close")))
bk["est_cost_usd"] = cost.reindex(bk.index).fillna(0) * A
bk = bk.sort_values(["side", "usd"], ascending=[True, False])
bk.to_csv(OUT / f"book_{t_live:%Y-%m-%d}.csv")
print(f"\n== Target book at the {t_live:%a %d %b %Y} close (data through {close.index[-1].date()}), AUM ${A / 1e6:.0f}M"
      + ("  (note: data does not end on this Sunday)" if gap else ""))
print(bk.round({"weight": 4, "usd": 0, "prev_usd": 0, "trade_usd": 0, "est_cost_usd": 0}).to_string())
lg, sh = bk[bk.side == "long"], bk[bk.side == "short"]
print(f"{len(lg)} longs ${lg.usd.sum():,.0f}, {len(sh)} shorts ${sh.usd.sum():,.0f}, BTC hedge ${bk.usd.get(btc, 0):,.0f}; "
      f"traded ${bk.trade_usd.abs().sum():,.0f} ({bk.trade_usd.abs().sum() / A:.0%} of AUM), "
      f"estimated cost ${bk.est_cost_usd.sum():,.0f} ({bk.est_cost_usd.sum() / A * 1e4:.1f} bps of AUM)")
