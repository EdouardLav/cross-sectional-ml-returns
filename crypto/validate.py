import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

STABLE = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "PAX", "DAI", "USDE", "EUR", "GBP", "AEUR", "USD1", "XUSD", "BFUSD",
          "RLUSD", "USTC", "USDS", "USDSB", "SUSD", "UST", "BKRW", "IDRT", "BIDR", "TRY", "BRL", "RUB", "UAH", "NGN", "ZAR"}

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--start", required=True)
ap.add_argument("--end", required=True)
ap.add_argument("--fee_bps", type=float, required=True)
ap.add_argument("--sizes", default="30,50,100")
ap.add_argument("--impact_k", type=float, default=1.0)
ap.add_argument("--band", type=float, default=0.4)
ap.add_argument("--trade_rate", type=float, default=1.0)
a = ap.parse_args()
D, OUT = Path(a.data), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
FEE = a.fee_bps / 1e4
AUMS = [1e6, 1e7, 5e7]

close = pd.read_parquet(D / "close.parquet")
close.index = pd.to_datetime(close.index)
qv = pd.read_parquet(D / "quote_volume.parquet").reindex_like(close)
fund = pd.read_parquet(D / "funding.parquet").reindex_like(close).fillna(0.0) if (D / "funding.parquet").exists() else None
base = lambda s: s[:-4] if s.endswith("USDT") else s
close = close[[c for c in close if base(c) not in STABLE]]
qv = qv[close.columns]
close = close.where(qv > 0)
adv0 = qv.rolling(30, min_periods=20).median()
spread = (2 * 2e-4 * np.sqrt(1e8 / adv0)).clip(lower=1e-4, upper=5e-3)
if fund is not None:
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
dates = close.index[close.index.dayofweek == 6]
dates = dates[(dates >= a.start) & (dates <= a.end) & (dates <= close.index[-8])]


def fwd(s, t):
    stop = min(t + pd.Timedelta(days=7), last[s])
    seg = close[s].loc[t:stop].dropna()
    if len(seg) < 2:
        return np.nan
    r = seg.iloc[-1] / seg.iloc[0] - 1
    if fund is not None:
        r -= fund[s].loc[t + pd.Timedelta(days=1):stop].sum()
    return r


def run(N):
    rows = []
    for t in dates:
        elig = (age.loc[t] >= 60) & close.loc[t].notna() & adv.loc[t].notna() & vol.loc[t].notna()
        for s in adv.loc[t][elig].nlargest(N).index:
            rows.append({"date": t, "sym": s, "fwd": fwd(s, t), "vol": vol.at[t, s], "max": mx.at[t, s],
                         "funding": fund[s].loc[:t].tail(7).mean() if fund is not None else np.nan})
    p = pd.DataFrame(rows).dropna(subset=["fwd", "vol", "max"])
    rk = lambda x: (x.rank() - x.rank().mean()) / x.rank().std()
    p["score"] = -(p.groupby("date").vol.transform(rk) + p.groupby("date")["max"].transform(rk)) / 2
    ic = p.groupby("date").apply(lambda g: pd.Series({
        "score": spearmanr(g.score, g.fwd).statistic, "vol": spearmanr(g.vol, g.fwd).statistic,
        "max": spearmanr(g["max"], g.fwd).statistic,
        "funding": spearmanr(g.funding, g.fwd).statistic if fund is not None else np.nan}))
    ic = ic[p.groupby("date").size().reindex(ic.index) >= 20]
    icsum = pd.DataFrame({"mean": ic.mean(), "t": ic.mean() / ic.std() * np.sqrt(ic.count()), "weeks": ic.count()})
    bk, prev, prev_a, pL, pS = [], pd.Series(dtype=float), pd.Series(dtype=float), set(), set()
    for t, g in p.groupby("date"):
        if len(g) < 20:
            continue
        pct = g.set_index("sym").score.rank(pct=True)
        L = set(pct.index[pct > 0.8]) | {x for x in pL if x in pct.index and pct[x] > 1 - a.band}
        S = set(pct.index[pct <= 0.2]) | {x for x in pS if x in pct.index and pct[x] <= a.band}
        pL, pS = L, S
        L, S = np.array(sorted(L)), np.array(sorted(S))
        iv = 1 / vol.loc[t]
        wt = pd.concat([iv[L] / iv[L].sum(), -iv[S] / iv[S].sum()]).groupby(level=0).sum()
        u = wt.index.union(prev_a.index)
        act = prev_a.reindex(u).fillna(0) + a.trade_rate * (wt.reindex(u).fillna(0) - prev_a.reindex(u).fillna(0))
        act = act[act.abs() >= 1e-3]
        fa = g.set_index("sym").fwd.reindex(act.index)
        for s in fa.index[fa.isna()]:
            fa[s] = fwd(s, t)
        act = act[fa.notna()]
        prev_a = act.copy()
        w = act.copy()
        h = -(w * beta.loc[t].reindex(w.index).fillna(1.0)).sum()
        w[btc] = w.get(btc, 0.0) + h
        f = fa.reindex(w.index)
        if pd.isna(f.get(btc)):
            f[btc] = fwd(btc, t)
        gross = (w * f).sum()
        dw = (w.reindex(w.index.union(prev.index)).fillna(0) - prev.reindex(w.index.union(prev.index)).fillna(0)).abs()
        sp = spread.loc[t].reindex(dw.index).fillna(spread.loc[t].median())
        sg = vol.loc[t].reindex(dw.index).fillna(vol.loc[t].median())
        ad = adv.loc[t].reindex(dw.index).fillna(adv.loc[t].median())
        r = {"date": t, "gross": gross, "turnover": dw.sum(), "btc": f[btc], "n": len(g)}
        for A in AUMS:
            r[f"cost_{int(A / 1e6)}m"] = (dw * (FEE + sp / 2 + a.impact_k * sg * np.sqrt(dw * A / ad))).sum()
            r[f"net_{int(A / 1e6)}m"] = gross - r[f"cost_{int(A / 1e6)}m"]
        bk.append(r)
        prev = w
    b = pd.DataFrame(bk).set_index("date")
    shp = lambda x: x.mean() / x.std() * np.sqrt(52)
    reg = lambda y: sm.OLS(y, sm.add_constant(b.btc)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    rg, rn = reg(b.gross), reg(b.net_10m)
    summ = {"N": N, "weeks": len(b), "avg_names": b.n.mean(), "sharpe_gross": shp(b.gross), "alpha_t_gross": rg.tvalues.iloc[0],
            "beta_btc": rg.params.iloc[1], "turnover": b.turnover.mean(), "ann_vol": b.gross.std() * np.sqrt(52),
            "maxdd_gross": ((1 + b.gross).cumprod() / (1 + b.gross).cumprod().cummax() - 1).min()}
    for A in AUMS:
        m = int(A / 1e6)
        summ[f"cost_bps_week_{m}m"] = b[f"cost_{m}m"].mean() * 1e4
        summ[f"sharpe_net_{m}m"] = shp(b[f"net_{m}m"])
    summ["alpha_t_net_10m"] = rn.tvalues.iloc[0]
    summ["yearly_net_10m"] = b.net_10m.groupby(b.index.year).apply(shp).round(2).to_dict()
    b.to_csv(OUT / f"book_n{N}.csv")
    icsum.to_csv(OUT / f"ic_n{N}.csv")
    return icsum, summ


pd.set_option("display.width", 250, "display.max_colwidth", 200)
allsum = []
for N in [int(x) for x in a.sizes.split(",")]:
    icsum, summ = run(N)
    allsum.append(summ)
    print(f"\n== top {N}: rank IC with next-week return")
    print(icsum.round(3).to_string())
res = pd.DataFrame(allsum).set_index("N")
res.to_csv(OUT / "summary.csv")
print("\n== BTC-hedged inverse-vol quintile book (score = low volatility and low MAX)")
print(res.drop(columns="yearly_net_10m").round(2).T.to_string())
print(res.yearly_net_10m.to_string())
ic50 = pd.read_csv(OUT / "ic_n50.csv", index_col=0) if (OUT / "ic_n50.csv").exists() else None
if ic50 is not None:
    v1 = ic50.loc["score", "t"] >= 2
    v2 = res.loc[50, "alpha_t_gross"] >= 2
    print(f"\nRULE V1 (score IC t >= 2, top 50): {'PASS' if v1 else 'FAIL'} ({ic50.loc['score', 't']:.2f})")
    print(f"RULE V2 (gross alpha t vs BTC >= 2, top 50): {'PASS' if v2 else 'FAIL'} ({res.loc[50, 'alpha_t_gross']:.2f})")
    print(f"RULE C1 (net Sharpe >= 0.5 at $10M, top 50): {'PASS' if res.loc[50, 'sharpe_net_10m'] >= 0.5 else 'FAIL'} "
          f"({res.loc[50, 'sharpe_net_10m']:.2f})")
