import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--book", required=True)
ap.add_argument("--start", required=True)
ap.add_argument("--end", required=True)
ap.add_argument("--trials", default="")
a = ap.parse_args()
D = Path(a.data)
STABLE = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "PAX", "DAI", "USDE", "EUR", "GBP", "AEUR", "USD1", "XUSD", "BFUSD",
          "RLUSD", "USTC", "USDS", "USDSB", "SUSD", "UST", "BKRW", "IDRT", "BIDR", "TRY", "BRL", "RUB", "UAH", "NGN", "ZAR"}

close = pd.read_parquet(D / "close.parquet")
close.index = pd.to_datetime(close.index)
qv = pd.read_parquet(D / "quote_volume.parquet").reindex_like(close)
fund = pd.read_parquet(D / "funding.parquet").reindex_like(close).fillna(0.0) if (D / "funding.parquet").exists() else None
close = close[[c for c in close if (c[:-4] if c.endswith("USDT") else c) not in STABLE]]
qv = qv[close.columns]
close = close.where(qv > 0)
if fund is not None:
    fund = fund[close.columns]
ret = close.pct_change(fill_method=None)
ret = ret.where(ret.abs() < 3)
vol = ret.rolling(30, min_periods=20).std()
mx = ret.rolling(30, min_periods=20).max()
adv = qv.rolling(30, min_periods=20).median()
mom = close / close.shift(21) - 1
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


rows = []
for t in dates:
    elig = (age.loc[t] >= 60) & close.loc[t].notna() & adv.loc[t].notna() & vol.loc[t].notna()
    for s in adv.loc[t][elig].nlargest(100).index:
        rows.append({"date": t, "sym": s, "fwd": fwd(s, t), "vol": vol.at[t, s], "max": mx.at[t, s],
                     "adv": adv.at[t, s], "mom": mom.at[t, s]})
p = pd.DataFrame(rows).dropna(subset=["fwd", "vol", "max", "adv"])
rk = lambda x: (x.rank() - x.rank().mean()) / x.rank().std()
p["score"] = -(p.groupby("date").vol.transform(rk) + p.groupby("date")["max"].transform(rk)) / 2


def spread(g, col, hi_minus_lo=True):
    g = g.dropna(subset=[col])
    q = g[col].rank(pct=True)
    hi, lo = g.fwd[q > 0.7].mean(), g.fwd[q <= 0.3].mean()
    return hi - lo if hi_minus_lo else lo - hi


fac = p.groupby("date").apply(lambda g: pd.Series({"LIQ": spread(g, "adv", False), "MOM": spread(g, "mom")}))
fac["BTC"] = pd.Series({t: fwd("BTCUSDT", t) for t in fac.index})
book = pd.read_csv(a.book, index_col=0, parse_dates=True)
df = book.join(fac, how="inner").dropna(subset=["BTC", "LIQ", "MOM"])

print(f"\n== A1: factor attribution of the frozen book ({len(df)} weeks)")
for y in ["gross", "net_10m"]:
    m = sm.OLS(df[y], sm.add_constant(df[["BTC", "LIQ", "MOM"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    print(f"{y:8s} alpha {m.params['const'] * 52:+.1%}/yr (t {m.tvalues['const']:.2f}) | "
          + " | ".join(f"{k} {m.params[k]:+.2f} (t {m.tvalues[k]:.2f})" for k in ["BTC", "LIQ", "MOM"]) + f" | R2 {m.rsquared:.2f}")
    if y == "gross":
        a1 = m.tvalues["const"]
print(f"factor Sharpe (ann.): LIQ {fac.LIQ.mean() / fac.LIQ.std() * np.sqrt(52):.2f}, MOM {fac.MOM.mean() / fac.MOM.std() * np.sqrt(52):.2f}; "
      f"corr(book gross, LIQ) {df.gross.corr(df.LIQ):.2f}, corr(book gross, MOM) {df.gross.corr(df.MOM):.2f}")

print("\n== A2: rank IC of the score within liquidity terciles of the top-100 universe")
p["terc"] = p.groupby("date").adv.transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=["least liquid", "middle", "most liquid"]))
ics = p.groupby(["date", "terc"], observed=True).apply(lambda g: spearmanr(g.score, g.fwd).statistic).unstack()
res = pd.DataFrame({"mean_ic": ics.mean(), "t": ics.mean() / ics.std() * np.sqrt(ics.count()), "weeks": ics.count()})
print(res.round(3).to_string())
a2 = (res.mean_ic > 0).all() and (res.t >= 2).sum() >= 2

a3 = None
if a.trials:
    srs = []
    for f in sorted({f for pat in a.trials.split(",") for f in glob.glob(pat)}):
        s = pd.read_csv(f, index_col=0)
        for c in ["sharpe_net_10m", "sharpe_net", "sharpe"]:
            if c in s:
                srs += list(s[c].dropna().values)
                break
    srs = np.array(srs)
    N, V = len(srs), np.var(srs / np.sqrt(52), ddof=1)
    g = 0.5772156649
    sr0 = np.sqrt(V) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e)))
    x = book.net_10m.dropna()
    sr, T = x.mean() / x.std(), len(x)
    sk, ku = x.skew(), x.kurt() + 3
    dsr = norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    print(f"\n== A3: Deflated Sharpe: {N} configurations counted, annualized Sharpe {sr * np.sqrt(52):.2f}, "
          f"expected best of {N} null trials {sr0 * np.sqrt(52):.2f}, DSR = {dsr:.2f}")
    a3 = dsr

print(f"\nRULE A1 (gross alpha t >= 2 after BTC, LIQ, MOM): {'PASS' if a1 >= 2 else 'FAIL'} ({a1:.2f})")
print(f"RULE A2 (IC > 0 in all terciles, t >= 2 in two): {'PASS' if a2 else 'FAIL'}")
if a3 is not None:
    print(f"A3 Deflated Sharpe (reported): {a3:.2f}")
