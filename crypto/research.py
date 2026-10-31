import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

D = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[3]) if len(sys.argv) > 3 else 50
COST = 10e-4
STABLE = {"USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "USDPUSDT", "DAIUSDT", "USDEUSDT", "EURUSDT",
          "GBPUSDT", "AEURUSDT", "USD1USDT", "XUSDUSDT", "BFUSDUSDT", "RLUSDUSDT", "USTCUSDT"}

close = pd.read_parquet(D / "close.parquet")
qv = pd.read_parquet(D / "quote_volume.parquet")
fund = pd.read_parquet(D / "funding.parquet")
close = close.drop(columns=[c for c in close if c in STABLE])
qv, fund = qv.reindex_like(close), fund.reindex_like(close).fillna(0.0)
close.index = pd.to_datetime(close.index)
qv.index = fund.index = close.index
close = close.where(qv > 0)
close = close.loc["2020-01-01":"2026-09-30"]
qv, fund = qv.loc[close.index], fund.loc[close.index]
ret = close.pct_change(fill_method=None)
ret = ret.where(ret.abs() < 3)

last = close.apply(lambda c: c.last_valid_index())
first = close.apply(lambda c: c.first_valid_index())
dates = close.index[close.index.dayofweek == 6]
dates = dates[(dates >= "2020-06-01") & (dates <= close.index[-8])]

qv30 = qv.rolling(30, min_periods=20).median()
age = close.notna().cumsum()
lr = np.log1p(ret.fillna(0)).where(close.notna())


def window_ret(a, b):
    return np.expm1(lr.rolling(b - a, min_periods=int(0.8 * (b - a))).sum().shift(a))


sig = {
    "mom_28_7": window_ret(7, 28),
    "rev_7": window_ret(0, 7),
    "vol_30": ret.rolling(30, min_periods=20).std(),
    "max_30": ret.rolling(30, min_periods=20).max(),
    "funding_7": fund.rolling(7, min_periods=5).mean(),
    "volume_ratio": qv.rolling(7, min_periods=5).mean() / qv.rolling(30, min_periods=20).mean(),
    "size": np.log(qv30),
}
SIGNS = {"mom_28_7": 1, "rev_7": -1, "vol_30": -1, "max_30": -1, "funding_7": -1, "volume_ratio": 0, "size": 0}

rows = []
for t in dates:
    elig = (age.loc[t] >= 60) & close.loc[t].notna() & qv30.loc[t].notna()
    univ = qv30.loc[t][elig].nlargest(N).index
    end = t + pd.Timedelta(days=7)
    for s in univ:
        stop = min(end, last[s])
        if stop <= t:
            continue
        seg = close[s].loc[t:stop].dropna()
        if len(seg) < 2:
            continue
        p0, p1 = seg.iloc[0], seg.iloc[-1]
        fr = fund.loc[t + pd.Timedelta(days=1):stop, s].sum()
        r = {"date": t, "sym": s, "fwd_price": p1 / p0 - 1, "fwd_funding": fr,
             "delisted": stop < end}
        for k, v in sig.items():
            r[k] = v.at[t, s]
        rows.append(r)
pan = pd.DataFrame(rows).dropna(subset=list(sig) + ["fwd_price"])
pan["fwd"] = pan.fwd_price - pan.fwd_funding
feat = list(sig)
z = pan.groupby("date")[feat].transform(lambda x: (x.rank() - x.rank().mean()) / x.rank().std())
pan[[f"z_{k}" for k in feat]] = z.values
pan.to_parquet(OUT / "panel.parquet")

ic = pan.groupby("date").apply(lambda g: pd.Series({k: spearmanr(g[k], g.fwd).statistic for k in feat}))
ics = pd.DataFrame({"mean_ic": ic.mean(), "t": ic.mean() / ic.std() * np.sqrt(ic.count()), "hit": (ic > 0).mean()})
ics.to_csv(OUT / "univariate_ic.csv")

pan["composite"] = sum(SIGNS[k] * pan[f"z_{k}"] for k in feat if SIGNS[k])
oos_start = pd.Timestamp("2022-01-01")
preds = []
for y in range(2022, 2027):
    tr = pan[pan.date < pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=7)]
    te = pan[(pan.date >= pd.Timestamp(f"{y}-01-01")) & (pan.date < pd.Timestamp(f"{y + 1}-01-01"))]
    if te.empty:
        continue
    yt = tr.groupby("date").fwd.rank(pct=True) - 0.5
    m = Ridge(alpha=1000.0).fit(tr[[f"z_{k}" for k in feat]], yt)
    p = te[["date", "sym"]].copy()
    p["ridge"] = m.predict(te[[f"z_{k}" for k in feat]])
    preds.append(p)
pan = pan.merge(pd.concat(preds), on=["date", "sym"], how="left")
pan.to_parquet(OUT / "panel.parquet")


def book(df, score, q=0.2):
    out, prev_l, prev_s = [], set(), set()
    for t, g in df.dropna(subset=[score]).groupby("date"):
        g = g.sort_values(score)
        k = max(int(len(g) * q), 1)
        S, L = g.head(k), g.tail(k)
        ls = L.fwd.mean() - S.fwd.mean()
        tl, ts = set(L.sym), set(S.sym)
        to = 0.5 * (len(tl - prev_l) / len(tl) + len(ts - prev_s) / len(ts)) if prev_l else 1.0
        out.append({"date": t, "long": L.fwd.mean(), "short": S.fwd.mean(), "ls": ls, "turnover": to,
                    "net": ls - to * 2 * COST * 2, "n": len(g)})
        prev_l, prev_s = tl, ts
    return pd.DataFrame(out).set_index("date")


btc = close["BTCUSDT"].reindex(dates).pct_change().shift(-1)
summ = []
for name, score, start in [("composite_full", "composite", None), ("composite_oos", "composite", oos_start),
                           ("ridge_oos", "ridge", oos_start)] + [(f"{k}_oos", f"z_{k}", oos_start) for k in feat]:
    sub = pan if start is None else pan[pan.date >= start]
    sc = score
    if score.startswith("z_") and SIGNS[score[2:]] < 0:
        sub = sub.assign(**{score: -sub[score]})
    b = book(sub, sc)
    b.to_csv(OUT / f"book_{name}.csv")
    x = sm.add_constant(btc.reindex(b.index).fillna(0))
    reg = sm.OLS(b.ls, x).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    ann = 52
    summ.append({"book": name, "weeks": len(b), "ann_ret": b.ls.mean() * ann, "ann_vol": b.ls.std() * np.sqrt(ann),
                 "sharpe": b.ls.mean() / b.ls.std() * np.sqrt(ann), "sharpe_net": b.net.mean() / b.net.std() * np.sqrt(ann),
                 "turnover": b.turnover.mean(), "beta_btc": reg.params.iloc[1], "alpha_t": reg.tvalues.iloc[0],
                 "maxdd": ((1 + b.ls).cumprod() / (1 + b.ls).cumprod().cummax() - 1).min(),
                 "hit": (b.ls > 0).mean(), "long": b.long.mean() * ann, "short": b.short.mean() * ann})
summ = pd.DataFrame(summ).set_index("book")
summ.to_csv(OUT / "summary.csv")
pd.set_option("display.width", 220)
print(f"universe top {N}, rebalance weekly, {pan.date.nunique()} weeks, {pan.sym.nunique()} symbols ever in universe, "
      f"{int(pan.delisted.sum())} delisting exits")
print(ics.round(3).to_string())
print(summ.round(3).to_string())
