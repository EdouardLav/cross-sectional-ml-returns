import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
COST = 10e-4
close = pd.read_parquet(D / "close.parquet")
qv = pd.read_parquet(D / "quote_volume.parquet").reindex_like(close)
fund = pd.read_parquet(D / "funding.parquet").reindex_like(close).fillna(0.0)
close.index = qv.index = fund.index = pd.to_datetime(close.index)
close = close.where(qv > 0)
ret = close.pct_change(fill_method=None)
ret = ret.where(ret.abs() < 3)
btc = ret["BTCUSDT"]
beta = ret.rolling(90, min_periods=60).cov(btc).div(btc.rolling(90, min_periods=60).var(), axis=0)
vol = ret.rolling(30, min_periods=20).std()
pan = pd.read_parquet(OUT / "panel.parquet")

ic_price = pan.groupby("date").apply(lambda g: spearmanr(g.funding_7, g.fwd_price).statistic)
ic_fund = pan.groupby("date").apply(lambda g: spearmanr(g.funding_7, -g.fwd_funding).statistic)
print(f"funding_7 IC on price only: {ic_price.mean():.3f} (t {ic_price.mean() / ic_price.std() * np.sqrt(len(ic_price)):.2f}); "
      f"on funding paid: {ic_fund.mean():.3f}")


def btc_fwd(t):
    seg = close["BTCUSDT"].loc[t:t + pd.Timedelta(days=7)].dropna()
    return seg.iloc[-1] / seg.iloc[0] - 1 - fund["BTCUSDT"].loc[t + pd.Timedelta(days=1):t + pd.Timedelta(days=7)].sum()


def risk_book(df, score, q=0.2):
    rows, prev = [], pd.Series(dtype=float)
    for t, g in df.dropna(subset=[score]).groupby("date"):
        g = g.sort_values(score)
        k = max(int(len(g) * q), 1)
        S, L = g.head(k), g.tail(k)
        iv = 1 / vol.loc[t].reindex(pd.concat([L.sym, S.sym])).clip(lower=0.01)
        wl = iv[L.sym.values] / iv[L.sym.values].sum()
        ws = -iv[S.sym.values] / iv[S.sym.values].sum()
        w = pd.concat([wl, ws]).groupby(level=0).sum()
        b = beta.loc[t].reindex(w.index).fillna(1.0)
        h = -(w * b).sum()
        w["HEDGE_BTC"] = h
        f = g.set_index("sym").fwd.reindex(w.index)
        f["HEDGE_BTC"] = btc_fwd(t)
        gross = (w * f).sum()
        to = (w.reindex(w.index.union(prev.index)).fillna(0) - prev.reindex(w.index.union(prev.index)).fillna(0)).abs().sum()
        rows.append({"date": t, "gross": gross, "turnover": to, "net": gross - to * COST, "hedge": h,
                     "long": (wl * f[wl.index]).sum(), "short": (ws * f[ws.index]).sum()})
        prev = w
    return pd.DataFrame(rows).set_index("date")


bt = pd.Series({t: btc_fwd(t) for t in pan.date.unique()}).sort_index()
out = []
for name, sc, start in [("composite_full", "composite", None), ("composite_oos", "composite", "2022-01-01"),
                        ("ridge_oos", "ridge", "2022-01-01"), ("vol_only_oos", "z_vol_30", "2022-01-01")]:
    sub = pan if start is None else pan[pan.date >= start]
    if sc == "z_vol_30":
        sub = sub.assign(z_vol_30=-sub.z_vol_30)
    b = risk_book(sub, sc)
    b.to_csv(OUT / f"riskbook_{name}.csv")
    x = sm.add_constant(bt.reindex(b.index))
    reg = sm.OLS(b.net, x).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    yearly = b.net.groupby(b.index.year).apply(lambda r: r.mean() / r.std() * np.sqrt(52)).round(2).to_dict()
    out.append({"book": name, "weeks": len(b), "ann_ret_net": b.net.mean() * 52, "ann_vol": b.net.std() * np.sqrt(52),
                "sharpe_gross": b.gross.mean() / b.gross.std() * np.sqrt(52),
                "sharpe_net": b.net.mean() / b.net.std() * np.sqrt(52), "turnover": b.turnover.mean(),
                "beta_btc": reg.params.iloc[1], "alpha_t": reg.tvalues.iloc[0],
                "maxdd": ((1 + b.net).cumprod() / (1 + b.net).cumprod().cummax() - 1).min(), "yearly": yearly})
res = pd.DataFrame(out).set_index("book")
pd.set_option("display.width", 250, "display.max_colwidth", 120)
print(res.drop(columns="yearly").round(3).to_string())
print(res.yearly.to_string())
res.to_csv(OUT / "riskbook_summary.csv")
