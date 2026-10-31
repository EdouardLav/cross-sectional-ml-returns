import argparse, json, logging, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

from src.data_loader import compute_returns, winsorize_cross_section, get_sp100_tickers
from src.features import (compute_momentum_12_1, compute_reversal_1m, compute_realized_volatility,
                          compute_max_return, compute_return_skewness, compute_abnormal_volume,
                          standardize_cross_section, features_to_panel)
import src.walk_forward as wfm
from src.walk_forward import walk_forward
_bm = wfm.build_models
if os.environ.get("MODELS"):
    keep = os.environ["MODELS"].split(",")
    wfm.build_models = lambda: {k: v for k, v in _bm().items() if k in keep}
from src.evaluation import evaluate_all
from src.transaction_costs import run_transaction_cost_analysis
from src.risk_analysis import run_risk_analysis
from config import data_cfg

SECONDARY = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA", "DISCK": "DISCA", "UA": "UAA", "CMCSK": "CMCSA"}


def month_end(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp("M")
    return df.groupby(level=0).last()


ALT = {"FB": "META", "ANTM": "ELV", "ABC": "COR", "PKI": "RVTY", "RE": "EG", "FISV": "FI", "FI": "FISV",
       "WLTW": "WTW", "HRS": "LHX", "UTX": "RTX", "BBT": "TFC", "JEC": "J", "HCP": "DOC", "PEAK": "DOC",
       "SYMC": "GEN", "NLOK": "GEN", "CTL": "LUMN", "MYL": "VTRS", "COG": "CTRA", "HFC": "DINO", "LB": "BBWI",
       "FBHS": "FBIN", "CDAY": "DAY", "FLT": "CPAY", "BLL": "BALL", "GPS": "GAP", "DISCA": "WBD", "VIAC": "PARA",
       "VIAB": "PARA", "CBS": "PARA", "PARA": "PSKY", "DWDP": "DD", "KORS": "CPRI", "BHGE": "BKR", "TMK": "GL",
       "FII": "FHI", "ESV": "VAL", "WYND": "TNL", "ADS": "BFH", "AABA": "YHOO", "BK": "BNY", "GOOG": "GOOGL"}


def build_members(history_csv):
    df = pd.read_csv(history_csv, parse_dates=["date"]).sort_values("date")
    snap = df.set_index("date").tickers.str.split(",")
    months = pd.date_range("2004-06-30", "2025-12-31", freq="ME")
    rows = {m: set(snap[snap.index <= m].iloc[-1]) for m in months}
    allt = sorted(set().union(*rows.values()))
    mem = pd.DataFrame(False, index=months, columns=allt)
    for m, ts in rows.items():
        mem.loc[m, list(ts)] = True
    sym = pd.DataFrame([(t, t.replace(".", "-"), ALT.get(t, "")) for t in allt], columns=["ticker", "symbol", "alt"])
    return mem, sym


def load_inputs(datadir):
    datadir = Path(datadir)
    close = month_end(pd.read_parquet(datadir / "close.parquet"))
    adj = month_end(pd.read_parquet(datadir / "adj_close.parquet"))
    vol = month_end(pd.read_parquet(datadir / "volume.parquet"))
    mem, sym = build_members(datadir / "sp500_history.csv")
    return close, adj, vol, mem, sym


def resolve_symbols(mem, adj, sym):
    have = set(adj.columns)
    first = adj.apply(lambda c: c.first_valid_index())
    last = adj.apply(lambda c: c.last_valid_index())
    out = {}
    for t, s, a in sym.itertuples(index=False):
        months = mem.index[mem[t]] if t in mem else []
        if len(months) == 0:
            continue
        lo, hi = months.min(), months.max()
        best = None
        for c in [s, a]:
            if c and c in have and first[c] <= hi and last[c] >= lo:
                best = c
                break
        out[t] = best
    return out


def pit_members(mem, mapping, drop_secondary=True):
    cols = {}
    for t in mem.columns:
        s = mapping.get(t)
        if s is None:
            continue
        if drop_secondary and t in SECONDARY:
            continue
        cols.setdefault(s, mem[t])
        cols[s] = cols[s] | mem[t]
    return pd.DataFrame(cols)


def top_by_dollar_volume(candidates, close, vol, adj, n=100, window=6):
    dv = (close * vol).rolling(window, min_periods=window).mean()
    idx = candidates.index.intersection(dv.index)
    cols = candidates.columns.intersection(dv.columns)
    dv = dv.loc[idx, cols].where(candidates.loc[idx, cols])
    dv = dv.where(adj.loc[idx, cols].notna())
    rank = dv.rank(axis=1, ascending=False, method="first")
    return (rank <= n)


def run(universe, mask, adj, vol, french, outdir, start="2005-01-01", end="2025-12-31"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler("pipeline.log", mode="w")], force=True)
    tick = sorted(mask.columns[mask.any()])
    prices = adj.loc[start:end, tick]
    returns = compute_returns(prices)
    returns = returns.where(returns.abs() <= data_cfg.max_abs_return)
    returns = returns.loc[:, returns.notna().any()]
    returns = winsorize_cross_section(returns)
    volume = vol.loc[returns.index, returns.columns]
    raw = {
        "momentum_12_1": compute_momentum_12_1(returns),
        "reversal_1m": compute_reversal_1m(returns),
        "realized_vol": compute_realized_volatility(returns),
        "max_return": compute_max_return(returns),
        "return_skewness": compute_return_skewness(returns),
        "abnormal_volume": compute_abnormal_volume(volume),
    }
    m = mask.reindex(index=returns.index, columns=returns.columns).fillna(False).astype(bool)
    feats = {k: standardize_cross_section(v.where(m)) for k, v in raw.items()}
    common = returns.index.intersection(french.index)
    returns = returns.loc[common]
    feats = {k: v.loc[common] for k, v in feats.items()}
    panel = features_to_panel(feats, returns)
    panel.to_parquet("panel.parquet")
    wf = walk_forward(panel, feature_names=list(feats))
    pred = wf["predictions"]
    pred.to_parquet("predictions.parquet")
    Path("results").mkdir(exist_ok=True)
    ev = evaluate_all(pred, french_factors=french.loc[common])
    for k, v in ev.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(f"results/{k}.csv")
    run_transaction_cost_analysis(pred, ev["portfolio_returns"])
    run_risk_analysis(pred, ev["portfolio_returns"], french.loc[common], headline_model="Ridge")
    m.to_parquet("membership.parquet")
    json.dump({"universe": universe, "panel": len(panel), "predictions": len(pred),
               "tickers": len(tick)}, open("meta.json", "w"))
    print(universe, "done", len(panel), len(pred))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, choices=["A", "B", "C"])
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--french", default=str(REPO / "data" / "french_factors.parquet"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--window", type=int, default=6)
    a = ap.parse_args()
    close, adj, vol, mem, sym = load_inputs(a.data)
    french = pd.read_parquet(a.french)
    french.index = pd.to_datetime(french.index).to_period("M").to_timestamp("M")
    if a.universe == "A":
        t = [x if x in adj.columns else {"BK": "BNY"}.get(x, x) for x in get_sp100_tickers()]
        t = [x for x in t if x in adj.columns]
        mask = pd.DataFrame(True, index=adj.index, columns=t)
    else:
        mapping = resolve_symbols(mem, adj, sym)
        if a.universe == "C":
            cand = pit_members(mem, mapping)
        else:
            today = mem.iloc[-1]
            cur = [mapping.get(t) for t in today.index[today] if t not in SECONDARY and mapping.get(t)]
            cand = pd.DataFrame(True, index=mem.index, columns=sorted(set(cur)))
        mask = top_by_dollar_volume(cand, close, vol, adj, n=a.n, window=a.window)
    run(a.universe, mask, adj, vol, french, a.out)
