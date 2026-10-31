import os, time
from io import StringIO
from pathlib import Path
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())
import pandas as pd
import requests
import yfinance as yf

COMMIT = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
URL = f"https://raw.githubusercontent.com/fja05680/sp500/{COMMIT}/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
ALT = {"FB": "META", "ANTM": "ELV", "ABC": "COR", "PKI": "RVTY", "RE": "EG", "FISV": "FI", "FI": "FISV",
       "WLTW": "WTW", "HRS": "LHX", "UTX": "RTX", "BBT": "TFC", "JEC": "J", "HCP": "DOC", "PEAK": "DOC",
       "SYMC": "GEN", "NLOK": "GEN", "CTL": "LUMN", "MYL": "VTRS", "COG": "CTRA", "HFC": "DINO", "LB": "BBWI",
       "FBHS": "FBIN", "CDAY": "DAY", "FLT": "CPAY", "BLL": "BALL", "GPS": "GAP", "DISCA": "WBD", "VIAC": "PARA",
       "VIAB": "PARA", "CBS": "PARA", "PARA": "PSKY", "DWDP": "DD", "KORS": "CPRI", "BHGE": "BKR", "TMK": "GL",
       "FII": "FHI", "ESV": "VAL", "WYND": "TNL", "ADS": "BFH", "AABA": "YHOO", "BK": "BNY", "GOOG": "GOOGL"}

d = Path(__file__).resolve().parent
out = d / "data"
out.mkdir(exist_ok=True)
cache = out / "cache"
cache.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(out / "tzcache"))
hist = pd.read_csv(StringIO(requests.get(URL, timeout=60).text), parse_dates=["date"])
hist.to_csv(out / "sp500_history.csv", index=False)
hist = hist[(hist.date >= "2004-01-01") & (hist.date <= "2025-12-31")]
tickers = sorted({t for row in hist.tickers for t in row.split(",")})
syms = sorted({t.replace(".", "-") for t in tickers} | {ALT[t] for t in tickers if t in ALT})
print(f"{len(tickers)} historical members, {len(syms)} Yahoo symbols to try", flush=True)


def fetch(sym):
    for attempt in range(3):
        try:
            h = yf.Ticker(sym).history(start="2004-01-01", end="2026-01-01", interval="1mo",
                                       auto_adjust=False, actions=False)
            return h
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    return None


for k, s in enumerate(syms, 1):
    f = cache / f"{s}.parquet"
    miss = cache / f"{s}.missing"
    if f.exists() or miss.exists():
        continue
    h = fetch(s)
    if h is not None and not h.empty and "Close" in h:
        h.index = pd.to_datetime(h.index).tz_localize(None)
        h[[c for c in ["Close", "Adj Close", "Volume"] if c in h]].to_parquet(f)
    else:
        miss.touch()
    if k % 25 == 0:
        print(f"{k}/{len(syms)}", flush=True)
    time.sleep(0.4)

fields = {"Close": {}, "Adj Close": {}, "Volume": {}}
for f in sorted(cache.glob("*.parquet")):
    h = pd.read_parquet(f)
    for c in fields:
        if c in h:
            fields[c][f.stem] = h[c]
for c, cols in fields.items():
    pd.DataFrame(cols).to_parquet(out / f"{c.replace(' ', '_').lower()}.parquet")
n_ok = len(list(cache.glob("*.parquet")))
print(f"done: {n_ok} symbols with data out of {len(syms)}")
print("missing:", " ".join(sorted(p.stem for p in cache.glob("*.missing"))))
