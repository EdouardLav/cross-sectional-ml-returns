import io, os, sys, time, zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
import pandas as pd
import requests

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision/"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
d = Path(__file__).resolve().parent / "data"
raw = d / "raw"
raw.mkdir(parents=True, exist_ok=True)
sess = requests.Session()


def s3_list(prefix, delimiter=True):
    keys, prefixes, marker = [], [], ""
    while True:
        params = {"prefix": prefix, "marker": marker}
        if delimiter:
            params["delimiter"] = "/"
        for attempt in range(5):
            try:
                r = sess.get(S3, params=params, timeout=30)
                r.raise_for_status()
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        root = ET.fromstring(r.content)
        keys += [k.text for k in root.findall("s:Contents/s:Key", NS)]
        prefixes += [p.text for p in root.findall("s:CommonPrefixes/s:Prefix", NS)]
        if root.findtext("s:IsTruncated", namespaces=NS) != "true":
            return keys, prefixes
        marker = root.findtext("s:NextMarker", namespaces=NS) or (keys[-1] if keys else prefixes[-1])


def read_zip(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None)
    if not str(df.iloc[0, 0]).replace(".", "").replace("-", "").isdigit():
        df = df.iloc[1:]
    return df


def get(key):
    for attempt in range(4):
        try:
            r = sess.get(FILES + key, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return read_zip(r.content)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def symbol_job(sym):
    out = raw / f"{sym}.parquet"
    if out.exists():
        return sym, "cached"
    kk, _ = s3_list(f"data/futures/um/monthly/klines/{sym}/1d/", delimiter=False)
    fk, _ = s3_list(f"data/futures/um/monthly/fundingRate/{sym}/", delimiter=False)
    kk = [k for k in kk if k.endswith(".zip") and k[-11:-4] >= "2019-09"]
    fk = [k for k in fk if k.endswith(".zip") and k[-11:-4] >= "2019-09"]
    kl = [x for x in (get(k) for k in kk) if x is not None]
    if not kl:
        (raw / f"{sym}.none").touch()
        return sym, "none"
    k = pd.concat(kl).iloc[:, :11]
    k.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
                 "taker_buy_volume", "taker_buy_quote_volume"]
    k = k.apply(pd.to_numeric, errors="coerce")
    k["date"] = pd.to_datetime(k.open_time, unit="ms").dt.normalize()
    k = k.drop_duplicates("date").set_index("date").sort_index()
    fl = [x for x in (get(f) for f in fk) if x is not None]
    if fl:
        f = pd.concat(fl)
        f = f.iloc[:, [0, f.shape[1] - 1]]
        f.columns = ["calc_time", "funding_rate"]
        f = f.apply(pd.to_numeric, errors="coerce")
        f["date"] = pd.to_datetime(f.calc_time, unit="ms").dt.normalize()
        k["funding"] = f.groupby("date").funding_rate.sum()
        k["funding_n"] = f.groupby("date").funding_rate.count()
    k.to_parquet(out)
    return sym, len(k)


if __name__ == "__main__":
    _, pref = s3_list("data/futures/um/monthly/klines/")
    syms = sorted(p.rstrip("/").split("/")[-1] for p in pref)
    syms = [s for s in syms if s.endswith("USDT") and "_" not in s]
    print(f"{len(syms)} USDT perpetual symbols (listed and delisted)", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(symbol_job, s) for s in syms if not (raw / f"{s}.none").exists()]
        for fu in as_completed(futs):
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(futs)}", flush=True)
    fields = {"close": {}, "quote_volume": {}, "funding": {}, "high": {}, "low": {}, "count": {}}
    for p in sorted(raw.glob("*.parquet")):
        k = pd.read_parquet(p)
        for c in fields:
            if c in k:
                fields[c][p.stem] = k[c]
    for c, cols in fields.items():
        pd.DataFrame(cols).sort_index().to_parquet(d / f"{c}.parquet")
    n = len(list(raw.glob("*.parquet")))
    print(f"done: {n} symbols with data")
