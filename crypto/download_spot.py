import io, os, re, time, zipfile
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
LAST_MONTH = "2020-06"
d = Path(__file__).resolve().parent / "spot_data"
raw = d / "raw"
raw.mkdir(parents=True, exist_ok=True)
sess = requests.Session()
STABLE = {"USDT", "USDC", "BUSD", "TUSD", "PAX", "USDS", "USDSB", "DAI", "SUSD", "EUR", "GBP", "BKRW", "IDRT", "TRY", "NGN", "RUB", "ZAR", "UAH", "BIDR", "BRL", "AUD"}


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


def get(key):
    for attempt in range(4):
        try:
            r = sess.get(FILES + key, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f, header=None)
            if not str(df.iloc[0, 0]).isdigit():
                df = df.iloc[1:]
            return df
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def job(sym):
    out = raw / f"{sym}.parquet"
    if out.exists() or (raw / f"{sym}.none").exists():
        return
    kk, _ = s3_list(f"data/spot/monthly/klines/{sym}/1d/", delimiter=False)
    kk = [k for k in kk if k.endswith(".zip") and k[-11:-4] <= LAST_MONTH]
    parts = [x for x in (get(k) for k in kk) if x is not None]
    if not parts:
        (raw / f"{sym}.none").touch()
        return
    k = pd.concat(parts).iloc[:, [0, 2, 3, 4, 7]].apply(pd.to_numeric, errors="coerce")
    k.columns = ["open_time", "high", "low", "close", "quote_volume"]
    k["date"] = pd.to_datetime(k.open_time, unit="ms").dt.normalize()
    k.drop_duplicates("date").set_index("date").sort_index().drop(columns="open_time").to_parquet(out)


if __name__ == "__main__":
    _, pref = s3_list("data/spot/monthly/klines/")
    syms = sorted(p.rstrip("/").split("/")[-1] for p in pref)
    lev = re.compile(r"(UP|DOWN|BULL|BEAR)(USDT|BTC)$")
    cand = [s for s in syms if (s.endswith("USDT") or s.endswith("BTC")) and not lev.search(s)
            and s[:-4 if s.endswith("USDT") else -3] not in STABLE]
    print(f"{len(cand)} BTC and USDT spot pairs to check (listed and delisted)", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(job, s) for s in cand]
        for i, _ in enumerate(as_completed(futs), 1):
            if i % 50 == 0:
                print(f"{i}/{len(futs)}", flush=True)
    data = {p.stem: pd.read_parquet(p) for p in raw.glob("*.parquet")}
    btc = data["BTCUSDT"].close
    close, qv = {}, {}
    bases = sorted({s[:-4] if s.endswith("USDT") else s[:-3] for s in data})
    for b in bases:
        u, x = data.get(b + "USDT"), data.get(b + "BTC")
        if b == "BTC":
            close["BTCUSDT"], qv["BTCUSDT"] = u.close, u.quote_volume
            continue
        if x is not None:
            px = x.close * btc.reindex(x.index)
            v = (x.quote_volume * btc.reindex(x.index)).add(u.quote_volume if u is not None else 0, fill_value=0)
        else:
            px, v = u.close, u.quote_volume
        close[b + "USDT"], qv[b + "USDT"] = px, v
    close, qv = pd.DataFrame(close).sort_index(), pd.DataFrame(qv).sort_index()
    close.to_parquet(d / "close.parquet")
    qv.to_parquet(d / "quote_volume.parquet")
    last = close.apply(lambda c: c.last_valid_index())
    print(f"done: {close.shape[1]} coins, {close.index.min().date()} to {close.index.max().date()}, "
          f"{int((last < '2020-06-01').sum())} delisted or renamed before June 2020")
    print("check:", {s: str(last.get(s, "absent"))[:10] for s in ["BCCUSDT", "VENUSDT", "HSRUSDT", "ICNUSDT", "TRIGUSDT", "SALTUSDT"]})
