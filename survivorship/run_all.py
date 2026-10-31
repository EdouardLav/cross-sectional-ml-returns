import os, subprocess, sys, time
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)
JOBS = [("C_lin", "C", [], "Ridge,Lasso"), ("C_n50", "C", ["--n", "50"], "Ridge,Lasso"),
        ("C_w3", "C", ["--window", "3"], "Ridge,Lasso"), ("C_n50_w3", "C", ["--n", "50", "--window", "3"], "Ridge,Lasso"),
        ("B_lin", "B", [], "Ridge,Lasso"), ("A_lin", "A", [], "Ridge,Lasso"),
        ("C", "C", [], ""), ("B", "B", [], ""), ("A", "A", [], "")]
only = sys.argv[1:]
for name, u, extra, models in JOBS:
    if only and name not in only:
        continue
    if (RUNS / name / "meta.json").exists():
        print(f"skip {name} (already done)", flush=True)
        continue
    env = dict(os.environ, MODELS=models) if models else {k: v for k, v in os.environ.items() if k != "MODELS"}
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "run_universe.py"), "--universe", u, "--out", str(RUNS / name)] + extra,
                       env=env, capture_output=True, text=True)
    status = "ok" if r.returncode == 0 else "FAILED"
    print(f"{name}: {status} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if r.returncode:
        print(r.stderr[-2000:], flush=True)
rows = []
for d in sorted(RUNS.glob("*/results")):
    ic = pd.read_csv(d / "ic_summary.csv", index_col=0)
    fa = pd.read_csv(d / "factor_alpha.csv", index_col=0)
    ps = pd.read_csv(d / "portfolio_summary.csv", index_col=0)
    for m in ic.index:
        rows.append({"run": d.parent.name, "model": m, "IC": ic.loc[m, "mean_IC"], "IC_t": ic.loc[m, "t_stat"],
                     "Sharpe": ps.loc[m, "sharpe"], "Carhart": fa.loc[m, "Carhart_alpha_annual"],
                     "Carhart_t": fa.loc[m, "Carhart_t_HAC"]})
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))
