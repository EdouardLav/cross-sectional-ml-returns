# Cross-Sectional ML Return Prediction

**Can machine learning models capture nonlinear relationships between stock characteristics and future returns, beyond what linear models achieve?**

Monthly cross-sectional US equity return prediction with Ridge, Lasso, Random Forest, and XGBoost. Every result below comes from a fully nested pipeline: hyperparameters are selected by an inner walk-forward on training data only, evaluation uses an outer expanding walk-forward, alphas are benchmarked against the Carhart 4-factor model with Newey-West standard errors, and transaction costs use realized turnover. The project also documents three self-audits (a look-ahead bias, a degenerate-selection bug, and a one-month misalignment in the factor regressions) whose lessons matter as much as the numbers.

## Headline Results (out-of-sample 2015–2024, 120 months)

| Model | Mean IC (t) | Sharpe gross | Sharpe net 15bps | Turnover | Carhart alpha (t, HAC) | Break-even |
|-------|------------|--------------|------------------|----------|------------------------|------------|
| **Ridge** | **0.042 (2.45)** | **1.00** | **0.94** | 20.6% | **14.3% (2.60)** | >100 bps |
| Lasso | 0.041 (2.25) | 0.92 | 0.87 | 18.2% | 14.6% (2.79) | >100 bps |
| Random Forest | 0.023 (1.60) | 0.65 | 0.52 | 50.8% | 6.8% (1.30) | 85 bps |
| XGBoost | 0.016 (1.28) | 0.21 | -0.01 | 60.0% | -0.2% (-0.04) | 15 bps |

### Key Findings

1. **The alpha survives the momentum control, and the book is long volatility.** The feature set includes momentum 12-1 and short-term reversal, so the natural objection is that the "alpha" is just UMD exposure. It is not: Ridge's UMD beta is -0.16 with t = -0.90, and adding UMD moves the alpha from 13.5% (FF3) to 14.3% (Carhart, t_HAC = 2.60). The book is not market-neutral, though: Carhart betas are 0.36 on the market (t = 3.5) and 0.44 on SMB. Its ranking loads positively on realized volatility and MAX (rank correlations of 0.88 and 0.93), so it is long volatile, lottery-like names and short defensives, the opposite sign to the published low-volatility and MAX anomalies. In a current-constituent universe, that is exactly where survivorship bias bites hardest (see Known Limitations).
2. **Both linear models beat both tree models; the two linear models are indistinguishable from each other.** Ridge (1.00) vs Lasso (0.92) is well inside sampling error (Lo-2002 SE ≈ 0.32 each; 95% CIs [0.36, 1.63] and [0.29, 1.55] overlap almost entirely). The linear-vs-tree gap is the robust statement.
3. **XGBoost has no deployable edge**: net of 15 bps its Sharpe is -0.01 and its break-even is exactly the cost assumption (15 bps). Random Forest's Carhart alpha is not significant at 5% (t = 1.30) and its Sharpe is entirely concentrated post-2020 (0.06 pre / 1.08 post), a regime dependence the linear models do not share (Ridge: 0.81 / 1.15).
4. **Symmetric nested tuning did not rescue the trees.** All four models were tuned by the same inner walk-forward on the same folds. The inner CV drives every model toward maximal regularization (Ridge to the heavy-shrinkage plateau, RF to its most-regularized corner in 9/10 folds), which is itself evidence about the signal-to-noise ratio.
5. **Out-of-sample R-squared is ~0.1% for the linear models and negative for the trees** — the textbook shape of monthly equity prediction: negligible level predictability, monetizable ranking ability (positive IC).

## Three Self-Audits

### Audit 1: Look-ahead via a time-constant feature
An early version included a static log-market-cap feature (today's cap applied to all historical dates). A feature constant in time per ticker is a **ticker identifier**: trees can split on it and memorize which stocks turned out to be the 2015–2024 winners, since today's megacap status embeds exactly that rally. Removing it halved Random Forest's IC (0.054 to 0.027) and erased XGBoost's alpha, while linear models barely moved — they could only extract a bounded "long megacaps" tilt, not per-stock memorization. Walk-forward validation cannot catch this class of leak: the split was correct, the feature construction was contaminated. Note that split-based feature importance also failed to flag it (the feature ranked last in RF's Gini importance): ablation, not importance, is the leak detector.

### Audit 2: Degenerate selection in nested CV
The first nested-CV run let Lasso select alpha = 0.01 on early folds — a penalty that zeroes every coefficient. Constant predictions cannot rank a cross-section: decile assignment degenerated to row order (an accidental alphabetical portfolio, visible as a two-year stretch of zero turnover followed by a 100% turnover spike). Fix: a **feasibility rule** — any configuration producing (near-)constant predictions on any inner validation year is disqualified (scored -inf), not treated as a "zero-signal" candidate. A strategy that cannot form deciles is outside the feasible set.

### Audit 3: Date alignment of the factor regressions
Predictions are formed at the end of month t and earn the return of month t+1, but the first version kept the formation date on the return series and regressed each return on the factors of month t. Regressing on last month's factors gives near-zero betas, so the reported "alpha" was close to the raw mean return (Carhart 17.9%, market beta -0.02, R-squared 0.02). The giveaway was that R-squared: a book sorted on volatility is not beta-neutral by accident, and the long leg's correlation with the market goes from -0.13 to 0.87 once shifted by one month. Fix: every portfolio, leg and turnover series is now indexed by the month in which the return is earned. Corrected Ridge Carhart alpha: 14.3% (t_HAC = 2.60), market beta 0.36. Sharpe, IC, turnover and costs are unaffected.

## Methodology

### Universe & Data
- **Universe:** S&P 100 constituents (101 tickers after dropping the GOOG duplicate share class). Survivorship bias acknowledged: current list, not point-in-time constituents.
- **Data snapshot:** yfinance, downloaded June 2026; historical adjustments are retroactive, so reruns may differ slightly. Kenneth French Data Library for Mkt-RF, SMB, HML, RF, and UMD.
- **Period:** 2005–2025 monthly; out-of-sample 2015–2024. Panel: 22,993 stock-month observations, 11,964 OOS predictions.
- **Cleaning:** extreme returns (|r| > 200%) flagged; cross-sectional winsorization at 1%/99% per month.

### Features (cross-sectionally standardized each month)
| Feature | Definition | Reference |
|---------|-----------|-----------|
| Momentum 12-1 | Cumulative return t-12 to t-1, skipping last month | Jegadeesh & Titman (1993) |
| Short-term reversal | 1-month lagged return | Jegadeesh (1990) |
| Realized volatility | Rolling 12-month return std | Ang, Hodrick, Xing, Zhang (2006) |
| Maximum return | Max monthly return, past 12 months | Bali, Cakici, Whitelaw (2011) |
| Return skewness | Rolling 12-month skewness | — |
| Abnormal volume | 3-month / 12-month avg volume | Campbell, Grossman, Wang (1993) |

*A static log-market-cap feature was tested and removed — see Audit 1.*

### Nested Walk-Forward Cross-Validation
- **Outer loop (evaluation):** expanding window, train [2006, Y-1], predict year Y, for Y in 2015–2024.
- **Inner loop (selection):** the same annual expanding walk-forward, restricted to the outer training block, scores each hyperparameter configuration by mean validation IC (Spearman) — the quantity the strategy monetizes, not MSE. The outer test year never influences the selection. All four models are tuned symmetrically.
- **Feasibility rule:** configurations producing degenerate (near-constant) predictions on any inner validation year are disqualified.
- **Selected values:** Ridge pins to the heavy-shrinkage plateau (as alpha grows, the Ridge ranking converges to the covariance-weighted score X'y, and IC is scale-invariant — the inner CV is saying "use maximal shrinkage"); Lasso is stable at 0.001 on all 10 folds; RF selects its most-regularized corner (depth 3, min leaf 50) in 9/10 folds. The trajectory is in `figures/alpha_trajectory.png` and `results/selected_hyperparams.csv`.

### Evaluation
- **IC:** monthly Spearman rank correlation, predicted vs realized.
- **Portfolios:** long top decile, short bottom decile, equal-weight, monthly rebalance. Each return is dated by the month in which it is earned (formation month + 1), and factor regressions use the factors of that same month.
- **Factor models:** FF3 and Carhart 4-factor, Newey-West (HAC, 6 lags) standard errors, OLS t-stats reported alongside for transparency.
- **Costs:** realized turnover from position diffs; 15 bps one-way charged round-trip on traded fraction; break-even swept 0–100 bps.
- **Robustness:** OOS R-squared, Lo (2002) Sharpe confidence intervals, sub-period split (2015–2019 vs 2020–2024), worst months with leg attribution, long/short leg-split alphas.

## Risk Profile (Ridge, headline model)

- **Worst months:** Oct 2018 (-12.2%, long leg -14.3% as the market fell 7.6%), Aug 2024 (-9.4%, the defensive short leg rallied +7.5%), Jan 2016 (-8.5%, long leg -12.8%).
- **Max drawdown:** -29.5%, from September 2018 to September 2019, ending with the September 2019 momentum-to-value rotation. February and March 2020 are not among the worst months (+0.9% and -0.6%).
- **Leg split (Carhart, HAC):** long leg +12.2%/yr alpha (t = 3.24, market beta 1.19); short leg -2.1%/yr (t = -0.84), so shorting those names adds a little, insignificantly. The edge sits in the long leg.
- **Exposure:** the long decile is mostly volatile technology and consumer names (AMD, TSLA, NVDA, NFLX), the short decile staples, utilities and telecom (KO, PG, JNJ, PEP, VZ).

## Figures

![Cumulative Returns](figures/cumulative_returns.png)
![Break-Even](figures/break_even_analysis.png)
![Alpha Trajectory](figures/alpha_trajectory.png)
![IC Comparison](figures/ic_comparison.png)
![IC Timeseries](figures/ic_timeseries.png)
![Decile Returns](figures/decile_returns_Ridge.png)
![Feature Importance](figures/feature_importance.png)
![Turnover](figures/turnover_timeseries.png)

## Project Structure

```
cross-sectional-ml-returns/
├── config.py                  # All parameters, documented
├── main.py                    # Full pipeline (7 steps)
├── src/
│   ├── data_loader.py         # Tickers, yfinance, French factors incl. UMD
│   ├── features.py            # 6 features + cross-sectional standardization
│   ├── models.py              # Model definitions
│   ├── nested_cv.py           # Inner walk-forward selection + feasibility rule
│   ├── walk_forward.py        # Outer expanding walk-forward
│   ├── evaluation.py          # IC, portfolios, FF3/Carhart with HAC errors
│   ├── transaction_costs.py   # Realized turnover, net Sharpe, break-even
│   ├── risk_analysis.py       # OOS R2, Sharpe CIs, sub-periods, legs
│   └── utils.py               # Plots
├── data/ · results/ · figures/
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py            # ~10 min (nested CV); --skip-data to reuse parquet
```

## Known Limitations

| Limitation | How to address |
|-----------|----------------|
| Survivorship bias (current S&P 100), amplified by a long-volatility book: NVDA, TSLA and AMD make 43% of long-leg returns, and the Sharpe falls to 0.62 without them | CRSP/Compustat point-in-time constituents |
| No size/value features (dropped for leakage) | Point-in-time fundamentals |
| Flat 15 bps cost | Per-stock impact model (Almgren-Chriss) |
| ~10 names per decile | Larger universe; quintile robustness check |
| Equal-weight, no sector or beta neutrality (market beta 0.36) | Constrained optimizer |
| P&L computed on winsorized returns (conservative: Ridge Sharpe is 1.10 on raw returns) | Winsorize the training target only |

## References

Gu, Kelly & Xiu (2020, RFS) · Jegadeesh & Titman (1993) · Jegadeesh (1990) · Ang, Hodrick, Xing & Zhang (2006) · Bali, Cakici & Whitelaw (2011) · Campbell, Grossman & Wang (1993) · Carhart (1997) · Newey & West (1987) · Lo (2002) · Varma & Simon (2006)
