# Crypto low-risk book: pre-registered validation

Written on 31 October 2026, before downloading or looking at the validation data.

## What was already seen
Binance USDT perpetuals, June 2020 to September 2026: volatility and MAX rank next-week returns (t from -7 to -8 in top 30, 50 and 100); the portfolio results varied with universe size. The code below was smoke-tested on Binance perps from July 2020 to March 2021, outside both test windows; that check showed turnover-driven costs, and the no-trade band was added then, before any test below was run.

## Fixed specification
- Score: average of within-date ranks of minus 30-day volatility and minus 30-day MAX of daily returns. Signs fixed (Ang et al. 2006; Bali, Cakici, Whitelaw 2011; the perps study).
- Universe: top 50 coins by 30-day median USD volume, listed for at least 60 days, delisted coins included. Top 30 and top 100 reported as robustness.
- Portfolio: weekly at Sunday close; long top quintile, short bottom quintile; no-trade band (a held name exits only beyond the 40th / 60th percentile); inverse-volatility weights, each leg sums to one; 90-day BTC beta hedged with BTC.
- Returns: next 7 days close to close, net of funding for perpetuals; a delisted coin exits at its last traded price.
- Costs: fee (perps 5 bps, spot 10 bps) + half-spread of 2 bps x sqrt($100M / 30-day median volume), between 0.5 and 25 bps + square-root impact (k = 1, daily volatility, trade / daily volume), at $1M, $10M and $50M.

## Tests and decision rules
1. Validation on never-seen data: Binance spot, BTC- and USDT-quoted pairs converted to USD, delisted pairs included, weekly from October 2017 to May 2020 (before the perps sample starts). Short selling was not broadly available on these coins then, so this tests the signal and the portfolio on paper, not implementability.
   - V1: rank IC of the score with next-week return, top 50, t-stat >= 2.
   - V2: gross alpha t-stat against BTC (Newey-West, 4 lags) >= 2, top 50. About 140 weeks, so V2 needs an annual Sharpe near 1.2: low power, stated in advance.
2. Cost realism on already-seen data: Binance perps, January 2022 to September 2026, same specification.
   - C1: net Sharpe >= 0.5 at $10M, top 50: deployable at that size.

Every result is reported, including failures.
