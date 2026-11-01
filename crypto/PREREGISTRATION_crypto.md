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

## Addendum, 31 October 2026: one cost-reduction change, fixed before running

The first validation found the signal (V1 pass) but costs bind: about 90% weekly turnover, net Sharpe 0.65 at $10M on perps, net alpha t = 1.2. One change is tested, chosen on theory before any run with it: partial rebalancing, trading a quarter of the way to the target weights each week (Garleanu and Pedersen, 2013), since the score is measured over 30 days and moves slowly. The BTC hedge is still reset weekly on the actual positions. Signal, universe, band and cost model are unchanged. The code was checked on July 2020 to March 2021 perps only: rate 1 reproduces the first validation exactly, rate 0.25 halves turnover.

Runs reported in full: perps January 2022 to September 2026 with taker fees (5 bps) and with maker fees (2 bps, an assumption about execution), and Binance spot October 2017 to May 2020 (paper).

- C2: net alpha t-stat against BTC >= 2 at $10M, top 50, perps, taker fees.
- C3: net Sharpe at $10M >= 0.5 in all three universe sizes (top 30, 50, 100), perps, taker fees.

## Addendum 2, 31 October 2026: attribution and multiple testing, fixed before running

No change to the frozen specification (quarter-step rebalancing, top 50). Checks reported whatever the outcome (`attribution.py`), on perpetuals 2022 to September 2026 and on spot October 2017 to May 2020:

- A1, factor attribution: weekly regression of the book (gross, and net at $10M) on BTC, a liquidity factor (equal-weight least liquid 30% minus most liquid 30% of the top-100 universe, by 30-day median volume) and a momentum factor (past 21-day return, top 30% minus bottom 30%), Newey-West with 4 lags. Rule: gross alpha t-stat >= 2 on perpetuals.
- A2, not a liquidity sort in disguise: rank IC of the score within liquidity terciles of the top-100 universe. Rule: positive in all three terciles, t-stat >= 2 in at least two.
- A3, multiple testing: Deflated Sharpe (Bailey and Lopez de Prado, 2014) of the frozen book's net Sharpe at $10M, with N = every book configuration run on the perpetuals so far, counted from the result files. Reported, no threshold.
- Forward test: the frozen specification run on October 2026 Binance data once it is published, then weekly, reported whatever the outcome.

## Addendum 3, 1 November 2026: non-crypto perpetuals, fixed before running

Found while building the first live book from the frozen code (positions at the 27 September 2026 close): since late 2025 Binance lists USDT perpetuals on equities, ETFs (leveraged ones included), commodities, FX and pre-IPO companies, plus tokenized gold (PAXG, XAUT). Their low volatility puts them in the long leg of the frozen book: at least 15 to 25% of the long leg from October 2025 to May 2026, 35% in June 2026 and 76% in September 2026, and about a third of the short leg in August and September 2026 (leveraged ETFs, memory stocks). The study is about crypto, and these underlyings trade on weekdays only, so their Sunday close is a stale price. Only portfolio weights were looked at before writing this addendum; no return of a book without these names has been computed.

Change: the universe excludes perpetuals whose underlying is not a crypto asset, defined as (a) the names in `tradfi.txt`, identified by name (equities, ETFs, commodities, FX, pre-IPO, tokenized gold), and (b) any perpetual whose mean absolute daily return on Saturdays and Sundays is below 46% of its weekday mean, with at least 8 weekends of data (`make_crypto_only.py`). On data through September 2026 this removes 171 of 900 symbols; no crypto coin with 8 weekends or more falls below the threshold (the closest are TAIKO at 0.47 and ANC at 0.48; BTC is at 0.58). Each run prints the current top-100 names removed by the test only, and the kept ones below 0.55, for a check by name; additions to `tradfi.txt` are dated commits.

Reported whatever the outcome: (1) the frozen specification as pre-registered, on all USDT perpetuals, and (2) the crypto-only universe, for the backtest (perpetuals 2022 to September 2026, taker 5 bps, quarter-step, top 30/50/100, rules C1 to C3 and A1 to A3, `run_validation3.sh`) and for the forward test (`run_forward.sh`, weekly, from the 27 September 2026 rebalance). From now on the crypto-only universe is the main specification, whichever of the two does better. The spot 2017-2020 validation is unaffected (no such listings then). The three configurations of this run are added to the Deflated Sharpe count.
