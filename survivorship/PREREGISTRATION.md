# Survivorship test: pre-registered design

Written on 29 October 2026, before any out-of-sample result of this test was seen.

## Question
Does the S&P 100 study's result survive once the universe is rebuilt point-in-time?

## Universes
- A: current S&P 100 list (the published study), rerun on fresh data.
- B: current S&P 500 members, top 100 by trailing 6-month average dollar volume each month.
- C (main test): S&P 500 members as of each month (fja05680/sp500, commit a2430f2), top 100 by trailing 6-month average dollar volume, using only data known at the formation date.

## Fixed in advance
- Same code, features, nested walk-forward, hyperparameter grids, costs and evaluation as the published study. No change after results are seen.
- Data: Yahoo Finance monthly prices. Names Yahoo no longer carries (acquired, delisted) are missing; coverage is reported by year.
- Robustness, reported whatever the outcome: C with top 50 instead of 100, and with a 3-month instead of 6-month volume window (linear models only), and the 2020-2024 sub-period, where data coverage exceeds 90%.

## Decision rule
The signal is said to survive if, on C over 2015-2024, Ridge has an IC t-stat of at least 2 and a Carhart alpha t-stat (Newey-West) of at least 2. Otherwise the published alpha is attributed to the universe. Every variant is reported, including those that go against the conclusion.

## Addendum, 1 November 2026: universe D from CRSP, written before the run

Data: CRSP monthly stock file (permno level, NYSE, AMEX and NASDAQ ordinary common shares), 1963 to September 2026, delisted stocks included, with delisting returns in the monthly return. The file stays out of the repository (WRDS licence); `build_crsp.py` turns it into the monthly matrices the pipeline reads.

Universe D: at each month-end, the 500 largest stocks by market capitalization (a proxy for the S&P 500, as this file carries no index membership), then the top 100 by trailing 6-month average dollar volume, using only data known at that date. No name is missing: 385 stocks enter D between 2005 and 2025, 111 of them later delisted (Lehman included, with its -99.7% last month).

Fixed in advance: same code as A, B and C (`run_universe.run` and `top_by_dollar_volume`, unchanged; `run_crsp.py` only builds the mask) and the same decision rule: the signal survives if, on D over 2015-2024, Ridge has an IC t-stat of at least 2 and a Carhart alpha t-stat (Newey-West) of at least 2. As in A to C, returns are winsorized at 1% and 99% within each month, delisting returns included. Reported whatever the outcome: all four models; top 50, a 3-month volume window, and both (linear models); sub-periods 2015-2019 and 2020-2024. D becomes the main test because it has no data gap; C stays reported.
