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
