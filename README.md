# RiskLens — Market Risk VaR & CVaR Framework

End-to-end market risk system built in Python.
Basel III / FRTB compliant. 2015–2024 · $10M notional · 8 assets.

## Portfolio
- Equities: NVDA · JPM · XOM · JNJ · SPY (via Stooq)
- FX: EUR/USD · GBP/USD · JPY/USD (via FRED API)
- 2,516 trading days · 60% equities / 40% FX

## What's Inside
| Phase | What I Built |
|-------|-------------|
| 1 | Data pipeline — log returns, dollar P&L |
| 2 | VaR models — Historical, Parametric, Monte Carlo |
| 3 | CVaR / Expected Shortfall — FRTB 97.5% |
| 4 | Backtesting — Kupiec + Christoffersen tests |
| 5 | Stress testing — COVID, GFC, hypothetical shocks |
| 6 | Interactive Plotly dashboard |

## Key Findings
- Historical VaR 99% = $217K — 13% above Parametric (fat tails confirmed)
- CVaR is 1.35–1.44× VaR — justifies FRTB switch to ES
- 41 breaches vs 25.2 expected — Basel RED zone (4× capital multiplier)
- NVDA = 39.9% of tail risk at only 15% weight
- GFC hypothetical = −$2.515M instant loss (25% of notional)

## Live Dashboard
[View interactive dashboard](https://kush2071.github.io/RiskLens/var_dashboard.html)

## Tech Stack
Python · pandas · NumPy · SciPy · Plotly · FRED API · Stooq · Basel III / FRTB
