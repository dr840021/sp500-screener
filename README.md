# S&P 500 Stock Screener & Portfolio Allocator

An interactive Streamlit web application that screens the S&P 500 universe for fundamentally strong stocks and generates portfolio allocations based on user-defined capital and weighting strategy.

**[Live demo](#)** *(coming soon — Streamlit Cloud deployment in progress)*

## What it does

- **Scrapes** the current S&P 500 constituent list from Wikipedia
- **Pulls fundamentals** for each stock via the yfinance API (P/E, ROE, profit margin, debt/equity, dividend yield, market cap)
- **Filters** the universe using five configurable criteria
- **Ranks candidates** with a hybrid score combining fundamental quality (P/E, margin, ROE) and live technical momentum (RSI, distance from moving averages, volume ratio)
- **Allocates capital** across the top N picks using equal-weight, score-weighted, or market-cap-weighted strategies
- **Exports** allocations to CSV and live-refreshing intraday data for further analysis

## Why these filters?

Filter choice is the most important screening decision. Cheap stocks are often cheap for a reason ("value traps"), so I paired the P/E ratio filter with profit margin and ROE thresholds to ensure low-priced stocks are also fundamentally healthy. The hybrid scoring then layers in live technical momentum so the ranking responds to actual market movement, not just stale quarterly fundamentals.

## Tech stack

- **Python 3.11** — core language
- **Streamlit** — web UI framework
- **pandas** — data manipulation
- **yfinance** — Yahoo Finance API wrapper
- **requests + lxml** — Wikipedia scraping

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit web interface |
| `screener_core.py` | Reusable screener + allocation logic (importable module) |
| `sp500_screener.py` | One-shot CLI version that exports Tableau-ready CSVs |
| `intraday_screener.py` | Auto-refreshing version with live 5-minute price data and technical indicators |
| `requirements.txt` | Dependency list |

## Limitations & honest trade-offs

- **yfinance is unofficial.** Yahoo occasionally rate-limits or changes their data format, breaking the library temporarily. Production deployments should use a paid API like Polygon, Alpha Vantage, or IEX Cloud.
- **Sector concentration.** The current filters tend to surface financials and consumer staples disproportionately — a future version should add sector caps to enforce diversification.
- **Yahoo data is delayed 15-20 minutes.** The intraday refresh is not real-time and is unsuitable for active trading decisions.
- **Educational tool only — not investment advice.** This is a research aid for identifying screening candidates, not a buy signal.

## About me

Devin Roper — Junior at Ohio University, majoring in MIS and Business Analytics. Starting an MS in Business Analytics at OU in Fall 2026. Built this as a hands-on exercise in financial data pipelines and interactive analytics tools.

[LinkedIn](https://www.linkedin.com/in/devin-roper1/)
