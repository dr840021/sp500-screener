"""
S&P 500 Stock Screener — Full Pipeline
---------------------------------------
1. Scrapes the current S&P 500 constituent list from Wikipedia
2. Pulls fundamentals + 1 year of price history for each ticker via yfinance
   (yfinance reads from finance.yahoo.com — same source you linked)
3. Applies filters and ranks candidates
4. Exports four Tableau-ready CSVs:
     - screener_results.csv        (one row per stock, all metrics)
     - tableau_metrics_long.csv    (long format — perfect for Tableau filters)
     - sector_summary.csv          (aggregated by sector)
     - price_history.csv           (daily close prices for time-series charts)

Performance notes:
  - Fundamentals are fetched in parallel with a thread pool (MAX_WORKERS).
    The single biggest speedup vs. the sequential version.
  - Price history uses yfinance's batched download, which is already fast.

Author: Devin [Last Name]
Disclaimer: Educational tool only. Not investment advice.
"""

import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# How many tickers to pull. Set to None for the full S&P 500.
LIMIT = 50

# How much price history to download for the time-series file
PRICE_HISTORY_PERIOD = "1y"   # options: 1mo, 3mo, 6mo, 1y, 2y, 5y, max

# Parallelism. Yahoo tolerates ~20 concurrent requests well in practice.
# If you start seeing 429 / rate-limit errors, dial this down to 10.
MAX_WORKERS = 20

FILTERS = {
    "max_pe_ratio": 30,
    "min_dividend_yield": 0.0,
    "min_market_cap": 10e9,
    "max_debt_to_equity": 200,
    "min_profit_margin": 0.05,
}


# ---------------------------------------------------------------
# 1. SCRAPE S&P 500 TICKERS FROM WIKIPEDIA
# ---------------------------------------------------------------

def get_sp500_tickers() -> pd.DataFrame:
    """
    Scrapes the current S&P 500 constituent list from Wikipedia.
    Uses requests with a browser User-Agent header to avoid 403 Forbidden.
    """
    import requests
    from io import StringIO

    print("Scraping S&P 500 list from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    df = tables[0]

    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
    })

    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

    cols = ["ticker", "name", "sector", "sub_industry"]
    print(f"  ✓ Found {len(df)} S&P 500 companies")
    return df[cols]


# ---------------------------------------------------------------
# 2. FETCH FUNDAMENTALS (PARALLEL)
# ---------------------------------------------------------------

def fetch_stock_data(ticker: str) -> dict | None:
    """Pull key metrics for a single ticker from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        if not info or "currentPrice" not in info:
            return None

        return {
            "ticker": ticker,
            "price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "dividend_yield": info.get("dividendYield", 0) or 0,
            "profit_margin": info.get("profitMargins", 0) or 0,
            "operating_margin": info.get("operatingMargins", 0) or 0,
            "debt_to_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity", 0) or 0,
            "roa": info.get("returnOnAssets", 0) or 0,
            "revenue_growth": info.get("revenueGrowth", 0) or 0,
            "earnings_growth": info.get("earningsGrowth", 0) or 0,
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
            "analyst_target": info.get("targetMeanPrice"),
        }
    except Exception:
        return None


def build_fundamentals_dataset(tickers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loop through tickers IN PARALLEL using a thread pool.
    yfinance calls are I/O-bound (HTTP requests), so threads work well —
    we get a ~MAX_WORKERS-x speedup over the sequential version.
    """
    tickers = tickers_df["ticker"].tolist()
    print(f"\nFetching fundamentals for {len(tickers)} tickers "
          f"(parallel, {MAX_WORKERS} workers)...")

    start = datetime.now()
    rows = []
    completed = 0
    total = len(tickers)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        # Submit all tasks up front
        future_to_ticker = {
            pool.submit(fetch_stock_data, t): t for t in tickers
        }
        # Collect results as they finish
        for future in as_completed(future_to_ticker):
            completed += 1
            data = future.result()
            if data:
                rows.append(data)
            if completed % 25 == 0 or completed == total:
                elapsed = (datetime.now() - start).total_seconds()
                print(f"  Progress: {completed}/{total}  "
                      f"({elapsed:.1f}s elapsed)")

    fundamentals = pd.DataFrame(rows)
    merged = fundamentals.merge(tickers_df, on="ticker", how="left")
    elapsed = (datetime.now() - start).total_seconds()
    print(f"  ✓ Successfully pulled {len(merged)} stocks in {elapsed:.1f}s")
    return merged


# ---------------------------------------------------------------
# 3. FETCH PRICE HISTORY (for Tableau time-series charts)
# ---------------------------------------------------------------

def fetch_price_history(tickers: list, period: str = "1y") -> pd.DataFrame:
    """
    Pull daily close prices for all tickers in one batched call.
    yfinance.download() already parallelizes internally when threads=True.
    Returns long-format DataFrame: ticker, date, close, volume.
    """
    print(f"\nFetching {period} of price history for {len(tickers)} tickers...")
    raw = yf.download(
        tickers,
        period=period,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,  # let yfinance parallelize this too
    )

    # Reshape from wide (one column per ticker) to long format
    records = []
    for ticker in tickers:
        try:
            sub = raw[ticker][["Close", "Volume"]].reset_index()
            sub["ticker"] = ticker
            sub = sub.rename(
                columns={"Date": "date", "Close": "close", "Volume": "volume"})
            records.append(sub)
        except (KeyError, AttributeError):
            continue  # Ticker had no data

    history = pd.concat(records, ignore_index=True)
    print(f"  ✓ Got {len(history):,} daily price records")
    return history[["ticker", "date", "close", "volume"]]


# ---------------------------------------------------------------
# 4. APPLY FILTERS + RANK
# ---------------------------------------------------------------

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    initial = len(df)
    f = df.copy()
    f = f[f["pe_ratio"].between(0, filters["max_pe_ratio"])]
    f = f[f["dividend_yield"] >= filters["min_dividend_yield"]]
    f = f[f["market_cap"] >= filters["min_market_cap"]]
    f = f[(f["debt_to_equity"].isna()) |
          (f["debt_to_equity"] <= filters["max_debt_to_equity"])]
    f = f[f["profit_margin"] >= filters["min_profit_margin"]]
    print(f"\nFiltered {initial} → {len(f)} candidates")
    return f


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    r = df.copy()
    r["score"] = (
        (1 / r["pe_ratio"].clip(lower=0.1)) * 30
        + r["profit_margin"] * 100 * 0.4
        + r["roe"] * 100 * 0.3
    )
    r["rank"] = r["score"].rank(ascending=False, method="dense").astype(int)
    return r.sort_values("score", ascending=False)


# ---------------------------------------------------------------
# 5. TABLEAU-READY EXPORTS
# ---------------------------------------------------------------

def export_tableau_files(ranked: pd.DataFrame, history: pd.DataFrame):
    """Write four CSVs optimized for Tableau ingestion."""

    # File 1: Main results (one row per stock — wide format)
    ranked.to_csv(OUTPUT_DIR / "screener_results.csv", index=False)

    # File 2: Long format for flexible Tableau pivoting
    metric_cols = [
        "pe_ratio", "forward_pe", "peg_ratio", "dividend_yield",
        "profit_margin", "operating_margin", "debt_to_equity",
        "roe", "roa", "revenue_growth", "earnings_growth", "beta",
    ]
    long_df = ranked.melt(
        id_vars=["ticker", "name", "sector", "sub_industry"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    long_df.to_csv(OUTPUT_DIR / "tableau_metrics_long.csv", index=False)

    # File 3: Sector summary (aggregated)
    sector_summary = (
        ranked.groupby("sector")
        .agg(
            company_count=("ticker", "count"),
            avg_pe=("pe_ratio", "mean"),
            avg_profit_margin=("profit_margin", "mean"),
            avg_roe=("roe", "mean"),
            avg_dividend_yield=("dividend_yield", "mean"),
            total_market_cap=("market_cap", "sum"),
        )
        .round(4)
        .reset_index()
        .sort_values("avg_roe", ascending=False)
    )
    sector_summary.to_csv(OUTPUT_DIR / "sector_summary.csv", index=False)

    # File 4: Price history for time-series charts
    history.to_csv(OUTPUT_DIR / "price_history.csv", index=False)

    print(f"\n✓ Exported 4 files to {OUTPUT_DIR.resolve()}/")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:35s} ({size_kb:>7.1f} KB)")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    print(f"=== S&P 500 Screener — "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")

    # Step 1: Get the universe
    sp500 = get_sp500_tickers()
    if LIMIT:
        sp500 = sp500.head(LIMIT)
        print(f"  (Limited to first {LIMIT} for this run — "
              f"set LIMIT=None for all 500)")

    # Step 2: Pull fundamentals (parallel)
    fundamentals = build_fundamentals_dataset(sp500)

    # Step 3: Filter and rank
    filtered = apply_filters(fundamentals, FILTERS)
    ranked = rank_candidates(filtered)

    # Step 4: Pull price history for the candidates that passed
    if not ranked.empty:
        history = fetch_price_history(
            ranked["ticker"].tolist(), PRICE_HISTORY_PERIOD)
    else:
        history = pd.DataFrame(columns=["ticker", "date", "close", "volume"])

    # Step 5: Print top 10 to console
    if not ranked.empty:
        print("\n=== TOP 10 CANDIDATES ===")
        cols = ["rank", "ticker", "name", "sector", "price",
                "pe_ratio", "profit_margin", "roe", "score"]
        print(ranked[cols].head(10).to_string(index=False))
    else:
        print("\nNo stocks passed filters. Try loosening criteria.")

    # Step 6: Export everything
    export_tableau_files(ranked, history)

    print("\n=== NEXT STEPS ===")
    print("  1. Open Tableau → Connect → Text File → screener_results.csv")
    print("  2. Add price_history.csv as a second data source for time-series")
    print("  3. Use 'sector' as a filter/color dimension")
    print("  4. Build views: scatter (PE vs ROE), bar (top 10 by score), line (price history)")


if __name__ == "__main__":
    main()
