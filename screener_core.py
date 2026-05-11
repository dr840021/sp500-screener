"""
screener_core.py
----------------
Reusable screener logic. Used by the Streamlit app (app.py) and importable
into Jupyter notebooks for analysis.

Separating "logic" from "presentation" is a real-world software pattern —
the same core functions power the script, the web app, and any future tools.
"""

from __future__ import annotations
import time
import requests
from io import StringIO
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------
# DATA COLLECTION
# ---------------------------------------------------------------

def get_sp500_tickers() -> pd.DataFrame:
    """Scrape current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    df = pd.read_html(StringIO(response.text))[0]

    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
    })
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df[["ticker", "name", "sector", "sub_industry"]]


def fetch_stock_data(ticker: str) -> dict | None:
    """Pull key metrics for a single ticker."""
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
            "dividend_yield": info.get("dividendYield", 0) or 0,
            "profit_margin": info.get("profitMargins", 0) or 0,
            "debt_to_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity", 0) or 0,
            "beta": info.get("beta"),
        }
    except Exception:
        return None


def build_fundamentals_dataset(tickers_df: pd.DataFrame,
                                progress_callback=None) -> pd.DataFrame:
    """
    Build the dataset. Optional progress_callback(current, total) gets called
    on each ticker — Streamlit uses this to update its progress bar.
    """
    rows = []
    total = len(tickers_df)
    for i, row in enumerate(tickers_df.itertuples(), 1):
        if progress_callback:
            progress_callback(i, total)
        data = fetch_stock_data(row.ticker)
        if data:
            rows.append(data)
        time.sleep(0.1)
    fundamentals = pd.DataFrame(rows)
    return fundamentals.merge(tickers_df, on="ticker", how="left")


# ---------------------------------------------------------------
# SCREENING + RANKING
# ---------------------------------------------------------------

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    f = df.copy()
    f = f[f["pe_ratio"].between(0, filters["max_pe_ratio"])]
    f = f[f["dividend_yield"] >= filters["min_dividend_yield"]]
    f = f[f["market_cap"] >= filters["min_market_cap"]]
    f = f[(f["debt_to_equity"].isna()) | (f["debt_to_equity"] <= filters["max_debt_to_equity"])]
    f = f[f["profit_margin"] >= filters["min_profit_margin"]]
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
# PORTFOLIO ALLOCATION
# ---------------------------------------------------------------

def allocate_portfolio(
    candidates: pd.DataFrame,
    capital: float,
    n_stocks: int = 5,
    strategy: str = "equal_weight",
    allow_fractional: bool = False,
) -> pd.DataFrame:
    """
    Given screener candidates and starting capital, compute a portfolio allocation.

    strategy options:
      - "equal_weight": split capital evenly across N stocks
      - "score_weighted": allocate more to higher-scoring stocks
      - "market_cap_weighted": allocate proportionally to market cap (like an index)
    """
    if candidates.empty or capital <= 0:
        return pd.DataFrame()

    top = candidates.head(n_stocks).copy()

    if strategy == "equal_weight":
        top["weight"] = 1 / len(top)
    elif strategy == "score_weighted":
        top["weight"] = top["score"] / top["score"].sum()
    elif strategy == "market_cap_weighted":
        top["weight"] = top["market_cap"] / top["market_cap"].sum()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    top["target_dollars"] = top["weight"] * capital

    if allow_fractional:
        top["shares"] = (top["target_dollars"] / top["price"]).round(4)
    else:
        top["shares"] = (top["target_dollars"] / top["price"]).astype(int)

    top["actual_dollars"] = top["shares"] * top["price"]
    top["leftover_cash"] = top["target_dollars"] - top["actual_dollars"]

    return top[[
        "ticker", "name", "sector", "price", "weight",
        "target_dollars", "shares", "actual_dollars", "score",
    ]].reset_index(drop=True)


def portfolio_summary(allocation: pd.DataFrame, capital: float,
                       candidates: pd.DataFrame) -> dict:
    """Return aggregate stats about the allocated portfolio."""
    if allocation.empty:
        return {}

    invested = allocation["actual_dollars"].sum()
    cash_left = capital - invested

    # Pull through the per-stock metrics from the candidates table
    detail = allocation.merge(
        candidates[["ticker", "pe_ratio", "dividend_yield", "beta"]],
        on="ticker", how="left",
    )

    weighted_pe = (detail["pe_ratio"] * detail["weight"]).sum()
    weighted_div = (detail["dividend_yield"] * detail["weight"]).sum()
    weighted_beta = (detail["beta"].fillna(1.0) * detail["weight"]).sum()
    annual_dividend_income = invested * weighted_div

    sector_breakdown = (
        detail.groupby("sector")["actual_dollars"].sum()
        .sort_values(ascending=False)
    )

    return {
        "total_capital": capital,
        "invested": invested,
        "cash_left": cash_left,
        "num_positions": len(allocation),
        "weighted_pe": weighted_pe,
        "weighted_dividend_yield": weighted_div,
        "weighted_beta": weighted_beta,
        "annual_dividend_income": annual_dividend_income,
        "sector_breakdown": sector_breakdown,
    }
