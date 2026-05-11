"""
S&P 500 Intraday Screener — Auto-Refresh Every 10 Minutes
----------------------------------------------------------
Runs the screener in a continuous loop:
  1. Scrapes S&P 500 list (once at startup — that doesn't change)
  2. Every REFRESH_INTERVAL seconds:
       - Pulls latest fundamentals
       - Pulls last 1 day of 5-minute price candles
       - Re-runs filters + ranking
       - Overwrites the CSVs with fresh data
       - Appends to a running snapshot log (so you can see how the top picks change over the day)

Stop the script with Ctrl+C in the terminal.

Author: Devin [Last Name]
Disclaimer: Educational tool only. Not investment advice.
            Yahoo data is delayed 15-20 minutes — this is NOT real-time data.
"""

import yfinance as yf
import pandas as pd
import time
import requests
from io import StringIO
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# How many tickers to screen. Keep small — running every 10 min hits Yahoo a lot.
LIMIT = 20

# Refresh interval in seconds (600 = 10 minutes)
REFRESH_INTERVAL = 600

# Intraday price history settings
# Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h
# Valid periods for 5m: up to 60 days
INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD = "1d"   # last trading day

# Delay between per-ticker calls (slower = less likely to get rate-limited)
PER_TICKER_DELAY = 0.3

FILTERS = {
    "max_pe_ratio": 30,
    "min_dividend_yield": 0.0,
    "min_market_cap": 10e9,
    "max_debt_to_equity": 200,
    "min_profit_margin": 0.05,
}


# ---------------------------------------------------------------
# 1. SCRAPE S&P 500 TICKERS (RUN ONCE AT STARTUP)
# ---------------------------------------------------------------

def get_sp500_tickers() -> pd.DataFrame:
    """Scrape S&P 500 list from Wikipedia."""
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
        "Symbol": "ticker", "Security": "name",
        "GICS Sector": "sector", "GICS Sub-Industry": "sub_industry",
    })
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df[["ticker", "name", "sector", "sub_industry"]]


# ---------------------------------------------------------------
# 2. FETCH FUNDAMENTALS (called every refresh cycle)
# ---------------------------------------------------------------

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
            "dividend_yield": info.get("dividendYield", 0) or 0,
            "profit_margin": info.get("profitMargins", 0) or 0,
            "debt_to_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity", 0) or 0,
            "beta": info.get("beta"),
            "day_change_pct": info.get("regularMarketChangePercent", 0) or 0,
            "volume": info.get("volume"),
        }
    except Exception:
        return None


def build_fundamentals_dataset(tickers_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in tickers_df.itertuples():
        data = fetch_stock_data(row.ticker)
        if data:
            rows.append(data)
        time.sleep(PER_TICKER_DELAY)
    fundamentals = pd.DataFrame(rows)
    return fundamentals.merge(tickers_df, on="ticker", how="left")


# ---------------------------------------------------------------
# 3. FETCH INTRADAY PRICE DATA (5-minute candles)
# ---------------------------------------------------------------

def fetch_intraday_history(tickers: list) -> pd.DataFrame:
    """
    Pull intraday 5-minute candles for all tickers.
    Returns long-format DataFrame: ticker, datetime, open, high, low, close, volume.
    """
    if not tickers:
        return pd.DataFrame(columns=["ticker", "datetime", "open", "high",
                                      "low", "close", "volume"])

    raw = yf.download(
        tickers,
        period=INTRADAY_PERIOD,
        interval=INTRADAY_INTERVAL,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    records = []
    for ticker in tickers:
        try:
            if len(tickers) > 1:
                sub = raw[ticker][["Open", "High", "Low", "Close", "Volume"]].reset_index()
            else:
                sub = raw[["Open", "High", "Low", "Close", "Volume"]].reset_index()
            sub["ticker"] = ticker
            # Yahoo uses "Datetime" for intraday, "Date" for daily
            time_col = "Datetime" if "Datetime" in sub.columns else "Date"
            sub = sub.rename(columns={
                time_col: "datetime", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume",
            })
            records.append(sub[["ticker", "datetime", "open", "high",
                                "low", "close", "volume"]])
        except (KeyError, AttributeError):
            continue

    if not records:
        return pd.DataFrame(columns=["ticker", "datetime", "open", "high",
                                      "low", "close", "volume"])
    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------
# 4. TECHNICAL INDICATORS (computed from intraday price history)
# ---------------------------------------------------------------

def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Relative Strength Index — momentum on a 0-100 scale.
    >70 = overbought, <30 = oversold, ~50 = neutral.
    Standard formula by J. Welles Wilder, 1978.
    """
    if len(prices) < period + 1:
        return 50.0  # neutral default when we don't have enough data
    delta = prices.diff().dropna()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(period).mean().iloc[-1]
    avg_loss = losses.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_indicators(history: pd.DataFrame) -> pd.DataFrame:
    """
    For each ticker in the intraday history, compute technical indicators
    from its price series. Returns one row per ticker.
    """
    rows = []
    for ticker, group in history.groupby("ticker"):
        group = group.sort_values("datetime")
        closes = group["close"].dropna()
        volumes = group["volume"].dropna()
        if len(closes) < 5:
            continue  # not enough data for meaningful indicators

        latest_price = closes.iloc[-1]
        first_price = closes.iloc[0]
        ma_20 = closes.rolling(20).mean().iloc[-1] if len(closes) >= 20 else closes.mean()
        ma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else closes.mean()

        rows.append({
            "ticker": ticker,
            "rsi": compute_rsi(closes),
            "pct_from_ma20": (latest_price - ma_20) / ma_20 * 100,
            "pct_from_ma50": (latest_price - ma_50) / ma_50 * 100,
            "session_change_pct": (latest_price - first_price) / first_price * 100,
            "volatility": closes.pct_change().std() * 100,  # % std dev of returns
            "volume_avg": volumes.mean(),
            "volume_latest": volumes.iloc[-1],
            "volume_ratio": volumes.iloc[-1] / volumes.mean() if volumes.mean() > 0 else 1,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# 5. FILTERS + HYBRID RANKING
# ---------------------------------------------------------------

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    f = df.copy()
    f = f[f["pe_ratio"].between(0, filters["max_pe_ratio"])]
    f = f[f["dividend_yield"] >= filters["min_dividend_yield"]]
    f = f[f["market_cap"] >= filters["min_market_cap"]]
    f = f[(f["debt_to_equity"].isna()) |
          (f["debt_to_equity"] <= filters["max_debt_to_equity"])]
    f = f[f["profit_margin"] >= filters["min_profit_margin"]]
    return f


def rank_candidates(df: pd.DataFrame,
                     indicators: pd.DataFrame = None) -> pd.DataFrame:
    """
    Hybrid scoring: 50% fundamental quality + 50% live technical momentum.

    Fundamental score rewards: low P/E, high profit margin, high ROE.
    Technical score rewards: positive session momentum, RSI near 50 (not extreme),
                              price above 20-period MA, healthy volume.

    The technical half makes the ranking actually move cycle-to-cycle.
    """
    if df.empty:
        return df

    r = df.copy()

    # --- Fundamental score (slow-changing) ---
    r["fundamental_score"] = (
        (1 / r["pe_ratio"].clip(lower=0.1)) * 30
        + r["profit_margin"] * 100 * 0.4
        + r["roe"] * 100 * 0.3
    )

    # --- Technical score (changes every cycle) ---
    if indicators is not None and not indicators.empty:
        r = r.merge(indicators, on="ticker", how="left")

        # Sub-components, each 0-25 points:
        # 1. Session momentum — reward stocks up today, penalize big drops
        r["tech_momentum"] = r["session_change_pct"].fillna(0).clip(-5, 5) * 5

        # 2. Trend strength — reward being above the 20-period MA
        r["tech_trend"] = r["pct_from_ma20"].fillna(0).clip(-3, 3) * 8

        # 3. RSI sweet spot — penalize overbought (>70) and oversold (<30)
        rsi = r["rsi"].fillna(50)
        r["tech_rsi"] = 25 - abs(rsi - 50) * 0.7  # max 25 at RSI=50

        # 4. Volume confirmation — unusual volume is interesting (good or bad)
        r["tech_volume"] = (r["volume_ratio"].fillna(1) - 1).clip(-1, 2) * 5

        r["technical_score"] = (
            r["tech_momentum"] + r["tech_trend"] +
            r["tech_rsi"] + r["tech_volume"]
        )
    else:
        # No price history available — score gets carried by fundamentals only
        r["technical_score"] = 0
        for col in ["rsi", "pct_from_ma20", "session_change_pct", "volume_ratio"]:
            r[col] = None

    # --- Combined score ---
    # Normalize each side to 0-100 before combining so they're comparable
    def _normalize(s: pd.Series) -> pd.Series:
        if s.max() == s.min():
            return pd.Series([50] * len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min()) * 100

    r["fund_norm"] = _normalize(r["fundamental_score"])
    r["tech_norm"] = _normalize(r["technical_score"])
    r["score"] = r["fund_norm"] * 0.5 + r["tech_norm"] * 0.5

    r["rank"] = r["score"].rank(ascending=False, method="dense").astype(int)
    return r.sort_values("score", ascending=False)


# ---------------------------------------------------------------
# 5. EXPORT (overwrite current CSVs + append to snapshot log)
# ---------------------------------------------------------------

def export_files(ranked: pd.DataFrame, history: pd.DataFrame,
                  cycle_timestamp: datetime):
    """Write current state and append to a historical snapshot log."""

    # Current state files (overwritten each cycle)
    ranked.to_csv(OUTPUT_DIR / "screener_results.csv", index=False)
    history.to_csv(OUTPUT_DIR / "intraday_prices.csv", index=False)

    # Snapshot log — append-only, tracks how the screener output changes over time
    snapshot = ranked.copy()
    snapshot["snapshot_time"] = cycle_timestamp
    snapshot_path = OUTPUT_DIR / "screener_snapshots.csv"

    if snapshot_path.exists():
        snapshot.to_csv(snapshot_path, mode="a", header=False, index=False)
    else:
        snapshot.to_csv(snapshot_path, index=False)


# ---------------------------------------------------------------
# 6. SINGLE REFRESH CYCLE
# ---------------------------------------------------------------

def run_cycle(sp500_universe: pd.DataFrame, cycle_num: int) -> bool:
    """
    Run one complete refresh cycle. Returns True if successful, False on error.
    """
    start = datetime.now()
    print(f"\n{'='*60}")
    print(f"  CYCLE #{cycle_num} — {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    try:
        # Step 1: Pull fundamentals (price + key metrics)
        print(f"  [1/5] Fetching fundamentals for {len(sp500_universe)} tickers...")
        fundamentals = build_fundamentals_dataset(sp500_universe)
        print(f"        ✓ Got {len(fundamentals)} stocks")

        # Step 2: Apply quality filters (fundamentals only — fast cut)
        print(f"  [2/5] Applying quality filters...")
        filtered = apply_filters(fundamentals, FILTERS)
        print(f"        ✓ {len(filtered)} candidates passed filters")

        # Step 3: Pull intraday price candles for the filtered universe
        print(f"  [3/5] Fetching {INTRADAY_INTERVAL} intraday data...")
        if not filtered.empty:
            history = fetch_intraday_history(filtered["ticker"].tolist())
            print(f"        ✓ Got {len(history):,} price candles")
        else:
            history = pd.DataFrame()

        # Step 4: Compute technical indicators from intraday prices
        print(f"  [4/5] Computing technical indicators...")
        indicators = compute_indicators(history) if not history.empty else pd.DataFrame()
        print(f"        ✓ Indicators computed for {len(indicators)} tickers")

        # Step 5: Rank using hybrid (fundamental + technical) score
        print(f"  [5/5] Ranking with hybrid score...")
        ranked = rank_candidates(filtered, indicators)
        export_files(ranked, history, start)
        print(f"        ✓ Updated CSVs in {OUTPUT_DIR.resolve()}/")

        # Show top 5 with the new technical info
        if not ranked.empty:
            print(f"\n  TOP 5 RIGHT NOW (50% fundamentals + 50% technicals):")
            print(f"  {'Rank':<5}{'Ticker':<8}{'Price':>10}{'Day %':>9}"
                  f"{'RSI':>7}{'vs MA20':>10}{'Vol×':>7}{'Score':>8}")
            print(f"  {'-'*64}")
            for _, row in ranked.head(5).iterrows():
                rsi = row.get("rsi", 50) or 50
                ma20 = row.get("pct_from_ma20", 0) or 0
                vol = row.get("volume_ratio", 1) or 1
                day_pct = row.get("session_change_pct", row.get("day_change_pct", 0)) or 0
                arrow = "▲" if day_pct >= 0 else "▼"
                print(f"  #{row['rank']:<4}{row['ticker']:<8}"
                      f"${row['price']:>9.2f}  {arrow}{day_pct:+5.2f}%"
                      f"{rsi:>7.0f}{ma20:>+8.2f}%"
                      f"{vol:>6.2f}x{row['score']:>8.1f}")

        elapsed = (datetime.now() - start).total_seconds()
        print(f"\n  Cycle completed in {elapsed:.1f}s")
        return True

    except Exception as e:
        print(f"  ⚠ Cycle failed: {e}")
        return False


# ---------------------------------------------------------------
# MAIN — RUN FOREVER UNTIL CTRL+C
# ---------------------------------------------------------------

def main():
    print("="*60)
    print("  S&P 500 INTRADAY SCREENER — AUTO-REFRESH MODE")
    print("="*60)
    print(f"  Refresh interval: {REFRESH_INTERVAL}s ({REFRESH_INTERVAL//60} min)")
    print(f"  Tickers per cycle: {LIMIT}")
    print(f"  Intraday interval: {INTRADAY_INTERVAL}")
    print(f"  Output folder: {OUTPUT_DIR.resolve()}")
    print(f"\n  ⚠ Yahoo data is delayed 15-20 minutes — NOT real-time")
    print(f"  ⚠ US markets open 9:30 AM - 4:00 PM ET, Mon-Fri")
    print(f"\n  Press Ctrl+C in this terminal to stop.\n")

    # Get the universe ONCE — the S&P 500 list doesn't change minute to minute
    print("  Loading S&P 500 list from Wikipedia...")
    sp500 = get_sp500_tickers()
    if LIMIT:
        sp500 = sp500.head(LIMIT)
    print(f"  ✓ Screening top {len(sp500)} S&P 500 stocks\n")

    cycle_num = 1
    try:
        while True:
            success = run_cycle(sp500, cycle_num)

            # Sleep until next cycle (countdown shown to user)
            next_run = datetime.now()
            print(f"\n  Next refresh in {REFRESH_INTERVAL//60} minutes "
                  f"(~{next_run.strftime('%H:%M')} + {REFRESH_INTERVAL//60}min). "
                  f"Press Ctrl+C to stop.")
            time.sleep(REFRESH_INTERVAL)
            cycle_num += 1

    except KeyboardInterrupt:
        print(f"\n\n  Stopped after {cycle_num} cycle(s). "
              f"Final outputs are in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
