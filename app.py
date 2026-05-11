"""
app.py — S&P 500 Stock Screener Web App
----------------------------------------
A Streamlit interface for the stock screener.

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → connect at https://share.streamlit.io
"""

import streamlit as st
import pandas as pd

from screener_core import (
    get_sp500_tickers,
    build_fundamentals_dataset,
    apply_filters,
    rank_candidates,
    allocate_portfolio,
    portfolio_summary,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="S&P 500 Stock Screener",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# DATA LOADING (cached so we don't re-pull on every interaction)
# ============================================================

@st.cache_data(ttl=3600)  # cache for 1 hour
def load_screener_data(universe_size: int) -> pd.DataFrame:
    """Pull and cache the screener dataset. Re-runs once per hour."""
    sp500 = get_sp500_tickers()
    if universe_size < len(sp500):
        sp500 = sp500.head(universe_size)

    progress = st.progress(0.0, text="Fetching fundamentals...")

    def update(current, total):
        progress.progress(current / total,
                          text=f"Fetching fundamentals... {current}/{total}")

    df = build_fundamentals_dataset(sp500, progress_callback=update)
    progress.empty()
    return df


# ============================================================
# HEADER
# ============================================================

st.title("📈 S&P 500 Stock Screener & Portfolio Allocator")
st.markdown(
    "Screen the S&P 500 for quality stocks, then allocate your capital across "
    "the top candidates. **Educational tool — not investment advice.**"
)


# ============================================================
# SIDEBAR — user inputs
# ============================================================

st.sidebar.header("⚙️ Screener Settings")

universe_size = st.sidebar.slider(
    "How many S&P 500 stocks to screen?",
    min_value=25, max_value=500, value=50, step=25,
    help="Larger = more thorough but slower. 50 takes ~2 min, 500 takes ~10 min."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Filter Criteria")

max_pe = st.sidebar.slider("Max P/E Ratio", 5, 50, 30,
    help="Lower = cheaper. Most stocks fall in the 15-25 range.")
min_div_yield = st.sidebar.slider("Min Dividend Yield (%)", 0.0, 5.0, 0.0, 0.1,
    help="0 includes non-dividend payers. 2%+ for income-focused investors.") / 100
min_market_cap_b = st.sidebar.slider("Min Market Cap ($B)", 1, 100, 10,
    help="Larger companies = more stable. $10B+ is 'large cap'.")
max_de = st.sidebar.slider("Max Debt-to-Equity", 0, 500, 200,
    help="Lower = less leveraged. Anything over 200 is meaningfully indebted.")
min_margin = st.sidebar.slider("Min Profit Margin (%)", 0.0, 30.0, 5.0, 0.5,
    help="Higher = better pricing power.") / 100

filters = {
    "max_pe_ratio": max_pe,
    "min_dividend_yield": min_div_yield,
    "min_market_cap": min_market_cap_b * 1e9,
    "max_debt_to_equity": max_de,
    "min_profit_margin": min_margin,
}

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Portfolio Settings")

capital = st.sidebar.number_input(
    "Starting Capital ($)", min_value=10.0, max_value=10_000_000.0,
    value=10000.0, step=100.0,
)

n_stocks = st.sidebar.slider("Number of stocks in portfolio", 1, 20, 5)

strategy = st.sidebar.selectbox(
    "Allocation Strategy",
    options=["equal_weight", "score_weighted", "market_cap_weighted"],
    format_func=lambda s: {
        "equal_weight": "Equal Weight (split evenly)",
        "score_weighted": "Score Weighted (favor top picks)",
        "market_cap_weighted": "Market Cap Weighted (like an index)",
    }[s],
)

allow_fractional = st.sidebar.checkbox(
    "Allow fractional shares",
    value=True,
    help="Many brokers (Fidelity, Schwab, Robinhood) support fractional shares.",
)


# ============================================================
# RUN THE SCREENER
# ============================================================

if st.sidebar.button("🚀 Run Screener", type="primary", use_container_width=True):
    with st.spinner("Loading data... (cached after first run)"):
        try:
            fundamentals = load_screener_data(universe_size)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

    if fundamentals.empty:
        st.error("No data returned. Try again — Yahoo occasionally rate-limits.")
        st.stop()

    filtered = apply_filters(fundamentals, filters)
    ranked = rank_candidates(filtered)

    # Stash results in session_state so other widgets can use them
    st.session_state["fundamentals"] = fundamentals
    st.session_state["ranked"] = ranked

# ============================================================
# DISPLAY RESULTS
# ============================================================

if "ranked" in st.session_state:
    ranked = st.session_state["ranked"]
    fundamentals = st.session_state["fundamentals"]

    # --- Summary metrics across the top ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Universe Screened", len(fundamentals))
    col2.metric("Passed Filters", len(ranked))
    col3.metric("Avg P/E", f"{ranked['pe_ratio'].mean():.1f}" if not ranked.empty else "—")
    col4.metric("Avg ROE", f"{ranked['roe'].mean()*100:.1f}%" if not ranked.empty else "—")

    if ranked.empty:
        st.warning("No stocks passed your filters. Try loosening criteria in the sidebar.")
        st.stop()

    # --- Tabs for different views ---
    tab1, tab2, tab3 = st.tabs(["💼 Your Portfolio", "🏆 Top Candidates", "📊 Sector Breakdown"])

    # === TAB 1: Portfolio allocation ===
    with tab1:
        allocation = allocate_portfolio(
            ranked, capital=capital, n_stocks=n_stocks,
            strategy=strategy, allow_fractional=allow_fractional,
        )

        if allocation.empty:
            st.warning("Couldn't allocate. Try increasing capital or loosening filters.")
        else:
            summary = portfolio_summary(allocation, capital, ranked)

            st.subheader(f"Recommended Allocation for ${capital:,.0f}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Invested", f"${summary['invested']:,.2f}")
            c2.metric("Cash Left", f"${summary['cash_left']:,.2f}")
            c3.metric("Portfolio P/E", f"{summary['weighted_pe']:.1f}")
            c4.metric("Annual Dividend Income", f"${summary['annual_dividend_income']:,.2f}")

            display = allocation.copy()
            display["price"] = display["price"].map("${:.2f}".format)
            display["target_dollars"] = display["target_dollars"].map("${:,.2f}".format)
            display["actual_dollars"] = display["actual_dollars"].map("${:,.2f}".format)
            display["weight"] = (display["weight"] * 100).map("{:.1f}%".format)
            display["score"] = display["score"].map("{:.1f}".format)

            st.dataframe(display, use_container_width=True, hide_index=True)

            st.caption(
                f"📌 Strategy: **{strategy.replace('_', ' ').title()}** · "
                f"Fractional shares: **{'Yes' if allow_fractional else 'No'}** · "
                f"Portfolio beta: **{summary['weighted_beta']:.2f}**"
            )

            csv = allocation.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Allocation as CSV", csv,
                               file_name="my_portfolio.csv", mime="text/csv")

    # === TAB 2: Full ranked list ===
    with tab2:
        st.subheader("All Stocks That Passed Your Filters")
        st.caption(f"{len(ranked)} stocks, ranked by composite score (value + profitability + capital efficiency)")
        st.dataframe(
            ranked[["rank", "ticker", "name", "sector", "price",
                    "pe_ratio", "dividend_yield", "profit_margin", "roe", "score"]],
            use_container_width=True, hide_index=True,
        )

    # === TAB 3: Sector breakdown ===
    with tab3:
        st.subheader("Sector Distribution of Candidates")
        sector_counts = ranked["sector"].value_counts()
        st.bar_chart(sector_counts)

        st.subheader("Average Metrics by Sector")
        sector_avgs = (
            ranked.groupby("sector")
            .agg(count=("ticker", "count"),
                 avg_pe=("pe_ratio", "mean"),
                 avg_roe=("roe", "mean"),
                 avg_margin=("profit_margin", "mean"))
            .round(3)
            .sort_values("avg_roe", ascending=False)
        )
        st.dataframe(sector_avgs, use_container_width=True)

else:
    st.info("👈 Configure your settings in the sidebar and click **Run Screener** to begin.")
    st.markdown("""
    ### How this works
    1. **Screener pulls fundamentals** for S&P 500 stocks (P/E, ROE, margins, etc.)
    2. **Your filters narrow the universe** to quality candidates
    3. **Scoring ranks them** by a value + profitability composite
    4. **Allocator splits your capital** across the top N picks

    ### What this is *not*
    This tool doesn't predict prices, time the market, or guarantee returns. It's a
    research aid for identifying fundamentally strong stocks — not a buy signal.
    Always do your own research before investing real money.
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Built with Python, Streamlit, and yfinance · "
    "Data: Yahoo Finance via yfinance · "
    "Educational use only · Not investment advice"
)
