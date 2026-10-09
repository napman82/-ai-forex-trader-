from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AI Forex Trader — Test Dashboard", page_icon="📈", layout="wide")

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "NZD/USD": "NZDUSD=X",
}
INTERVALS = {
    "5 minutes": ("5m", "5d"),
    "15 minutes": ("15m", "60d"),
    "1 hour": ("60m", "730d"),
    "1 day": ("1d", "5y"),
}


def fetch_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, interval=interval, period=period, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if len(needed) != 4:
        return pd.DataFrame()
    df = df[needed].copy()
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().sort_index()


def indicators(df: pd.DataFrame, fast: int, slow: int, atr_period: int, adx_period: int) -> pd.DataFrame:
    out = df.copy()
    out["SMA_fast"] = out["Close"].rolling(fast).mean()
    out["SMA_slow"] = out["Close"].rolling(slow).mean()
    prev_close = out["Close"].shift(1)
    tr = pd.concat([
        out["High"] - out["Low"],
        (out["High"] - prev_close).abs(),
        (out["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(atr_period).mean()
    up = out["High"].diff()
    down = -out["Low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_adx = tr.rolling(adx_period).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(adx_period).mean() / atr_adx
    minus_di = 100 * minus_dm.rolling(adx_period).mean() / atr_adx
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    out["ADX"] = dx.rolling(adx_period).mean()
    return out


def quote_to_usd_factor(pair: str, price: float) -> float:
    """USD value of one unit of quote currency, approximated from the pair price."""
    return (1.0 / price) if pair.startswith("USD/") and price > 0 else 1.0


def run_backtest(df: pd.DataFrame, pair: str, initial_cash: float, risk_pct: float,
                 fast: int, slow: int, adx_min: float, atr_stop: float,
                 reward_risk: float, spread_pips: float, pip_size: float,
                 max_daily_loss_pct: float, max_drawdown_pct: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = df.copy().dropna(subset=["Open", "High", "Low", "Close", "SMA_fast", "SMA_slow", "ATR", "ADX"])
    if len(data) < 3:
        return pd.DataFrame(), pd.DataFrame(), {"error": "Not enough candles after indicators. Try a longer interval/history."}

    # A signal is calculated at candle close and an order is simulated at the next candle open.
    data["signal"] = 0
    data.loc[(data["SMA_fast"] > data["SMA_slow"]) & (data["ADX"] >= adx_min), "signal"] = 1
    data.loc[(data["SMA_fast"] < data["SMA_slow"]) & (data["ADX"] >= adx_min), "signal"] = -1

    cash = float(initial_cash)
    position = None
    trades = []
    equity_rows = []
    peak_equity = cash
    current_day = None
    day_start_equity = cash
    daily_paused = False
    drawdown_paused = False
    spread = float(spread_pips) * float(pip_size)

    def marked_equity(mark_price: float) -> float:
        if position is None:
            return cash
        quote_pnl = position["side"] * position["units"] * (mark_price - position["entry"])
        return cash + quote_pnl * quote_to_usd_factor(pair, mark_price)

    rows = list(data.iterrows())
    for i, (timestamp, row) in enumerate(rows):
        price = float(row["Open"])
        day = pd.Timestamp(timestamp).date()
        if current_day != day:
            current_day = day
            day_start_equity = marked_equity(price)
            daily_paused = False

        # Exits are checked against the full candle range. If stop and target both hit,
        # assume the stop was hit first (conservative, but still only an approximation).
        if position is not None:
            hi, lo = float(row["High"]), float(row["Low"])
            stop_hit = lo <= position["stop"] if position["side"] == 1 else hi >= position["stop"]
            target_hit = hi >= position["target"] if position["side"] == 1 else lo <= position["target"]
            exit_price, reason = None, None
            if stop_hit:
                exit_price, reason = position["stop"], "Stop loss"
            elif target_hit:
                exit_price, reason = position["target"], "Take profit"
            elif i == len(rows) - 1:
                exit_price, reason = float(row["Close"]), "End of data"
            # Reverse signal exits at this candle's open; signal is from previous candle.
            elif i > 0 and int(data["signal"].iloc[i - 1]) == -position["side"]:
                exit_price, reason = price, "Opposite signal"
            if exit_price is not None:
                # Spread is charged on entry and exit in quote currency.
                effective_exit = exit_price - position["side"] * spread
                quote_pnl = position["side"] * position["units"] * (effective_exit - position["entry_effective"])
                pnl_usd = quote_pnl * quote_to_usd_factor(pair, max(effective_exit, 1e-12))
                cash += pnl_usd
                trades.append({
                    "Entry time": position["entry_time"], "Exit time": timestamp,
                    "Pair": pair, "Side": "LONG" if position["side"] == 1 else "SHORT",
                    "Units": round(position["units"], 2), "Entry": position["entry"],
                    "Exit": exit_price, "P&L (USD)": pnl_usd, "Reason": reason,
                    "Equity after trade": cash,
                })
                position = None

        equity_now = marked_equity(float(row["Close"]))
        peak_equity = max(peak_equity, equity_now)
        drawdown_pct = (peak_equity - equity_now) / peak_equity * 100 if peak_equity > 0 else 0
        daily_loss_pct = (day_start_equity - equity_now) / day_start_equity * 100 if day_start_equity > 0 else 0
        if daily_loss_pct >= max_daily_loss_pct:
            daily_paused = True
        if drawdown_pct >= max_drawdown_pct:
            drawdown_paused = True

        # New positions use the previous completed candle's signal and ATR.
        if position is None and i > 0 and not daily_paused and not drawdown_paused:
            sig = int(data["signal"].iloc[i - 1])
            prior_atr = float(data["ATR"].iloc[i - 1])
            if sig != 0 and np.isfinite(prior_atr) and prior_atr > 0:
                stop_distance = prior_atr * atr_stop
                risk_cash = max(0.0, cash) * risk_pct / 100.0
                risk_per_unit_usd = stop_distance * quote_to_usd_factor(pair, price)
                units = risk_cash / risk_per_unit_usd if risk_per_unit_usd > 0 else 0.0
                # Prevent nonsensical sizing from corrupt/near-zero prices.
                if np.isfinite(units) and units > 0 and units < 1e12:
                    entry_effective = price + sig * spread
                    position = {
                        "side": sig, "units": units, "entry": price,
                        "entry_effective": entry_effective, "entry_time": timestamp,
                        "stop": price - sig * stop_distance,
                        "target": price + sig * stop_distance * reward_risk,
                    }
        equity_rows.append({"Time": timestamp, "Equity": marked_equity(float(row["Close"])),
                            "Close": float(row["Close"]), "Signal": int(row["signal"])})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows).set_index("Time") if equity_rows else pd.DataFrame()
    final_equity = float(equity_df["Equity"].iloc[-1]) if not equity_df.empty else cash
    if not trades_df.empty:
        wins = int((trades_df["P&L (USD)"] > 0).sum())
        losses = int((trades_df["P&L (USD)"] <= 0).sum())
        win_rate = 100 * wins / len(trades_df)
        net_profit = float(trades_df["P&L (USD)"].sum())
    else:
        wins = losses = 0
        win_rate = 0.0
        net_profit = 0.0
    eq = equity_df["Equity"] if not equity_df.empty else pd.Series([initial_cash])
    dd = ((eq.cummax() - eq) / eq.cummax().replace(0, np.nan) * 100).max()
    stats = {"Initial balance": initial_cash, "Final equity": final_equity,
             "Net profit (closed trades)": net_profit, "Return %": (final_equity / initial_cash - 1) * 100 if initial_cash else 0,
             "Trades": len(trades_df), "Wins": wins, "Losses": losses, "Win rate %": win_rate,
             "Max observed drawdown %": float(dd) if pd.notna(dd) else 0.0,
             "Trading paused by drawdown limit": drawdown_paused}
    return trades_df, equity_df, stats


st.title("📈 Forex Strategy Test Dashboard")
st.warning("TEST MODE ONLY — this app does not place broker orders and does not contain a trained AI. Market data from Yahoo Finance can be delayed, incomplete, or inaccurate. Backtests are simplified simulations, not proof of future profitability. Do not use real money based on these results.")

with st.sidebar:
    st.header("Market & strategy")
    pair = st.selectbox("Forex pair", list(PAIRS.keys()))
    interval_label = st.selectbox("Chart interval", list(INTERVALS.keys()), index=2)
    interval, period = INTERVALS[interval_label]
    initial_cash = st.number_input("Starting balance (USD)", min_value=100.0, max_value=1_000_000.0, value=10000.0, step=500.0)
    risk_pct = st.slider("Risk budget per trade (%)", 0.1, 2.0, 0.5, 0.1)
    fast = st.slider("Fast SMA", 3, 50, 10)
    slow = st.slider("Slow SMA", 10, 200, 30)
    adx_min = st.slider("Minimum ADX", 5, 50, 20)
    atr_period = st.slider("ATR period", 5, 40, 14)
    adx_period = st.slider("ADX period", 5, 40, 14)
    atr_stop = st.slider("Stop distance (ATR multiple)", 0.5, 5.0, 1.5, 0.25)
    reward_risk = st.slider("Target / risk ratio", 0.5, 5.0, 2.0, 0.25)
    spread_pips = st.number_input("Estimated spread (pips)", min_value=0.0, max_value=20.0, value=1.5, step=0.1)
    max_daily_loss_pct = st.slider("Pause new trades at daily loss (%)", 1.0, 20.0, 5.0, 0.5)
    max_drawdown_pct = st.slider("Pause new trades at drawdown (%)", 5.0, 50.0, 20.0, 1.0)
    refresh = st.button("Refresh market data", type="primary")

pip_size = 0.01 if pair.endswith("/JPY") else 0.0001

@st.cache_data(ttl=60)
def cached_fetch(ticker, interval, period, _refresh_token):
    return fetch_data(ticker, interval, period)

raw = cached_fetch(PAIRS[pair], interval, period, int(refresh))
if raw.empty:
    st.error("No market data was returned. Try another pair or interval, or refresh later.")
    st.stop()
if fast >= slow:
    st.error("Fast SMA must be smaller than Slow SMA. Change the settings in the sidebar.")
    st.stop()

idata = indicators(raw, fast, slow, atr_period, adx_period)
latest = idata.dropna(subset=["SMA_fast", "SMA_slow", "ATR", "ADX"])
if latest.empty:
    st.error("There is not enough data to calculate the indicators with these settings.")
    st.stop()
last = latest.iloc[-1]
if last["SMA_fast"] > last["SMA_slow"] and last["ADX"] >= adx_min:
    signal_text = "LONG BIAS"
elif last["SMA_fast"] < last["SMA_slow"] and last["ADX"] >= adx_min:
    signal_text = "SHORT BIAS"
else:
    signal_text = "NO TRADE"

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{pair} latest close", f"{last['Close']:.5f}" if not pair.endswith("/JPY") else f"{last['Close']:.3f}")
c2.metric("Rule-based signal", signal_text)
c3.metric("ADX", f"{last['ADX']:.1f}")
c4.metric("ATR", f"{last['ATR']:.5f}" if not pair.endswith("/JPY") else f"{last['ATR']:.3f}")
st.caption(f"Data timestamp: {latest.index[-1]} · Retrieved at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. A signal is an educational indicator, not a recommendation.")

st.subheader("Price chart and moving averages")
chart_df = idata[["Close", "SMA_fast", "SMA_slow"]].dropna().tail(500)
st.line_chart(chart_df)

st.subheader("Historical strategy simulation")
trades_df, equity_df, stats = run_backtest(
    idata, pair, initial_cash, risk_pct, fast, slow, adx_min, atr_stop,
    reward_risk, spread_pips, pip_size, max_daily_loss_pct, max_drawdown_pct,
)
if "error" in stats:
    st.info(stats["error"])
else:
    a, b, c, d = st.columns(4)
    a.metric("Final simulated equity", f"${stats['Final equity']:,.2f}")
    b.metric("Simulated return", f"{stats['Return %']:.2f}%")
    c.metric("Closed trades", f"{stats['Trades']}")
    d.metric("Max observed drawdown", f"{stats['Max observed drawdown %']:.2f}%")
    st.caption("Model assumptions: next-candle-open entries; simplified ATR stops/targets; conservative stop-first rule if stop and target are both touched; estimated spread; no commissions, swaps, slippage, financing, margin calls, or broker execution effects. USD-base pair P&L is converted approximately using the exit/mark price. Results may differ materially from live trading.")
    if not equity_df.empty:
        st.line_chart(equity_df[["Equity"]])
    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        st.download_button("Download trade log CSV", trades_df.to_csv(index=False).encode("utf-8"), file_name="forex_test_trades.csv", mime="text/csv")
    else:
        st.info("No trades were closed during this simulation. Try a different pair, interval, or settings; do not optimize only for the best historical result.")
    if not equity_df.empty:
        st.download_button("Download equity curve CSV", equity_df.to_csv().encode("utf-8"), file_name="forex_test_equity.csv", mime="text/csv")

with st.expander("What this version can and cannot do"):
    st.markdown("""
    **It can:** download historical candles from Yahoo Finance, calculate SMA/ATR/ADX-style indicators, display a rule-based bias, and run a simplified historical simulation.

    **It cannot:** submit orders, connect to a broker, run continuously in the background, guarantee data quality, learn from trades, or guarantee profits. This is not yet an autonomous AI trader.

    **Next testing steps:** compare a few pairs and timeframes, inspect individual trades, test different historical periods, and use a broker demo/paper account separately before considering any live trading. Keep API keys and passwords out of this app and out of GitHub.
    """)
