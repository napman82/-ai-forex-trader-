from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AI Forex Lab v2", page_icon="📈", layout="wide")

PAIRS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "CAD=X", "USD/CHF": "CHF=X", "NZD/USD": "NZDUSD=X",
}
INTERVALS = {
    "5 minutes": ("5m", "5d"), "15 minutes": ("15m", "60d"),
    "1 hour": ("60m", "730d"), "1 day": ("1d", "5y"),
}

st.title("📈 AI Forex Lab — Version 2")
st.caption("Mobile-friendly research prototype • paper trading only • no live broker orders")
st.warning(
    "This is an experimental research tool, not a validated or profitable trading system. "
    "It does not place live trades. Market data may be delayed or incomplete. Backtests are simulations."
)

def fetch_data(symbol, interval, period):
    try:
        df = yf.download(symbol, interval=interval, period=period, auto_adjust=False,
                         progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    required = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    return df[required].dropna()

def calc_indicators(df, fast, slow, adx_period, atr_period):
    x = df.copy()
    x["SMA_Fast"] = x["Close"].rolling(fast).mean()
    x["SMA_Slow"] = x["Close"].rolling(slow).mean()
    prev = x["Close"].shift(1)
    tr = pd.concat([(x["High"] - x["Low"]), (x["High"] - prev).abs(),
                    (x["Low"] - prev).abs()], axis=1).max(axis=1)
    x["TR"] = tr
    x["ATR"] = tr.rolling(atr_period).mean()
    up = x["High"].diff()
    down = -x["Low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    smoothed_tr = tr.rolling(adx_period).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(adx_period).mean() / smoothed_tr
    minus_di = 100 * minus_dm.rolling(adx_period).mean() / smoothed_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["ADX"] = dx.rolling(adx_period).mean()
    return x

def signals(x, adx_floor):
    long = (x["SMA_Fast"] > x["SMA_Slow"]) & (x["ADX"] >= adx_floor)
    short = (x["SMA_Fast"] < x["SMA_Slow"]) & (x["ADX"] >= adx_floor)
    return pd.Series(np.select([long, short], [1, -1], default=0), index=x.index, name="Signal")

def simulate(df, initial_cash, risk_pct, stop_atr, target_r, spread_pips, pip_size,
             daily_loss_limit_pct, max_drawdown_pct):
    """
    Transparent OHLC-bar simulation. Uses next-bar open for entries and checks stop/target
    against each bar's high/low. If both stop and target are touched in one bar, assumes
    the stop happened first (conservative). This still omits real execution details.
    """
    x = df.copy()
    cash = float(initial_cash)
    equity = cash
    peak_equity = cash
    daily_start_equity = cash
    current_day = None
    halted = False
    side = 0
    units = 0.0
    entry = stop = target = None
    entry_time = None
    trades = []
    curve = []
    fee_rate = (spread_pips * pip_size)

    for i in range(1, len(x)):
        ts = x.index[i]
        bar = x.iloc[i]
        prev_sig = int(x["Signal"].iloc[i-1]) if pd.notna(x["Signal"].iloc[i-1]) else 0
        o, h, l, c = map(float, [bar["Open"], bar["High"], bar["Low"], bar["Close"]])
        atr = float(x["ATR"].iloc[i-1]) if pd.notna(x["ATR"].iloc[i-1]) else np.nan
        day = pd.Timestamp(ts).date()

        if day != current_day:
            current_day = day
            daily_start_equity = equity
            # Daily halt resets on the next data date; this is only a historical simulation.
            halted = False

        # Mark-to-market equity for an open position at bar close.
        unrealized = side * units * (c - entry) if side and entry is not None else 0.0
        equity = cash + unrealized
        peak_equity = max(peak_equity, equity)
        daily_loss = (equity / daily_start_equity - 1) if daily_start_equity else 0.0
        drawdown = (equity / peak_equity - 1) if peak_equity else 0.0
        if daily_loss <= -daily_loss_limit_pct / 100 or drawdown <= -max_drawdown_pct / 100:
            halted = True

        # Check protective exits on current bar.
        if side != 0:
            exit_price = None
            exit_reason = None
            if side == 1:
                stop_hit = l <= stop
                target_hit = h >= target
            else:
                stop_hit = h >= stop
                target_hit = l <= target
            if stop_hit:
                exit_price, exit_reason = stop, "Stop loss"
            elif target_hit:
                exit_price, exit_reason = target, "Take profit"
            elif prev_sig == -side:
                exit_price, exit_reason = o, "Opposite signal"
            if exit_price is not None:
                # Adverse spread approximation on exit.
                exit_price = exit_price - fee_rate if side == 1 else exit_price + fee_rate
                pnl = side * units * (exit_price - entry)
                cash += pnl
                trades.append({
                    "Entry time": entry_time, "Exit time": ts, "Side": "Long" if side == 1 else "Short",
                    "Units": round(units, 2), "Entry": entry, "Exit": exit_price,
                    "P&L ($)": pnl, "Exit reason": exit_reason,
                    "Return on risk budget (%)": (pnl / max(abs(entry-stop)*units, 1e-12))*100
                })
                side, units, entry, stop, target, entry_time = 0, 0.0, None, None, None, None
                equity = cash
                peak_equity = max(peak_equity, equity)
                drawdown = equity / peak_equity - 1 if peak_equity else 0

        # Enter at current bar open using signal from prior completed bar.
        if side == 0 and not halted and prev_sig != 0 and np.isfinite(atr) and atr > 0:
            entry = o + fee_rate if prev_sig == 1 else o - fee_rate
            stop_distance = stop_atr * atr
            risk_cash = max(0.0, cash * risk_pct / 100)
            units = risk_cash / stop_distance if stop_distance > 0 else 0.0
            if units > 0:
                side = prev_sig
                stop = entry - side * stop_distance
                target = entry + side * stop_distance * target_r
                entry_time = ts
        unrealized = side * units * (c - entry) if side and entry is not None else 0.0
        equity = cash + unrealized
        peak_equity = max(peak_equity, equity)
        curve.append({"Time": ts, "Equity": equity, "Cash": cash,
                      "Drawdown %": (equity / peak_equity - 1) * 100 if peak_equity else 0,
                      "Open side": side, "Open units": units, "Close": c})
    curve_df = pd.DataFrame(curve).set_index("Time") if curve else pd.DataFrame()
    return curve_df, pd.DataFrame(trades), cash, side, units, entry, stop, target

with st.sidebar:
    st.header("Settings")
    pair = st.selectbox("Currency pair", list(PAIRS), index=0)
    interval_label = st.selectbox("Timeframe", list(INTERVALS), index=1)
    interval, period = INTERVALS[interval_label]
    st.subheader("Signal strategy")
    fast = st.slider("Fast moving average", 5, 30, 12)
    slow = st.slider("Slow moving average", 20, 80, 32)
    adx_floor = st.slider("Minimum ADX", 10, 40, 20)
    atr_period = st.slider("ATR period", 5, 30, 14)
    st.subheader("Paper risk controls")
    initial_cash = st.number_input("Starting simulated balance ($)", 100.0, 1_000_000.0, 10000.0, 500.0)
    risk_pct = st.slider("Risk budget per trade (%)", 0.1, 1.0, 0.5, 0.1)
    stop_atr = st.slider("Stop distance (ATR multiple)", 0.5, 4.0, 1.5, 0.25)
    target_r = st.slider("Take-profit multiple (R)", 0.5, 5.0, 2.0, 0.25)
    spread_pips = st.slider("Assumed spread (pips)", 0.0, 5.0, 1.2, 0.1)
    daily_loss_limit = st.slider("Daily loss pause threshold (%)", 0.5, 5.0, 2.0, 0.5)
    max_dd_limit = st.slider("Max drawdown pause threshold (%)", 2.0, 25.0, 10.0, 1.0)

if fast >= slow:
    st.error("Fast moving average must be smaller than slow moving average.")
    st.stop()

with st.spinner(f"Loading {pair} data..."):
    raw = fetch_data(PAIRS[pair], interval, period)
if raw.empty or len(raw) < max(slow, atr_period, 50):
    st.error("Market data is unavailable or there is not enough history for these settings. Try another pair or timeframe.")
    st.stop()

pip_size = 0.01 if "JPY" in pair else 0.0001
ind = calc_indicators(raw, fast, slow, 14, atr_period)
ind["Signal"] = signals(ind, adx_floor)
valid = ind.dropna(subset=["SMA_Fast", "SMA_Slow", "ATR", "ADX"]).copy()
if len(valid) < 20:
    st.error("Not enough valid indicator rows to analyze.")
    st.stop()

latest = valid.iloc[-1]
bias = "BUY BIAS" if latest["Signal"] == 1 else "SELL BIAS" if latest["Signal"] == -1 else "NO TRADE"
regime = "Trend filter passed" if latest["ADX"] >= adx_floor else "Weak/ranging conditions"
atr_ratio = valid["ATR"] / valid["Close"]
vol_state = "Elevated" if latest["ATR"] / latest["Close"] >= atr_ratio.quantile(.8) else "Typical"

a, b, c, d = st.columns(4)
a.metric("Latest reference price", f"{latest['Close']:.5f}")
b.metric("Rule-based signal", bias)
c.metric("ADX", f"{latest['ADX']:.1f}", regime)
d.metric("Volatility", vol_state, f"ATR {latest['ATR']:.5f}")

st.subheader(f"{pair} price trend")
st.line_chart(valid[["Close", "SMA_Fast", "SMA_Slow"]].tail(250), height=300)

tab1, tab2, tab3, tab4 = st.tabs(["Backtest & risk", "Strategy review", "Paper monitor", "Data"])
with tab1:
    st.caption("OHLC-bar simulation with ATR stops/targets and position sizing based on a risk budget. It is still simplified and cannot reproduce real fills.")
    curve, trades, cash, side, units, entry, stop, target = simulate(
        valid, initial_cash, risk_pct, stop_atr, target_r, spread_pips, pip_size,
        daily_loss_limit, max_dd_limit
    )
    if curve.empty:
        st.info("Not enough rows to simulate.")
    else:
        end_equity = float(curve["Equity"].iloc[-1])
        total_return = end_equity / initial_cash - 1
        max_dd = float(curve["Drawdown %"].min())
        wins = int((trades["P&L ($)"] > 0).sum()) if not trades.empty else 0
        count = len(trades)
        win_rate = wins / count * 100 if count else 0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ending simulated equity", f"${end_equity:,.2f}")
        m2.metric("Net return", f"{total_return*100:.2f}%")
        m3.metric("Max drawdown", f"{max_dd:.2f}%")
        m4.metric("Closed trades / win rate", f"{count} / {win_rate:.1f}%")
        st.line_chart(curve[["Equity"]], height=250)
        if side:
            st.info(f"Simulated open position: {'LONG' if side > 0 else 'SHORT'} • units {units:,.2f} • entry {entry:.5f} • stop {stop:.5f} • target {target:.5f}")
        if not trades.empty:
            st.dataframe(trades.tail(100), use_container_width=True)
            st.download_button("Download trade log", trades.to_csv(index=False).encode(), "trade_log.csv", "text/csv")
        else:
            st.info("No closed simulated trades in this sample.")
        st.download_button("Download equity curve", curve.to_csv().encode(), "equity_curve.csv", "text/csv")

with tab2:
    st.subheader("Transparent strategy review")
    st.write(f"**Current market classification:** {regime}; volatility is {vol_state.lower()}.")
    if latest["Signal"] == 1:
        st.write("The rule set has a bullish bias: fast SMA is above slow SMA and the ADX filter passes.")
    elif latest["Signal"] == -1:
        st.write("The rule set has a bearish bias: fast SMA is below slow SMA and the ADX filter passes.")
    else:
        st.write("No directional signal meets the selected rules.")
    st.info("This is not yet a trained predictive AI. Version 2 focuses on transparent rules and more explicit simulated risk controls.")
    if st.button("Research candidate settings"):
        results = []
        for f in [8, 12, 16]:
            for s in [24, 32, 40]:
                if f >= s: continue
                for adx in [15, 20, 25]:
                    candidate = calc_indicators(raw, f, s, 14, atr_period)
                    candidate["Signal"] = signals(candidate, adx)
                    candidate = candidate.dropna(subset=["SMA_Fast", "SMA_Slow", "ATR", "ADX"])
                    if len(candidate) < 50: continue
                    cv, tr, *_ = simulate(candidate, initial_cash, risk_pct, stop_atr, target_r,
                                          spread_pips, pip_size, daily_loss_limit, max_dd_limit)
                    if cv.empty: continue
                    ret = cv["Equity"].iloc[-1] / initial_cash - 1
                    dd = cv["Drawdown %"].min() / 100
                    score = ret + 0.5 * dd
                    results.append({"Fast SMA": f, "Slow SMA": s, "ADX floor": adx,
                                    "Return %": ret*100, "Max drawdown %": dd*100,
                                    "Research score": score})
        if results:
            table = pd.DataFrame(results).sort_values("Research score", ascending=False)
            st.dataframe(table, use_container_width=True)
            st.download_button("Download research candidates", table.to_csv(index=False).encode(),
                               "candidate_settings.csv", "text/csv")
            st.warning("Research scan results are vulnerable to overfitting. Do not automatically deploy the top row; test it on a separate untouched period and paper trade.")
        else:
            st.warning("No candidate results were available.")

with tab3:
    st.subheader("Paper monitor — no real orders")
    st.write("This view reports the latest rule-based state and a historical simulation summary. It does not run unattended in the background.")
    st.metric("Latest rule-based state", bias)
    st.metric("Reference time (data timestamp)", str(valid.index[-1]))
    st.write("Risk guardrails used in backtest:")
    st.write(f"- Risk budget per simulated trade: **{risk_pct:.2f}%** of simulated cash")
    st.write(f"- Daily loss pause threshold: **{daily_loss_limit:.1f}%**")
    st.write(f"- Drawdown pause threshold: **{max_dd_limit:.1f}%**")
    st.write(f"- Stop: **{stop_atr:.2f} ATR** • Target: **{target_r:.2f}R**")
    st.warning("These thresholds govern this historical simulation only; this page is not a persistent broker-connected paper account.")

with tab4:
    st.dataframe(valid.tail(100).sort_index(ascending=False), use_container_width=True)
    st.download_button("Download analyzed market data", valid.to_csv().encode(), "market_data.csv", "text/csv")

st.divider()
st.caption("Next planned stage: reliable data feed, persistent event-driven paper account, walk-forward validation, monitoring/alerts, then optional trained model evaluation. Live trading is intentionally excluded.")
