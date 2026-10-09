# AI Forex Lab v2 (Starter)

## What's new
- Browser-based Streamlit dashboard
- Rule-based trend signal and market-condition summary
- ATR-based stops and take-profit levels in an OHLC historical simulation
- Position sizing based on a configurable risk budget
- Simulated daily-loss and drawdown pause thresholds
- Trade log, equity curve, and bounded candidate-setting research
- CSV exports

## What this is not
- Not a validated or profitable strategy
- Not a trained predictive AI model
- Not a persistent, unattended paper-trading service
- Not connected to a broker and cannot place live orders
- Not a complete execution simulator; no reliable bid/ask feed, margin, swaps, gaps, liquidity, partial fills, or broker-specific order rules

The historical simulation uses bar data and simplified spread assumptions. If both stop and target are touched within the same bar, it assumes the stop was hit first. Results can still be materially different from real trading.

## Run locally
Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Mobile access
The app can be deployed to a hosted Python service and then opened in Safari. Hosting setup is a separate step; this ZIP does not deploy itself.

## Development roadmap
1. Test and review accounting against hand-checked examples.
2. Add robust data quality checks and a more realistic persistent paper broker.
3. Add walk-forward and untouched out-of-sample validation.
4. Add alerts and uptime monitoring.
5. Evaluate whether a trained model improves results out-of-sample.
6. Consider supervised live execution only after extensive testing, with independent risk controls and broker-specific safeguards.
