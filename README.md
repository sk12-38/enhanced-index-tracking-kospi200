# Enhanced Index Tracking on KOSPI200

This project reproduces and extends the deep learning enhanced index tracking framework of Dai & Li (2024) using KOSPI200 market data.

## Strategies

- IT: Index Tracking
- EIT: Enhanced Index Tracking
- EIT-CVaR: Enhanced Index Tracking with CVaR constraint
- RO: Re-Optimization baseline

## Pipeline

1. Load KOSPI200 price, index, and market capitalization data
2. Estimate bull/bear regimes using a 2-state Gaussian HMM
3. Build short-term return, volatility, beta, and regime features
4. Train neural network policy variants
5. Run rolling backtests with transaction costs
6. Compare TE, MER, IR, CVaR, Sharpe, MDD, ATC, and final wealth

## Structure

```text
src/          Python modules
notebooks/    Main experiment notebook
results/      Summary metrics and representative figures
