# Deep Learning for Enhanced Index Tracking on KOSPI200

This project reproduces and extends the deep learning enhanced index tracking framework of Dai and Li (2024) on Korean equity market data. The main experiment tracks the KOSPI200 index using a small stock universe and compares index tracking, enhanced index tracking, CVaR-constrained enhanced index tracking, and a re-optimization baseline.

## Project Summary

The research pipeline is implemented around `Index_tracking_v3.ipynb` and the supporting Python modules in this directory. The notebook runs the full workflow:

1. Load KOSPI200 adjusted close, index, and market capitalization data.
2. Build a tradable stock universe based on market capitalization and data availability.
3. Estimate bull and bear market regimes with two-state Gaussian HMM models.
4. Construct short-term stock and index features, including regime probability, rolling mean return, volatility, beta, and current portfolio weights.
5. Train neural network policy variants for IT, EIT, and EIT-CVaR objectives.
6. Compare neural network policies against a rolling re-optimization baseline.
7. Evaluate tracking error, excess return, information ratio, CVaR, Sharpe ratio, maximum drawdown, average transaction cost, and final wealth.
8. Export summary tables and selected figures.

## Strategies

| Strategy | Description |
|---|---|
| IT | Standard index tracking objective that minimizes tracking error against the benchmark index. |
| EIT | Enhanced index tracking objective that balances tracking error and excess return. |
| EIT-CVaR | Enhanced index tracking with a CVaR penalty to control downside tail risk. |
| RO | Re-optimization baseline that solves a rolling optimization problem instead of learning a policy network. |

## Policy Variants

The neural policy network is tested with four architectural variants:

| Variant | Description |
|---|---|
| NN-ST | Short-term feature block only. |
| NN-IR | Index regime feature block. |
| NN-ISR | Index and stock regime feature blocks. |
| NN-All | Full model using regime, score, and memory-style information. |

## Main Files

| Path | Role |
|---|---|
| `Index_tracking_v3.ipynb` | Main research notebook and experiment driver. |
| `config.py` | Central experiment configuration: data paths, training years, cost levels, model hyperparameters, and policy variants. |
| `data_loader.py` | Loads KOSPI200 data, aligns dates, computes returns, and builds the stock universe. |
| `hmm_model.py` | Implements two-state Gaussian HMM regime estimation and smoothed bull probability extraction. |
| `features.py` | Builds rolling return, volatility, beta, regime, and portfolio-state features. |
| `policy_network.py` | Defines the neural network policy architecture and variant flags. |
| `loss.py` | Defines IT, EIT, and EIT-CVaR objective functions. |
| `trainer.py` | Handles bootstrap path generation, policy training, portfolio simulation, and seed control. |
| `backtester.py` | Implements the rolling backtest and re-optimization baseline. |
| `ro_optimizer.py` | Implements the re-optimization weight solver. |
| `simulator.py` | Provides NumPy and PyTorch portfolio transition functions. |
| `evaluation.py` | Computes performance metrics and diagnostic plots. |
| `make_figures.py` | Runs policy comparisons and saves summary figures/tables. |
| `parity_check.py` | Checks consistency between NumPy and PyTorch portfolio simulation steps. |

## Data Requirements

The experiment expects the following CSV files:

| File | Description |
|---|---|
| `KOSPI200_adj_close.csv` | Adjusted close prices for the stock universe. |
| `KOSPI200_index.csv` | KOSPI200 benchmark index level. |
| `KOSPI200_mkt_cap.csv` | Market capitalization data for universe selection. |

Each file should use dates as the index. Stock-level files should have ticker columns, while the index file should contain the benchmark index series.

If this repository is published without raw data, place these files in the project root or update the paths in `config.py`:

```python
PRICE_PATH = "KOSPI200_adj_close.csv"
INDEX_PATH = "KOSPI200_index.csv"
MCAP_PATH = "KOSPI200_mkt_cap.csv"
```

## Setup

Create a virtual environment and install the required packages:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If `requirements.txt` is not included, install the core dependencies manually:

```powershell
pip install torch pandas numpy scipy matplotlib seaborn tqdm scikit-learn hmmlearn
```

## How to Run

1. Place the required KOSPI200 CSV files in the project directory, or update `config.py` with your data paths.
2. Open `Index_tracking_v3.ipynb`.
3. Run the notebook from top to bottom.
4. For a quick smoke test, set `QUICK_TEST = True` in the notebook before the rolling backtest section.
5. For the full experiment, use `QUICK_TEST = False`.

The default full experiment uses:

| Setting | Value |
|---|---|
| Training start year | 2000 |
| Test period | 2017 to 2022 |
| Universe size | Top 5 stocks by market capitalization in the reference setup |
| Rebalancing frequency | 5 trading days |
| Transaction cost levels | `rho = 0.0`, `rho = 0.005` |
| Random seed | 42 |

## Outputs

The notebook and helper scripts can generate:

| Output | Description |
|---|---|
| `recomputed_metrics.csv` | Recomputed performance metrics across objectives, policies, cost levels, and normalization settings. |
| `kospi200_cvar_curve_daily.csv` | Daily CVaR calibration output. |
| `kospi200_expanding_c_values.csv` | Expanding-window CVaR threshold values. |
| `figures/section6_auto/` | Saved wealth curves, paper-style figures, daily return tables, and portfolio weight histories. |
| `checkpoints/` | Model checkpoints for normalization-off runs. |
| `checkpoints_norm/` | Model checkpoints for normalization-on runs. |

Generated outputs and checkpoints can usually be excluded from Git because they are reproducible from the notebook.

## Suggested GitHub Cleanup

For a lightweight public repository, keep:

- `Index_tracking_v3.ipynb`
- Python modules: `*.py`
- `README.md`
- `requirements.txt`
- A small number of representative figures or summary CSVs, if desired

Consider excluding:

- Raw market data if redistribution is restricted
- `__pycache__/`
- `*.pyc`
- `checkpoints/`
- `checkpoints_norm/`
- Rendered notebook exports such as `*.html` and `*.pdf`
- Large generated weight-history CSVs under `figures/`

## Notes

This project is research code. It is intended for empirical backtesting and reproducibility study, not for production trading or investment advice. Performance results depend on data availability, universe construction, transaction cost assumptions, and notebook configuration.

This work is also being developed into a research paper in collaboration with a master's student and a professor.
