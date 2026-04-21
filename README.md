# Deep Learning for Enhanced Index Tracking on KOSPI200

## 한국어

이 저장소는 Dai and Li (2024)의 deep learning enhanced index tracking framework를 한국 주식시장 데이터에 적용해 재현하고 확장한 연구 프로젝트입니다. KOSPI200 지수를 소규모 주식 universe로 추적하면서 index tracking, enhanced index tracking, CVaR 제약 enhanced index tracking, rolling re-optimization baseline을 비교합니다.

### 프로젝트 요약

주요 실험은 `Index_tracking_v3.ipynb`와 보조 Python module을 중심으로 구성됩니다.

1. KOSPI200 수정주가, 지수, 시가총액 데이터를 불러옵니다.
2. 시가총액과 데이터 가용성을 기준으로 투자 가능한 종목 universe를 구성합니다.
3. 2상태 Gaussian HMM으로 bull/bear 시장 국면을 추정합니다.
4. 국면 확률, rolling 평균 수익률, 변동성, beta, 현재 포트폴리오 비중 등 단기 feature를 생성합니다.
5. IT, EIT, EIT-CVaR 목적함수에 대한 neural network policy를 학습합니다.
6. 학습된 policy와 rolling re-optimization baseline을 비교합니다.
7. tracking error, excess return, information ratio, CVaR, Sharpe ratio, maximum drawdown, 거래비용, final wealth를 평가합니다.

### 전략

| 전략 | 설명 |
| --- | --- |
| IT | benchmark index와의 tracking error를 최소화하는 표준 index tracking 목적함수 |
| EIT | tracking error와 excess return을 함께 고려하는 enhanced index tracking 목적함수 |
| EIT-CVaR | downside tail risk를 제어하기 위해 CVaR penalty를 포함한 enhanced index tracking |
| RO | policy network 대신 rolling optimization 문제를 푸는 re-optimization baseline |

### 주요 파일

- `Index_tracking_v3.ipynb`: 메인 연구 노트북 및 실험 driver
- `config.py`: 데이터 경로, 학습 기간, 비용 수준, hyperparameter, policy variant 설정
- `data_loader.py`: KOSPI200 데이터 로딩, 날짜 정렬, 수익률 계산, universe 구성
- `hmm_model.py`: 2상태 Gaussian HMM 국면 추정
- `features.py`: rolling return, volatility, beta, regime, portfolio-state feature 생성
- `policy_network.py`: neural network policy architecture 정의
- `loss.py`: IT, EIT, EIT-CVaR objective 정의
- `trainer.py`: bootstrap path 생성, policy 학습, portfolio simulation
- `backtester.py`: rolling backtest와 re-optimization baseline 구현
- `evaluation.py`: 성과 지표와 diagnostic plot 계산

### 실행 방법

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

필요한 KOSPI200 CSV 파일을 프로젝트 루트에 두거나 `config.py`의 경로를 수정한 뒤, `Index_tracking_v3.ipynb`를 처음부터 실행합니다.

## English

This repository reproduces and extends the deep learning enhanced index tracking framework of Dai and Li (2024) on Korean equity market data. The main experiment tracks the KOSPI200 index using a small stock universe and compares index tracking, enhanced index tracking, CVaR-constrained enhanced index tracking, and a rolling re-optimization baseline.

### Project Summary

The main workflow is implemented around `Index_tracking_v3.ipynb` and supporting Python modules.

1. Load KOSPI200 adjusted close, index, and market capitalization data.
2. Build a tradable stock universe based on market capitalization and data availability.
3. Estimate bull and bear market regimes with two-state Gaussian HMM models.
4. Construct short-term stock and index features, including regime probability, rolling mean return, volatility, beta, and current portfolio weights.
5. Train neural network policy variants for IT, EIT, and EIT-CVaR objectives.
6. Compare neural network policies against a rolling re-optimization baseline.
7. Evaluate tracking error, excess return, information ratio, CVaR, Sharpe ratio, maximum drawdown, transaction cost, and final wealth.

### Strategies

| Strategy | Description |
| --- | --- |
| IT | Standard index tracking objective that minimizes tracking error against the benchmark index |
| EIT | Enhanced index tracking objective that balances tracking error and excess return |
| EIT-CVaR | Enhanced index tracking with a CVaR penalty to control downside tail risk |
| RO | Re-optimization baseline that solves a rolling optimization problem instead of learning a policy network |

### Main Files

- `Index_tracking_v3.ipynb`: Main research notebook and experiment driver
- `config.py`: Data paths, training years, cost levels, hyperparameters, and policy variants
- `data_loader.py`: Data loading, date alignment, return calculation, and universe construction
- `hmm_model.py`: Two-state Gaussian HMM regime estimation
- `features.py`: Rolling return, volatility, beta, regime, and portfolio-state feature construction
- `policy_network.py`: Neural network policy architecture
- `loss.py`: IT, EIT, and EIT-CVaR objectives
- `trainer.py`: Bootstrap path generation, policy training, and portfolio simulation
- `backtester.py`: Rolling backtest and re-optimization baseline
- `evaluation.py`: Performance metrics and diagnostic plots

### How to Run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Place the required KOSPI200 CSV files in the project root or update the paths in `config.py`, then run `Index_tracking_v3.ipynb` from top to bottom.

### Notes

This project is research code for empirical backtesting and reproducibility study. It is not intended for production trading or investment advice.
