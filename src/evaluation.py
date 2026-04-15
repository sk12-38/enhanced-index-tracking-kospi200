"""
evaluation.py
=============
논문 Table 1~4: 성과 지표 계산 및 결과 출력 모듈.

계산 지표:
  - TE   : Tracking Error = √(252 · mean((r_p - r_i)²))
  - MER  : Mean Excess Return = 252 · mean(r_p - r_i)
  - SR   : Sharpe Ratio = MER / (√252 · std(r_p - r_i))
  - MDD  : Maximum Drawdown
  - CVaR : CVaR(5%), 연율화
  - FW   : Final Wealth (초기 1.0 기준 누적)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 개별 지표 함수
# ─────────────────────────────────────────────────────────────────────────────

def annualized_tracking_error(port_rets: np.ndarray, index_rets: np.ndarray) -> float:
    """연율화 Tracking Error = √(252 · mean((r_p - r_i)²))"""
    diff = port_rets - index_rets
    return float(np.sqrt(252 * np.mean(diff ** 2)))


def annualized_excess_return(port_rets: np.ndarray, index_rets: np.ndarray) -> float:
    """연율화 Mean Excess Return = 252 · mean(r_p - r_i)"""
    return float(252 * np.mean(port_rets - index_rets))


def sharpe_ratio(port_rets: np.ndarray, index_rets: np.ndarray) -> float:
    """
    Information Ratio (Sharpe 변형):
    SR = MER / (√252 · std(r_p - r_i))
    """
    diff  = port_rets - index_rets
    te_daily = np.std(diff, ddof=1)
    if te_daily < 1e-10:
        return 0.0
    return float((252 * np.mean(diff)) / (np.sqrt(252) * te_daily))


def maximum_drawdown(port_rets: np.ndarray) -> float:
    """최대 낙폭 = max(1 - cumulative_wealth / peak_wealth)"""
    cumulative = np.cumprod(1 + port_rets)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = 1 - cumulative / running_max
    return float(np.max(drawdowns))


def cvar_5pct(port_rets: np.ndarray, alpha: float = 0.05) -> float:
    """
    CVaR(5%), 연율화.
    CVaR = -mean(하위 α 분위 수익률) × 252
    """
    k = max(1, int(len(port_rets) * alpha))
    sorted_rets = np.sort(port_rets)
    return float(-sorted_rets[:k].mean() * 252)


def final_wealth(port_rets: np.ndarray) -> float:
    """최종 부 = 초기 1.0 기준 누적 수익률."""
    return float(np.prod(1 + port_rets))


def compute_metrics(
    port_rets: np.ndarray,
    index_rets: np.ndarray,
    label: str = "",
) -> dict:
    """
    전체 성과 지표 딕셔너리 반환.

    Returns
    -------
    dict with keys: TE, MER, SR, MDD, CVaR5, FW, Label
    """
    return {
        "Label": label,
        "TE":    annualized_tracking_error(port_rets, index_rets),
        "MER":   annualized_excess_return(port_rets,  index_rets),
        "SR":    sharpe_ratio(port_rets,              index_rets),
        "MDD":   maximum_drawdown(port_rets),
        "CVaR5": cvar_5pct(port_rets),
        "FW":    final_wealth(port_rets),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 결과 테이블 생성
# ─────────────────────────────────────────────────────────────────────────────

def build_results_table(
    results_dict: dict,
    rho: float = 0.0,
) -> pd.DataFrame:
    """
    여러 전략의 결과를 하나의 DataFrame으로 정리합니다.

    Parameters
    ----------
    results_dict : {전략명: pd.DataFrame with ['port_ret', 'index_ret']}
    rho          : 거래비용률

    Returns
    -------
    table : pd.DataFrame (전략 × 지표)
    """
    rows = []
    for strategy, df in results_dict.items():
        pr = df["port_ret"].values
        ir = df["index_ret"].values
        m  = compute_metrics(pr, ir, label=f"{strategy} (ρ={rho})")
        rows.append(m)

    table = pd.DataFrame(rows).set_index("Label")
    table = table[["TE", "MER", "SR", "MDD", "CVaR5", "FW"]]
    table.columns = ["TE", "MER (%)", "Sharpe", "MDD", "CVaR(5%)", "Final Wealth"]
    table["MER (%)"] = table["MER (%)"] * 100       # 퍼센트 표시
    table["MDD"]     = table["MDD"]     * 100
    table["CVaR(5%)"]= table["CVaR(5%)"]* 100
    return table


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_wealth(
    results_dict: dict,
    rho: float = 0.0,
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
):
    """
    전략별 누적 부 곡선 플롯.

    Parameters
    ----------
    results_dict : {전략명: pd.DataFrame with ['port_ret', 'index_ret']}
    rho          : 거래비용률
    """
    fig, ax = plt.subplots(figsize=figsize)

    for strategy, df in results_dict.items():
        cumw = np.cumprod(1 + df["port_ret"].values)
        ax.plot(df.index, cumw, label=strategy, linewidth=1.5)

    # 인덱스 (첫 번째 결과에서)
    first_df = next(iter(results_dict.values()))
    idx_cumw = np.cumprod(1 + first_df["index_ret"].values)
    ax.plot(first_df.index, idx_cumw, label="KOSPI200 Index",
            color="black", linestyle="--", linewidth=1.5)

    ax.set_title(f"누적 부 (ρ={rho})", fontsize=14)
    ax.set_xlabel("날짜")
    ax.set_ylabel("누적 부 (초기=1)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_excess_returns(
    results_dict: dict,
    rho: float = 0.0,
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None,
):
    """초과수익(r_p - r_i) 롤링 연율화 플롯."""
    fig, axes = plt.subplots(len(results_dict), 1, figsize=figsize, sharex=True)
    if len(results_dict) == 1:
        axes = [axes]

    for ax, (strategy, df) in zip(axes, results_dict.items()):
        excess = df["port_ret"] - df["index_ret"]
        rolling_er = excess.rolling(63).mean() * 252
        ax.plot(df.index, rolling_er, linewidth=1.2)
        ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(f"{strategy} - 롤링 초과수익 (연율화, 63일)", fontsize=11)
        ax.set_ylabel("초과수익")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("날짜")
    plt.suptitle(f"초과수익 추이 (ρ={rho})", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_regime_probs(
    index_bull_prob: np.ndarray,
    dates: pd.DatetimeIndex,
    index_rets: np.ndarray,
    figsize: tuple = (14, 5),
):
    """인덱스 레짐 확률 및 수익률 시각화."""
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(dates, index_bull_prob, linewidth=1.2, color="steelblue")
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=0.8)
    axes[0].fill_between(dates, index_bull_prob, 0.5,
                          where=(index_bull_prob >= 0.5), alpha=0.3, color="green", label="bull")
    axes[0].fill_between(dates, index_bull_prob, 0.5,
                          where=(index_bull_prob < 0.5),  alpha=0.3, color="red",   label="bear")
    axes[0].set_title("KOSPI200 인덱스 레짐 확률 (bull 확률)")
    axes[0].set_ylabel("P(bull)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    cumret = np.cumprod(1 + index_rets)
    axes[1].plot(dates, cumret, linewidth=1.2, color="black")
    axes[1].set_title("KOSPI200 누적 수익")
    axes[1].set_ylabel("누적 수익 (초기=1)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_results_comparison(tables: dict):
    """
    ρ=0 / ρ=0.005 결과 테이블을 나란히 출력합니다.

    Parameters
    ----------
    tables : {'ρ=0': pd.DataFrame, 'ρ=0.005': pd.DataFrame}
    """
    print("\n" + "="*70)
    print("성과 지표 비교 (논문 Table 재현)")
    print("="*70)

    for label, table in tables.items():
        print(f"\n[ 거래비용: {label} ]")
        pd.set_option("display.float_format", "{:.4f}".format)
        print(table.to_string())

    print("="*70)
