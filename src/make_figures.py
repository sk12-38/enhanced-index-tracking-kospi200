"""
make_figures.py
===============
Dai & Li (2024) 논문 스타일 비교 그림 생성 모듈.

주요 함수
---------
run_all_variants()        : 4가지 정책 변형 전체 백테스트 실행
plot_wealth_curves()      : 누적 부 비교 (Figure 1 스타일)
plot_te_mer_bars()        : TE / MER 막대 비교 (Table 1 스타일)
plot_risk_bars()          : CVaR / MDD 막대 비교 (Table 1 스타일)
plot_weight_dispersion()  : 비중 분산 / 회전율 추이 (검증용)
validate_variant_weights(): 변형 간 비중 차이 검증 (Task D)

사용 예시
---------
from make_figures import run_all_variants, plot_wealth_curves

# 4가지 변형 실행
results_rho0 = run_all_variants(
    prices, index, mcap,
    strategy='EIT', rho=0.0,
    test_start_year=2010, test_end_year=2022,
)

# 비교 그림 생성
plot_wealth_curves(results_rho0, rho=0.0)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional

from backtester import RollingBacktest
from evaluation import compute_metrics
import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# 전체 변형 실행기
# ─────────────────────────────────────────────────────────────────────────────

def run_all_variants(
    prices: pd.DataFrame,
    index: pd.Series,
    mcap: pd.DataFrame,
    strategy: str = "EIT",
    rho: float = 0.0,
    top_n: int = cfg.TOP_N_DEFAULT,
    train_start_year: int = cfg.TRAIN_START_YEAR,
    test_start_year: int = cfg.TEST_START_YEAR,
    test_end_year: int = cfg.TEST_END_YEAR,
    device: str = "cpu",
    save_dir: str = "checkpoints",
    seed: int = cfg.RANDOM_SEED,
    verbose: bool = True,
    variants: list = None,
    use_normalization: bool = False,
    return_weight_history: bool = False,
) -> dict:
    """
    4가지 정책 변형을 순서대로 실행하고 결과를 딕셔너리로 반환합니다.

    Parameters
    ----------
    variants         : None이면 cfg.POLICY_VARIANTS 전체 실행
    use_normalization: True이면 학습/추론 시 z-score 정규화 적용

    Returns
    -------
    return_weight_history=False:
        results : {variant_name: pd.DataFrame with ['port_ret', 'index_ret']}
    return_weight_history=True:
        (results, weight_histories, turnover_histories)
        weight_histories  : {variant_name: pd.DataFrame(T×N_tickers)}
        turnover_histories: {variant_name: pd.Series(T,) — 리밸런싱 시점만 유효, 그 외 NaN}
    """
    if variants is None:
        variants = cfg.POLICY_VARIANTS

    results = {}
    weight_histories = {} if return_weight_history else None
    turnover_histories = {} if return_weight_history else None
    for variant in variants:
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# 변형: {variant} | 전략: {strategy} | ρ={rho} | norm={use_normalization}")
            print(f"{'#'*60}")

        bt = RollingBacktest(
            prices=prices,
            index=index,
            mcap=mcap,
            strategy=strategy,
            rho=rho,
            top_n=top_n,
            train_start_year=train_start_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
            device=device,
            save_dir=save_dir,
            seed=seed,
            verbose=verbose,
            policy_variant=variant,
            use_normalization=use_normalization,
        )
        results[variant] = bt.run()
        if return_weight_history:
            weight_histories[variant] = getattr(bt, "weight_history_", pd.DataFrame())
            turnover_histories[variant] = getattr(bt, "turnover_history_", pd.Series(dtype=float))

    if return_weight_history:
        return results, weight_histories, turnover_histories
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 신규: 목적함수별 전체 정책 실행 (4 variants + RO)
# ─────────────────────────────────────────────────────────────────────────────

def run_objective_all_policies(
    prices: pd.DataFrame,
    index: pd.Series,
    mcap: pd.DataFrame,
    strategy: str = "EIT",
    rho: float = 0.0,
    top_n: int = cfg.TOP_N_DEFAULT,
    train_start_year: int = cfg.TRAIN_START_YEAR,
    test_start_year: int = cfg.TEST_START_YEAR,
    test_end_year: int = cfg.TEST_END_YEAR,
    device: str = "cpu",
    seed: int = cfg.RANDOM_SEED,
    verbose: bool = True,
    use_normalization: bool = True,
    return_weight_history: bool = False,
) -> dict:
    """
    단일 목적함수(strategy)에 대해 4가지 정책 변형 + RO 베이스라인을 실행한다.

    Returns
    -------
    return_weight_history=False:
        results : {variant_name: df, 'RO': df}
                  각 df는 columns=['port_ret', 'index_ret'], index=DatetimeIndex
    return_weight_history=True:
        (results, weight_histories, turnover_histories)
        weight_histories   : {variant_name: weight_df, 'RO': weight_df}
        turnover_histories : {variant_name: pd.Series (리밸런싱 시점만 유효), 'RO': Series}
    """
    from backtester import ROBaseline

    if return_weight_history:
        results, weight_histories, turnover_histories = run_all_variants(
            prices=prices, index=index, mcap=mcap,
            strategy=strategy, rho=rho, top_n=top_n,
            train_start_year=train_start_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
            device=device, seed=seed, verbose=verbose,
            use_normalization=use_normalization,
            return_weight_history=True,
        )
    else:
        results = run_all_variants(
            prices=prices, index=index, mcap=mcap,
            strategy=strategy, rho=rho, top_n=top_n,
            train_start_year=train_start_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
            device=device, seed=seed, verbose=verbose,
            use_normalization=use_normalization,
        )

    if verbose:
        print(f"\n{'#'*60}")
        print(f"# RO 베이스라인 | 전략: {strategy} | ρ={rho}")
        print(f"{'#'*60}")

    ro = ROBaseline(
        prices=prices, index=index, mcap=mcap,
        strategy=strategy, rho=rho, top_n=top_n,
        train_start_year=train_start_year,
        test_start_year=test_start_year,
        test_end_year=test_end_year,
        device=device, seed=seed, verbose=verbose,
    )
    results["RO"] = ro.run()
    if return_weight_history:
        weight_histories["RO"] = getattr(ro, "weight_history_", pd.DataFrame())
        turnover_histories["RO"] = getattr(ro, "turnover_history_", pd.Series(dtype=float))

    if return_weight_history:
        return results, weight_histories, turnover_histories
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 신규: 단일 목적함수 누적 부 그림
# ─────────────────────────────────────────────────────────────────────────────

def plot_objective_wealth(
    results_dict: dict,
    strategy: str,
    rho: float = 0.0,
    top_n: int = cfg.TOP_N_DEFAULT,
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
):
    """
    단일 목적함수에 대한 누적 부 비교 그림 1장.

    - 4 policy 변형 + RO: 실선
    - Index: 검정 점선
    - 제목: '{strategy} | rho={rho} | top_n={N} | rebal={REBAL_FREQ}d'

    Parameters
    ----------
    results_dict : {label: pd.DataFrame with ['port_ret', 'index_ret']}
    strategy     : 목적함수 이름 (제목 표시용)
    rho          : 거래비용 (제목 표시용)
    save_path    : 저장 경로 (None이면 저장 안 함)
    """
    fig, ax = plt.subplots(figsize=figsize)

    first_df = None
    for label, df in results_dict.items():
        wealth = np.cumprod(1 + df["port_ret"].values)
        ax.plot(df.index, wealth, label=label, linewidth=1.6)
        if first_df is None:
            first_df = df

    if first_df is not None:
        idx_wealth = np.cumprod(1 + first_df["index_ret"].values)
        ax.plot(
            first_df.index, idx_wealth,
            label="Index",
            color="black",
            linestyle="--",
            linewidth=1.4,
        )

    title = (
        f"{strategy}  |  rho={rho}"
        f"  |  top_n={top_n}"
        f"  |  rebal={cfg.REBAL_FREQ}d"
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("날짜", fontsize=11)
    ax.set_ylabel("누적 부 (초기=1)", fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Figure] 저장: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 신규: IT / EIT / EIT-CVaR 세 목적함수 전체 실행 + 그림 저장
# ─────────────────────────────────────────────────────────────────────────────

def run_and_plot_all_objectives(
    prices: pd.DataFrame,
    index: pd.Series,
    mcap: pd.DataFrame,
    rho: float = 0.0,
    save_dir: str = "figures",
    top_n: int = cfg.TOP_N_DEFAULT,
    train_start_year: int = cfg.TRAIN_START_YEAR,
    test_start_year: int = cfg.TEST_START_YEAR,
    test_end_year: int = cfg.TEST_END_YEAR,
    device: str = "cpu",
    seed: int = cfg.RANDOM_SEED,
    verbose: bool = True,
    use_normalization: bool = True,
) -> dict:
    """
    IT / EIT / EIT-CVaR 3개 목적함수에 대해 각각:
      - 4 policy variants + RO 실행
      - 누적 부 그림 저장 (figures/wealth_{strategy}_rho{rho}.png)

    Returns
    -------
    all_results : {strategy: {variant/RO: df}}
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    objectives  = ["IT", "EIT", "EIT-CVaR"]
    all_results = {}

    for strategy in objectives:
        if verbose:
            print(f"\n{'='*60}")
            print(f"[목적함수] {strategy}  |  ρ={rho}  |  norm={use_normalization}")
            print(f"{'='*60}")

        res = run_objective_all_policies(
            prices=prices, index=index, mcap=mcap,
            strategy=strategy, rho=rho, top_n=top_n,
            train_start_year=train_start_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
            device=device, seed=seed, verbose=verbose,
            use_normalization=use_normalization,
        )
        all_results[strategy] = res

        save_path = os.path.join(save_dir, f"wealth_{strategy}_rho{rho}.png")
        plot_objective_wealth(
            res, strategy=strategy, rho=rho, top_n=top_n,
            save_path=save_path,
        )

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: 누적 부 곡선 비교
# ─────────────────────────────────────────────────────────────────────────────

def plot_wealth_curves(
    results_dict: dict,
    rho: float = 0.0,
    include_index: bool = True,
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
):
    """
    정책 변형별 누적 부 비교 곡선.

    논문 Figure와 동일하게:
      - 각 변형(NN-ST, NN-IR, NN-ISR, NN-All)이 선으로 표현
      - 인덱스 벤치마크는 검정 점선으로 표현
      - rho=0 / rho=0.005 별도 호출로 두 그림 생성

    Parameters
    ----------
    results_dict : {label: pd.DataFrame with ['port_ret', 'index_ret']}
    rho          : 거래비용 (제목 표시용)
    include_index: True이면 인덱스 벤치마크 선 포함
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, df in results_dict.items():
        wealth = np.cumprod(1 + df["port_ret"].values)
        ax.plot(df.index, wealth, label=label, linewidth=1.6)

    if include_index:
        first_df = next(iter(results_dict.values()))
        idx_wealth = np.cumprod(1 + first_df["index_ret"].values)
        ax.plot(
            first_df.index, idx_wealth,
            label="Index",
            color="black",
            linestyle="--",
            linewidth=1.4,
        )

    rho_label = f"ρ={rho}" if rho > 0 else "비용 없음 (ρ=0)"
    ax.set_title(f"누적 부 비교  [{rho_label}]", fontsize=13)
    ax.set_xlabel("날짜", fontsize=11)
    ax.set_ylabel("누적 부 (초기=1)", fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: TE / MER 막대 비교
# ─────────────────────────────────────────────────────────────────────────────

def plot_te_mer_bars(
    results_dict: dict,
    rho: float = 0.0,
    figsize: tuple = (10, 4),
    save_path: Optional[str] = None,
):
    """
    Tracking Error와 Mean Excess Return(연율화) 막대 그래프.

    각 변형을 x축에, TE(왼쪽)와 MER(오른쪽) 두 패널로 표시합니다.
    """
    labels = list(results_dict.keys())
    te_vals  = []
    mer_vals = []

    for label in labels:
        df = results_dict[label]
        m  = compute_metrics(df["port_ret"].values, df["index_ret"].values)
        te_vals.append(m["TE"] * 100)           # 퍼센트 표시
        mer_vals.append(m["MER"] * 100)

    x = np.arange(len(labels))
    width = 0.5

    # ── TE 그림 ─────────────────────────────────────────────────────────────
    fig_te, ax_te = plt.subplots(figsize=figsize)
    ax_te.bar(x, te_vals, width=width)
    ax_te.set_xticks(x)
    ax_te.set_xticklabels(labels, fontsize=11)
    ax_te.set_ylabel("Tracking Error (%)", fontsize=11)
    rho_label = f"ρ={rho}" if rho > 0 else "ρ=0"
    ax_te.set_title(f"Tracking Error 비교  [{rho_label}]", fontsize=13)
    ax_te.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig_te.savefig(save_path.replace(".png", "_TE.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # ── MER 그림 ────────────────────────────────────────────────────────────
    fig_mer, ax_mer = plt.subplots(figsize=figsize)
    bar_colors = ["tab:green" if v >= 0 else "tab:red" for v in mer_vals]
    ax_mer.bar(x, mer_vals, width=width, color=bar_colors)
    ax_mer.axhline(0, color="black", linewidth=0.8)
    ax_mer.set_xticks(x)
    ax_mer.set_xticklabels(labels, fontsize=11)
    ax_mer.set_ylabel("Mean Excess Return (%)", fontsize=11)
    ax_mer.set_title(f"Mean Excess Return 비교  [{rho_label}]", fontsize=13)
    ax_mer.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig_mer.savefig(save_path.replace(".png", "_MER.png"), dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: 위험 지표 막대 비교 (CVaR / MDD)
# ─────────────────────────────────────────────────────────────────────────────

def plot_risk_bars(
    results_dict: dict,
    rho: float = 0.0,
    figsize: tuple = (10, 4),
    save_path: Optional[str] = None,
):
    """
    CVaR(5%)와 최대 낙폭(MDD) 막대 그래프.
    값이 낮을수록 좋은 지표 (위험 최소화 관점).
    """
    labels    = list(results_dict.keys())
    cvar_vals = []
    mdd_vals  = []

    for label in labels:
        df = results_dict[label]
        m  = compute_metrics(df["port_ret"].values, df["index_ret"].values)
        cvar_vals.append(m["CVaR5"] * 100)
        mdd_vals.append(m["MDD"]   * 100)

    x     = np.arange(len(labels))
    width = 0.5
    rho_label = f"ρ={rho}" if rho > 0 else "ρ=0"

    # ── CVaR 그림 ───────────────────────────────────────────────────────────
    fig_cv, ax_cv = plt.subplots(figsize=figsize)
    ax_cv.bar(x, cvar_vals, width=width)
    ax_cv.set_xticks(x)
    ax_cv.set_xticklabels(labels, fontsize=11)
    ax_cv.set_ylabel("CVaR 5% (연율화, %)", fontsize=11)
    ax_cv.set_title(f"CVaR(5%) 비교  [{rho_label}]", fontsize=13)
    ax_cv.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig_cv.savefig(save_path.replace(".png", "_CVaR.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # ── MDD 그림 ────────────────────────────────────────────────────────────
    fig_md, ax_md = plt.subplots(figsize=figsize)
    ax_md.bar(x, mdd_vals, width=width)
    ax_md.set_xticks(x)
    ax_md.set_xticklabels(labels, fontsize=11)
    ax_md.set_ylabel("MDD (%)", fontsize=11)
    ax_md.set_title(f"최대 낙폭(MDD) 비교  [{rho_label}]", fontsize=13)
    ax_md.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig_md.savefig(save_path.replace(".png", "_MDD.png"), dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 (선택): 비중 분산 / 회전율 추이
# ─────────────────────────────────────────────────────────────────────────────

def plot_weight_dispersion(
    weight_records: dict,
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
):
    """
    리밸런싱 시점별 비중 단면 표준편차 및 회전율 추이.

    Parameters
    ----------
    weight_records : {
        variant_name: pd.DataFrame(
            index=rebalance_dates,
            columns=stock_tickers,
            data=weights,
        )
    }

    Note
    ----
    weight_records 수집이 필요합니다.
    run_all_variants()가 아닌 별도 루프에서 weights를 기록해야 합니다.
    """
    fig_std, ax_std = plt.subplots(figsize=figsize)
    fig_to,  ax_to  = plt.subplots(figsize=figsize)

    for label, wdf in weight_records.items():
        w_arr = wdf.values                                # (T_rebal, N)

        # 단면 표준편차: 각 리밸런싱 시점의 비중 분산
        cross_std = wdf.std(axis=1)
        ax_std.plot(wdf.index, cross_std, label=label, linewidth=1.4)

        # 회전율: |w_t - w_{t-1}|의 합
        diff   = np.abs(np.diff(w_arr, axis=0)).sum(axis=1)
        ax_to.plot(wdf.index[1:], diff, label=label, linewidth=1.4)

    ax_std.set_title("비중 단면 표준편차 (리밸런싱 시점)", fontsize=13)
    ax_std.set_xlabel("날짜", fontsize=11)
    ax_std.set_ylabel("std(w)", fontsize=11)
    ax_std.legend(fontsize=10)
    ax_std.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_path:
        fig_std.savefig(save_path.replace(".png", "_std.png"), dpi=150, bbox_inches="tight")
    plt.show()

    ax_to.set_title("포트폴리오 회전율 (리밸런싱 시점)", fontsize=13)
    ax_to.set_xlabel("날짜", fontsize=11)
    ax_to.set_ylabel("턴오버", fontsize=11)
    ax_to.legend(fontsize=10)
    ax_to.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_path:
        fig_to.savefig(save_path.replace(".png", "_turnover.png"), dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Task D: 변형 간 비중 차이 검증
# ─────────────────────────────────────────────────────────────────────────────

def validate_variant_weights(
    prices: pd.DataFrame,
    index: pd.Series,
    mcap: pd.DataFrame,
    strategy: str = "EIT",
    rho: float = 0.0,
    sample_year: int = cfg.TEST_START_YEAR,
    top_n: int = cfg.TOP_N_DEFAULT,
    device: str = "cpu",
    seed: int = cfg.RANDOM_SEED,
    n_rebalance_samples: int = 5,
):
    """
    Task D: 4가지 변형이 실제로 다른 비중을 생성하는지 검증.

    단일 샘플 연도에 대해:
    - 변형 쌍별 평균 절대 비중 차이 출력
    - 변형별 비중 단면 표준편차 출력 (균등비중 붕괴 여부 확인)

    Parameters
    ----------
    n_rebalance_samples : 검증에 사용할 리밸런싱 시점 수 (처음 N개)
    """
    import torch
    import torch.nn.functional as F
    from policy_network import PolicyNetwork, VARIANT_FLAGS
    from data_loader import build_universe, get_period_data
    from hmm_model import HMMCollection
    from features import FeatureBuilder

    print(f"\n{'='*60}")
    print(f"[검증] 변형 간 비중 차이 | 전략={strategy} | 연도={sample_year}")
    print(f"{'='*60}")

    train_start = pd.Timestamp(f"{cfg.TRAIN_START_YEAR}-01-01")
    train_end   = pd.Timestamp(f"{sample_year - 1}-12-31")
    test_start  = pd.Timestamp(f"{sample_year}-01-01")
    test_end    = pd.Timestamp(f"{sample_year}-12-31")

    universe = build_universe(
        prices, mcap,
        train_start=train_start, train_end=train_end,
        top_n=top_n,
    )
    N = len(universe)

    train_sr, train_ir, _ = get_period_data(
        prices, index, mcap,
        start=train_start, end=train_end, universe=universe,
    )
    full_sr, full_ir, _ = get_period_data(
        prices, index, mcap,
        start=train_start, end=test_end, universe=universe,
    )

    hmm = HMMCollection(random_state=seed)
    hmm.fit_all(train_ir.values, train_sr, verbose=False)

    full_ib    = hmm.index_bull_prob(full_ir.values, smooth=cfg.REGIME_SMOOTH)
    full_sb    = hmm.stock_bull_probs(full_sr, smooth=cfg.REGIME_SMOOTH).values
    full_ib_s  = pd.Series(full_ib, index=full_ir.index)
    full_sb_df = pd.DataFrame(full_sb, index=full_sr.index, columns=universe)
    feat_builder = FeatureBuilder(full_sr, full_ir, window=cfg.WINDOW_ST)

    test_mask  = (full_sr.index >= test_start) & (full_sr.index <= test_end)
    test_dates = full_sr.index[test_mask]

    # 유효한 리밸런싱 날짜 수집
    valid_dates = []
    for i, date in enumerate(test_dates):
        if i % cfg.REBAL_FREQ != 0:
            continue
        if date not in feat_builder.mean_rets.index:
            continue
        if feat_builder.mean_rets.loc[date].isna().any():
            continue
        valid_dates.append(date)
        if len(valid_dates) >= n_rebalance_samples:
            break

    if not valid_dates:
        print("  [경고] 유효한 리밸런싱 날짜가 없습니다.")
        return

    # 각 변형별 비중 수집
    prev_w_np = np.ones(N) / N
    variant_weights = {v: [] for v in cfg.POLICY_VARIANTS}

    for variant in cfg.POLICY_VARIANTS:
        # 간단한 초기 비중으로 forward 테스트 (실제 학습 없이 초기화된 네트워크 사용)
        policy = PolicyNetwork.from_variant(variant, n_stocks=N).to(device)
        policy.eval()

        for date in valid_dates:
            idx_prob  = float(full_ib_s.loc[date])
            stk_probs = full_sb_df.loc[date].values
            idx_mean = float(np.nan_to_num(feat_builder.index_mean_rets.loc[date], nan=0.0))
            idx_vol = float(np.nan_to_num(feat_builder.index_vols.loc[date], nan=0.0))
            w = policy.predict(
                index_regime_prob=idx_prob,
                stock_features_np=np.stack([
                    feat_builder.mean_rets.loc[date].values,
                    feat_builder.vols.loc[date].values,
                    feat_builder.betas.loc[date].values,
                    np.full(N, idx_mean, dtype=float),
                    np.full(N, idx_vol, dtype=float),
                ], axis=1),
                stock_regime_probs_np=stk_probs,
                prev_weights_np=prev_w_np,
                device=device,
            )
            variant_weights[variant].append(w)

    # ── 비중 단면 표준편차 (균등비중 붕괴 여부) ────────────────────────────
    print("\n[비중 단면 표준편차] (값이 0에 가까우면 균등비중에 가까움)")
    uniform_std = float(np.std(np.ones(N) / N))
    print(f"  균등비중 기준 std = {uniform_std:.5f} (N={N})")
    for variant in cfg.POLICY_VARIANTS:
        ws = np.array(variant_weights[variant])   # (n_samples, N)
        mean_std = float(ws.std(axis=1).mean())
        print(f"  {variant:8s}: mean cross-sectional std = {mean_std:.5f}")

    # ── 변형 쌍별 평균 절대 비중 차이 ────────────────────────────────────
    print("\n[변형 쌍별 평균 절대 비중 차이]")
    variants = cfg.POLICY_VARIANTS
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            va, vb = variants[i], variants[j]
            ws_a = np.array(variant_weights[va])
            ws_b = np.array(variant_weights[vb])
            mean_abs_diff = float(np.abs(ws_a - ws_b).mean())
            print(f"  {va:8s} vs {vb:8s}: mean |Δw| = {mean_abs_diff:.5f}")

    print(f"\n{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 요약 메트릭 테이블 (단일 dict 입력)
# ─────────────────────────────────────────────────────────────────────────────

def build_metrics_table(results_dict: dict, rho: float = 0.0) -> pd.DataFrame:
    """
    모든 변형의 성과 지표를 하나의 DataFrame으로 정리합니다.

    Parameters
    ----------
    results_dict : {label: pd.DataFrame with ['port_ret', 'index_ret']}

    Returns
    -------
    table : pd.DataFrame (변형 × 지표)
    """
    rows = []
    for label, df in results_dict.items():
        m = compute_metrics(df["port_ret"].values, df["index_ret"].values)
        rows.append({
            "Variant":      label,
            "TE (%)":       round(m["TE"]    * 100, 4),
            "MER (%)":      round(m["MER"]   * 100, 4),
            "Sharpe":       round(m["SR"],          4),
            "MDD (%)":      round(m["MDD"]   * 100, 4),
            "CVaR5 (%)":    round(m["CVaR5"] * 100, 4),
            "Final Wealth": round(m["FW"],          4),
        })
    return pd.DataFrame(rows).set_index("Variant")


# ─────────────────────────────────────────────────────────────────────────────
# 편의 함수: rho=0 / rho=0.005 전체 그림 세트 생성
# ─────────────────────────────────────────────────────────────────────────────

def make_all_figures(
    results_rho0: dict,
    results_rho005: dict,
    save_dir: Optional[str] = None,
):
    """
    논문 스타일 전체 그림 세트 생성.

    Parameters
    ----------
    results_rho0   : run_all_variants() 결과 (rho=0.0)
    results_rho005 : run_all_variants() 결과 (rho=0.005)
    save_dir       : None이면 저장 안 함
    """
    def _path(name):
        if save_dir is None:
            return None
        import os
        os.makedirs(save_dir, exist_ok=True)
        return f"{save_dir}/{name}.png"

    # ── 누적 부 곡선 ─────────────────────────────────────────────────────────
    plot_wealth_curves(results_rho0,   rho=0.0,   save_path=_path("wealth_rho0"))
    plot_wealth_curves(results_rho005, rho=0.005, save_path=_path("wealth_rho005"))

    # ── TE / MER 막대 ────────────────────────────────────────────────────────
    plot_te_mer_bars(results_rho0,   rho=0.0,   save_path=_path("te_mer_rho0"))
    plot_te_mer_bars(results_rho005, rho=0.005, save_path=_path("te_mer_rho005"))

    # ── 위험 지표 막대 ────────────────────────────────────────────────────────
    plot_risk_bars(results_rho0,   rho=0.0,   save_path=_path("risk_rho0"))
    plot_risk_bars(results_rho005, rho=0.005, save_path=_path("risk_rho005"))

    # ── 메트릭 테이블 출력 ────────────────────────────────────────────────────
    print("\n[성과 지표 — ρ=0]")
    print(build_metrics_table(results_rho0,   rho=0.0).to_string())

    print("\n[성과 지표 — ρ=0.005]")
    print(build_metrics_table(results_rho005, rho=0.005).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 비중 이력 시각화 (논문 "weight evolution by policy" 섹션 재현)
# ─────────────────────────────────────────────────────────────────────────────

def plot_weight_stacked(
    weight_df: pd.DataFrame,
    cash_series: Optional[pd.Series] = None,
    policy_name: str = "",
    top_k: int = 10,
    strategy: str = "",
    rho: float = 0.0,
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
) -> tuple:
    """
    포트폴리오 비중 추이 — 스택 면적 차트 (stackplot).

    파라미터
    --------
    weight_df   : (T × N) DataFrame, DatetimeIndex, 열=종목 코드/이름
                  여러 연도 연결 시 유니버스가 달라 NaN이 있을 수 있음 → 0으로 채움.
    cash_series : (T,) Series, 암묵적 현금 비중 (None이면 생략).
                  rho=0 이면 항상 0 → 생략해도 무방.
    policy_name : 제목 및 파일명에 사용할 정책 이름.
    top_k       : 평균 비중 상위 K 종목 개별 표시; 나머지는 'Others' 집계.
    strategy    : 제목 표시용 목적함수 이름.
    rho         : 제목 표시용 거래비용률.
    save_path   : 저장 경로 (None이면 저장 안 함).

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
    """
    import os

    # NaN → 0 (다중 연도 연결 시 유니버스 차이)
    wdf = weight_df.fillna(0.0)

    # 평균 비중 기준 상위 top_k 종목 선정
    avg_w       = wdf.mean(axis=0).sort_values(ascending=False)
    top_tickers = avg_w.index[:top_k].tolist()
    others      = wdf.drop(columns=top_tickers, errors="ignore").sum(axis=1)  # (T,)
    top_data    = wdf[top_tickers]   # (T, top_k)

    # stackplot 배열: 하단부터 [top_k 종목..., Others, (Cash)]
    # 상위 종목을 역순(평균 비중 작은 것부터) 으로 쌓아 큰 종목이 바닥에 오게 함
    stack_labels = list(reversed(top_tickers)) + ["Others"]
    stack_arrays = [top_data[s].values for s in reversed(top_tickers)] + [others.values]

    if cash_series is not None:
        cs = cash_series.reindex(wdf.index).fillna(0.0)
        stack_arrays.append(cs.values)
        stack_labels.append("Cash")

    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(wdf.index, stack_arrays, labels=stack_labels)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("날짜", fontsize=11)
    ax.set_ylabel("포트폴리오 비중", fontsize=11)

    title_parts = [f"비중 추이: {policy_name}"]
    if strategy:
        title_parts.append(f"전략={strategy}")
    title_parts.append(f"ρ={rho}")
    ax.set_title("  |  ".join(title_parts), fontsize=13)

    # 범례: 역순 → 차트 위쪽 항목이 범례 상단에 오도록
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1], labels[::-1],
        fontsize=8, loc="upper left",
        bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    )
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Figure] 저장: {save_path}")
    plt.show()
    return fig, ax


def plot_cash_turnover(
    cash_series: pd.Series,
    turnover_series: Optional[pd.Series] = None,
    policy_name: str = "",
    strategy: str = "",
    rho: float = 0.0,
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
) -> tuple:
    """
    현금 비중 & 회전율 시계열 선/막대 그래프.

    레이아웃
    --------
    - 상단 패널 : 현금 비중 시계열 (선 그래프)
    - 하단 패널 : 리밸런싱 시점 L1 회전율 (막대 그래프, NaN 구간 제외)
                  turnover_series=None 이면 단일 패널.

    파라미터
    --------
    cash_series     : (T,) Series, 현금 비중 (bt.cash_history_).
    turnover_series : (T,) Series, 회전율 (bt.turnover_history_); NaN=비리밸런싱.
    save_path       : 저장 경로 (None이면 저장 안 함).

    Returns
    -------
    fig, axes : matplotlib Figure, list of Axes
    """
    import os

    has_turnover = turnover_series is not None
    nrows  = 2 if has_turnover else 1
    height = figsize[1] * nrows
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1,
        figsize=(figsize[0], height),
        sharex=True,
    )
    if nrows == 1:
        axes = [axes]

    # ── 현금 비중 패널 ──────────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.plot(cash_series.index, cash_series.fillna(0.0).values, linewidth=1.4)
    ax0.set_ylabel("현금 비중", fontsize=11)
    ax0.set_ylim(bottom=0.0)
    ax0.grid(True, alpha=0.25)

    title_parts = [f"현금·회전율: {policy_name}"]
    if strategy:
        title_parts.append(f"전략={strategy}")
    title_parts.append(f"ρ={rho}")
    ax0.set_title("  |  ".join(title_parts), fontsize=13)

    # ── 회전율 패널 (옵션) ──────────────────────────────────────────────────
    if has_turnover:
        ax1  = axes[1]
        to   = turnover_series.dropna()   # 리밸런싱 시점만
        ax1.bar(to.index, to.values, width=2)
        ax1.set_ylabel("회전율 (L1)", fontsize=11)
        ax1.set_xlabel("날짜", fontsize=11)
        ax1.grid(True, alpha=0.25)

    plt.tight_layout()

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Figure] 저장: {save_path}")
    plt.show()
    return fig, list(axes)


def plot_all_policy_weights(
    bts_dict: dict,
    save_dir: str = "figures/weights",
    top_k: int = 10,
    strategy: str = "",
    rho: float = 0.0,
    include_cash_turnover: bool = True,
) -> None:
    """
    여러 정책의 비중 추이를 순서대로 시각화한다.

    각 정책에 대해 자동으로 생성되는 그림
    ----------------------------------------
    1) 스택 면적 차트 (비중 추이)
       파일명: weights_stack_{policy}_{strategy}_rho{rho}.png
    2) 현금 비중 & 회전율 (include_cash_turnover=True)
       파일명: cash_turnover_{policy}_{strategy}_rho{rho}.png

    파라미터
    --------
    bts_dict : {policy_name: RollingBacktest 또는 ROBaseline 인스턴스}
               인스턴스에 .weight_history_, .cash_history_, .turnover_history_ 속성이
               있어야 함 (record_weights=True 로 run() 호출 후 자동 설정됨).
    save_dir : 그림 저장 디렉토리.
    top_k    : 스택 차트에 개별 표시할 최대 종목 수.
    strategy : 제목/파일명에 사용할 목적함수 이름.
    rho      : 제목/파일명에 사용할 거래비용률.
    include_cash_turnover : False면 스택 차트만 생성.

    사용 예시
    ---------
    from make_figures import plot_all_policy_weights

    bts = {}
    results = {}
    for variant in ['NN-ST', 'NN-IR', 'NN-ISR', 'NN-All']:
        bt = RollingBacktest(..., policy_variant=variant, record_weights=True)
        results[variant] = bt.run()
        bts[variant] = bt

    ro = ROBaseline(..., record_weights=True)
    results['RO'] = ro.run()
    bts['RO'] = ro

    plot_all_policy_weights(bts, strategy='EIT', rho=0.005)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    for name, bt in bts_dict.items():
        if not hasattr(bt, "weight_history_"):
            print(
                f"[경고] '{name}': weight_history_ 없음. "
                "record_weights=True 로 run() 을 먼저 호출하세요."
            )
            continue

        cash_s = getattr(bt, "cash_history_",     None)
        trn_s  = getattr(bt, "turnover_history_", None)

        # ── (1) 스택 면적 차트 ───────────────────────────────────────────
        sp1 = os.path.join(
            save_dir, f"weights_stack_{name}_{strategy}_rho{rho}.png"
        )
        plot_weight_stacked(
            weight_df   = bt.weight_history_,
            cash_series = cash_s,
            policy_name = name,
            top_k       = top_k,
            strategy    = strategy,
            rho         = rho,
            save_path   = sp1,
        )

        # ── (2) 현금 비중 & 회전율 ───────────────────────────────────────
        if include_cash_turnover and cash_s is not None:
            sp2 = os.path.join(
                save_dir, f"cash_turnover_{name}_{strategy}_rho{rho}.png"
            )
            plot_cash_turnover(
                cash_series     = cash_s,
                turnover_series = trn_s,
                policy_name     = name,
                strategy        = strategy,
                rho             = rho,
                save_path       = sp2,
            )
