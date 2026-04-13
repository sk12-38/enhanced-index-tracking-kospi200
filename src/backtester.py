"""
backtester.py
=============
논문 Section 5: 롤링 연도별 백테스트 및 RO 베이스라인.

롤링 백테스트 프로토콜:
  - 각 테스트 연도마다:
    1. 훈련 시작 ~ 테스트 연도 전년 말까지 누적 데이터로 HMM 재피팅
    2. 정책망 재학습
    3. 테스트 연도 내 5일마다 리밸런싱

RO 베이스라인:
  - 각 리밸런싱 시점마다:
    - 과거 2년 수익률 데이터로 EIT/EIT-CVaR 목적함수 직접 최적화
    - 블록 부트스트랩 200 에포크
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Optional

from data_loader import load_price_data, build_universe, get_period_data
from hmm_model import HMMCollection
from features import FeatureBuilder, FeatureNormalizer, build_path_features
from policy_network import PolicyNetwork, VARIANT_FLAGS
from trainer import train_policy, set_seed, block_bootstrap_paths, simulate_portfolio_returns
from loss import get_loss_fn
from ro_optimizer import ro_optimize_weights
from simulator import paper_step_numpy
import config as cfg


def _carry_state_to_universe(
    a_hold_init: np.ndarray,
    c_hold_init: float,
    tickers_init: list[str],
    universe: list[str],
    rho: float,
) -> tuple[np.ndarray, float]:
    """
    연도 경계에서 이전 장부를 새 유니버스 축으로 정렬한다.

    규칙:
      - 겹치는 티커: 기존 보유액 유지
      - 제외 티커: 경계에서 청산 후 거래비용(rho * |trade|) 차감, 순현금 유입
      - 신규 티커: 0으로 시작
    """
    a_prev = np.asarray(a_hold_init, dtype=float)
    c_new = float(c_hold_init)

    idx_new = {t: i for i, t in enumerate(universe)}
    a_new = np.zeros(len(universe), dtype=float)
    removed_vals = []

    n_prev = min(len(a_prev), len(tickers_init))
    for i in range(n_prev):
        t = tickers_init[i]
        val = float(a_prev[i])
        j = idx_new.get(t)
        if j is None:
            removed_vals.append(val)
        else:
            a_new[j] = val

    if removed_vals:
        removed_arr = np.asarray(removed_vals, dtype=float)
        liquidation_cost = rho * float(np.abs(removed_arr).sum())
        c_new = c_new + float(removed_arr.sum()) - liquidation_cost
        if c_new < 0.0 and c_new > -1e-10:
            c_new = 0.0

    return a_new, c_new


class RollingBacktest:
    """
    연도별 롤링 백테스트 엔진.

    Usage
    -----
    bt = RollingBacktest(
        prices, index, mcap,
        strategy='EIT', rho=0.0, top_n=20,
        policy_variant='NN-All',  # 'NN-ST' | 'NN-IR' | 'NN-ISR' | 'NN-All'
    )
    results = bt.run()
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        index: pd.Series,
        mcap: pd.DataFrame,
        strategy: str = "EIT",
        rho: float = 0.0,
        top_n: int = cfg.TOP_N_DEFAULT,
        train_start_year: int = cfg.TRAIN_START_YEAR,
        test_start_year: int  = cfg.TEST_START_YEAR,
        test_end_year: int    = cfg.TEST_END_YEAR,
        device: str = "cpu",
        save_dir: str = "checkpoints",
        seed: int = cfg.RANDOM_SEED,
        verbose: bool = True,
        policy_variant: str = cfg.POLICY_VARIANT,
        use_normalization: bool = False,
        record_weights: bool = True,
    ):
        self.prices             = prices
        self.index              = index
        self.mcap               = mcap
        self.strategy           = strategy
        self.rho                = rho
        self.top_n              = top_n
        self.train_start_year   = train_start_year
        self.test_start_year    = test_start_year
        self.test_end_year      = test_end_year
        self.device             = device
        self.save_dir           = save_dir
        self.seed               = seed
        self.verbose            = verbose
        self.policy_variant     = policy_variant
        self.use_normalization  = use_normalization
        self.record_weights     = record_weights

        if policy_variant not in VARIANT_FLAGS:
            raise ValueError(
                f"알 수 없는 policy_variant: '{policy_variant}'. "
                f"선택 가능: {list(VARIANT_FLAGS.keys())}"
            )

        # use_normalization=True 이면 별도 디렉토리 사용
        if use_normalization and save_dir == "checkpoints":
            self.save_dir = cfg.CHECKPOINT_DIR_NORM

        os.makedirs(self.save_dir, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """
        전체 롤링 백테스트 실행.

        Returns
        -------
        results : DataFrame (test_dates × ['port_ret', 'index_ret'])

        Side-effects (record_weights=True 일 때 설정되는 인스턴스 속성)
        ------------------------------------------------------------------
        weight_history_   : DataFrame (T × N_tickers), 각 날짜의 주식 비중
                            다중 연도에서 유니버스가 달라지면 NaN으로 채워짐.
        cash_history_     : Series (T,), 암묵적 현금 비중 = 1 - sum(w_stock)
        turnover_history_ : Series (T,), 리밸런싱 시점 L1 회전율 (그 외 NaN)
        alpha_history_    : Series (T,), 리밸런싱 시점 실현 가능성 alpha (그 외 NaN)
        cost_history_     : Series (T,), 리밸런싱 시점 거래비용(달러, 그 외 NaN)
        value_history_    : Series (T,), 일별 포트폴리오 가치
        """
        all_port_rets  = []
        all_index_rets = []
        all_dates      = []
        _all_wdfs:     list = []
        _all_cash:     list = []
        _all_turnover: list = []
        _all_alpha:    list = []
        _all_cost:     list = []
        _all_value:    list = []

        carry_state = None
        for test_year in range(self.test_start_year, self.test_end_year + 1):
            if self.verbose:
                print(f"\n{'='*60}")
                print(
                    f"[Backtest] 테스트 연도: {test_year} | 변형: {self.policy_variant}"
                    f" | 전략: {self.strategy} | ρ={self.rho}"
                )

            df_year, a_end, c_end, tickers_end = self._run_year(
                test_year=test_year,
                initial_state=carry_state,
            )
            winfo = df_year.attrs.get("winfo")

            all_port_rets.extend(df_year["port_ret"].tolist())
            all_index_rets.extend(df_year["index_ret"].tolist())
            all_dates.extend(df_year.index.tolist())
            carry_state = (a_end, c_end, tickers_end)

            if winfo is not None:
                _all_wdfs.append(pd.DataFrame(
                    winfo["weights"],
                    index=pd.DatetimeIndex(df_year.index),
                    columns=winfo["tickers"],
                ))
                _all_cash.extend(winfo["cash"].tolist())
                _all_turnover.extend(winfo["turnover"].tolist())
                _all_alpha.extend(winfo["alpha"].tolist())
                _all_cost.extend(winfo["cost"].tolist())
                _all_value.extend(winfo["value"].tolist())

        results = pd.DataFrame({
            "port_ret":  all_port_rets,
            "index_ret": all_index_rets,
        }, index=pd.DatetimeIndex(all_dates))

        if self.record_weights and _all_wdfs:
            self.weight_history_   = pd.concat(_all_wdfs)
            _idx = pd.DatetimeIndex(all_dates)
            self.cash_history_     = pd.Series(_all_cash,     index=_idx, name="w_cash")
            self.turnover_history_ = pd.Series(_all_turnover, index=_idx, name="turnover")
            self.alpha_history_    = pd.Series(_all_alpha,     index=_idx, name="alpha")
            self.cost_history_     = pd.Series(_all_cost,      index=_idx, name="trading_cost")
            self.value_history_    = pd.Series(_all_value,     index=_idx, name="portfolio_value")

        return results

    def _run_year(
        self,
        test_year: int,
        initial_state: Optional[tuple[np.ndarray, float, list[str]]] = None,
    ) -> tuple:
        """단일 테스트 연도 백테스트."""
        train_start = pd.Timestamp(f"{self.train_start_year}-01-01")
        train_end   = pd.Timestamp(f"{test_year - 1}-12-31")
        test_start  = pd.Timestamp(f"{test_year}-01-01")
        test_end    = pd.Timestamp(f"{test_year}-12-31")

        # ── 유니버스 구성 ──────────────────────────────────────────────────
        universe = build_universe(
            self.prices, self.mcap,
            train_start=train_start, train_end=train_end,
            top_n=self.top_n,
        )

        # ── 훈련 데이터 ────────────────────────────────────────────────────
        train_sr, train_ir, _ = get_period_data(
            self.prices, self.index, self.mcap,
            start=train_start, end=train_end, universe=universe,
        )

        # ── HMM 피팅 (모든 변형 공통: train_policy가 레짐 배열을 요구) ─────
        hmm = HMMCollection(random_state=self.seed)
        hmm.fit_all(train_ir.values, train_sr, verbose=self.verbose)

        train_ib = hmm.index_bull_prob(train_ir.values, smooth=cfg.REGIME_SMOOTH)
        train_sb = hmm.stock_bull_probs(train_sr, smooth=cfg.REGIME_SMOOTH).values

        # ── 테스트 기간 피처 빌더 ──────────────────────────────────────────
        full_sr, full_ir, _ = get_period_data(
            self.prices, self.index, self.mcap,
            start=train_start, end=test_end, universe=universe,
        )
        feat_builder = FeatureBuilder(full_sr, full_ir, window=cfg.WINDOW_ST)

        # ── 피처 정규화 (선택) ─────────────────────────────────────────────
        normalizer = None
        if self.use_normalization:
            normalizer = FeatureNormalizer()
            normalizer.fit_from_builder(feat_builder, train_end)
            if self.verbose:
                normalizer.log_stats()

        # ── 정책망 학습 ────────────────────────────────────────────────────
        # 변형에 맞는 PolicyNetwork 생성 (from_variant 사용)
        policy = PolicyNetwork.from_variant(
            self.policy_variant, n_stocks=len(universe)
        ).to(self.device)

        # 체크포인트 이름: 변형명 + norm 태그 포함
        norm_tag = "_normTrue" if self.use_normalization else ""
        save_path = os.path.join(
            self.save_dir,
            f"{self.policy_variant}_{self.strategy}_rho{self.rho}"
            f"{norm_tag}_top{self.top_n}_{test_year}.pt"
        )

        policy = train_policy(
            policy,
            stock_rets=train_sr.values,
            index_rets=train_ir.values,
            index_bull=train_ib,
            stock_bull=train_sb,
            strategy=self.strategy,
            rho=self.rho,
            save_path=save_path,
            device=self.device,
            seed=self.seed,
            verbose=self.verbose,
            normalizer=normalizer,
        )

        # 테스트 기간 HMM 추론 (파라미터 고정)
        full_ib = hmm.index_bull_prob(full_ir.values, smooth=cfg.REGIME_SMOOTH)
        full_sb = hmm.stock_bull_probs(full_sr, smooth=cfg.REGIME_SMOOTH).values
        full_ib_s  = pd.Series(full_ib, index=full_ir.index)
        full_sb_df = pd.DataFrame(full_sb, index=full_sr.index, columns=universe)

        # ── 테스트 기간 리밸런싱 루프 ──────────────────────────────────────
        test_dates_mask = (full_sr.index >= test_start) & (full_sr.index <= test_end)
        test_dates      = full_sr.index[test_dates_mask]

        port_rets_list  = []
        index_rets_list = []
        used_dates      = []
        # 비중/현금/거래 이력 추적용 (record_weights=True 일 때만 사용)
        _wgt_list = [] if self.record_weights else None
        _csh_list = [] if self.record_weights else None
        _trn_list = [] if self.record_weights else None
        _alp_list = [] if self.record_weights else None
        _cst_list = [] if self.record_weights else None
        _val_list = [] if self.record_weights else None

        # 변형별 플래그: score block 사용하는 경우에만 short-term NaN 체크
        need_st = getattr(policy, "use_score", True)

        # 연도 간 상태 이월:
        # - 첫 테스트 연도는 all-cash로 시작
        # - 이후 연도는 직전 연도 종료 장부(a_hold, c_hold)를 새 유니버스 축으로 정렬
        n_assets = len(universe)
        if initial_state is None:
            a_hold = np.zeros(n_assets, dtype=float)
            c_hold = 1.0
        else:
            a_hold, c_hold = _carry_state_to_universe(
                a_hold_init=initial_state[0],
                c_hold_init=initial_state[1],
                tickers_init=initial_state[2],
                universe=list(universe),
                rho=self.rho,
            )

        policy.eval()
        for i, date in enumerate(test_dates):
            # 매 반복 초기화: 리밸런싱이 실제로 발생한 경우에만 덮어씌워짐
            _rebal_turnover = np.nan
            _rebal_alpha    = np.nan
            _rebal_cost     = np.nan

            if date not in full_sr.index:
                continue

            v_pre = float(c_hold + a_hold.sum())
            if v_pre <= 0.0:
                continue

            w_pre = a_hold / v_pre

            rebalance_now = False
            w_target = None
            if i % cfg.REBAL_FREQ == 0:
                if date not in feat_builder.mean_rets.index:
                    continue

                idx_prob  = float(full_ib_s.loc[date])
                stk_probs = full_sb_df.loc[date].values

                # 단기 통계를 사용하는 변형(NN-ST, NN-All)만 NaN 구간 스킵
                if need_st:
                    row_mean = feat_builder.mean_rets.loc[date]
                    if row_mean.isna().any():
                        continue

                idx_mean = float(np.nan_to_num(feat_builder.index_mean_rets.loc[date], nan=0.0))
                idx_vol = float(np.nan_to_num(feat_builder.index_vols.loc[date], nan=0.0))
                raw_feat = np.stack([
                    feat_builder.mean_rets.loc[date].values,
                    feat_builder.vols.loc[date].values,
                    feat_builder.betas.loc[date].values,
                    np.full(len(universe), idx_mean, dtype=float),
                    np.full(len(universe), idx_vol, dtype=float),
                ], axis=1)   # (N, 5)
                feat_in = (
                    normalizer.transform_stock_features(raw_feat)
                    if normalizer is not None else raw_feat
                )

                w_target = policy.predict(
                    index_regime_prob     = idx_prob,
                    stock_features_np     = feat_in,
                    stock_regime_probs_np = stk_probs,
                    prev_weights_np       = w_pre,
                    device                = self.device,
                )
                rebalance_now = True
                _rebal_alpha = 1.0
                if self.verbose:
                    sum_w_tgt = float(np.sum(w_target))
                    print(
                        f"[Rebalance] {date.date()} | v_pre={v_pre:.6f} "
                        f"| sum_w_target={sum_w_tgt:.6f}"
                    )

            r_stocks = full_sr.loc[date].values
            r_idx    = float(full_ir.loc[date])
            w_stock_next, w_cash_next, port_ret, turnover, cost_frac = paper_step_numpy(
                w_stock_pre=w_pre,
                w_cash_pre=float(c_hold / max(v_pre, 1e-12)),
                r_stock_next=r_stocks,
                rho=self.rho,
                rebalance=rebalance_now,
                w_stock_target=w_target,
            )
            v_post = float(v_pre * (1.0 + port_ret))
            a_hold = np.asarray(w_stock_next, dtype=float) * v_post
            c_hold = float(w_cash_next) * v_post
            if rebalance_now:
                _rebal_turnover = float(turnover)
                _rebal_cost = float(cost_frac * v_pre)

            port_rets_list.append(port_ret)
            index_rets_list.append(r_idx)
            used_dates.append(date)

            w_post = np.asarray(w_stock_next, dtype=float)
            w_cash_post = float(w_cash_next)

            if self.record_weights:
                _wgt_list.append(w_post.copy())
                _csh_list.append(w_cash_post)
                _trn_list.append(_rebal_turnover)                 # NaN: 비리밸런싱일
                _alp_list.append(_rebal_alpha)                    # NaN: 비리밸런싱일
                _cst_list.append(_rebal_cost)                     # NaN: 비리밸런싱일
                _val_list.append(v_post)

        winfo = None
        if self.record_weights and _wgt_list:
            winfo = {
                "weights":  np.array(_wgt_list, dtype=np.float32),   # (T, N)
                "cash":     np.array(_csh_list, dtype=np.float32),   # (T,)
                "turnover": np.array(_trn_list, dtype=np.float64),   # (T,), NaN: 비리밸런싱
                "alpha":    np.array(_alp_list, dtype=np.float64),   # (T,), NaN: 비리밸런싱
                "cost":     np.array(_cst_list, dtype=np.float64),   # (T,), NaN: 비리밸런싱
                "value":    np.array(_val_list, dtype=np.float64),   # (T,)
                "tickers":  list(universe),
            }

        df_year = pd.DataFrame({
            "port_ret": port_rets_list,
            "index_ret": index_rets_list,
        }, index=pd.DatetimeIndex(used_dates))
        df_year.attrs["winfo"] = winfo
        return df_year, a_hold.copy(), float(c_hold), list(universe)


class ROBaseline:
    """
    Re-Optimization (RO) 베이스라인.

    각 리밸런싱 시점마다:
    1. 과거 2년(≈504 거래일) 수익률 데이터 사용
    2. 동일 목적함수(EIT 또는 EIT-CVaR)로 직접 최적화
    3. 블록 부트스트랩 200 에포크
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        index: pd.Series,
        mcap: pd.DataFrame,
        strategy: str = "EIT",
        rho: float = 0.0,
        top_n: int = cfg.TOP_N_DEFAULT,
        train_start_year: int = cfg.TRAIN_START_YEAR,
        test_start_year: int  = cfg.TEST_START_YEAR,
        test_end_year: int    = cfg.TEST_END_YEAR,
        device: str = "cpu",
        seed: int = cfg.RANDOM_SEED,
        verbose: bool = True,
        record_weights: bool = True,
    ):
        self.prices           = prices
        self.index            = index
        self.mcap             = mcap
        self.strategy         = strategy
        self.rho              = rho
        self.top_n            = top_n
        self.train_start_year = train_start_year
        self.test_start_year  = test_start_year
        self.test_end_year    = test_end_year
        self.device           = device
        self.seed             = seed
        self.verbose          = verbose
        self.record_weights   = record_weights

    def run(self) -> pd.DataFrame:
        """전체 롤링 RO 백테스트 실행."""
        all_port_rets  = []
        all_index_rets = []
        all_dates      = []
        _all_wdfs:     list = []
        _all_cash:     list = []
        _all_turnover: list = []
        _all_alpha:    list = []
        _all_cost:     list = []
        _all_value:    list = []

        carry_state = None
        for test_year in range(self.test_start_year, self.test_end_year + 1):
            if self.verbose:
                print(f"\n[RO] 테스트 연도: {test_year} | ρ={self.rho}")

            df_year, a_end, c_end, tickers_end = self._run_year(
                test_year=test_year,
                initial_state=carry_state,
            )
            winfo = df_year.attrs.get("winfo")

            all_port_rets.extend(df_year["port_ret"].tolist())
            all_index_rets.extend(df_year["index_ret"].tolist())
            all_dates.extend(df_year.index.tolist())
            carry_state = (a_end, c_end, tickers_end)

            if winfo is not None:
                _all_wdfs.append(pd.DataFrame(
                    winfo["weights"],
                    index=pd.DatetimeIndex(df_year.index),
                    columns=winfo["tickers"],
                ))
                _all_cash.extend(winfo["cash"].tolist())
                _all_turnover.extend(winfo["turnover"].tolist())
                _all_alpha.extend(winfo["alpha"].tolist())
                _all_cost.extend(winfo["cost"].tolist())
                _all_value.extend(winfo["value"].tolist())

        results = pd.DataFrame({
            "port_ret":  all_port_rets,
            "index_ret": all_index_rets,
        }, index=pd.DatetimeIndex(all_dates))

        if self.record_weights and _all_wdfs:
            self.weight_history_   = pd.concat(_all_wdfs)
            _idx = pd.DatetimeIndex(all_dates)
            self.cash_history_     = pd.Series(_all_cash,     index=_idx, name="w_cash")
            self.turnover_history_ = pd.Series(_all_turnover, index=_idx, name="turnover")
            self.alpha_history_    = pd.Series(_all_alpha,     index=_idx, name="alpha")
            self.cost_history_     = pd.Series(_all_cost,      index=_idx, name="trading_cost")
            self.value_history_    = pd.Series(_all_value,     index=_idx, name="portfolio_value")

        return results

    def _run_year(
        self,
        test_year: int,
        initial_state: Optional[tuple[np.ndarray, float, list[str]]] = None,
    ) -> tuple:
        """단일 테스트 연도 RO 백테스트. 논문: lookback 수익률만으로 정적 비중 직접 최적화."""
        train_start = pd.Timestamp(f"{self.train_start_year}-01-01")
        train_end   = pd.Timestamp(f"{test_year - 1}-12-31")
        test_start  = pd.Timestamp(f"{test_year}-01-01")
        test_end    = pd.Timestamp(f"{test_year}-12-31")

        universe = build_universe(
            self.prices, self.mcap,
            train_start=train_start, train_end=train_end,
            top_n=self.top_n,
        )

        full_sr, full_ir, _ = get_period_data(
            self.prices, self.index, self.mcap,
            start=train_start, end=test_end, universe=universe,
        )

        test_mask   = (full_sr.index >= test_start) & (full_sr.index <= test_end)
        test_dates  = full_sr.index[test_mask]

        port_rets_list  = []
        index_rets_list = []
        used_dates      = []
        _wgt_list = [] if self.record_weights else None
        _csh_list = [] if self.record_weights else None
        _trn_list = [] if self.record_weights else None
        _alp_list = [] if self.record_weights else None
        _cst_list = [] if self.record_weights else None
        _val_list = [] if self.record_weights else None

        n_assets = len(universe)
        if initial_state is None:
            a_hold = np.zeros(n_assets, dtype=float)
            c_hold = 1.0
        else:
            # RO 베이스라인도 정책모델과 동일하게 연도 경계 state carry를 적용
            a_hold, c_hold = _carry_state_to_universe(
                a_hold_init=initial_state[0],
                c_hold_init=initial_state[1],
                tickers_init=initial_state[2],
                universe=list(universe),
                rho=self.rho,
            )
        first_rebalance_done = False

        for i, date in enumerate(test_dates):
            _rebal_turnover = np.nan
            _rebal_alpha    = np.nan
            _rebal_cost     = np.nan

            if date not in full_sr.index:
                continue

            v_pre = float(c_hold + a_hold.sum())
            if v_pre <= 0.0:
                continue

            w_pre = a_hold / v_pre

            rebalance_now = False
            w_target = None
            if i % cfg.REBAL_FREQ == 0:
                ro_end_idx   = full_sr.index.get_loc(date)
                ro_start_idx = max(0, ro_end_idx - cfg.RO_LOOKBACK)
                ro_sr = full_sr.iloc[ro_start_idx:ro_end_idx].values
                ro_ir = full_ir.iloc[ro_start_idx:ro_end_idx].values.flatten()

                if len(ro_ir) < 20:
                    continue

                log_first = self.verbose and not first_rebalance_done
                w_target = ro_optimize_weights(
                    ro_sr, ro_ir,
                    strategy=self.strategy,
                    device=self.device,
                    seed=self.seed,
                    epochs=cfg.RO_EPOCHS,
                    path_len=cfg.PATH_LEN,
                    batch_size=cfg.BATCH_SIZE,
                    block_size=cfg.RO_BLOCK,
                    verbose=False,
                    log_first_rebalance=log_first,
                )
                first_rebalance_done = True

                rebalance_now = True
                _rebal_alpha = 1.0
                if self.verbose:
                    sum_w_tgt = float(np.sum(w_target))
                    print(
                        f"[RO Rebalance] {date.date()} | v_pre={v_pre:.6f} "
                        f"| sum_w_target={sum_w_tgt:.6f}"
                    )

            r_stocks = full_sr.loc[date].values
            r_idx    = float(full_ir.loc[date])
            w_stock_next, w_cash_next, port_ret, turnover, cost_frac = paper_step_numpy(
                w_stock_pre=w_pre,
                w_cash_pre=float(c_hold / max(v_pre, 1e-12)),
                r_stock_next=r_stocks,
                rho=self.rho,
                rebalance=rebalance_now,
                w_stock_target=w_target,
            )
            v_post = float(v_pre * (1.0 + port_ret))
            a_hold = np.asarray(w_stock_next, dtype=float) * v_post
            c_hold = float(w_cash_next) * v_post
            if rebalance_now:
                _rebal_turnover = float(turnover)
                _rebal_cost = float(cost_frac * v_pre)

            port_rets_list.append(port_ret)
            index_rets_list.append(r_idx)
            used_dates.append(date)

            if self.record_weights:
                w_post = np.asarray(w_stock_next, dtype=float)
                _wgt_list.append(w_post.copy())
                _csh_list.append(float(w_cash_next))
                _trn_list.append(_rebal_turnover)
                _alp_list.append(_rebal_alpha)
                _cst_list.append(_rebal_cost)
                _val_list.append(v_post)

        winfo = None
        if self.record_weights and _wgt_list:
            winfo = {
                "weights":  np.array(_wgt_list, dtype=np.float32),
                "cash":     np.array(_csh_list, dtype=np.float32),
                "turnover": np.array(_trn_list, dtype=np.float64),
                "alpha":    np.array(_alp_list, dtype=np.float64),
                "cost":     np.array(_cst_list, dtype=np.float64),
                "value":    np.array(_val_list, dtype=np.float64),
                "tickers":  list(universe),
            }

        df_year = pd.DataFrame({
            "port_ret": port_rets_list,
            "index_ret": index_rets_list,
        }, index=pd.DatetimeIndex(used_dates))
        df_year.attrs["winfo"] = winfo
        return df_year, a_hold.copy(), float(c_hold), list(universe)
