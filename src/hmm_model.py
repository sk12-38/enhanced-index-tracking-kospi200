"""
hmm_model.py
============
논문 Section 3.1: 2-상태 가우시안 HMM (Hidden Markov Model).

- Baum-Welch (EM) 알고리즘으로 파라미터 추정
- 순방향(Forward) 알고리즘으로 필터링된 상태 확률 계산
- 2일 이동평균 스무딩 후 bull/bear 레짐 분류
- 인덱스용 HMM과 각 종목별 HMM을 분리하여 피팅

bull 상태: P(state_1) ≥ 0.5
bear 상태: P(state_1) < 0.5
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional


class GaussianHMM2State:
    """
    2-상태 가우시안 HMM.

    상태 0: state_0 (bear 또는 bull, 학습 후 결정)
    상태 1: state_1
    레짐 판별: 수익률 평균이 높은 상태 = bull (state reorder)
    """

    def __init__(self, n_iter: int = 100, tol: float = 1e-4, random_state: int = 42):
        self.n_iter       = n_iter
        self.tol          = tol
        self.random_state = random_state

        # 파라미터 (fit 후 설정)
        self.pi      = None   # 초기 상태 확률 (2,)
        self.A       = None   # 전이 행렬 (2, 2)
        self.mu      = None   # 방출 평균 (2,)
        self.sigma   = None   # 방출 표준편차 (2,)
        self.bull_state = 1   # bull 상태 인덱스 (fit 후 결정)

    def _emission_prob(self, x: np.ndarray) -> np.ndarray:
        """각 관측값에 대한 상태별 방출 확률 B[t, k] = N(x_t | mu_k, sigma_k)"""
        T = len(x)
        B = np.zeros((T, 2))
        for k in range(2):
            B[:, k] = norm.pdf(x, loc=self.mu[k], scale=self.sigma[k])
        # 수치 안정성: 0이 되지 않도록 클리핑
        B = np.clip(B, 1e-300, None)
        return B

    def _forward(self, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward 알고리즘 (스케일링 버전).

        Returns
        -------
        alpha_hat : (T, 2), 스케일링된 순방향 변수
        c         : (T,),   스케일링 상수
        """
        T = B.shape[0]
        alpha_hat = np.zeros((T, 2))
        c = np.zeros(T)

        alpha_hat[0] = self.pi * B[0]
        c[0] = alpha_hat[0].sum()
        alpha_hat[0] /= c[0] + 1e-300

        for t in range(1, T):
            alpha_hat[t] = (alpha_hat[t - 1] @ self.A) * B[t]
            c[t] = alpha_hat[t].sum()
            alpha_hat[t] /= c[t] + 1e-300

        return alpha_hat, c

    def _backward(self, B: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Backward 알고리즘 (스케일링 버전).

        Returns
        -------
        beta_hat : (T, 2), 스케일링된 후방향 변수
        """
        T = B.shape[0]
        beta_hat = np.zeros((T, 2))
        beta_hat[-1] = 1.0 / (c[-1] + 1e-300)

        for t in range(T - 2, -1, -1):
            beta_hat[t] = (self.A @ (B[t + 1] * beta_hat[t + 1])) / (c[t] + 1e-300)

        return beta_hat

    def fit(self, x: np.ndarray) -> "GaussianHMM2State":
        """
        Baum-Welch (EM) 알고리즘으로 파라미터 추정.

        Parameters
        ----------
        x : (T,), 1차원 수익률 시계열
        """
        rng = np.random.default_rng(self.random_state)
        T = len(x)

        # 파라미터 초기화 (k-means 스타일)
        sorted_x = np.sort(x)
        self.mu    = np.array([sorted_x[:T // 2].mean(), sorted_x[T // 2:].mean()])
        self.sigma = np.array([max(sorted_x[:T // 2].std(), 1e-6),
                               max(sorted_x[T // 2:].std(), 1e-6)])
        self.pi = np.array([0.5, 0.5])
        self.A  = np.array([[0.95, 0.05], [0.05, 0.95]])

        prev_loglik = -np.inf

        for _ in range(self.n_iter):
            # ── E-step ───────────────────────────────────────────────────────
            B = self._emission_prob(x)
            alpha_hat, c = self._forward(B)
            beta_hat      = self._backward(B, c)

            # 감마: P(s_t = k | x_{1:T})
            gamma = alpha_hat * beta_hat
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            # 크사이: P(s_t=i, s_{t+1}=j | x_{1:T})
            xi = np.zeros((T - 1, 2, 2))
            for t in range(T - 1):
                xi[t] = (alpha_hat[t, :, None]
                         * self.A
                         * B[t + 1, None, :]
                         * beta_hat[t + 1, None, :])
                xi[t] /= xi[t].sum() + 1e-300

            # ── M-step ───────────────────────────────────────────────────────
            self.pi = gamma[0]
            self.A  = xi.sum(axis=0) / (gamma[:-1].sum(axis=0)[:, None] + 1e-300)
            self.A  = self.A / self.A.sum(axis=1, keepdims=True)  # 행 정규화

            for k in range(2):
                w = gamma[:, k]
                self.mu[k]    = (w * x).sum() / (w.sum() + 1e-300)
                self.sigma[k] = np.sqrt((w * (x - self.mu[k]) ** 2).sum()
                                        / (w.sum() + 1e-300))
                self.sigma[k] = max(self.sigma[k], 1e-6)

            # 로그우도 수렴 확인
            loglik = np.log(c + 1e-300).sum()
            if abs(loglik - prev_loglik) < self.tol:
                break
            prev_loglik = loglik

        # bull 상태 = 평균 수익률이 높은 상태
        self.bull_state = int(np.argmax(self.mu))

        return self

    def filtered_probs(self, x: np.ndarray) -> np.ndarray:
        """
        순방향 알고리즘으로 필터링된 상태 확률 계산.

        Returns
        -------
        probs : (T, 2), P(s_t = k | x_{1:t}) for k in {0, 1}
        """
        B = self._emission_prob(x)
        alpha_hat, _ = self._forward(B)
        return alpha_hat

    def bull_prob(self, x: np.ndarray) -> np.ndarray:
        """
        bull 상태 확률 반환.

        Returns
        -------
        (T,), P(s_t = bull_state | x_{1:t})
        """
        probs = self.filtered_probs(x)
        return probs[:, self.bull_state]


def smooth_regime(bull_prob: np.ndarray, window: int = 2) -> np.ndarray:
    """
    논문 Section 3.1: 레짐 확률에 이동평균 스무딩 적용 후 bull/bear 분류.

    Parameters
    ----------
    bull_prob : (T,), bull 상태 확률
    window    : 이동평균 윈도우 (논문: 2일)

    Returns
    -------
    smoothed : (T,), 스무딩된 bull 확률 (첫 window-1 값은 누적 평균)
    """
    s = pd.Series(bull_prob)
    smoothed = s.rolling(window=window, min_periods=1).mean().values
    return smoothed


def is_bull(smoothed_prob: np.ndarray) -> np.ndarray:
    """스무딩된 bull 확률 → bull(True)/bear(False) 분류."""
    return smoothed_prob >= 0.5


class HMMCollection:
    """
    인덱스 HMM 1개 + 종목별 HMM N개를 관리하는 컨테이너.

    Usage
    -----
    coll = HMMCollection()
    coll.fit_all(index_rets, stock_rets)
    idx_prob  = coll.index_bull_prob(index_rets)
    stk_probs = coll.stock_bull_probs(stock_rets)
    """

    def __init__(self, n_iter: int = 100, random_state: int = 42):
        self.n_iter       = n_iter
        self.random_state = random_state
        self.hmm_index    = None
        self.hmm_stocks   = {}       # {종목명: GaussianHMM2State}

    def fit_all(
        self,
        index_rets: np.ndarray,
        stock_rets: pd.DataFrame,
        verbose: bool = True,
    ) -> "HMMCollection":
        """
        인덱스 및 모든 종목에 대해 HMM 피팅.

        Parameters
        ----------
        index_rets : (T,), 인덱스 단순수익률
        stock_rets : (T × N), 종목 단순수익률
        """
        if verbose:
            print("[HMM] 인덱스 HMM 피팅 중...")
        self.hmm_index = GaussianHMM2State(
            n_iter=self.n_iter, random_state=self.random_state
        ).fit(index_rets)

        if verbose:
            print(f"[HMM] 종목 HMM 피팅 중... ({stock_rets.shape[1]}개 종목)")

        for col in stock_rets.columns:
            self.hmm_stocks[col] = GaussianHMM2State(
                n_iter=self.n_iter, random_state=self.random_state
            ).fit(stock_rets[col].values)

        if verbose:
            print("[HMM] 피팅 완료.")
        return self

    def index_bull_prob(
        self, index_rets: np.ndarray, smooth: int = 2
    ) -> np.ndarray:
        """인덱스 bull 확률 (스무딩 적용)."""
        raw = self.hmm_index.bull_prob(index_rets)
        return smooth_regime(raw, window=smooth)

    def stock_bull_probs(
        self, stock_rets: pd.DataFrame, smooth: int = 2
    ) -> pd.DataFrame:
        """
        종목별 bull 확률 DataFrame (T × N).

        훈련 시와 다른 기간의 데이터를 넘겨도 파라미터는 고정된 상태로 추론.
        """
        result = {}
        for col in stock_rets.columns:
            if col not in self.hmm_stocks:
                raise KeyError(f"종목 '{col}'의 HMM이 피팅되지 않았습니다.")
            raw = self.hmm_stocks[col].bull_prob(stock_rets[col].values)
            result[col] = smooth_regime(raw, window=smooth)
        return pd.DataFrame(result, index=stock_rets.index)
