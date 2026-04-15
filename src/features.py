"""
features.py
===========
Feature pipeline for paper-style policy blocks.

Score block requires per-stock 5D input:
  [mu_i^(k), sigma_i^(k), beta_i^(k), mu_I^(k), sigma_I^(k)]
where k is the rolling window length (default 63).
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional
import config as cfg


def _safe_stats(df: pd.DataFrame) -> tuple:
    """
    DataFrame 전체 값(모든 행·열)의 스칼라 mean/std를 반환한다.
    std는 cfg.NORM_EPS 이상으로 클램프하여 division-by-zero를 방지한다.
    NaN은 계산에서 제외된다.
    """
    vals = df.values.astype(float)
    mu   = float(np.nanmean(vals))
    sig  = float(np.nanstd(vals, ddof=1))
    sig  = max(sig, cfg.NORM_EPS)
    return mu, sig


class FeatureNormalizer:
    """train_zscore normalizer for stock feature tensor (N, 5)."""

    def __init__(self):
        self.mu_mean = 0.0
        self.sig_mean = 1.0
        self.mu_vol = 0.0
        self.sig_vol = 1.0
        self.mu_beta = 0.0
        self.sig_beta = 1.0
        self.mu_idx_mean = 0.0
        self.sig_idx_mean = 1.0
        self.mu_idx_vol = 0.0
        self.sig_idx_vol = 1.0
        self._fitted  = False

    def fit_from_builder(
        self,
        builder: "FeatureBuilder",
        train_end: pd.Timestamp,
    ) -> None:
        """
        FeatureBuilder의 롤링 통계에서 학습 기간(≤ train_end) 값만으로
        mu/sigma를 추정한다.
        """
        mask = builder.mean_rets.index <= train_end
        self.mu_mean, self.sig_mean = _safe_stats(builder.mean_rets[mask].dropna())
        self.mu_vol, self.sig_vol = _safe_stats(builder.vols[mask].dropna())
        self.mu_beta, self.sig_beta = _safe_stats(builder.betas[mask].dropna())
        idx_mean_df = pd.DataFrame({"idx_mean": builder.index_mean_rets[mask]})
        idx_vol_df = pd.DataFrame({"idx_vol": builder.index_vols[mask]})
        self.mu_idx_mean, self.sig_idx_mean = _safe_stats(idx_mean_df.dropna())
        self.mu_idx_vol, self.sig_idx_vol = _safe_stats(idx_vol_df.dropna())
        self._fitted = True

    def transform_stock_features(self, feat: np.ndarray) -> np.ndarray:
        """(N,5) array [mu_i, sigma_i, beta_i, mu_I, sigma_I] z-score normalize."""
        if not self._fitted:
            return feat
        out = feat.copy().astype(np.float32)
        out[:, 0] = (feat[:, 0] - self.mu_mean) / self.sig_mean
        out[:, 1] = (feat[:, 1] - self.mu_vol) / self.sig_vol
        out[:, 2] = (feat[:, 2] - self.mu_beta) / self.sig_beta
        out[:, 3] = (feat[:, 3] - self.mu_idx_mean) / self.sig_idx_mean
        out[:, 4] = (feat[:, 4] - self.mu_idx_vol) / self.sig_idx_vol
        return out

    def transform_path_tensor(self, stk_feat: torch.Tensor) -> torch.Tensor:
        """(B,N,5) tensor normalization."""
        if not self._fitted:
            return stk_feat
        out = stk_feat.clone()
        out[..., 0] = (stk_feat[..., 0] - self.mu_mean) / self.sig_mean
        out[..., 1] = (stk_feat[..., 1] - self.mu_vol) / self.sig_vol
        out[..., 2] = (stk_feat[..., 2] - self.mu_beta) / self.sig_beta
        out[..., 3] = (stk_feat[..., 3] - self.mu_idx_mean) / self.sig_idx_mean
        out[..., 4] = (stk_feat[..., 4] - self.mu_idx_vol) / self.sig_idx_vol
        return out

    def log_stats(self) -> None:
        """피처별 mu/sigma 요약 출력 (진단용)."""
        print(f"[Normalizer] mean_ret  : mu={self.mu_mean:.6f}  sigma={self.sig_mean:.6f}")
        print(f"[Normalizer] vol       : mu={self.mu_vol:.6f}  sigma={self.sig_vol:.6f}")
        print(f"[Normalizer] beta      : mu={self.mu_beta:.6f}  sigma={self.sig_beta:.6f}")
        print(f"[Normalizer] index_mean: mu={self.mu_idx_mean:.6f}  sigma={self.sig_idx_mean:.6f}")
        print(f"[Normalizer] index_vol : mu={self.mu_idx_vol:.6f}  sigma={self.sig_idx_vol:.6f}")


def compute_rolling_stats(
    stock_rets: pd.DataFrame,
    index_rets: pd.Series,
    window: int = 63,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    63일 롤링 윈도우 단기 통계 계산.

    Parameters
    ----------
    stock_rets : (T × N), 종목 단순수익률
    index_rets : (T,),    인덱스 단순수익률
    window     : 롤링 윈도우 (거래일)

    Returns
    -------
    mean_rets : (T × N), stock rolling mean
    vols      : (T × N), stock rolling vol
    betas     : (T × N), stock rolling beta
    idx_mean  : (T,), index rolling mean
    idx_vol   : (T,), index rolling vol
    """
    mean_rets = stock_rets.rolling(window, min_periods=window).mean()
    vols      = stock_rets.rolling(window, min_periods=window).std()

    # 베타 = Cov(r_k, r_i) / Var(r_i), 롤링 윈도우 내에서 계산
    betas = pd.DataFrame(index=stock_rets.index, columns=stock_rets.columns, dtype=float)
    idx_arr = index_rets.values

    for t in range(window - 1, len(stock_rets)):
        idx_window = idx_arr[t - window + 1 : t + 1]
        var_i = np.var(idx_window, ddof=1)
        if var_i < 1e-12:
            betas.iloc[t] = 0.0
            continue
        for col in stock_rets.columns:
            stk_window = stock_rets[col].values[t - window + 1 : t + 1]
            cov = np.cov(stk_window, idx_window, ddof=1)[0, 1]
            betas.at[stock_rets.index[t], col] = cov / var_i

    idx_mean = index_rets.rolling(window, min_periods=window).mean().fillna(0.0)
    idx_vol = index_rets.rolling(window, min_periods=window).std().fillna(0.0)
    return mean_rets, vols, betas, idx_mean, idx_vol


class FeatureBuilder:
    """
    매 거래일(또는 리밸런싱일)에 정책망 입력 피처를 구성합니다.

    사전에 rolling stats를 전체 기간에 걸쳐 계산해두고,
    특정 시점 t의 피처 벡터를 반환합니다.
    """

    def __init__(
        self,
        stock_rets: pd.DataFrame,
        index_rets: pd.Series,
        window: int = 63,
    ):
        self.stock_rets = stock_rets
        self.index_rets = index_rets
        self.window     = window
        self.n_stocks   = stock_rets.shape[1]

        print(f"[Features] 롤링 통계 계산 중 (window={window})...")
        self.mean_rets, self.vols, self.betas, self.index_mean_rets, self.index_vols = compute_rolling_stats(
            stock_rets, index_rets, window
        )
        print("[Features] 완료.")

    def get_feature_at(
        self,
        t: pd.Timestamp,
        index_bull_prob: float,
        stock_bull_probs: np.ndarray,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """
        시점 t의 피처 벡터 반환.

        Parameters
        ----------
        t                 : 현재 날짜
        index_bull_prob   : float, P(bull | index) at t
        stock_bull_probs  : (N,),  각 종목의 bull 확률 at t
        current_weights   : (N,),  현재(이전 리밸런싱) 포트폴리오 비중

        Returns
        -------
        dict with stock 5D score features and regime probs.
        """
        if t not in self.mean_rets.index:
            raise KeyError(f"시점 {t}이 피처 인덱스에 없습니다.")

        mean_r = self.mean_rets.loc[t].values.astype(float)
        vol    = self.vols.loc[t].values.astype(float)
        beta   = self.betas.loc[t].values.astype(float)

        # NaN → 0 처리 (초기 window 미충족 구간)
        mean_r = np.nan_to_num(mean_r, nan=0.0)
        vol    = np.nan_to_num(vol,    nan=0.0)
        beta   = np.nan_to_num(beta,   nan=1.0)

        idx_mean = float(np.nan_to_num(self.index_mean_rets.loc[t], nan=0.0))
        idx_vol = float(np.nan_to_num(self.index_vols.loc[t], nan=0.0))

        return {
            "index_regime": float(index_bull_prob),
            "stock_regime": np.array(stock_bull_probs, dtype=float),
            "mean_rets":    mean_r,
            "vols":         vol,
            "betas":        beta,
            "index_mean":   idx_mean,
            "index_vol":    idx_vol,
            "prev_weights": np.array(current_weights, dtype=float),
        }

    def get_feature_tensor(
        self,
        t: pd.Timestamp,
        index_bull_prob: float,
        stock_bull_probs: np.ndarray,
        current_weights: np.ndarray,
    ):
        """
        get_feature_at의 결과를 PolicyNetwork 입력용 텐서로 변환.

        Returns
        -------
        (index_regime_tensor, stock_features_tensor, prev_weights_tensor)
        shapes: (1,), (N, 5), (N,)
        """
        import torch
        feat = self.get_feature_at(t, index_bull_prob, stock_bull_probs, current_weights)

        idx_regime = torch.tensor([feat["index_regime"]], dtype=torch.float32)

        idx_mean = np.full_like(feat["mean_rets"], feat["index_mean"], dtype=float)
        idx_vol = np.full_like(feat["mean_rets"], feat["index_vol"], dtype=float)
        # 종목별 score 피처: [mu_i, sigma_i, beta_i, mu_I, sigma_I]
        stk_feats = np.stack(
            [feat["mean_rets"], feat["vols"], feat["betas"], idx_mean, idx_vol],
            axis=1,
        )
        stk_feats_t = torch.tensor(stk_feats, dtype=torch.float32)

        prev_w = torch.tensor(feat["prev_weights"], dtype=torch.float32)

        return idx_regime, stk_feats_t, prev_w


def build_path_features(
    stock_rets_path: np.ndarray,
    index_rets_path: np.ndarray,
    index_bull_path: np.ndarray,
    stock_bull_path: np.ndarray,
    prev_weights_path: np.ndarray,
    window: int = 63,
) -> dict:
    """
    블록 부트스트랩 경로(path) 전체에 대한 피처 배치 구성.
    trainer.py에서 시뮬레이션 경로마다 호출됩니다.

    Parameters
    ----------
    stock_rets_path   : (path_len, N)
    index_rets_path   : (path_len,)
    index_bull_path   : (path_len,)   - 경로 내 인덱스 bull 확률
    stock_bull_path   : (path_len, N) - 경로 내 종목 bull 확률
    prev_weights_path : (path_len, N) - 경로 내 이전 비중 (rolling)
    window            : 롤링 윈도우

    Returns
    -------
    dict of numpy arrays for entire path
    """
    T, N = stock_rets_path.shape

    # 롤링 통계 (경로 내에서만 계산)
    mean_r = np.zeros((T, N))
    vol    = np.zeros((T, N))
    beta   = np.zeros((T, N))

    for t in range(T):
        s = max(0, t - window + 1)
        chunk_stk = stock_rets_path[s : t + 1]
        chunk_idx = index_rets_path[s : t + 1]

        if len(chunk_stk) < 2:
            mean_r[t] = 0.0
            vol[t]    = 0.0
            beta[t]   = 1.0
            continue

        mean_r[t] = chunk_stk.mean(axis=0)
        vol[t]    = chunk_stk.std(axis=0, ddof=1) + 1e-8
        var_i = chunk_idx.var(ddof=1) + 1e-12
        for k in range(N):
            cov = np.cov(chunk_stk[:, k], chunk_idx, ddof=1)[0, 1]
            beta[t, k] = cov / var_i

    idx_mean = np.zeros(T)
    idx_vol = np.zeros(T)
    for t in range(T):
        s = max(0, t - window + 1)
        chunk_idx = index_rets_path[s : t + 1]
        if len(chunk_idx) < 2:
            idx_mean[t] = 0.0
            idx_vol[t] = 0.0
        else:
            idx_mean[t] = float(np.mean(chunk_idx))
            idx_vol[t] = float(np.std(chunk_idx, ddof=1))

    return {
        "index_regime":  index_bull_path,          # (T,)
        "stock_regime":  stock_bull_path,           # (T, N)
        "mean_rets":     mean_r,                    # (T, N)
        "vols":          vol,                       # (T, N)
        "betas":         beta,                      # (T, N)
        "index_mean":    idx_mean,                  # (T,)
        "index_vol":     idx_vol,                   # (T,)
        "prev_weights":  prev_weights_path,         # (T, N)
    }
