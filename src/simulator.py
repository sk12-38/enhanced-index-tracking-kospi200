"""
simulator.py
============
Paper-consistent trading simulator utilities.

Implements Eq. (2), Eq. (4), Eq. (5) style mechanics with:
  - stock weights w_S and cash weight w_0
  - turnover on rebalance only
  - transaction cost applied to stock turnover only
  - no alpha scaling / no haircut / no feasibility bisection
"""

from __future__ import annotations

import numpy as np
import torch


def _safe_denom_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.where(np.abs(x) < eps, eps, x)


def _safe_denom_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.where(torch.abs(x) < eps, torch.full_like(x, eps), x)


def paper_step_numpy(
    w_stock_pre: np.ndarray,
    w_cash_pre: np.ndarray,
    r_stock_next: np.ndarray,
    rho: float,
    rebalance: bool,
    w_stock_target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One-step portfolio evolution under paper equations using numpy arrays.

    Parameters
    ----------
    w_stock_pre : (N,) or (B, N), pre-rebalance stock weights
    w_cash_pre : scalar or (B,), pre-rebalance cash weights
    r_stock_next : same batch shape as w_stock_pre, next-day stock simple returns
    rho : transaction cost rate
    rebalance : whether trade occurs at this step
    w_stock_target : post-rebalance stock weights if rebalance=True

    Returns
    -------
    w_stock_next : drifted stock weights at next step
    w_cash_next : drifted cash weights at next step
    port_ret : portfolio simple return for the step
    turnover : L1 turnover at rebalance (0 if non-rebalance)
    cost_frac : rho * turnover (fraction of portfolio value)
    """
    ws_pre = np.asarray(w_stock_pre, dtype=float)
    wc_pre = np.asarray(w_cash_pre, dtype=float)
    rs = np.asarray(r_stock_next, dtype=float)

    if ws_pre.ndim == 1:
        ws_pre = ws_pre[None, :]
        rs = rs[None, :]
        wc_pre = np.asarray([float(wc_pre)], dtype=float)
        squeeze_out = True
    else:
        squeeze_out = False

    if rebalance:
        if w_stock_target is None:
            raise ValueError("w_stock_target is required when rebalance=True")
        ws_tilde = np.asarray(w_stock_target, dtype=float)
        if ws_tilde.ndim == 1:
            ws_tilde = ws_tilde[None, :]
        ws_tilde = np.clip(ws_tilde, 0.0, None)
        turnover = np.abs(ws_tilde - ws_pre).sum(axis=1)
        wc_tilde = 1.0 - ws_tilde.sum(axis=1) - rho * turnover
    else:
        ws_tilde = ws_pre
        wc_tilde = wc_pre
        turnover = np.zeros(ws_pre.shape[0], dtype=float)

    gross = wc_tilde + np.sum(ws_tilde * (1.0 + rs), axis=1)
    port_ret = gross - 1.0
    denom = _safe_denom_np(gross)
    ws_next = ws_tilde * (1.0 + rs) / denom[:, None]
    wc_next = wc_tilde / denom
    cost_frac = rho * turnover

    if squeeze_out:
        return ws_next[0], float(wc_next[0]), float(port_ret[0]), float(turnover[0]), float(cost_frac[0])
    return ws_next, wc_next, port_ret, turnover, cost_frac


def paper_step_torch(
    w_stock_pre: torch.Tensor,
    w_cash_pre: torch.Tensor,
    r_stock_next: torch.Tensor,
    rho: float,
    rebalance: bool,
    w_stock_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One-step portfolio evolution under paper equations using torch tensors.
    Keeps gradient flow through w_stock_target and returns.
    """
    ws_pre = w_stock_pre
    wc_pre = w_cash_pre
    rs = r_stock_next

    if rebalance:
        if w_stock_target is None:
            raise ValueError("w_stock_target is required when rebalance=True")
        ws_tilde = torch.clamp(w_stock_target, min=0.0)
        turnover = torch.sum(torch.abs(ws_tilde - ws_pre), dim=-1)
        wc_tilde = 1.0 - torch.sum(ws_tilde, dim=-1) - rho * turnover
    else:
        ws_tilde = ws_pre
        wc_tilde = wc_pre
        turnover = torch.zeros(ws_pre.shape[0], dtype=ws_pre.dtype, device=ws_pre.device)

    gross = wc_tilde + torch.sum(ws_tilde * (1.0 + rs), dim=-1)
    port_ret = gross - 1.0
    denom = _safe_denom_torch(gross)
    ws_next = ws_tilde * (1.0 + rs) / denom.unsqueeze(-1)
    wc_next = wc_tilde / denom
    cost_frac = rho * turnover
    return ws_next, wc_next, port_ret, turnover, cost_frac
