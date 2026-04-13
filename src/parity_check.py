"""
parity_check.py
===============
Lightweight parity checks for paper-consistent simulator.
"""

import numpy as np
import torch

from simulator import paper_step_numpy, paper_step_torch


def check_step_parity(seed: int = 42, tol: float = 1e-8) -> bool:
    """Check numpy vs torch simulator parity under identical inputs."""
    rng = np.random.default_rng(seed)
    b, n = 8, 5
    w_pre = rng.uniform(0.0, 0.2, size=(b, n))
    w_cash = 1.0 - w_pre.sum(axis=1)
    r_next = rng.normal(0.0002, 0.01, size=(b, n))
    w_tgt = np.clip(rng.uniform(0.0, 0.2, size=(b, n)), 0.0, None)

    ws_np, wc_np, rp_np, to_np, c_np = paper_step_numpy(
        w_stock_pre=w_pre,
        w_cash_pre=w_cash,
        r_stock_next=r_next,
        rho=0.005,
        rebalance=True,
        w_stock_target=w_tgt,
    )
    ws_t, wc_t, rp_t, to_t, c_t = paper_step_torch(
        w_stock_pre=torch.tensor(w_pre, dtype=torch.float64),
        w_cash_pre=torch.tensor(w_cash, dtype=torch.float64),
        r_stock_next=torch.tensor(r_next, dtype=torch.float64),
        rho=0.005,
        rebalance=True,
        w_stock_target=torch.tensor(w_tgt, dtype=torch.float64),
    )
    ok = (
        np.allclose(ws_np, ws_t.numpy(), atol=tol, rtol=0.0)
        and np.allclose(wc_np, wc_t.numpy(), atol=tol, rtol=0.0)
        and np.allclose(rp_np, rp_t.numpy(), atol=tol, rtol=0.0)
        and np.allclose(to_np, to_t.numpy(), atol=tol, rtol=0.0)
        and np.allclose(c_np, c_t.numpy(), atol=tol, rtol=0.0)
    )
    return bool(ok)


if __name__ == "__main__":
    result = check_step_parity()
    print(f"[ParityCheck] simulator numpy/torch parity: {'PASS' if result else 'FAIL'}")
