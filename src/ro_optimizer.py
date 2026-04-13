"""
ro_optimizer.py
===============
Dai & Li (2024) 논문에 맞는 Re-Optimization (RO) 베이스라인용 정적 비중 최적화.

- 각 리밸런싱 시점에서 lookback 수익률만 사용
- theta (softmax → w) 직접 최적화, PolicyNetwork 미사용
- loss.py의 loss_IT / loss_EIT / loss_EIT_CVaR 사용
- 블록 부트스트랩 + Adam
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer import block_bootstrap_returns, set_seed
from loss import get_loss_fn
import config as cfg


def ro_optimize_weights(
    R_s: np.ndarray,
    R_i: np.ndarray,
    strategy: str = "EIT",
    device: str = "cpu",
    seed: int = cfg.RANDOM_SEED,
    epochs: int = cfg.RO_EPOCHS,
    path_len: int = cfg.PATH_LEN,
    batch_size: int = cfg.BATCH_SIZE,
    block_size: int = cfg.RO_BLOCK,
    lr: float = 1e-2,
    verbose: bool = False,
    log_first_rebalance: bool = False,
) -> np.ndarray:
    """
    Lookback 수익률 (R_s, R_i)로 RO 정적 비중을 최적화합니다.
    theta in R^(N+1) softmax 후 cash(0번) 드롭 → stock weights (sum<=1).

    Parameters
    ----------
    R_s : (T, N), lookback 종목 수익률
    R_i : (T,),   lookback 인덱스 수익률
    strategy : 'IT' | 'EIT' | 'EIT-CVaR'
    device, seed, epochs, path_len, batch_size, block_size, lr
    verbose : 에포크별 로깅 여부
    log_first_rebalance : True면 최적화 후 비중 요약 출력 (mean, std, top-5)

    Returns
    -------
    w : (N,) numpy, long-only stock weights (sum<=1)
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    T, N = R_s.shape
    if T < 20:
        return np.ones(N) / N

    loss_fn = get_loss_fn(strategy)
    theta = nn.Parameter(torch.zeros(N + 1, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([theta], lr=lr)

    for epoch in range(epochs):
        batch_stock, batch_index = block_bootstrap_returns(
            R_s, R_i,
            path_len=min(path_len, T),
            batch_size=batch_size,
            block_size=block_size,
            rng=rng,
        )
        b_stock = torch.tensor(batch_stock, device=device, dtype=torch.float32)
        b_index = torch.tensor(batch_index, device=device, dtype=torch.float32)

        w_full = F.softmax(theta, dim=0)
        w = w_full[1:]  # drop cash
        batch_port = (b_stock * w[None, None, :]).sum(dim=2)

        B = batch_port.shape[0]
        loss_list = [loss_fn(batch_port[b], b_index[b]) for b in range(B)]
        loss = torch.stack(loss_list).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  RO epoch {epoch+1}/{epochs} loss={loss.item():.6f}")

    with torch.no_grad():
        w_np = F.softmax(theta, dim=0)[1:].cpu().numpy()

    if log_first_rebalance:
        mean_w = float(w_np.mean())
        std_w = float(w_np.std())
        top5_idx = np.argsort(w_np)[-5:][::-1]
        top5_vals = w_np[top5_idx]
        print(f"  [RO weights] mean={mean_w:.4f} std={std_w:.4f} top5={top5_vals}")

    return w_np
