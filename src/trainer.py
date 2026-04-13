"""
trainer.py
==========
논문 Section 4.2: 학습 프로토콜 구현.

- 블록 부트스트랩으로 훈련 배치 생성
- Adam 옵티마이저, 학습률 1e-2, 50 에포크
- 검증 손실 최소 모델 저장 (early stopping 없이 best 저장)
- 재현성을 위한 시드 고정
"""

import os
import copy
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from typing import Callable, Optional

from policy_network import PolicyNetwork
from loss import get_loss_fn
from simulator import paper_step_torch
import config as cfg


def set_seed(seed: int = 42):
    """재현성을 위한 전역 시드 고정."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def block_bootstrap_returns(
    stock_rets: np.ndarray,
    index_rets: np.ndarray,
    path_len: int = 200,
    batch_size: int = 128,
    block_size: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """
    블록 부트스트랩으로 수익률만 사용하는 훈련 배치 생성 (RO 베이스라인용).
    HMM/레짐 피처 없이 stock_rets, index_rets만 사용.

    Parameters
    ----------
    stock_rets : (T, N), 종목 수익률
    index_rets : (T,),   인덱스 수익률
    path_len   : 각 경로 길이 (거래일)
    batch_size : 배치 크기 (경로 수)
    block_size : 블록 길이 (거래일)
    rng        : 난수 생성기

    Returns
    -------
    batch_stock : (B, path_len, N)
    batch_index : (B, path_len)
    """
    if rng is None:
        rng = np.random.default_rng()

    T, N = stock_rets.shape
    n_blocks_per_path = (path_len + block_size - 1) // block_size

    batch_stock = np.zeros((batch_size, path_len, N), dtype=np.float32)
    batch_index = np.zeros((batch_size, path_len), dtype=np.float32)

    for b in range(batch_size):
        path_stk = []
        path_idx = []
        for _ in range(n_blocks_per_path):
            start = rng.integers(0, max(1, T - block_size))
            end = min(start + block_size, T)
            path_stk.append(stock_rets[start:end])
            path_idx.append(index_rets[start:end])
        path_stk = np.concatenate(path_stk, axis=0)[:path_len]
        path_idx = np.concatenate(path_idx, axis=0)[:path_len]
        actual_len = len(path_idx)
        batch_stock[b, :actual_len] = path_stk
        batch_index[b, :actual_len] = path_idx

    return batch_stock, batch_index


def block_bootstrap_paths(
    stock_rets: np.ndarray,
    index_rets: np.ndarray,
    index_bull: np.ndarray,
    stock_bull: np.ndarray,
    path_len: int = 200,
    batch_size: int = 128,
    block_size: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """
    블록 부트스트랩으로 훈련 배치를 생성합니다.

    논문: path_len=200일, batch_size=128 경로, block_size=10일

    Parameters
    ----------
    stock_rets  : (T, N), 종목 수익률
    index_rets  : (T,),   인덱스 수익률
    index_bull  : (T,),   인덱스 bull 확률
    stock_bull  : (T, N), 종목 bull 확률
    path_len    : 각 경로 길이 (거래일)
    batch_size  : 배치 크기 (경로 수)
    block_size  : 블록 길이 (거래일)

    Returns
    -------
    batch_stock  : (B, path_len, N)
    batch_index  : (B, path_len)
    batch_ibull  : (B, path_len)
    batch_sbull  : (B, path_len, N)
    """
    if rng is None:
        rng = np.random.default_rng()

    T, N = stock_rets.shape
    n_blocks_per_path = (path_len + block_size - 1) // block_size

    batch_stock = np.zeros((batch_size, path_len, N), dtype=np.float32)
    batch_index = np.zeros((batch_size, path_len),    dtype=np.float32)
    batch_ibull = np.zeros((batch_size, path_len),    dtype=np.float32)
    batch_sbull = np.zeros((batch_size, path_len, N), dtype=np.float32)

    for b in range(batch_size):
        path_stk   = []
        path_idx   = []
        path_ibull = []
        path_sbull = []

        for _ in range(n_blocks_per_path):
            # 블록 시작 인덱스 샘플링 (끝에서 block_size 이전까지)
            start = rng.integers(0, max(1, T - block_size))
            end   = min(start + block_size, T)

            path_stk.append(stock_rets[start:end])
            path_idx.append(index_rets[start:end])
            path_ibull.append(index_bull[start:end])
            path_sbull.append(stock_bull[start:end])

        # 이어 붙이기 후 path_len으로 자르기
        path_stk   = np.concatenate(path_stk,   axis=0)[:path_len]
        path_idx   = np.concatenate(path_idx,   axis=0)[:path_len]
        path_ibull = np.concatenate(path_ibull, axis=0)[:path_len]
        path_sbull = np.concatenate(path_sbull, axis=0)[:path_len]

        actual_len = len(path_idx)
        batch_stock[b, :actual_len] = path_stk
        batch_index[b, :actual_len] = path_idx
        batch_ibull[b, :actual_len] = path_ibull
        batch_sbull[b, :actual_len] = path_sbull

    return batch_stock, batch_index, batch_ibull, batch_sbull


def _compute_rolling_stats_batch(
    batch_stock: np.ndarray,
    batch_index: np.ndarray,
    window: int,
    eps: float = 1e-8,
) -> tuple:
    """
    (B, T, N) 배치 경로 전체에 대해 t 기준 롤링 통계를 계산한다.

    백테스터의 FeatureBuilder와 동일한 정의:
      mean_ret[t] = mean(r[t-window+1 : t+1])
      vol[t]      = std (r[t-window+1 : t+1], ddof=1)
      beta[t]     = Cov(r_k, r_idx) / Var(r_idx)

    시간 차원(T)은 Python 루프, B·N 차원은 numpy 벡터화.

    Returns
    -------
    mean_arr : (B, T, N)
    vol_arr  : (B, T, N)
    beta_arr : (B, T, N)
    idx_mean_arr : (B, T)
    idx_vol_arr  : (B, T)
    """
    B, T, N = batch_stock.shape
    mean_arr = np.zeros((B, T, N), dtype=np.float32)
    vol_arr  = np.full((B, T, N), eps, dtype=np.float32)
    beta_arr = np.ones((B, T, N), dtype=np.float32)

    for t in range(T):
        s         = max(0, t - window + 1)
        chunk_stk = batch_stock[:, s : t + 1, :]   # (B, L, N) — 뷰, 복사 없음
        chunk_idx = batch_index[:, s : t + 1]       # (B, L)
        L         = chunk_stk.shape[1]

        if L < 2:
            continue

        mean_arr[:, t, :] = chunk_stk.mean(axis=1)
        vol_arr[:, t, :]  = np.maximum(chunk_stk.std(axis=1, ddof=1), eps)

        idx_mean = chunk_idx.mean(axis=1, keepdims=True)              # (B, 1)
        idx_var  = np.maximum(chunk_idx.var(axis=1, ddof=1), 1e-12)  # (B,)
        idx_cen  = chunk_idx - idx_mean                               # (B, L)
        stk_cen  = chunk_stk - chunk_stk.mean(axis=1, keepdims=True) # (B, L, N)
        cov      = (stk_cen * idx_cen[:, :, None]).sum(axis=1) / (L - 1)  # (B, N)
        beta_arr[:, t, :] = cov / idx_var[:, None]

    idx_mean_arr = np.zeros((B, T), dtype=np.float32)
    idx_vol_arr = np.zeros((B, T), dtype=np.float32)
    for t in range(T):
        s = max(0, t - window + 1)
        idx_chunk = batch_index[:, s : t + 1]
        if idx_chunk.shape[1] < 2:
            continue
        idx_mean_arr[:, t] = idx_chunk.mean(axis=1)
        idx_vol_arr[:, t] = idx_chunk.std(axis=1, ddof=1).astype(np.float32)
    return mean_arr, vol_arr, beta_arr, idx_mean_arr, idx_vol_arr


def simulate_portfolio_returns(
    policy: PolicyNetwork,
    batch_stock: np.ndarray,
    batch_index: np.ndarray,
    batch_ibull: np.ndarray,
    batch_sbull: np.ndarray,
    rho: float = 0.0,
    device: str = "cpu",
    normalizer=None,
    rebal_freq: int = cfg.REBAL_FREQ,
    window: int = cfg.WINDOW_ST,
    use_memory_override: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Paper-consistent path simulation:
      - all-cash initial condition
      - rebalance only every rebal_freq
      - Eq.(2)/(4)/(5) mechanics via shared simulator

    Parameters
    ----------
    batch_* : (B, T, ...) numpy 배열
    rho         : 거래비용률
    rebal_freq  : 리밸런싱 주기 (일), 기본값 cfg.REBAL_FREQ
    window      : 롤링 통계 윈도우 (일), 기본값 cfg.WINDOW_ST
    normalizer  : FeatureNormalizer (None이면 정규화 생략)

    Returns
    -------
    port_rets  : (B, T), 포트폴리오 일간 수익률 텐서
    index_rets : (B, T), 인덱스 수익률 텐서
    """
    B, T, N = batch_stock.shape

    mean_arr, vol_arr, beta_arr, idx_mean_arr, idx_vol_arr = _compute_rolling_stats_batch(
        batch_stock, batch_index, window
    )

    # 전체 배치를 텐서로 일괄 변환 (루프 안에서 중복 변환 방지)
    batch_stock_t = torch.tensor(batch_stock, dtype=torch.float32, device=device)  # (B,T,N)
    batch_ibull_t = torch.tensor(batch_ibull, dtype=torch.float32, device=device)  # (B,T)
    batch_sbull_t = torch.tensor(batch_sbull, dtype=torch.float32, device=device)  # (B,T,N)
    mean_arr_t    = torch.tensor(mean_arr,    dtype=torch.float32, device=device)
    vol_arr_t     = torch.tensor(vol_arr,     dtype=torch.float32, device=device)
    beta_arr_t    = torch.tensor(beta_arr,    dtype=torch.float32, device=device)
    idx_mean_t    = torch.tensor(idx_mean_arr, dtype=torch.float32, device=device)
    idx_vol_t     = torch.tensor(idx_vol_arr, dtype=torch.float32, device=device)

    port_rets_list = []

    # Initial condition: all cash
    w_stock = torch.zeros(B, N, dtype=torch.float32, device=device)
    w_cash = torch.ones(B, dtype=torch.float32, device=device)

    for t in range(T):
        stk_rets = batch_stock_t[:, t, :]   # (B, N)
        w_pre = w_stock

        if t % rebal_freq == 0:
            idx_prob = batch_ibull_t[:, t]           # (B,)
            stk_bull = batch_sbull_t[:, t, :]        # (B, N)
            mean_t   = mean_arr_t[:, t, :]           # (B, N)
            vol_t    = vol_arr_t[:, t, :]            # (B, N)
            beta_t   = beta_arr_t[:, t, :]           # (B, N)
            idx_mean_cur = idx_mean_t[:, t].unsqueeze(-1).expand(-1, N)
            idx_vol_cur = idx_vol_t[:, t].unsqueeze(-1).expand(-1, N)
            stk_feat = torch.stack([mean_t, vol_t, beta_t, idx_mean_cur, idx_vol_cur], dim=-1)  # (B,N,5)

            if normalizer is not None:
                stk_feat = normalizer.transform_path_tensor(stk_feat)

            w_proposed = policy(
                idx_prob.unsqueeze(-1),
                stk_feat,
                stk_bull,
                w_pre.detach(),
                use_memory_override=use_memory_override,
            )
            w_target = torch.clamp(w_proposed, min=0.0)
            rebal = True
        else:
            w_target = None
            rebal = False

        w_stock, w_cash, port_ret_t, _, _ = paper_step_torch(
            w_stock_pre=w_stock,
            w_cash_pre=w_cash,
            r_stock_next=stk_rets,
            rho=rho,
            rebalance=rebal,
            w_stock_target=w_target,
        )

        port_rets_list.append(port_ret_t)

    port_rets  = torch.stack(port_rets_list, dim=1)   # (B, T)
    index_rets = torch.tensor(batch_index, dtype=torch.float32, device=device)

    return port_rets, index_rets


def train_policy(
    policy: PolicyNetwork,
    stock_rets: np.ndarray,
    index_rets: np.ndarray,
    index_bull: np.ndarray,
    stock_bull: np.ndarray,
    strategy: str = "EIT",
    rho: float = 0.0,
    path_len: int = cfg.PATH_LEN,
    batch_size: int = cfg.BATCH_SIZE,
    epochs: int = cfg.EPOCHS,
    lr: float = cfg.LR,
    block_size: int = cfg.BLOCK_SIZE,
    val_ratio: float = cfg.VAL_RATIO,
    save_path: Optional[str] = None,
    device: str = "cpu",
    seed: int = cfg.RANDOM_SEED,
    verbose: bool = True,
    normalizer=None,
) -> PolicyNetwork:
    """
    정책망 학습.

    Parameters
    ----------
    policy      : PolicyNetwork 인스턴스
    stock_rets  : (T, N) numpy, 종목 단순수익률
    index_rets  : (T,)   numpy, 인덱스 단순수익률
    index_bull  : (T,)   numpy, 인덱스 bull 확률
    stock_bull  : (T, N) numpy, 종목 bull 확률
    strategy    : 'IT' | 'EIT' | 'EIT-CVaR'
    rho         : 거래비용률
    save_path   : 최적 모델 저장 경로 (None이면 저장 안 함)
    verbose     : 진행 상황 출력 여부

    Returns
    -------
    best_policy : 검증 손실 최소 PolicyNetwork
    """
    set_seed(seed)
    policy = policy.to(device)
    loss_fn = get_loss_fn(strategy)
    rng = np.random.default_rng(seed)

    # 훈련/검증 분할 (시계열 끝 부분을 검증에 사용)
    T = len(index_rets)
    val_start = int(T * (1 - val_ratio))

    train_sr  = stock_rets[:val_start]
    train_ir  = index_rets[:val_start]
    train_ib  = index_bull[:val_start]
    train_sb  = stock_bull[:val_start]

    val_sr    = stock_rets[val_start:]
    val_ir    = index_rets[val_start:]
    val_ib    = index_bull[val_start:]
    val_sb    = stock_bull[val_start:]

    def _run_stage(
        n_epochs: int,
        stage_rho: float,
        use_memory_override: bool | None,
        optimizer: optim.Optimizer,
        desc: str,
    ) -> None:
        nonlocal policy
        best_val_loss = float("inf")
        best_state = copy.deepcopy(policy.state_dict())
        pbar = tqdm(range(n_epochs), desc=desc, disable=not verbose)
        for _ in pbar:
            policy.train()
            bs, bi, bib, bsb = block_bootstrap_paths(
                train_sr, train_ir, train_ib, train_sb,
                path_len=path_len, batch_size=batch_size,
                block_size=block_size, rng=rng,
            )
            port_rets, idx_rets = simulate_portfolio_returns(
                policy, bs, bi, bib, bsb,
                rho=stage_rho, device=device, normalizer=normalizer,
                use_memory_override=use_memory_override,
            )
            train_loss = torch.stack([loss_fn(port_rets[b], idx_rets[b]) for b in range(batch_size)]).mean()
            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            if len(val_ir) >= path_len:
                policy.eval()
                with torch.no_grad():
                    vbs, vbi, vbib, vbsb = block_bootstrap_paths(
                        val_sr, val_ir, val_ib, val_sb,
                        path_len=min(path_len, len(val_ir)),
                        batch_size=min(32, batch_size),
                        block_size=block_size, rng=rng,
                    )
                    vport, vidx = simulate_portfolio_returns(
                        policy, vbs, vbi, vbib, vbsb,
                        rho=stage_rho, device=device, normalizer=normalizer,
                        use_memory_override=use_memory_override,
                    )
                    val_loss = torch.stack([loss_fn(vport[b], vidx[b]) for b in range(vport.shape[0])]).mean().item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(policy.state_dict())
            else:
                val_loss = float("nan")
            if verbose:
                pbar.set_postfix({
                    "train_loss": f"{train_loss.item():.6f}",
                    "val_loss": f"{val_loss:.6f}" if not np.isnan(val_loss) else "N/A",
                    "rho": f"{stage_rho}",
                    "mem": "on" if use_memory_override else "off",
                })
        policy.load_state_dict(best_state)

    if rho > 0.0:
        # Step 1: rho=0, memory off, train Main/Score/Gate
        for p in policy.parameters():
            p.requires_grad = True
        opt1 = optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=lr)
        _run_stage(epochs, 0.0, False, opt1, f"[{strategy}] Step1 rho=0 mem=off")

        # Step 2: freeze Main/Score/Gate, train Memory only with rho>0
        for name, p in policy.named_parameters():
            p.requires_grad = ("memory_block" in name)
        mem_params = [p for p in policy.parameters() if p.requires_grad]
        if mem_params:
            opt2 = optim.Adam(mem_params, lr=lr)
            _run_stage(epochs, rho, True, opt2, f"[{strategy}] Step2 rho={rho} mem=on")
        for p in policy.parameters():
            p.requires_grad = True
    else:
        # rho=0: train first three blocks only (memory off)
        for p in policy.parameters():
            p.requires_grad = True
        opt = optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=lr)
        _run_stage(epochs, 0.0, False, opt, f"[{strategy}] rho=0 mem=off")

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        torch.save(policy.state_dict(), save_path)
        if verbose:
            print(f"[Trainer] 최적 모델 저장: {save_path}")

    return policy
