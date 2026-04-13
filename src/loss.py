"""
loss.py
=======
논문 Section 4: 손실 함수 구현.

IT (Index Tracking):
  L_IT = TE(r_p, r_i)

EIT (Enhanced Index Tracking):
  L_EIT = TE(r_p, r_i) − λ · ER(r_p, r_i)
  λ = 20

EIT-CVaR:
  L_EIT_CVaR = TE − λ · ER + P_CVaR
  P_CVaR = γ · Softplus_β(CVaR_α(r_p) − c)
  α=0.05, c=0.03, γ=1e6, β=2000

모든 함수는 PyTorch 텐서를 입력으로 받아 스칼라 텐서를 반환합니다.
"""

import torch
import torch.nn.functional as F


def tracking_error(r_p: torch.Tensor, r_i: torch.Tensor) -> torch.Tensor:
    """
    Tracking Error (TE) = √mean((r_p - r_i)²)

    Parameters
    ----------
    r_p : (T,) or (B, T), 포트폴리오 수익률
    r_i : (T,) or (B, T), 인덱스 수익률

    Returns
    -------
    TE : scalar tensor
    """
    diff = r_p - r_i
    return torch.sqrt(torch.mean(diff ** 2))


def excess_return(r_p: torch.Tensor, r_i: torch.Tensor) -> torch.Tensor:
    """
    Mean Excess Return (ER) = mean(r_p - r_i)

    Parameters
    ----------
    r_p : (T,) or (B, T)
    r_i : (T,) or (B, T)

    Returns
    -------
    ER : scalar tensor
    """
    return torch.mean(r_p - r_i)


def cvar(r_p: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    CVaR_α(r_p) = 하위 α 분위 평균 손실 (손실 = -수익률).

    CVaR > 0 이면 큰 손실을 의미합니다.

    Parameters
    ----------
    r_p   : (T,), 포트폴리오 수익률 시계열
    alpha : 신뢰수준 (하위 alpha 분위)

    Returns
    -------
    cvar_val : scalar tensor
    """
    losses = -r_p                                          # 손실 = -수익률
    k = max(1, int(len(losses) * alpha))
    sorted_losses, _ = torch.sort(losses, descending=True)
    cvar_val = sorted_losses[:k].mean()
    return cvar_val


def softplus_penalty(
    cvar_val: torch.Tensor,
    c: float = 0.03,
    gamma: float = 1e6,
    beta: float = 2000.0,
) -> torch.Tensor:
    """
    CVaR 소프트플러스 패널티:
    P_CVaR = γ · Softplus_β(CVaR_α - c)
           = γ · (1/β) · log(1 + exp(β · (CVaR_α - c)))

    CVaR가 허용 임계값 c 이하이면 패널티 ≈ 0.
    초과하면 γ 스케일의 큰 패널티.

    Parameters
    ----------
    cvar_val : scalar tensor, CVaR_α(r_p)
    c        : 허용 CVaR 임계값 (논문: 0.03)
    gamma    : 패널티 스케일 (논문: 1e6)
    beta     : Softplus 스케일 (논문: 2000)

    Returns
    -------
    penalty : scalar tensor
    """
    # Softplus_β(x) = (1/β) * log(1 + exp(β*x))
    x = beta * (cvar_val - c)
    softplus_val = F.softplus(x) / beta
    return gamma * softplus_val


def loss_IT(r_p: torch.Tensor, r_i: torch.Tensor) -> torch.Tensor:
    """
    IT 손실: Tracking Error만 최소화.

    L_IT = TE(r_p, r_i)
    """
    return tracking_error(r_p, r_i)


def loss_EIT(
    r_p: torch.Tensor,
    r_i: torch.Tensor,
    lam: float = 20.0,
) -> torch.Tensor:
    """
    EIT 손실: TE 최소화 + 초과수익 극대화.

    L_EIT = TE(r_p, r_i) − λ · ER(r_p, r_i)
    λ = 20 (논문 설정)
    """
    te = tracking_error(r_p, r_i)
    er = excess_return(r_p, r_i)
    return te - lam * er


def loss_EIT_CVaR(
    r_p: torch.Tensor,
    r_i: torch.Tensor,
    lam: float = 20.0,
    alpha: float = 0.05,
    c: float = 0.03,
    gamma: float = 1e6,
    beta: float = 2000.0,
) -> torch.Tensor:
    """
    EIT-CVaR 손실: EIT + CVaR 소프트플러스 패널티.

    L = TE − λ · ER + γ · Softplus_β(CVaR_α(r_p) − c)

    Parameters
    ----------
    r_p   : (T,), 포트폴리오 수익률
    r_i   : (T,), 인덱스 수익률
    lam   : 초과수익 가중치 λ (논문: 20)
    alpha : CVaR 신뢰수준 (논문: 0.05)
    c     : CVaR 허용 임계값 (논문: 0.03)
    gamma : 패널티 스케일 (논문: 1e6)
    beta  : Softplus 스케일 (논문: 2000)
    """
    te       = tracking_error(r_p, r_i)
    er       = excess_return(r_p, r_i)
    cvar_val = cvar(r_p, alpha=alpha)
    penalty  = softplus_penalty(cvar_val, c=c, gamma=gamma, beta=beta)
    return te - lam * er + penalty


def get_loss_fn(strategy: str):
    """
    전략명 → 손실 함수 반환.

    Parameters
    ----------
    strategy : 'IT' | 'EIT' | 'EIT-CVaR'

    Returns
    -------
    callable(r_p, r_i) → scalar tensor
    """
    if strategy == "IT":
        return loss_IT
    elif strategy == "EIT":
        return loss_EIT
    elif strategy == "EIT-CVaR":
        return loss_EIT_CVaR
    else:
        raise ValueError(f"알 수 없는 전략: '{strategy}'. 'IT', 'EIT', 'EIT-CVaR' 중 선택하세요.")
