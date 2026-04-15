"""
data_loader.py
==============
KOSPI200 데이터 로드 및 투자 유니버스 구성 모듈.

데이터 형식:
  - prices  : DataFrame (날짜 × 종목), 수정주가 (KRW, 정수)
  - index   : DataFrame (날짜 × 'KOSPI200'), 지수 레벨
  - mcap    : DataFrame (날짜 × 종목), 시가총액
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def load_price_data(
    price_path: str,
    index_path: str,
    mcap_path: str,
    base_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    CSV 파일에서 주가, 지수, 시가총액을 로드합니다.

    Returns
    -------
    prices : pd.DataFrame  (T × N), 수정주가
    index  : pd.Series     (T,),    KOSPI200 지수 레벨
    mcap   : pd.DataFrame  (T × N), 시가총액
    """
    if base_dir:
        price_path = str(Path(base_dir) / price_path)
        index_path = str(Path(base_dir) / index_path)
        mcap_path  = str(Path(base_dir) / mcap_path)

    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    index  = pd.read_csv(index_path, index_col=0, parse_dates=True).iloc[:, 0]
    mcap   = pd.read_csv(mcap_path,  index_col=0, parse_dates=True)

    prices.index = pd.to_datetime(prices.index)
    index.index  = pd.to_datetime(index.index)
    mcap.index   = pd.to_datetime(mcap.index)

    # 세 데이터셋의 날짜 교집합 사용
    common_dates = prices.index.intersection(index.index).intersection(mcap.index)
    prices = prices.loc[common_dates]
    index  = index.loc[common_dates]
    mcap   = mcap.loc[common_dates]

    print(f"[DataLoader] 로드 완료: {len(common_dates)}일 × {prices.shape[1]}종목")
    print(f"[DataLoader] 기간: {common_dates[0].date()} ~ {common_dates[-1].date()}")

    return prices, index, mcap


def build_universe(
    prices: pd.DataFrame,
    mcap: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    top_n: int = 20,
) -> list[str]:
    """
    논문 설정에 따른 투자 유니버스 구성.

    1. 훈련 기간 전체에 걸쳐 결측값 없는 종목만 선택
    2. 기준일(train_end) 시가총액 기준 상위 top_n 종목 선택

    Parameters
    ----------
    prices     : 전체 주가 DataFrame
    mcap       : 전체 시가총액 DataFrame
    train_start: 훈련 시작일
    train_end  : 훈련 종료일 (유니버스 기준일)
    top_n      : 편입 종목 수

    Returns
    -------
    list[str] : 선택된 종목 코드 목록
    """
    mask = (prices.index >= train_start) & (prices.index <= train_end)
    period_prices = prices.loc[mask]
    period_mcap   = mcap.loc[mask]

    # 훈련 기간 전체에 데이터 있는 종목만 유지
    valid_cols = period_prices.columns[period_prices.isna().sum() == 0].tolist()
    valid_cols = [c for c in valid_cols if c in period_mcap.columns]

    if len(valid_cols) < top_n:
        raise ValueError(
            f"유효 종목 수({len(valid_cols)})가 top_n({top_n})보다 작습니다. "
            f"훈련 기간 또는 top_n을 조정하세요."
        )

    # 기준일(train_end) 직전 유효 시총 기준 정렬
    ref_mcap = period_mcap[valid_cols].iloc[-1]
    universe = ref_mcap.nlargest(top_n).index.tolist()

    print(f"[Universe] top_n={top_n}, 유효종목={len(valid_cols)}, "
          f"기준일={train_end.date()}")
    print(f"[Universe] 편입 종목: {universe}")

    return universe


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    수정주가 → 일간 단순수익률 계산.
    첫 번째 행(NaN)은 제거합니다.
    """
    simple_ret = (prices / prices.shift(1) - 1.0).dropna(how="all")
    return simple_ret

def get_period_data(
    prices: pd.DataFrame,
    index: pd.Series,
    mcap: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    universe: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    특정 기간 + 유니버스 종목으로 슬라이싱.

    Returns
    -------
    stock_rets : DataFrame (T × N), 종목 단순수익률
    index_rets : Series   (T,),    지수 단순수익률
    mcap_slice : DataFrame (T × N), 시가총액
    """
    mask = (prices.index >= start) & (prices.index <= end)

    stock_prices = prices.loc[mask, universe]
    idx_prices   = index.loc[mask]
    mcap_slice   = mcap.loc[mask, universe]

    # 논문 재현 모드: 수익률 컨벤션은 단순수익률로 통일한다.
    stock_rets = compute_simple_returns(stock_prices)
    index_rets = compute_simple_returns(idx_prices.to_frame()).iloc[:, 0]

    # 날짜 정렬
    common = stock_rets.index.intersection(index_rets.index)
    stock_rets = stock_rets.loc[common]
    index_rets = index_rets.loc[common]
    mcap_slice = mcap_slice.loc[mcap_slice.index.isin(common)]

    return stock_rets, index_rets, mcap_slice
