# -*- coding: utf-8 -*-
"""本地推演筹码分布数据源

定位：
    补充数据源，不参与日线/实时行情主链路。
    东财没有公开 CYQ 接口，ak.stock_cyq_em 已失效；本源用
    OHLC + 换手率在本地推演筹码分布（三角形权重衰减换手模型）。

数据来源：
    1. baostock 日线（免 key，直接带换手率字段，优先）
    2. 腾讯 fqkline 日线（兜底，无换手率时按近 60 日成交量近似流通盘）

启用条件：
    无需任何 API key，由 DataFetcherManager 自动实例化。
    manager.get_chip_distribution() 会按优先级遍历所有实现了
    get_chip_distribution 的数据源，本源排在 AkshareFetcher 之前。
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from .base import (
    BaseFetcher,
    DataFetchError,
    _is_etf_code,
    _is_non_cn_request,
    normalize_stock_code,
)
from .realtime_types import ChipDistribution

logger = logging.getLogger(__name__)

# baostock 日线请求字段（turn 即换手率 %）
_BAOSTOCK_K_FIELDS = "date,open,high,low,close,turn,tradestatus"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126 Safari/537.36"
)


def _triangular_weights(hist, price_bins):
    """当日成交以三角形权重分布在 [低价, 高价]，峰值在中位价"""
    weights = _np().zeros(len(price_bins))
    for _open, high, low, _close, turnover in hist:
        peak = (high + low) / 2
        span = max(high - low, 1e-6)
        for i, p in enumerate(price_bins):
            if high >= p >= low:
                weights[i] += turnover * (1 - abs(p - peak) / (span / 2 + 1e-9))
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


def _np():
    import numpy as np

    return np


def _compute_chip_distribution(df: pd.DataFrame) -> dict:
    """核心推演：输入 OHLC + turnover(%)，返回各项筹码指标

    初始筹码播种为首日全部流通盘（避免从全零起步把窗口前的存量持仓清零），
    逐日按换手率衰减旧筹码、叠加当日三角形权重分布。
    """
    np = _np()

    df = df.dropna(subset=["close"]).copy()
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)

    if "turnover" in df.columns:
        df["turnover"] = df["turnover"].astype(float).fillna(0)
        df.loc[df["turnover"] <= 0, "turnover"] = 0.01
        hist_turns = df["turnover"].tolist()
    elif "volume" in df.columns:
        vol = df["volume"].astype(float).fillna(0)
        free_float = vol.iloc[-min(len(vol), 60):].sum() or 1.0
        df["turnover"] = (vol / free_float * 100).clip(0.01, 35)
        hist_turns = df["turnover"].tolist()
    else:
        raise DataFetchError("筹码推演需要 turnover 或 volume 列")

    hist = list(zip(df["open"], df["high"], df["low"], df["close"], hist_turns))
    pmin, pmax = df["low"].min(), df["high"].max()
    price_bins = np.linspace(pmin * 0.9, pmax * 1.1, 200)

    weights = np.ones(len(price_bins))
    weights /= weights.sum()

    for _o, h, l, _c, turn in hist:
        weights *= (1 - min(turn, 100) / 100)
        weights += _triangular_weights([(1, h, l, 1, turn)], price_bins)
        s = weights.sum()
        if s > 0:
            weights /= s

    cum = np.cumsum(weights)

    def pct_range(lo_pct, hi_pct):
        lo = price_bins[int(np.searchsorted(cum, lo_pct))]
        hi = price_bins[min(int(np.searchsorted(cum, hi_pct)), len(price_bins) - 1)]
        return float(lo), float(hi)

    price = float(df["close"].iloc[-1])
    profit_ratio = float(weights[price_bins <= price].sum())
    avg_cost = float((price_bins * weights).sum())
    c90_lo, c90_hi = pct_range(0.05, 0.95)
    c70_lo, c70_hi = pct_range(0.15, 0.85)

    def conc(lo, hi):
        mid = (lo + hi) / 2
        return (hi - lo) / abs(mid) if mid else 0.0

    return {
        "profit_ratio": profit_ratio,
        "avg_cost": avg_cost,
        "cost_90_low": c90_lo,
        "cost_90_high": c90_hi,
        "concentration_90": conc(c90_lo, c90_hi),
        "cost_70_low": c70_lo,
        "cost_70_high": c70_hi,
        "concentration_70": conc(c70_lo, c70_hi),
    }


class LocalChipFetcher(BaseFetcher):
    """本地推演筹码分布（补充源，仅提供 get_chip_distribution）"""

    name = "LocalChipFetcher"
    priority = 1  # 与 AkshareFetcher 同级；插入顺序在其之前，作为首选筹码源

    def __init__(self, lookback_days: int = 180) -> None:
        self.lookback_days = int(lookback_days)

    def is_available_for_request(self, capability: str = "") -> bool:
        """仅参与筹码分布链路，不进入日线/实时行情等主链路"""
        return capability == "chip"

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise DataFetchError("LocalChipFetcher 不提供日线行情数据（专用筹码补充源）")

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        raise DataFetchError("LocalChipFetcher 不提供日线行情数据（专用筹码补充源）")

    # ------------------------------------------------------------------
    # 行情获取（两级降级）
    # ------------------------------------------------------------------

    def _kline_baostock(self, code: str) -> Optional[pd.DataFrame]:
        """baostock 日线：直接带换手率字段（免 key），优先"""
        try:
            import baostock as bs
        except ImportError:
            return None
        prefix = (
            "sh"
            if code.startswith(("6", "5", "9"))
            else ("bj" if code.startswith(("4", "8", "92")) else "sz")
        )
        end = date.today()
        start = end - timedelta(days=int(self.lookback_days * 1.6))
        try:
            bs.login()
            rs = bs.query_history_k_data_plus(
                f"{prefix}.{code}",
                _BAOSTOCK_K_FIELDS,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                frequency="d",
                adjustflag="2",
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            bs.logout()
        except Exception as exc:
            logger.warning("[LocalChip] baostock 拉 K 线失败 %s: %s", code, exc)
            return None
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=_BAOSTOCK_K_FIELDS.split(","))
        df = df[df["tradestatus"] == "1"]
        for c in ("open", "high", "low", "close", "turn"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.rename(columns={"turn": "turnover"}).dropna(
            subset=["open", "high", "low", "close"]
        )

    def _kline_tencent(self, code: str) -> Optional[pd.DataFrame]:
        """腾讯日 K（前复权）兜底：无换手率，走成交量近似分支"""
        import requests

        prefix = (
            "sh"
            if code.startswith(("6", "5", "9"))
            else ("bj" if code.startswith(("4", "8", "92")) else "sz")
        )
        url = (
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
            f"param={prefix}{code},day,,,{self.lookback_days + 140},qfq"
        )
        try:
            r = requests.get(url, headers={"User-Agent": _UA}, timeout=15)
            data = r.json()["data"][f"{prefix}{code}"]
        except Exception as exc:
            logger.warning("[LocalChip] 腾讯拉 K 线失败 %s: %s", code, exc)
            return None
        rows = data.get("qfqday") or data.get("day") or []
        if not rows:
            return None
        df = pd.DataFrame(rows).iloc[:, :6]
        df.columns = ["date", "open", "close", "high", "low", "volume"]
        for c in ("open", "close", "high", "low", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        # 腾讯 qfq 日线 volume 单位：手（科创板为股）
        df["volume"] = df["volume"] * (100 if not code.startswith("68") else 1)
        return df

    # ------------------------------------------------------------------
    # 筹码分布协议入口
    # ------------------------------------------------------------------

    def get_chip_distribution(self, stock_code: str) -> Optional[ChipDistribution]:
        """
        本地推演筹码分布核心指标：
        获利比例 / 平均成本 / 90% 与 70% 成本区间及集中度。

        Returns:
            ChipDistribution（source=local_cyq），失败返回 None 交给下一数据源。
        """
        stock_code = normalize_stock_code(stock_code)
        if _is_non_cn_request(stock_code) or _is_etf_code(stock_code):
            logger.debug(f"[API跳过] {stock_code} 非A股个股，无筹码分布数据")
            return None

        code = stock_code.split(".")[-1]
        if not (code.isdigit() and len(code) == 6):
            return None

        try:
            df = self._kline_baostock(code)
            if df is None or len(df) < 30:
                df = self._kline_tencent(code)
        except Exception as exc:
            logger.warning("[LocalChip] %s 拉行情失败: %s", stock_code, exc)
            return None
        if df is None or len(df) < 30:
            logger.warning("[LocalChip] %s 行情数据不足 30 日，跳过", stock_code)
            return None

        try:
            metrics = _compute_chip_distribution(df)
        except Exception as exc:
            logger.warning("[LocalChip] %s 推演失败: %s", stock_code, exc)
            return None

        chip = ChipDistribution(
            code=stock_code,
            date=str(df["date"].iloc[-1]) if "date" in df.columns else "",
            source="local_cyq",
            **metrics,
        )
        logger.info(
            "[LocalChip] %s 推演完成: 获利比例=%.1f%%, 平均成本=%.2f, "
            "90集中度=%.2f%%, 70集中度=%.2f%%",
            stock_code,
            chip.profit_ratio * 100,
            chip.avg_cost,
            chip.concentration_90 * 100,
            chip.concentration_70 * 100,
        )
        return chip
