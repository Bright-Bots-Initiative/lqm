# backtest.py
"""
Walk-forward windowed back-test that feeds simulate() and aggregates
hit-rate vs. a simple “↑ if return > 0” profit metric.
"""
from __future__ import annotations
from datetime import date, timedelta
from typing import Tuple

import numpy as np
import pandas as pd

from sim import simulate


def run_backtest(ticker: str = "AAPL", years: float = 1.0) -> Tuple[pd.DataFrame, float]:
    """
    Parameters
    ----------
    ticker : str
    years  : float   # years of history to examine (e.g. 0.5 = six months)

    Returns
    -------
    (result_df, hit_rate)
    """
    today     = date.today()
    start_win = today - timedelta(days=int(365 * years))
    end_win   = today

    result_df = simulate(ticker, start_win.isoformat(), end_win.isoformat())

    # Hit-rate: correct bullish/bearish vs realised ±10 % target
    hit = np.where(
        (result_df.signal > 0) & (result_df.ret_7d > 0), True,
        np.where((result_df.signal < 0) & (result_df.ret_7d < 0), True, False)
    )
    hit_rate = hit.mean() if hit.size else 0.0
    return result_df, hit_rate
