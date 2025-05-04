# sim.py
"""
Generate one–step-ahead 7-day return samples plus model signal.

The simulation:
  • pulls price history with yfinance
  • builds a minimal feature vector for each day
  • calls OptionsPredictor.batch_predict() directly (no HTTP)
"""
from __future__ import annotations
from datetime import timedelta
from typing import List, Dict

import pandas as pd
import yfinance as yf

from model import OptionsPredictor

predictor = OptionsPredictor()            # single shared instance


def _build_features(close_px: float) -> List[Dict]:
    """Create a single-row feature list expected by OptionsPredictor."""
    return [{
        "spot_price":       close_px,
        "strike_price":     close_px * 1.05,
        "volatility":       0.25,                # stub – replace with IV if available
        "time_to_maturity": 7 / 365,
        "risk_free_rate":   0.03
    }]


def simulate(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Simulate daily signals & 7-day realised returns between start & end.

    Returns
    -------
    pd.DataFrame with columns
        date, ret_7d, signal  (1 = bullish, 0 = neutral, -1 = bearish)
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty or len(df) < 8:
        raise ValueError("Not enough historical data for simulation window.")

    rows = []
    for today, tomorrow in zip(df.index[:-7], df.index[7:]):       # 7-day horizon
        close_today = float(df.loc[today, "Close"])
        close_next  = float(df.loc[tomorrow, "Close"])

        feats = _build_features(close_today)
        preds, confs = predictor.batch_predict(feats)  # batch_predict returns lists
        signal, _conf = preds[0], confs[0]             # take first elements since we only have one set of features
        rows.append({
            "date":    today,
            "ret_7d":  (close_next - close_today) / close_today,
            "signal":  signal
        })

    return pd.DataFrame(rows)
