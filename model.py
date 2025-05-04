# model.py
"""
Large Quantitative Model (LQM) – batch, Firestore-budgeted, QPU-ready

• RULE weights can be tuned automatically by tune.py
• best_params.json or rules.json (whichever exists) overrides defaults
• Batch endpoint runs classical preview, then optional Cirq-sim "QPU" refinement
"""

from __future__ import annotations
from dataclasses import dataclass
import json, pathlib, logging
import pandas as pd, yfinance as yf
from firestore_util import use_shots
import qpu_client, circuits

log = logging.getLogger("model")

# ───────────────────────── 1. RULES (defaults) ──────────────────────────
RULES: dict[str, float] = {
    # weight parameters (−1 … +1)
    "ma_trend_weight":    0.4,
    "rsi_extreme_weight": 0.3,
    "vol_rank_weight":    0.2,
    "momentum_3d_weight": 0.1,
    # thresholds
    "rsi_high": 70,
    "rsi_low":  30,
    "iv_rank_high": 0.7,
    "iv_rank_low":  0.3,
    # optional circuit angles
    "angle_0": 1.5707,
    "angle_1": 1.5707,
}

# Try best_params.json first, then rules.json
for fname in ("best_params.json", "rules.json"):
    p = pathlib.Path(__file__).with_name(fname)
    if p.exists():
        log.info("Loading parameters from %s", fname)
        RULES.update(json.loads(p.read_text()))
        break

# ───────────────────────── 2. Helper functions ─────────────────────────
def fetch_last_n(ticker: str, n: int) -> pd.DataFrame:
    return yf.download(ticker, period=f"{n}d", progress=False)

def ma(df: pd.DataFrame, w: int) -> float:
    return df["Close"].tail(w).mean().item()

def rsi(df: pd.DataFrame, w: int = 14) -> float:
    delta = df["Close"].diff().dropna()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    if len(delta) < w:
        return 50.0
    avg_gain = gains.tail(w).mean().item()
    avg_loss = losses.tail(w).mean().item()
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def iv_rank(cur_iv: float, hist: pd.Series) -> float:
    return (hist < cur_iv).mean().item()

# ───────────────────────── 3. Predictor class ──────────────────────────
QPU_THRESHOLD   = 0.75            # send to QPU if preview conf ≥ 0.75
SHOTS_PER_FEAT  = qpu_client.shots_per_row

@dataclass
class OptionsPredictor:
    lookback_days: int = 20
    ticker: str = "AAPL"

    # ----- single classical prediction --------------------------------
    def _classical_predict(self) -> tuple[int, float]:
        df = fetch_last_n(self.ticker, self.lookback_days)
        close = df["Close"].iloc[-1].item()
        ma10  = ma(df, 10)
        mom3d = (close - df["Close"].iloc[-4].item()) / df["Close"].iloc[-4].item()
        rsi14 = rsi(df, 14)

        iv_today = 0.35
        iv_hist  = pd.Series([0.25, 0.28, 0.30, 0.27, 0.33, 0.38, 0.31])
        iv_rnk   = iv_rank(iv_today, iv_hist)

        score = 0.0
        score += RULES["ma_trend_weight"]    * (1 if close > ma10 else -1)
        score += RULES["momentum_3d_weight"] * (1 if mom3d > 0 else -1)

        if rsi14 >= RULES["rsi_high"]:
            score -= RULES["rsi_extreme_weight"]
        elif rsi14 <= RULES["rsi_low"]:
            score += RULES["rsi_extreme_weight"]

        if iv_rnk >= RULES["iv_rank_high"]:
            score -= RULES["vol_rank_weight"]
        elif iv_rnk <= RULES["iv_rank_low"]:
            score += RULES["vol_rank_weight"]

        prediction = 1 if score >= 0 else -1
        confidence = round(min(abs(score), 1.0), 2)
        return prediction, confidence

    # ----- batch API ---------------------------------------------------
    def batch_predict(self, feats: list[dict]) -> tuple[list[int], list[float]]:
        # 1) preview with classical logic
        pred, conf = self._classical_predict()
        preds = [pred]  * len(feats)
        confs = [conf]  * len(feats)

        # 2) determine which rows qualify for QPU refinement
        idx_qpu = [i for i, c in enumerate(confs) if c >= QPU_THRESHOLD]
        needed_shots = len(idx_qpu) * SHOTS_PER_FEAT

        # 3) fire to QPU if budget allows
        if idx_qpu and use_shots(needed_shots):
            circuits_batch = [circuits.build_circuit(feats[i]) for i in idx_qpu]
            hists = qpu_client.submit(circuits_batch)

            # convert histogram to prediction/confidence
            for j, i in enumerate(idx_qpu):
                ones   = hists[j].get("1", 0)
                zeros  = hists[j].get("0", 0)
                total  = max(ones + zeros, 1)
                p_up   = ones / total
                preds[i] = 1 if p_up >= 0.5 else -1
                confs[i] = round(abs(p_up - 0.5) * 2, 2)

        return preds, confs
