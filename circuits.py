# circuits.py
"""
Utility to turn one feature-dict into a Cirq circuit whose
measurement parity encodes bullish/bearish probability.
(This is a placeholder—you can swap for your proprietary circuit.)
"""

import cirq, math, hashlib

def build_circuit(feature: dict) -> cirq.Circuit:
    """
    Very small 2-qubit variational circuit:
        • encode scaled price diff into Ry rotation
        • simple entangler + measurement
    """
    q0, q1 = cirq.LineQubit.range(2)
    # Example encoding: (spot - strike) / strike → angle
    diff = (feature["spot_price"] - feature["strike_price"]) / feature["strike_price"]
    theta = float(diff) * math.pi
    c = cirq.Circuit(
        cirq.ry(theta).on(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, key="m")
    )
    # Give circuit a stable hash-based name to enable caching
    tag = hashlib.md5(str(feature).encode()).hexdigest()[:8]
    c._device = cirq.Device()  # Device field required by API
    c._name   = f"lqm_{tag}"
    return c
