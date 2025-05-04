# qpu_client.py
"""
Local Cirq simulator that pretends to be the real QPU.
Costs $0, but keeps the same shots_per_row constant so budgeting logic
and later QPU swap-in are seamless.
"""

import cirq, time, logging

shots_per_row = 100
log = logging.getLogger("qpu_client")

def submit(circuits: list[cirq.Circuit]):
    """Return histogram dicts using Cirq’s built-in simulator."""
    simulator = cirq.Simulator()
    results = []
    for c in circuits:
        r = simulator.run(c, repetitions=shots_per_row)
        ones  = r.histogram(key="m").get(1, 0)
        zeros = shots_per_row - ones
        results.append({"1": ones, "0": zeros})
        time.sleep(0.002)        # tiny pause to mimic network lag
    return results
