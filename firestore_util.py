# firestore_util.py
"""
Lightweight helpers for Firestore:
• get_db() returns a cached client
• use_shots(n) atomically checks & decrements the daily shot budget
• log_doc(col, doc) writes a document to a collection
"""

from google.cloud import firestore
from datetime import datetime, timezone

_DAILY_BUDGET = 2_500   # change if you buy more shots
_COLLECTION    = "lqm_logs"
_COUNTER_ID    = "shot_counter"

_client = None
def get_db():
    global _client
    if _client is None:
        _client = firestore.Client()
    return _client

def _today_key():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def use_shots(n: int) -> bool:
    """
    Atomically decrement today's counter by n.
    Returns True if allowed, False if would exceed budget.
    """
    db = get_db()
    doc = db.collection(_COLLECTION).document(_COUNTER_ID)
    today = _today_key()

    def txn(tx):
        snap = doc.get(transaction=tx)
        data = snap.to_dict() or {}
        used = data.get(today, 0)
        if used + n > _DAILY_BUDGET:
            return False
        data[today] = used + n
        tx.set(doc, data)
        return True

    return db.transaction()(txn)

def log_doc(collection: str, data: dict):
    get_db().collection(collection).add(data)
