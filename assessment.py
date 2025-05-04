# assessment.py
import pandas as pd
from datetime import datetime, timezone
from firestore_util import log_doc

class SelfAssessment:
    def __init__(self):
        self.logs=[]

    def log(self,features,pred,actual,conf):
        rec={
            "ts":datetime.now(timezone.utc).isoformat(),
            "pred":pred,"actual":actual,
            "correct":pred==actual,"conf":conf
        }
        self.logs.append(rec)
        log_doc("prediction_logs",rec)     # Firestore write

    def evaluate(self):
        if not self.logs:
            return {"total":0,"accuracy":None,"avg_confidence":None}
        df=pd.DataFrame(self.logs)
        return {
            "total":len(df),
            "accuracy":round(df["correct"].mean(),3),
            "avg_confidence":round(df["conf"].mean(),3)
        }
