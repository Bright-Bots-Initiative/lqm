# app.py  – batch-aware Flask API
from flask import Flask, request, jsonify
from model import OptionsPredictor
from assessment import SelfAssessment
import logging

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)          # silence access log

predictor = OptionsPredictor()
evaluator = SelfAssessment()

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST JSON  {"features":[{...}, {...}, ...]}
    ↳ returns   {"predictions":[1,-1,...], "confidences":[0.7,0.4,...]}
    """
    feats = request.get_json(force=True)["features"]
    preds, confs = predictor.batch_predict(feats)
    return jsonify({"predictions": preds, "confidences": confs})

@app.route("/evaluate", methods=["GET"])
def evaluate():
    return jsonify(evaluator.evaluate())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
