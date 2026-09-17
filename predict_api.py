"""
Prediction API for the AQI Prediction section.

Serves the actual trained model (aqi_rf_deploy_model.joblib) -- this endpoint does NOT
recompute AQI from the CPCB formula. It loads the model and calls .predict() directly.

Run:
    pip install flask flask-cors scikit-learn pandas joblib
    python predict_api.py

The frontend (aqi_predict_section.html) expects this running at:
    POST /api/predict-aqi
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)  # allow the frontend (possibly on a different port) to call this

MODEL_PATH = os.path.join(os.path.dirname(__file__), "aqi_rf_deploy_model.joblib")
model = joblib.load(MODEL_PATH)

# Exact feature names and order the model was trained on (do not reorder --
# a RandomForestRegressor fit on a DataFrame is not strictly order-sensitive
# by name, but we pass a DataFrame with these exact column names to be safe
# and explicit about what the model expects).
FEATURE_ORDER = list(model.feature_names_in_)

AQI_CATEGORIES = [
    (50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
    (300, "Poor"), (400, "Very Poor"), (float("inf"), "Severe"),
]


def categorize(aqi: float) -> str:
    for threshold, name in AQI_CATEGORIES:
        if aqi <= threshold:
            return name
    return "Severe"


@app.route("/api/predict-aqi", methods=["POST"])
def predict_aqi():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be JSON."}), 400

    missing = [f for f in FEATURE_ORDER if f not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        row = {f: float(payload[f]) for f in FEATURE_ORDER}
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric."}), 400

    X = pd.DataFrame([row], columns=FEATURE_ORDER)
    prediction = float(model.predict(X)[0])

    return jsonify({
        "predicted_aqi": round(prediction, 1),
        "category": categorize(prediction),
        "horizon": "6h",
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_features": FEATURE_ORDER})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
