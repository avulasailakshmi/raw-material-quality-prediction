"""Flask web app for raw material quality prediction."""
from flask import Flask, request, jsonify, send_from_directory
import pickle, json, os
import numpy as np

app = Flask(__name__)
ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(ROOT, "templates")
MODELS_DIR = os.path.join(ROOT, "models")
INDUSTRIES = {"Food Processing":"food_processing","Textile":"textile","Cosmetics":"cosmetics"}

def load_artifacts():
    arts = {}
    for label, name in INDUSTRIES.items():
        with open(os.path.join(MODELS_DIR, f"{name}_scaler.pkl"), "rb") as f: scaler = pickle.load(f)
        with open(os.path.join(MODELS_DIR, f"{name}_le.pkl"), "rb") as f: le = pickle.load(f)
        with open(os.path.join(MODELS_DIR, f"{name}_models.pkl"), "rb") as f: models = pickle.load(f)
        arts[label] = {"scaler": scaler, "le": le, "models": models}
    with open(os.path.join(MODELS_DIR, "summary.json"), encoding="utf-8") as f: summary = json.load(f)
    return arts, summary

ARTS, SUMMARY = load_artifacts()

@app.route("/")
def home(): return send_from_directory(TEMPLATES_DIR, "landing.html")
@app.route("/predict")
def predict_page(): return send_from_directory(TEMPLATES_DIR, "predict.html")
@app.route("/analytics")
def analytics_page(): return send_from_directory(TEMPLATES_DIR, "analytics.html")
@app.route("/about")
def about_page(): return send_from_directory(TEMPLATES_DIR, "about.html")
@app.route("/api/summary")
def summary(): return jsonify(SUMMARY)

@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    industry, model_name, features = body.get("industry"), body.get("model"), body.get("features")
    if industry not in ARTS: return jsonify({"error":"Unknown industry"}), 400
    if model_name not in ARTS[industry]["models"]: return jsonify({"error":"Unknown model"}), 400
    if not features: return jsonify({"error":"No features provided"}), 400
    art = ARTS[industry]
    x = art["scaler"].transform(np.array(features, dtype=float).reshape(1,-1))
    model = art["models"][model_name]
    encoded = model.predict(x)[0]
    label = art["le"].inverse_transform([encoded])[0]
    probabilities = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)[0]
        probabilities = {c: round(float(p[i])*100,1) for i,c in enumerate(art["le"].classes_)}
    return jsonify({"prediction":label,"probabilities":probabilities,"model":model_name,"industry":industry})

@app.route("/api/predict_all", methods=["POST"])
def predict_all():
    body = request.get_json(force=True)
    industry, features = body.get("industry"), body.get("features")
    if industry not in ARTS or not features: return jsonify({"error":"Invalid input"}), 400
    art = ARTS[industry]
    x = art["scaler"].transform(np.array(features,dtype=float).reshape(1,-1))
    results = {}
    for name, model in art["models"].items():
        encoded = model.predict(x)[0]
        results[name] = art["le"].inverse_transform([encoded])[0]
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1", port=5000)
