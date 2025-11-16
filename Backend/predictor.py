import sys, json, joblib, os
import numpy as np

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.joblib")
IMPUTER_PATH = os.path.join(ARTIFACT_DIR, "imputer.joblib")
ENCODERS_PATH = os.path.join(ARTIFACT_DIR, "encoders.joblib")

FEATURE_ORDER = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "cholestoral",
    "fasting_blood_sugar",
    "rest_ecg",
    "thalach",
    "exercise_induced_angina",
    "oldpeak",
    "slope",
    "vessels_colored_by_flourosopy",
    "thalassemia"
]

ALIAS_MAP = {
    "trestbps": "resting_blood_pressure",
    "trestbp": "resting_blood_pressure",
    "trestbps": "resting_blood_pressure",
    "chol": "cholestoral",
    "fbs": "fasting_blood_sugar",
    "restecg": "rest_ecg",
    "thalach": "thalach",
    "exang": "exercise_induced_angina",
    "ca": "vessels_colored_by_flourosopy",
    "thal": "thalassemia",
    "cp": "chest_pain_type",
}

THRESHOLD = 0.4  

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model.joblib not found; run save_model.py first")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, scaler, imputer, encoders

def map_input_keys(payload):
    mapped = {}
    for k, v in payload.items():
        key = k.lower()
        if key in FEATURE_ORDER:
            mapped[key] = v
        elif key in ALIAS_MAP:
            mapped[ALIAS_MAP[key]] = v
        else:
            key2 = key.replace("-", "_").replace(" ", "_")
            if key2 in FEATURE_ORDER:
                mapped[key2] = v
            else:
                mapped[key] = v  
    return mapped

def category_from_prob(p):
    if p < 0.33:
        return "Low"
    elif p < 0.66:
        return "Medium"
    else:
        return "High"

def main():
    try:
        raw = sys.stdin.read()
        if not raw:
            print(json.dumps({"error":"no input"}))
            return
        payload = json.loads(raw)
        model, scaler, imputer, encoders = load_artifacts()

        mapped = map_input_keys(payload)

        features = []
        for feat in FEATURE_ORDER:
            if feat in mapped:
                val = mapped[feat]
                if feat in encoders:
                    le = encoders[feat]
                    try:
                        if isinstance(val, (int, float)):
                            vnum = float(val)
                            features.append(vnum)
                        else:
                            vstr = str(val)
                            if vstr in list(le.classes_):
                                features.append(float(le.transform([vstr])[0]))
                            else:
                                try:
                                    features.append(float(vstr))
                                except:
                                    features.append(np.nan)
                    except Exception:
                        features.append(np.nan)
                else:
                    try:
                        features.append(float(val))
                    except:
                        features.append(np.nan)
            else:
                features.append(np.nan)

        X = np.array(features).reshape(1, -1)

        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            if len(probs) > 1:
                pos_prob = float(probs[1])
            else:
                pos_prob = float(probs[0])
        else:
            pred = model.predict(X_scaled)[0]
            pos_prob = float(pred)

        risk_pct = round(pos_prob * 100, 2)
        pred_class = 1 if pos_prob >= THRESHOLD else 0
        cat = category_from_prob(pos_prob)

        out = {
            "risk_pct": risk_pct,
            "class": cat,
            "raw_score": pos_prob,
            "pred_class": int(pred_class),
            "threshold": THRESHOLD
        }
        print(json.dumps(out))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
