from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path
import pandas as pd

app = Flask(__name__)
ROOT = Path(__file__).parent

# Load saved model and preprocessors
MODEL = joblib.load(ROOT / "models" / "model.pkl")
IMPUTER = joblib.load(ROOT / "models" / "imputer.pkl")
SCALER = joblib.load(ROOT / "models" / "scaler.pkl")

# Load CSV just to get feature names
X_sample = pd.read_csv(ROOT / "data" / "water_potability.csv")
FEATURES = list(X_sample.drop(columns=["Potability"]).columns)

@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        vals = [float(request.form.get(f, 0)) for f in FEATURES]
        x = np.array(vals).reshape(1, -1)

        # Impute missing values and scale
        x_imp = IMPUTER.transform(x)
        x_scaled = SCALER.transform(x_imp)

        # Make prediction (0 or 1)
        pred = MODEL.predict(x_scaled)[0]
        label = "Safe to Drink ✅" if pred == 1 else "Not Safe ❌"

        # Render result.html with just the label
        return render_template("result.html", label=label)

    except Exception as e:
        return f"Error: {e}", 400

if __name__ == "__main__":
    app.run(debug=True)
