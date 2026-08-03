"""Epic 5 web application for flood prediction.
Run `python app.py` and open http://127.0.0.1:5000.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "flood dataset.xlsx"
TARGET = "flood"

# The form uses friendly names while the model continues to receive the original
# dataset column names. Rainfall values use the same unit as the source dataset: mm.
FIELD_DETAILS = {
    "Temp": ("Temperature (°C)", "Current air temperature."),
    "Humidity": ("Humidity (%)", "Current relative humidity."),
    "Cloud Cover": ("Cloud cover (%)", "Percentage of the sky covered by clouds."),
    "ANNUAL": ("Annual rainfall (mm)", "Total rainfall recorded for the year."),
    "Jan-Feb": ("January–February rainfall (mm)", "Total rainfall during January and February."),
    "Mar-May": ("March–May rainfall (mm)", "Total pre-monsoon rainfall."),
    "Jun-Sep": ("June–September rainfall (mm)", "Total monsoon-season rainfall."),
    "Oct-Dec": ("October–December rainfall (mm)", "Total post-monsoon rainfall."),
    "avgjune": ("Average June rainfall (mm)", "Dataset-derived value: June rainfall divided by 3."),
    "sub": ("June–May rainfall difference (mm)", "Dataset-derived value: June rainfall minus May rainfall."),
}

app = Flask(__name__)


def load_application_model():
    """Train the app's prediction pipeline from the supplied flood dataset."""
    data = pd.read_excel(DATA_PATH)
    if TARGET not in data.columns:
        raise KeyError(f"The dataset must contain a '{TARGET}' column.")
    features = data.drop(columns=TARGET)
    if len(features.select_dtypes(exclude="number").columns) > 0:
        raise ValueError("The application form supports numeric feature columns only.")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(features, data[TARGET])
    return pipeline, list(features.columns), features.median().to_dict()


def form_fields():
    """Return fields with human-readable labels and help text for the template."""
    return [
        {"name": name, "label": FIELD_DETAILS.get(name, (name, ""))[0],
         "help": FIELD_DETAILS.get(name, (name, ""))[1]}
        for name in FEATURE_COLUMNS
    ]


MODEL, FEATURE_COLUMNS, DEFAULTS = load_application_model()


@app.get("/")
def home():
    return render_template("index.html", fields=form_fields(), defaults=DEFAULTS)


@app.post("/predict")
def predict():
    try:
        values = {field: float(request.form[field]) for field in FEATURE_COLUMNS}
        feature_row = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        prediction = int(MODEL.predict(feature_row)[0])
        probability = float(MODEL.predict_proba(feature_row)[0][1]) * 100
        return render_template("result.html", prediction=prediction, probability=probability)
    except (KeyError, TypeError, ValueError) as error:
        return render_template(
            "index.html", fields=form_fields(), defaults=DEFAULTS,
            error=f"Enter a valid number for every feature. ({error})",
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
