"""Flood prediction project structured by Epics 2, 3, and 4.

Run: python rising-water.py
Outputs: outputs/ (charts, comparison table, and best_flood_model.pkl)
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "flood dataset.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"
TARGET = "flood"
RANDOM_STATE = 42


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# =============================================================================
# Epic 2: Visualizing and Analysing the Data
#   1. Importing the Libraries       2. Reading the Dataset
#   3. Univariate Analysis           4. Multivariate Analysis
#   5. Descriptive Analysis
# =============================================================================
def read_dataset() -> pd.DataFrame:
    """Read the Excel file and check that its prediction column is available."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_excel(DATA_PATH)
    if data.empty:
        raise ValueError("The flood dataset is empty.")
    if TARGET not in data.columns:
        raise KeyError(f"Column '{TARGET}' was not found. Found: {list(data.columns)}")
    return data


def descriptive_analysis(data: pd.DataFrame) -> None:
    print(f"Dataset size: {data.shape[0]} rows × {data.shape[1]} columns")
    print("\nFirst five rows:\n", data.head().to_string(index=False))
    print("\nData types:\n", data.dtypes.to_string())
    print("\nDescriptive statistics:\n", data.describe(include="all").T.to_string())


def create_visualisations(data: pd.DataFrame) -> None:
    """Create the univariate distribution and multivariate heatmap charts."""
    numeric_data = data.select_dtypes(include=np.number)
    if numeric_data.empty:
        return
    OUTPUT_DIR.mkdir(exist_ok=True)

    feature = "Temp" if "Temp" in numeric_data.columns else numeric_data.columns[0]
    plt.figure(figsize=(8, 5))
    sns.histplot(numeric_data[feature], kde=True, color="#0f766e")
    plt.title(f"Univariate Analysis: {feature} Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "univariate_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="summer", vmin=-1, vmax=1,
                linewidths=.4, linecolor="white", square=True)
    plt.title("Multivariate Analysis: Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"Charts saved to: {OUTPUT_DIR}")


# =============================================================================
# Epic 3: Data Pre-processing
#   1. Handling Missing Values       2. Handling Outliers
#   3. Handling Categorical Values   4. Splitting Training/Test Sets
#   5. Feature Scaling
# =============================================================================
def clip_outliers(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IQR-clip numeric features using bounds learned only from training data."""
    train, test = x_train.copy(), x_test.copy()
    for column in train.select_dtypes(include=np.number).columns:
        lower_quartile, upper_quartile = train[column].quantile([.25, .75])
        iqr = upper_quartile - lower_quartile
        if pd.notna(iqr) and iqr > 0:
            lower, upper = lower_quartile - 1.5 * iqr, upper_quartile + 1.5 * iqr
            train[column] = train[column].clip(lower, upper)
            test[column] = test[column].clip(lower, upper)
    return train, test


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    transformers = [
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scaler", StandardScaler())]), numeric)
    ]
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers)


def prepare_data(data: pd.DataFrame):
    x, y = data.drop(columns=TARGET), data[TARGET]
    if y.isna().any():
        raise ValueError("The flood target contains missing values; correct them before training.")
    counts = y.value_counts()
    stratify = y if y.nunique() > 1 and counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.25, random_state=RANDOM_STATE, stratify=stratify
    )
    x_train, x_test = clip_outliers(x_train, x_test)
    return x_train, x_test, y_train, y_test


# =============================================================================
# Epic 4: Model Building
#   1. Decision Tree  2. Random Forest  3. KNN  4. XGBoost
#   5. Comparing Models  6. Evaluating Performance and Saving the Model
# =============================================================================
def model_definitions():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.05,
                                  random_state=RANDOM_STATE, eval_metric="logloss"),
    }


def train_and_compare(x_train, x_test, y_train, y_test):
    """Train every requested classifier through the same preprocessing workflow."""
    models, results = {}, []
    for name, classifier in model_definitions().items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(x_train)),
            ("model", classifier),
        ])
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, prediction)
        models[name] = pipeline
        results.append({"Model": name, "Accuracy": accuracy})
        print(f"{name:<16} accuracy: {accuracy:.4f}")
    return pd.DataFrame(results).sort_values("Accuracy", ascending=False), models


def evaluate_and_save(comparison, models, x_test, y_test) -> None:
    best_name = comparison.iloc[0]["Model"]
    best_model = models[best_name]
    prediction = best_model.predict(x_test)
    print(f"\nBest model: {best_name}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, prediction))
    print("\nClassification report:\n", classification_report(y_test, prediction, zero_division=0))
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    with (OUTPUT_DIR / "best_flood_model.pkl").open("wb") as file:
        pickle.dump(best_model, file)
    print(f"Model and results saved to: {OUTPUT_DIR}")


def main(skip_plots: bool = False) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    section("Epic 2: Visualizing and Analysing the Data")
    data = read_dataset()
    descriptive_analysis(data)
    if not skip_plots:
        create_visualisations(data)

    section("Epic 3: Data Pre-processing")
    print("Missing values:\n", data.isna().sum().to_string())
    x_train, x_test, y_train, y_test = prepare_data(data)
    print(f"Training records: {len(x_train)} | Test records: {len(x_test)}")
    print("Missing values are imputed, numeric outliers are IQR-clipped, categorical values are encoded, and numeric values are scaled.")

    section("Epic 4: Model Building")
    comparison, models = train_and_compare(x_train, x_test, y_train, y_test)
    print("\nModel comparison:\n", comparison.to_string(index=False))
    evaluate_and_save(comparison, models, x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flood prediction models.")
    parser.add_argument("--no-plots", action="store_true", help="Do not generate PNG charts.")
    main(skip_plots=parser.parse_args().no_plots)
