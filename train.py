from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATASET_PATH = DATA_DIR / "dataset.csv"
MODEL_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

DATASET_ID = 697
DATASET_NAME = "Predict Students' Dropout and Academic Success"
DATASET_URL = "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"
DATASET_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)
TARGET_COLUMN = "Target"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

FEATURES = [
    {
        "key": "admission_grade",
        "source": "Admission grade",
        "name": "Admission grade",
        "type": "number",
        "min": 95,
        "max": 200,
        "step": 0.1,
        "default": 126.1,
        "example": 140.0,
        "description": "Admission grade on the 0-200 scale.",
    },
    {
        "key": "age_at_enrollment",
        "source": "Age at enrollment",
        "name": "Age at enrollment",
        "type": "number",
        "min": 17,
        "max": 80,
        "step": 1,
        "default": 20,
        "example": 20,
        "description": "Student age (years) when enrolling.",
    },
    {
        "key": "gender",
        "source": "Gender",
        "name": "Gender",
        "type": "binary",
        "choices": [
            {"value": 0, "label": "Female"},
            {"value": 1, "label": "Male"},
        ],
        "default": 0,
        "example": 0,
        "description": "Student gender.",
    },
    {
        "key": "tuition_fees_up_to_date",
        "source": "Tuition fees up to date",
        "name": "Tuition fees up to date",
        "type": "binary",
        "choices": [
            {"value": 1, "label": "Yes"},
            {"value": 0, "label": "No"},
        ],
        "default": 1,
        "example": 1,
        "description": "Are tuition fees paid up to date?",
    },
    {
        "key": "scholarship_holder",
        "source": "Scholarship holder",
        "name": "Scholarship holder",
        "type": "binary",
        "choices": [
            {"value": 0, "label": "No"},
            {"value": 1, "label": "Yes"},
        ],
        "default": 0,
        "example": 0,
        "description": "Does the student hold a scholarship?",
    },
    {
        "key": "debtor",
        "source": "Debtor",
        "name": "Debtor",
        "type": "binary",
        "choices": [
            {"value": 0, "label": "No"},
            {"value": 1, "label": "Yes"},
        ],
        "default": 0,
        "example": 0,
        "description": "Does the student have outstanding debt?",
    },
    {
        "key": "sem1_approved",
        "source": "Curricular units 1st sem (approved)",
        "name": "1st semester units approved",
        "type": "number",
        "min": 0,
        "max": 30,
        "step": 1,
        "default": 5,
        "example": 7,
        "description": "Curricular units approved in the 1st semester.",
    },
    {
        "key": "sem1_grade",
        "source": "Curricular units 1st sem (grade)",
        "name": "1st semester average grade",
        "type": "number",
        "min": 0,
        "max": 20,
        "step": 0.1,
        "default": 12.3,
        "example": 14.0,
        "description": "Average grade in the 1st semester (0-20).",
    },
    {
        "key": "sem2_approved",
        "source": "Curricular units 2nd sem (approved)",
        "name": "2nd semester units approved",
        "type": "number",
        "min": 0,
        "max": 30,
        "step": 1,
        "default": 5,
        "example": 7,
        "description": "Curricular units approved in the 2nd semester.",
    },
    {
        "key": "sem2_grade",
        "source": "Curricular units 2nd sem (grade)",
        "name": "2nd semester average grade",
        "type": "number",
        "min": 0,
        "max": 20,
        "step": 0.1,
        "default": 12.2,
        "example": 14.0,
        "description": "Average grade in the 2nd semester (0-20).",
    },
]

SOURCE_COLUMNS = [feature["source"] for feature in FEATURES]
NUMERIC_COLUMNS = [
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]
BINARY_COLUMNS = [
    "Tuition fees up to date",
    "Scholarship holder",
    "Debtor",
    "Gender",
]
CLASS_DESCRIPTIONS = {
    "Dropout": "Student is likely to drop out.",
    "Enrolled": "Student is likely to remain enrolled.",
    "Graduate": "Student is likely to graduate.",
}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _download_from_zip() -> pd.DataFrame:
    with urllib.request.urlopen(DATASET_ZIP_URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_name = next(
            name for name in archive.namelist() if name.lower().endswith(".csv")
        )
        raw = archive.read(csv_name)
    frame = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    if TARGET_COLUMN not in frame.columns:
        frame = pd.read_csv(io.BytesIO(raw), sep=";")
    return frame


def load_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        print(f"Loading local dataset: {DATASET_PATH}")
        return pd.read_csv(DATASET_PATH)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading UCI dataset {DATASET_ID} via ucimlrepo ...")
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=DATASET_ID)
        features = dataset.data.features
        targets = dataset.data.targets
        frame = pd.concat([features, targets], axis=1)
    except Exception as primary_error:
        print(f"ucimlrepo failed: {primary_error}")
        print("Trying direct UCI archive download ...")
        try:
            frame = _download_from_zip()
        except Exception as fallback_error:
            raise RuntimeError(
                "Could not download the UCI dataset. Download it manually from "
                f"{DATASET_URL} and save the CSV as {DATASET_PATH}."
            ) from fallback_error

    frame.to_csv(DATASET_PATH, index=False)
    print(f"Saved dataset to {DATASET_PATH}")
    return frame


def print_eda(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 72)
    print(f"Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")
    print(f"Missing values (total): {int(df.isna().sum().sum())}")
    print("\nClass balance:")
    counts = df[TARGET_COLUMN].value_counts()
    for label, count in counts.items():
        print(f"  {label:<10} {count:>5} ({count / len(df):.1%})")
    print("\nSelected feature summary:")
    print(df[SOURCE_COLUMNS].describe().T.to_string(float_format=lambda x: f"{x:.2f}"))


def print_importance_screening(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("FEATURE IMPORTANCE SCREENING (Random Forest on all features)")
    print("=" * 72)
    X_all = df.drop(columns=[TARGET_COLUMN]).select_dtypes(include=[np.number])
    y_all = df[TARGET_COLUMN]
    screening_model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    )
    screening_model.fit(X_all, y_all)
    ranking = pd.Series(
        screening_model.feature_importances_, index=X_all.columns
    ).sort_values(ascending=False)
    for name, score in ranking.head(15).items():
        marker = " *" if name in SOURCE_COLUMNS else ""
        print(f"  {name:<52} {score:.4f}{marker}")
    print("  * = selected for the deployed model (domain-driven shortlist)")


def build_pipeline(model) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("binary", binary_transformer, BINARY_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[("preprocess", preprocessor), ("model", model)]
    )


def candidate_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": build_pipeline(
            LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
            )
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": build_pipeline(
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            )
        ),
    }


def compare_models(
    models: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[str, dict]:
    print("\n" + "=" * 72)
    print(f"MODEL COMPARISON - {CV_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 72)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }
    results = {}
    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1
        )
        results[name] = {
            "accuracy_mean": float(scores["test_accuracy"].mean()),
            "accuracy_std": float(scores["test_accuracy"].std()),
            "macro_f1_mean": float(scores["test_macro_f1"].mean()),
            "macro_f1_std": float(scores["test_macro_f1"].std()),
            "weighted_f1_mean": float(scores["test_weighted_f1"].mean()),
            "weighted_f1_std": float(scores["test_weighted_f1"].std()),
        }
        print(
            f"  {name:<20} accuracy={results[name]['accuracy_mean']:.4f} "
            f"+/- {results[name]['accuracy_std']:.4f}   "
            f"macro-F1={results[name]['macro_f1_mean']:.4f} "
            f"+/- {results[name]['macro_f1_std']:.4f}"
        )
    best_name = max(results, key=lambda key: results[key]["macro_f1_mean"])
    print(f"\nBest model by macro-F1: {best_name}")
    return best_name, results


def extract_importances(pipeline: Pipeline) -> dict[str, float]:
    estimator = pipeline.named_steps["model"]
    feature_names = list(
        pipeline.named_steps["preprocess"].get_feature_names_out()
    )
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        values = (
            np.abs(coefficients).mean(axis=0)
            if coefficients.ndim > 1
            else np.abs(coefficients)
        )
    else:
        return {}
    total = float(values.sum()) or 1.0
    ordered = sorted(
        zip(feature_names, values / total), key=lambda item: item[1], reverse=True
    )
    return {name: round(float(score), 4) for name, score in ordered}


def build_metadata(
    df: pd.DataFrame,
    best_name: str,
    cv_results: dict,
    metrics: dict,
    importances: dict,
) -> dict:
    features = []
    for feature in FEATURES:
        entry = {key: value for key, value in feature.items() if key != "source"}
        entry["source_column"] = feature["source"]
        features.append(entry)

    class_counts = {
        str(label): int(count)
        for label, count in df[TARGET_COLUMN].value_counts().items()
    }
    return {
        "api": {
            "title": "Student Success Prediction API",
            "version": "1.0.0",
        },
        "model": {
            "name": best_name,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "random_state": RANDOM_STATE,
            "framework": f"scikit-learn {__import__('sklearn').__version__}",
        },
        "dataset": {
            "id": DATASET_ID,
            "name": DATASET_NAME,
            "url": DATASET_URL,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target": TARGET_COLUMN,
            "class_counts": class_counts,
        },
        "target": {
            "name": TARGET_COLUMN,
            "classes": sorted(class_counts),
            "class_descriptions": CLASS_DESCRIPTIONS,
        },
        "features": features,
        "example_input": {feature["key"]: feature["example"] for feature in FEATURES},
        "metrics": metrics,
        "cross_validation": {
            "folds": CV_FOLDS,
            "scoring": "f1_macro",
            "results": cv_results,
        },
        "feature_importance": importances,
    }


def main() -> None:
    pd.set_option("display.width", 160)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    missing = [column for column in SOURCE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    print_eda(df)
    print_importance_screening(df)

    X = df[SOURCE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    models = candidate_models()
    best_name, cv_results = compare_models(models, X_train, y_train)

    best_pipeline = models[best_name]
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    labels = sorted(y.unique())

    print("\n" + "=" * 72)
    print(f"FINAL EVALUATION - {best_name} (held-out {TEST_SIZE:.0%} test set)")
    print("=" * 72)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(f"  labels: {labels}")
    print(pd.DataFrame(matrix, index=labels, columns=labels).to_string())

    best_pipeline.fit(X, y)
    importances = extract_importances(best_pipeline)
    print("\nFeature importance in final model:")
    for name, score in importances.items():
        print(f"  {name:<52} {score:.4f}")

    joblib.dump(best_pipeline, MODEL_PATH)

    metrics = {
        "test_accuracy": round(float(accuracy), 4),
        "test_macro_f1": round(float(macro_f1), 4),
        "test_weighted_f1": round(float(weighted_f1), 4),
        "test_size": TEST_SIZE,
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1_score": round(float(report[label]["f1-score"]), 4),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.tolist(),
        },
        "classification_report": report,
    }
    metadata = build_metadata(df, best_name, cv_results, metrics, importances)
    METADATA_PATH.write_text(
        json.dumps(_jsonable(metadata), indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("ARTIFACTS SAVED")
    print("=" * 72)
    print(f"Pipeline: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Dataset:  {DATASET_PATH}")


if __name__ == "__main__":
    main()
