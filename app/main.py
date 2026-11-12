from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
INDEX_PATH = BASE_DIR / "web" / "index.html"

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    raise RuntimeError(
        "Model artifacts are missing. Run `python train.py` before starting the API."
    )

METADATA = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
MODEL = joblib.load(MODEL_PATH)
FEATURES = METADATA["features"]
EXAMPLE_INPUT = METADATA["example_input"]
CLASS_NAMES = METADATA["target"]["classes"]
MODEL_NAME = METADATA["model"]["name"]

TAGS_METADATA = [
    {
        "name": "System",
        "description": "Availability and service checks.",
    },
    {
        "name": "Metadata",
        "description": "Model information used to build the prediction form.",
    },
    {
        "name": "Prediction",
        "description": "Predict a student's academic outcome.",
    },
]

app = FastAPI(
    title="Student Success Prediction API",
    description=(
        "A REST API that predicts whether a student will **drop out**, remain "
        "**enrolled**, or **graduate**, using a scikit-learn pipeline trained on "
        "the UCI *Predict Students' Dropout and Academic Success* dataset (id 697).\n\n"
        "The API returns calibrated class probabilities for all three outcomes. "
        "The full preprocessing pipeline is bundled inside the model artifact, so "
        "requests only need the ten raw input features.\n\n"
        "**Try it:** use `GET /metadata` to inspect the input schema, then call "
        "`POST /predict` with the example body shown in the docs."
    ),
    version=METADATA["api"]["version"],
    openapi_tags=TAGS_METADATA,
    contact={"name": "DIT323 - Artificial Intelligence as a Service"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentFeatures(BaseModel):
    admission_grade: float = Field(
        ...,
        ge=0,
        le=200,
        description="Admission grade on the 0-200 scale.",
        examples=[140.0],
    )
    age_at_enrollment: int = Field(
        ...,
        ge=15,
        le=100,
        description="Student age at enrollment (years).",
        examples=[20],
    )
    gender: Literal[0, 1] = Field(
        ...,
        description="0 = Female, 1 = Male.",
        examples=[0],
    )
    tuition_fees_up_to_date: Literal[0, 1] = Field(
        ...,
        description="1 = tuition fees are up to date, 0 = not up to date.",
        examples=[1],
    )
    scholarship_holder: Literal[0, 1] = Field(
        ...,
        description="1 = student holds a scholarship, 0 = no scholarship.",
        examples=[0],
    )
    debtor: Literal[0, 1] = Field(
        ...,
        description="1 = student has outstanding debt, 0 = no debt.",
        examples=[0],
    )
    sem1_approved: int = Field(
        ...,
        ge=0,
        le=50,
        description="Curricular units approved in the 1st semester.",
        examples=[7],
    )
    sem1_grade: float = Field(
        ...,
        ge=0,
        le=20,
        description="Average grade in the 1st semester (0-20).",
        examples=[14.0],
    )
    sem2_approved: int = Field(
        ...,
        ge=0,
        le=50,
        description="Curricular units approved in the 2nd semester.",
        examples=[7],
    )
    sem2_grade: float = Field(
        ...,
        ge=0,
        le=20,
        description="Average grade in the 2nd semester (0-20).",
        examples=[14.0],
    )

    model_config = ConfigDict(
        json_schema_extra={"example": EXAMPLE_INPUT},
        extra="forbid",
    )


class PredictionResponse(BaseModel):
    predicted_label: str = Field(
        ...,
        description="Most likely academic outcome.",
        examples=["Graduate"],
    )
    predicted_class_index: int = Field(
        ...,
        description="Index of the predicted class in the model class list.",
        examples=[2],
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability assigned to the predicted class.",
        examples=[0.8102],
    )
    probabilities: dict[str, float] = Field(
        ...,
        description="Probability for every class.",
        examples=[{"Dropout": 0.0376, "Enrolled": 0.1521, "Graduate": 0.8102}],
    )
    model_name: str = Field(
        ...,
        description="Classifier used by the deployed pipeline.",
        examples=["Random Forest"],
    )


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Health check")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metadata", tags=["Metadata"], summary="Get model and input metadata")
def metadata() -> JSONResponse:
    return JSONResponse(content=METADATA)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict student outcome",
    description=(
        "Accepts the ten selected student features and returns the predicted "
        "class plus probabilities for Dropout, Enrolled, and Graduate."
    ),
)
def predict(payload: StudentFeatures) -> PredictionResponse:
    row = {
        feature["source_column"]: getattr(payload, feature["key"])
        for feature in FEATURES
    }
    frame = pd.DataFrame([row], columns=[feature["source_column"] for feature in FEATURES])
    probabilities = MODEL.predict_proba(frame)[0]
    classes = [str(label) for label in MODEL.classes_]
    best_index = int(probabilities.argmax())
    probability_map = {
        label: round(float(probability), 6)
        for label, probability in zip(classes, probabilities)
    }
    return PredictionResponse(
        predicted_label=classes[best_index],
        predicted_class_index=best_index,
        confidence=round(float(probabilities[best_index]), 6),
        probabilities=probability_map,
        model_name=MODEL_NAME,
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    if not INDEX_PATH.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": "web/index.html not found."},
        )
    return FileResponse(INDEX_PATH, media_type="text/html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
