import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA = json.loads(
    (BASE_DIR / "artifacts" / "metadata.json").read_text(encoding="utf-8")
)
EXAMPLE = copy.deepcopy(METADATA["example_input"])
CLASSES = METADATA["target"]["classes"]

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"]["id"] == 697
    assert body["target"]["name"] == "Target"
    assert sorted(body["target"]["classes"]) == sorted(CLASSES)
    assert len(body["features"]) == 10
    assert set(EXAMPLE) == {feature["key"] for feature in body["features"]}
    assert body["metrics"]["test_accuracy"] > 0.6
    assert "confusion_matrix" in body["metrics"]


def test_predict_valid():
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] in CLASSES
    assert set(body["probabilities"]) == set(CLASSES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    max_label = max(body["probabilities"], key=body["probabilities"].get)
    assert body["predicted_label"] == max_label
    assert body["confidence"] == pytest.approx(
        body["probabilities"][max_label], abs=1e-5
    )
    assert 0 <= body["confidence"] <= 1


def test_predict_accepts_full_range_values():
    payload = copy.deepcopy(EXAMPLE)
    payload.update(
        {
            "admission_grade": 200,
            "age_at_enrollment": 80,
            "sem1_approved": 30,
            "sem1_grade": 20,
            "sem2_approved": 30,
            "sem2_grade": 20,
        }
    )
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_label"] in CLASSES


@pytest.mark.parametrize(
    "field,value",
    [
        ("age_at_enrollment", 5),
        ("age_at_enrollment", 120),
        ("admission_grade", -1),
        ("admission_grade", 250),
        ("sem1_grade", 25),
        ("sem2_grade", -0.5),
        ("sem1_approved", -1),
        ("gender", 3),
        ("tuition_fees_up_to_date", 2),
    ],
)
def test_predict_invalid_values(field, value):
    payload = copy.deepcopy(EXAMPLE)
    payload[field] = value
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_field():
    payload = copy.deepcopy(EXAMPLE)
    payload.pop("admission_grade")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_unknown_field():
    payload = copy.deepcopy(EXAMPLE)
    payload["favourite_colour"] = "green"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Student Success Predictor" in response.text
