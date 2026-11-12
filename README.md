# Student Success Prediction API

DIT323 - Artificial Intelligence as a Service

A complete end-to-end machine learning service that predicts a university
student's academic outcome: **Dropout**, **Enrolled**, or **Graduate**. The
project trains a scikit-learn pipeline, wraps it in a FastAPI REST API, and ships
a single-page web prototype that builds its form from the API metadata.

## Dataset

- **Name:** Predict Students' Dropout and Academic Success
- **Source:** UCI Machine Learning Repository, dataset id **697**
- **URL:** <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>
- **Rows:** 4,424 students
- **Columns:** 36 features + 1 target
- **Target:** `Target` with three classes: `Dropout`, `Enrolled`, `Graduate`
- **Class balance:** Graduate 2,209 (49.9%), Dropout 1,421 (32.1%), Enrolled 794 (17.9%)
- **Missing values:** none

The script downloads the dataset with `ucimlrepo` and stores a local copy at
`data/dataset.csv`. If `ucimlrepo` is unavailable it falls back to the direct UCI
archive ZIP; if both fail it prints instructions for placing the CSV manually.

## Problem

Given a compact student profile, estimate the probability of each academic
outcome at the end of the normal course duration. This is a **multiclass
classification** problem. A service like this can help academic advisers identify
at-risk students early, while keeping a human in the loop.

## Project structure

```text
.
├── data/dataset.csv
├── train.py
├── artifacts/
│   ├── model_pipeline.joblib
│   └── metadata.json
├── app/main.py
├── web/index.html
├── tests/test_api.py
├── requirements.txt
└── README.md
```

## Selected features

The deployed model uses ten features chosen for predictive value and a usable web
form. A Random Forest screening over all 36 features is printed by `train.py`; the
final shortlist follows the strongest signals plus sensible domain choices.

| Key | Source column | Type |
| --- | --- | --- |
| `admission_grade` | Admission grade | number, 0-200 |
| `age_at_enrollment` | Age at enrollment | number, 17-80 |
| `gender` | Gender | binary, 0 Female / 1 Male |
| `tuition_fees_up_to_date` | Tuition fees up to date | binary |
| `scholarship_holder` | Scholarship holder | binary |
| `debtor` | Debtor | binary |
| `sem1_approved` | Curricular units 1st sem (approved) | number, 0-30 |
| `sem1_grade` | Curricular units 1st sem (grade) | number, 0-20 |
| `sem2_approved` | Curricular units 2nd sem (approved) | number, 0-30 |
| `sem2_grade` | Curricular units 2nd sem (grade) | number, 0-20 |

All preprocessing (median/most-frequent imputation and standard scaling for
numeric features) lives inside the saved `Pipeline`. The API accepts raw values.

## Model selection

Three candidate pipelines were compared with 5-fold stratified cross-validation,
selecting by mean macro-F1:

| Model | CV accuracy | CV macro-F1 |
| --- | --- | --- |
| Logistic Regression | 0.6889 | 0.6543 |
| **Random Forest** | **0.7400** | **0.6682** |
| Gradient Boosting | 0.7451 | 0.6588 |

**Selected model:** Random Forest (400 trees, `min_samples_leaf=2`,
`class_weight="balanced_subsample"`).

### Held-out test results (20%, stratified)

- **Accuracy:** 0.7356
- **Macro-F1:** 0.6586
- **Weighted-F1:** 0.7281

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Dropout | 0.77 | 0.73 | 0.75 | 284 |
| Enrolled | 0.43 | 0.36 | 0.39 | 159 |
| Graduate | 0.80 | 0.88 | 0.84 | 442 |

Confusion matrix (rows = actual, columns = predicted):

```text
          Dropout  Enrolled  Graduate
Dropout       207        38        39
Enrolled       45        57        57
Graduate       18        37       387
```

`Enrolled` is the hardest class because it is the smallest and overlaps with the
other two outcomes. Metrics are regenerated every time `train.py` runs and are
written to `artifacts/metadata.json`.

## Installation

Python 3.10+ is recommended. From the project root:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If script execution is blocked, run the interpreter directly instead:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

This prints the EDA summary, class balance, feature-importance screening,
cross-validation comparison, held-out classification report, and confusion
matrix. It writes:

- `artifacts/model_pipeline.joblib` - fitted preprocessing + classifier pipeline
- `artifacts/metadata.json` - features, classes, ranges, choices, metrics,
  feature importance, and a ready-to-use example input

Training is deterministic (`random_state=42`).

## Run the API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or during development:

```bash
uvicorn app.main:app --reload
```

| URL | Purpose |
| --- | --- |
| <http://127.0.0.1:8000/> | Web prototype |
| <http://127.0.0.1:8000/docs> | Interactive Swagger UI |
| <http://127.0.0.1:8000/redoc> | ReDoc API reference |
| <http://127.0.0.1:8000/openapi.json> | OpenAPI schema |
| <http://127.0.0.1:8000/health> | Health check |
| <http://127.0.0.1:8000/metadata> | Model + form metadata |

CORS is enabled for all origins so the prototype can also be served separately
during development. The FastAPI app serves `web/index.html` at `/`.

## API reference

### `GET /health`

```json
{"status": "ok"}
```

### `GET /metadata`

Returns the contents of `artifacts/metadata.json`, including the input schema,
classes, model name, metrics, and example input.

### `POST /predict`

Request body (all fields required, unknown fields rejected):

```json
{
  "admission_grade": 140.0,
  "age_at_enrollment": 20,
  "gender": 0,
  "tuition_fees_up_to_date": 1,
  "scholarship_holder": 0,
  "debtor": 0,
  "sem1_approved": 7,
  "sem1_grade": 14.0,
  "sem2_approved": 7,
  "sem2_grade": 14.0
}
```

Response:

```json
{
  "predicted_label": "Graduate",
  "predicted_class_index": 2,
  "confidence": 0.810222,
  "probabilities": {
    "Dropout": 0.037633,
    "Enrolled": 0.152145,
    "Graduate": 0.810222
  },
  "model_name": "Random Forest"
}
```

Invalid values (for example an age of 5 or a grade above 20) return HTTP `422`
with a Pydantic validation error.

### curl examples

Windows PowerShell (use a file to avoid native argument quoting issues):

```powershell
'{"admission_grade":140.0,"age_at_enrollment":20,"gender":0,"tuition_fees_up_to_date":1,"scholarship_holder":0,"debtor":0,"sem1_approved":7,"sem1_grade":14.0,"sem2_approved":7,"sem2_grade":14.0}' | Set-Content -NoNewline body.json
curl.exe -s http://127.0.0.1:8000/health
curl.exe -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@body.json"
```

macOS / Linux / Git Bash:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"admission_grade":140.0,"age_at_enrollment":20,"gender":0,"tuition_fees_up_to_date":1,"scholarship_holder":0,"debtor":0,"sem1_approved":7,"sem1_grade":14.0,"sem2_approved":7,"sem2_grade":14.0}'
```

## Tests

```bash
python -m pytest -q
```

The suite covers:

- `GET /health`
- `GET /metadata` schema and values
- valid prediction, probability sum, predicted class consistency
- boundary values at the maximum end of the ranges
- invalid ranges and invalid binary values returning `422`
- missing and unknown fields returning `422`
- the served HTML prototype

Current result: **16 passed**.

## Web prototype

`web/index.html` is a single self-contained file (vanilla HTML/CSS/JS, no build
step). It:

- fetches `/metadata` and generates number inputs and binary segmented controls
- validates values against the metadata ranges before calling the API
- shows a loading state, API status indicator, and friendly error messages
- displays the predicted class and animated probability bars for all classes
- includes **Fill example**, **Reset**, and a model/dataset description
- is responsive down to mobile widths

Use **Fill example** and then **Predict outcome** for a ready-made demonstration
(example profile predicts **Graduate** with about 81% confidence).

## Suggested screenshots

1. **Prototype UI before prediction:** <http://127.0.0.1:8000/> - full page with
   the form, model metrics chips, and empty result panel.
2. **Prediction result:** click **Fill example**, then **Predict outcome**. The
   result panel shows `Graduate`, confidence `81.0%`, and three probability bars.
3. **Swagger UI:** <http://127.0.0.1:8000/docs> - expand `POST /predict`, show the
   generated example body, and execute it.
4. **Health check:** <http://127.0.0.1:8000/health> in a browser or terminal.
5. **Metadata:** <http://127.0.0.1:8000/metadata> showing the feature schema and
   metrics.

## Notes and limitations

- The dataset describes one Portuguese institution; results may not generalise.
- `Enrolled` has the weakest recall because of class imbalance and label overlap.
- The model is for coursework and decision support, not a guarantee of student
  outcomes.
- Artifacts must exist before the API starts. Run `python train.py` first.
