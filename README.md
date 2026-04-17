## Setup

### 1. Clone repository

```bash
git clone https://github.com/team-5-fraud-dectection/MLOps_Fraud_Detection.git
cd MLOps_Fraud_Detection
git checkout ldtesting
````

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data (DVC)

### Configure DVC remote (DagsHub)

```bash
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/rizerize-1/DVC_Fraud_Detection.s3
```

PowerShell:

```powershell
$DAGSHUB_TOKEN="YOUR_TOKEN"
dvc remote modify --local origin access_key_id $DAGSHUB_TOKEN
dvc remote modify --local origin secret_access_key $DAGSHUB_TOKEN
```

---

### Pull data and artifacts

```bash
dvc pull
```

---

## Run full pipeline

```bash
dvc repro
```

This will:

* preprocess data
* generate features
* balance dataset
* train models
* save best model + metrics

---

## MLflow (Experiment Tracking)

Start MLflow server:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Open UI:

```
http://127.0.0.1:5000
```

---

## API (Live Inference)

Run API:

```bash
uvicorn src.api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

### Test prediction

Use `sample_request.json`:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data @sample_request.json
```

---

## Docker

### Build image

```bash
docker build -t ieee-fraud-api .
```

### Run container

```bash
docker run -p 8000:8000 ieee-fraud-api
```

---

## Docker Compose

```bash
docker compose up --build
```

---

## Model Information

* Best model: XGBoost (based on AUPRC)
* Threshold tuned for optimal F1 score
* Model stored at:

```
models/model.pkl
```
