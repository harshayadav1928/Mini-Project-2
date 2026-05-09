# MLOpsFlow: From Data to Deployment

End-to-end MLOps project for automated customer churn prediction. It includes synthetic customer data generation, SQLite storage, preprocessing, model training, MLflow experiment tracking, FastAPI model serving, monitoring, an Airflow DAG, Docker Compose orchestration, CI checks, and a modern React user interface.

## What Is Included

- Customer churn dataset seeded in `data/sample_customers.csv`
- SQLite database generated at `data/mlopsflow.db`
- Scikit-learn training pipeline with optional XGBoost support
- MLflow experiment tracking and model artifact logging
- FastAPI REST API for predictions, health, metrics, and drift monitoring
- React + Vite frontend for easy churn prediction and dashboard viewing
- Airflow DAG for scheduled data generation, training, and monitoring
- Dockerfiles and `docker-compose.yml`
- Unit tests for preprocessing and API behavior

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_generation
python -m src.train
uvicorn api.main:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the frontend at `http://localhost:5173`.

## Docker Start

```bash
docker compose up --build
```

Services:

- Frontend: `http://localhost:5173`
- API: `http://localhost:8000`
- MLflow: `http://localhost:5000`

## Main API Endpoints

- `GET /health`
- `GET /metrics`
- `GET /customers/sample`
- `POST /predict`
- `POST /monitor/log`
- `GET /monitor/drift`

## Project Structure

```text
.
├── .github/workflows/ci.yml
├── airflow/dags/churn_pipeline.py
├── api/
├── data/
├── frontend/
├── models/
├── monitoring/
├── src/
├── tests/
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Typical Workflow

1. Generate or refresh customer data with `python -m src.data_generation`.
2. Train the model with `python -m src.train`.
3. Start the API with `uvicorn api.main:app --reload --port 8000`.
4. Start the frontend from `frontend/`.
5. Use the UI to submit customer details and view churn risk.

The project has no brand watermark and is ready to customize for reports, demonstrations, or portfolio submission.
