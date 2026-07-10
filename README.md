# Energy Demand Forecasting — MLOps Pipeline with MLflow

An end-to-end deep learning pipeline that forecasts household energy demand using the [UCI Household Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption). The project covers the full lifecycle — data ingestion, validation, transformation, model training/evaluation with MLflow tracking, REST API + dashboard serving, containerization, and automated CI/CD deployment to AWS.

## Table of Contents

- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Model Serving](#model-serving)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running with Docker](#running-with-docker)
- [CI/CD](#cicd)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)

## Architecture

```
 Data Ingestion → Data Validation → Data Transformation → Model Training → Model Evaluation
        │                                                        │
        │                                                        ▼
        │                                                     MLflow
        ▼
   artifacts/                                              (best model selection)
        │
        ▼
 ┌─────────────────┐        ┌──────────────────────┐
 │  FastAPI (8000)  │◄──────►│  PredictionPipeline  │
 └─────────────────┘        └──────────────────────┘
        ▲
        │
 ┌─────────────────┐
 │ Streamlit (8501) │
 └─────────────────┘
```

Both the API and the dashboard are built from a single Docker image and run as separate `docker-compose` services.

## Pipeline Stages

The training pipeline (`main.py`) runs five config-driven stages in sequence:

| Stage | Description |
|---|---|
| **1. Data Ingestion** | Downloads/stages the raw dataset. |
| **2. Data Validation** | Validates incoming data against `schema.yaml`. |
| **3. Data Transformation** | Datetime indexing, imputation, resampling, scaling, and sliding-window construction (`WINDOW_SIZE`). |
| **4. Model Training** | Trains LSTM/GRU models (TensorFlow/Keras) using hyperparameters from `params.yaml`. |
| **5. Model Evaluation** | Computes evaluation metrics (R², etc.) and logs runs to MLflow. |

Run the full pipeline locally:

```bash
python main.py
```

## Model Serving

### FastAPI (`fastapi_app.py`) — port `8000`

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check; reports the currently loaded model. |
| `/model-info` | GET | Active model name and evaluation score. |
| `/predict` | POST | Predict from pre-scaled input values. |
| `/forecast` | POST | Autoregressive multi-step forecast (up to 168 hours). |
| `/predict-raw` | POST | Predict from raw, unscaled input rows (runs the full preprocessing pipeline). |
| `/train` | POST | Triggers a retraining run (`main.py`) as a background subprocess. |
| `/train/status` | GET | Whether training is running, and the last exit code. |
| `/train/logs` | GET | Tail of the training subprocess output. |

Interactive API docs are available at `http://localhost:8000/docs` once running.

### Streamlit Dashboard (`streamlit_app.py`) — port `8501`

A UI for exploring predictions and forecasts interactively over the same prediction pipeline.

## Project Structure

```
├── src/dlProject_energy_demand_forcasting/
│   ├── components/        # Data ingestion, validation, transformation, training, evaluation
│   ├── config/             # ConfigurationManager
│   ├── entity/              # Config dataclasses
│   ├── pipeline/            # Stage runners + prediction_pipeline.py
│   └── utils/                # Logger, exceptions, helpers
├── config/config.yaml       # File paths and stage configuration
├── params.yaml              # Model/training hyperparameters
├── schema.yaml               # Expected data schema
├── main.py                   # Runs the full training pipeline
├── fastapi_app.py             # REST API serving layer
├── streamlit_app.py            # Dashboard
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/cicd.yml
```

## Getting Started

```bash
# clone and install dependencies (uv-managed)
git clone https://github.com/MainakDanOpRes/dlops_energy_forecasting_with_MLflow.git
cd dlops_energy_forecasting_with_MLflow
uv sync

# run the training pipeline
uv run python main.py

# start the API
uv run uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload

# start the dashboard (separate terminal)
uv run streamlit run streamlit_app.py
```

## Running with Docker

```bash
export IMAGE_URI=<your-image>
docker compose up -d --build
```

- API → `http://localhost:8000`
- Dashboard → `http://localhost:8501`

## CI/CD

GitHub Actions (`.github/workflows/cicd.yml`) runs a three-stage, tag-gated pipeline:

1. **Continuous Integration** — runs on every push to `main`: `uv sync`, `ruff check`, `pytest`.
2. **Continuous Delivery** — runs only on version tags (`v*.*.*`): builds and pushes the Docker image to Amazon ECR.
3. **Continuous Deployment** — runs only on version tags: a self-hosted runner on the target EC2 instance pulls the released image and redeploys via `docker compose up -d --force-recreate`.

To release a new version:

```bash
git tag v1.0.1
git push origin v1.0.1
```

## Configuration

All pipeline behavior is externalized rather than hardcoded:

- `config/config.yaml` — file paths and artifact locations per stage.
- `params.yaml` — model architecture and training hyperparameters.
- `schema.yaml` — expected columns/dtypes for validation.

## Tech Stack

TensorFlow/Keras · MLflow · FastAPI · Streamlit · Docker · Docker Compose · GitHub Actions · AWS (EC2, ECR) · uv · ruff · pytest

## License

will be added later

