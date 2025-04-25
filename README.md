# Iris Classifier Microservice

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Last Commit](https://img.shields.io/github/last-commit/javierhwulin/Iris-Classifier-Microservice)](https://github.com/javierhwulin/Iris-Classifier-Microservice/commits/main)
[![Open Issues](https://img.shields.io/github/issues/javierhwulin/Iris-Classifier-Microservice)](https://github.com/javierhwulin/Iris-Classifier-Microservice/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/javierhwulin/Iris-Classifier-Microservice)](https://github.com/javierhwulin/Iris-Classifier-Microservice/pulls)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-docker-orange)](#docker)

A production‑ready **FastAPI** microservice that exposes an ML model predicting *Iris* flower species (Setosa, Versicolor, Virginica). The service provides a REST API, automatic Swagger / OpenAPI docs, and a Docker image for easy deployment.

---

## About the Project

This microservice wraps a lightweight PyTorch MLP trained on the classic Iris dataset. It demonstrates modern backend practices—lazy model loading, health checks, typed request/response validation with Pydantic, CI via GitHub Actions, and multi‑stage Docker builds managed by the ultra‑fast **uv** package manager.

### Key Features

* **Predict Endpoint** – `POST /v1/predict` returns predicted class & confidence.
* **Health Endpoint** – `GET /v1/health` for readiness checks.
* **Automatic Docs** – Swagger UI (`/docs`) and Redoc (`/redoc`).
* **Model & Scaler Artifacts** – Stored under `models/` for reproducible inference.
* **Containerised** – Multi‑stage image (~1.06 GB) running as a non‑root user.
* **CI Pipeline** – uv tests + Docker smoke tests on every push.

---

## Tech Stack

| Layer            | Tech                                 |
|------------------|--------------------------------------|
| **Language**     | Python 3.12                          |
| **Web Framework**| FastAPI × Uvicorn                    |
| **ML**           | PyTorch 2 • scikit‑learn (scaler)     |
| **Package Mgr**  | **uv** (lockfile‑driven, blazing)    |
| **Runtime**      | Docker multi‑stage (slim)      |
| **CI/CD**        | GitHub Actions                       |

---

## Getting Started

### Prerequisites

* Python 3.12+
* Docker (24+)
* (Optional) `uv` installed globally: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Clone & Install

```bash
# 1 · Clone repo
git clone https://github.com/javierhwulin/Iris-Classifier-Microservice.git
cd Iris-Classifier-Microservice

# 2 · Create venv & install deps
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync     # installs from uv.lock
uv pip install -e .  # editable install for local imports

# 3 · Train model (optional – artefacts already committed)
python -m app.training.train
```

### Run Locally (dev)

```bash
uvicorn app.main:create_app --reload
open http://localhost:8000/docs
```

### Run Tests

```bash
pytest -q          # unit + API + accuracy tests
```

### Build & Run via Docker

```bash
# Build multi‑stage image
docker build -t iris-service:latest .

# Map host :8000 → container :80
docker run -d --name iris -p 8000:80 iris-service:latest

curl http://localhost:8000/health        # → {"status":"ok"}
```

---

## Usage Examples

### ➊ Prediction

```bash
curl -X POST http://localhost:8000/v1/predict \
     -H "Content-Type: application/json" \
     -d '{"sepal_length":5.1, "sepal_width":3.5, "petal_length":1.4, "petal_width":0.2}'
```
Response:
```json
{"class_name": "setosa", "confidence": 0.99}
```

### ➋ Health Check

```bash
curl http://localhost:8000/health   # {"status":"ok"}
```

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## Contact

Javier Hengda \| [javier.hwulin.devtech@gmail.com](mailto:javier.hwulin.devtech@gmail.com)

---

## Acknowledgements

* FastAPI · Pydantic · Uvicorn – for the blazing API stack.
* PyTorch & scikit‑learn – model & preprocessing.
* **uv** – dependency management at warp speed.
* The open‑source community for inspiration.

> *If you run into issues or have feature ideas, please [open an issue](https://github.com/your-org/iris-classifier/issues) or drop me a line.*

