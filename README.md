# MLFlow Proof of Concept (POC)

A local implementation of MLFlow with PostgreSQL backend and MinIO (S3-compatible) artifact storage, demonstrating model tracking, registry, and deployment workflows.

## Features

- 🏗️ Local MLFlow tracking server with PostgreSQL backend
- 📦 MinIO (S3-compatible) storage for model artifacts
- 📊 Example workflows for model training, tracking, and registry
- 🧪 Two demonstration notebooks:
  - Basic tracking quickstart (Iris dataset)
  - Advanced workflow with hyperparameter search (Diabetes dataset)
- 🐳 Docker-compose setup for easy deployment

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Poetry (for Python dependency management)

## Quick Start

1. **Start the services**:
```bash
  docker-compose up -d
```

2. **Access to  services**:

MLFlow UI: http://localhost:5000
MinIO Console: http://localhost:9001 (Login with admin/admin123)


```bash
# Start Jupyter Lab with Poetry (load only ipynb files)
 poetry run jupyter-lab --notebook-dir=./src/notebooks 
```


## Sample Notebooks

1. [Basic Tracking (Iris Dataset)](#basic-tracking-iris-dataset)
2. [Advanced Workflow (Diabetes Dataset)](#advanced-workflow-diabetes-dataset)


### Basic Tracking (Iris Dataset)

Simple logistic regression model that demonstrates:

* Experiment creation
* Parameter and metric logging
* Model registration


### Advanced Workflow (Diabetes Dataset)

Random forest regressor with hyperparameter search that features:

* Nested runs for hyperparameter optimization
* Automated model comparison and registration
* Model versioning with aliases


## Resources

Initial docker compose comes from here: https://github.com/bubulmet/mlflow-postgres-minio
