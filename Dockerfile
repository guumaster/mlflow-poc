FROM python:3.12-slim AS mlflow

# Install system dependencies (including build tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Set working directory
WORKDIR /usr/src/app

# Copy pyproject.toml
COPY pyproject.toml poetry.lock ./

# Install dependencies using Poetry
RUN poetry install --no-interaction --no-root


# ============ MLFlow Server ============
FROM mlflow AS  mlflow-server

# Set environment variables
ENV BACKEND_URI=/path/to/backend
ENV ARTIFACT_ROOT=/path/to/artifact

# Activate the virtual environment
ENV PATH="$PATH:/usr/src/app/.venv/bin"

COPY src src

EXPOSE 5000

# Run mlflow server
CMD ["sh", "-c", "poetry run mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --host 0.0.0.0"]

# ============ Model Server ============
FROM mlflow AS model-server

ENV MODEL_URI=/path/to/model
# Activate the virtual environment
ENV PATH="$PATH:/usr/src/app/.venv/bin"

EXPOSE 8000

# Start the MLFlow model server
CMD [ "sh", "-c", "poetry run mlflow models serve --model-uri \"${MODEL_URI}\" --host 0.0.0.0  --port 7000 --workers 4  --no-conda --enable-mlserver" ]



# ============ Model Server ============
FROM seldonio/mlserver:1.7.0-mlflow AS mlserver


RUN pip install boto3 mlflow==2.22.0 numpy==2.2.5 psutil==7.0.0 scipy==1.15.3
