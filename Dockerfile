FROM python:3.10-slim AS builder

# Security-conscious organizations should package/review uv themselves.
COPY --from=ghcr.io/astral-sh/uv:python3.10-alpine /usr/local/bin/uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

WORKDIR /usr/src/app

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies using the venv's Python explicitly
RUN uv venv .venv \
    && uv sync \
        --locked \
        --no-dev \
        --no-install-project



# ============ MLFlow Server ============
FROM python:3.10-slim  AS mlflow-server

WORKDIR /usr/src/app

COPY --from=builder /usr/src/app/.venv /usr/src/app/.venv

# Set environment variables
ENV BACKEND_URI=/path/to/backend
ENV ARTIFACT_ROOT=/path/to/artifact

# Activate the virtual environment
ENV PATH="/usr/src/app/.venv/bin:$PATH"

EXPOSE 5000

# Run mlflow server
CMD ["sh", "-c", "mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --host 0.0.0.0"]


# ============ Model Server ============
FROM builder AS model-server

WORKDIR /usr/src/app

ENV MODEL_URI=/path/to/your/model
ENV ARTIFACT_ROOT=/path/to/artifact

#COPY --from=builder /usr/src/app /usr/src/app

ENV PATH="/usr/src/app/.venv/bin:$PATH"

EXPOSE 7000

# Start the MLFlow model server
CMD [ "sh", "-c", "mlflow models serve --model-uri \"${MODEL_URI}\" --host 0.0.0.0  --port 7000 --workers 4  --no-conda --enable-mlserver" ]
