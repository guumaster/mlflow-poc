FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Set working directory
WORKDIR /usr/src/app
ENV PYTHONPATH=/usr/src/app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install all dependencies (including notebook dependencies)
RUN poetry install --no-interaction --no-root

# Add to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/notebooks/extensions"

# Copy necessary directories
COPY src/dagster_job /usr/src/app/src/dagster_job
COPY src/notebooks /usr/src/app/notebooks
COPY dagster/dagster.yaml /usr/src/app/dagster.yaml

# Expose ports
EXPOSE 3000

# Run Dagster using module syntax
CMD ["sh", "-c", "poetry run dagster dev -h 0.0.0.0 -p 3000 -m src.dagster_job.jobs"]
