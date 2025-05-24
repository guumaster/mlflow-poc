FROM python:3.12-slim AS builder

# Security-conscious organizations should package/review uv themselves.
COPY --from=ghcr.io/astral-sh/uv:python3.12-alpine /usr/local/bin/uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

WORKDIR /usr/src/app

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies using the venv's Python explicitly
RUN uv venv .venv \
    && uv sync --group dagster \
        --locked \
        --no-dev \
        --no-install-project


FROM python:3.12-slim AS dagster

# Install system dependencies
RUN apt-get update \
    && apt-get install -y docker.io \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/python/bin:$PATH"
ENV PATH="/usr/src/app/.venv/bin:$PATH"

WORKDIR /usr/src/app

COPY --from=builder /usr/src/app /usr/src/app

# Add to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/notebooks/extensions"

# Copy necessary directories
COPY src/dagster_job /usr/src/app/src/dagster_job
COPY src/notebooks /usr/src/app/notebooks
COPY dagster/dagster.yaml /usr/src/app/dagster.yaml

# Expose ports
EXPOSE 3000

CMD ["dagster", "dev", "-h", "0.0.0.0", "-p", "3000", "-m", "src.dagster_job.jobs"]
