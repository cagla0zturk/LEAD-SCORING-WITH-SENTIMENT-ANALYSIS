# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# LightGBM needs libgomp at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LOG_LEVEL=INFO

WORKDIR /app

# Install dependencies first to maximise layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project and install the package.
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY data/raw/Leads.csv ./data/raw/Leads.csv
RUN pip install --no-cache-dir -e .

# Build data artifacts and train models at image-build time so the container is
# immediately serveable. (For large/real workloads you would instead mount a model
# registry volume; see README "Production'a girse".)
RUN python -m scripts.prepare_data && python -m scripts.train_all --no-plots

EXPOSE 8000

# Basic container healthcheck against the API.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "lead_priority.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
