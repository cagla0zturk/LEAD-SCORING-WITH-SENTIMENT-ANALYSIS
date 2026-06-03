# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# LightGBM needs libgomp at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=20 \
    PIP_PREFER_BINARY=1 \
    LOG_LEVEL=INFO

WORKDIR /app

# Install the large CPU PyTorch wheel (~190 MB) FIRST, in its own layer with a pip cache
# mount + many retries. On a flaky network this means a dropped download only ever re-pulls
# torch (everything else stays cached), and a simple `docker build` re-run resumes from here.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0"

# Then the rest of the dependencies (also cached across rebuilds).
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the project and install the package.
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY data/raw/Leads.csv ./data/raw/Leads.csv
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .

# Build data artifacts and train models at image-build time so the container is
# immediately serveable. (For large/real workloads you would instead mount a model
# registry volume; see README "Production'a girse".)
RUN python -m scripts.prepare_data && python -m scripts.train_all --no-plots

EXPOSE 8000

# Basic container healthcheck against the API.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "lead_priority.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
