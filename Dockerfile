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

# (Optional) install any pre-downloaded wheels first. On flaky networks where pip cannot
# finish the ~190 MB torch download, drop a resumably-downloaded torch wheel into ./wheels/
# (see README "Ağ sorunluysa"). This layer is a no-op when ./wheels/ has no .whl files.
COPY wheels/ /wheels/
RUN --mount=type=cache,target=/root/.cache/pip \
    sh -c 'if ls /wheels/*.whl >/dev/null 2>&1; then pip install /wheels/*.whl; else echo "no prefetched wheels; using index"; fi'

# Install the large CPU PyTorch wheel (~190 MB) in its own layer with a pip cache mount.
# If torch was already installed from a prefetched wheel above, this is a no-op (no download).
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
