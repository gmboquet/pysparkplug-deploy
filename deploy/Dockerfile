# syntax=docker/dockerfile:1
#
# mixle-mlops platform gateway image (NOT the old single-model scoring server).
# Multi-stage, slim, non-root. Builds a wheel of the package, installs it plus the
# mixle core (from git, since the platform pins mixle>=0.2 which may not yet be on
# the index your build pulls from), and runs `mixle-serve`
# (uvicorn mixle_mlops.gateway.app:app on 0.0.0.0:8000).
#
# Build (from the repo root):
#   docker build -f deploy/Dockerfile -t mixle-mlops:latest .
#
# The mixle core source can be supplied three ways (see MIXLE_GIT below):
#   - default: pip install "mixle @ git+https://github.com/gmboquet/mixle@main"
#   - override the git ref:  --build-arg MIXLE_GIT="git+https://github.com/gmboquet/mixle@<sha>"
#   - or rely on the pinned "mixle>=0.2" from your configured index by passing
#     --build-arg MIXLE_GIT="mixle>=0.2"

############################
# Stage 1 — builder
############################
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# git is needed to pip-install mixle from a git ref.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Build a wheel of the platform package itself.
COPY pyproject.toml README.md ./
COPY mixle_mlops ./mixle_mlops
RUN python -m pip install --upgrade pip build \
    && python -m build --wheel --outdir /wheels .

# Install the platform wheel + mixle core into a self-contained venv we can copy out.
ARG MIXLE_GIT="git+https://github.com/gmboquet/mixle@main"
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --upgrade pip \
    && pip install "mixle @ ${MIXLE_GIT}" \
    && pip install /wheels/*.whl

############################
# Stage 2 — runtime
############################
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    MIXLE_DATA_DIR=/data \
    MIXLE_HOST=0.0.0.0 \
    MIXLE_PORT=8000

# Non-root user; /data is the writable volume for sqlite + the model registry.
RUN useradd --create-home --uid 10001 mixle \
    && mkdir -p /data \
    && chown -R mixle:mixle /data

COPY --from=builder /opt/venv /opt/venv

USER mixle
WORKDIR /home/mixle

EXPOSE 8000
VOLUME ["/data"]

# Container-level healthcheck hits the gateway's /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# `mixle-serve` -> uvicorn mixle_mlops.gateway.app:app on 0.0.0.0:8000.
CMD ["mixle-serve"]
