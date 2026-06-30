# mixle model server image (standalone mixle-mlops package).
#
#   docker build -t <registry>/mixle-mlops:latest .
#
# Installs mixle (the core model library) + this serving package. The image bundles no model --
# the model is loaded at runtime from the registry volume (seed it with `mixle-seed`).

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY mixle_mlops ./mixle_mlops

# mixle is not on PyPI yet; install it from git, then this package (which pulls fastapi/uvicorn).
# Once mixle is published, drop the explicit git line -- the dependency resolves from PyPI.
RUN pip install --no-cache-dir "mixle @ git+https://github.com/gmboquet/mixle.git" \
    && pip install --no-cache-dir .

ENV MIXLE_REGISTRY_ROOT=/models \
    MIXLE_MODEL_NAME=model \
    MIXLE_MODEL_ALIAS=production \
    MIXLE_ACTIVITY_LOG=/dev/stdout

EXPOSE 8000

# Console script from pyproject [project.scripts]; runs uvicorn on mixle_mlops.app:app.
CMD ["mixle-serve"]
