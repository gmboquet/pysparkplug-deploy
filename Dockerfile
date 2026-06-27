# pysparkplug model server image (standalone pysparkplug-deploy package).
#
#   docker build -t <registry>/pysparkplug-deploy:latest .
#
# Installs pysp-learn (the core model library) + this serving package. The image bundles no model --
# the model is loaded at runtime from the registry volume (seed it with `pysp-seed`).

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY pysparkplug_deploy ./pysparkplug_deploy

# pysp-learn is not on PyPI yet; install it from git, then this package (which pulls fastapi/uvicorn).
# Once pysp-learn is published, drop the explicit git line -- the dependency resolves from PyPI.
RUN pip install --no-cache-dir "pysp-learn @ git+https://github.com/gmboquet/pysparkplug.git" \
    && pip install --no-cache-dir .

ENV PYSP_REGISTRY_ROOT=/models \
    PYSP_MODEL_NAME=model \
    PYSP_MODEL_ALIAS=production \
    PYSP_ACTIVITY_LOG=/dev/stdout

EXPOSE 8000

# Console script from pyproject [project.scripts]; runs uvicorn on pysparkplug_deploy.app:app.
CMD ["pysp-serve"]
