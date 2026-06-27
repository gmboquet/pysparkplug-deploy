"""pysparkplug-deploy: a container/Kubernetes serving layer for pysparkplug models.

A thin FastAPI server over ``pysp.inference.ModelService`` plus seed/drift-retrain helpers and Kubernetes
manifests. Kept out of the core ``pysparkplug`` library on purpose -- the model primitives live there; the
deployment opinions (HTTP, Docker, k8s) live here.
"""

__version__ = "0.1.0"
