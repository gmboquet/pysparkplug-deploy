"""Train mixle models and fine-tune LLMs on rented GPUs (vast.ai)."""
from .jobspec import TrainingJob
from .launcher import launch, plan
from .vast import Offer, VastClient, VastError

__all__ = ["TrainingJob", "launch", "plan", "VastClient", "Offer", "VastError"]
