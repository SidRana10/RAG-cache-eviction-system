"""Wrapper for LLM calls"""
from abc import ABC, abstractmethod
from transformers import pipeline
from tenacity import retry, wait_random_exponential, stop_after_attempt

# custom modules
from ..utils.config import huggingface_config
from ..utils.helpers import timer


class ModelProvider(ABC):
    """Abstract base class for model providers."""

    @abstractmethod
    def load_model(self):
        """Load model to be used in pipeline (synchronous)."""
        raise NotImplementedError


class HUGGINGFACE(ModelProvider):
    """Hugging Face model provider (synchronous)."""

    _loaded = False

    def __init__(self):
        # huggingface_config should return (model_name, task, ...)
        cfg = huggingface_config()
        if not cfg or len(cfg) < 2:
            raise ValueError(
                "huggingface_config must return at least (model_name, task)")
        # assign explicitly
        self.model_name = cfg['model']
        self.task = cfg['task']
        self.pipeline = None

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _create_pipeline(self):
        """Internal helper that actually constructs the transformers pipeline."""
        return pipeline(self.task, model=self.model_name)

    @timer
    def load_model(self):
        """
        Public method to load the model into self.pipeline.
        """
        # Fast path: already loaded
        if HUGGINGFACE._loaded and self.pipeline is not None:
            return self.pipeline
        try:
            pipe = self._create_pipeline()
        except Exception as err:
            # raise a clear exception so callers can handle it
            raise RuntimeError(
                f"Could not load Hugging Face model \
                    '{self.model_name}': {err}"
                ) from err

        # store on instance and mark class as loaded
        self.pipeline = pipe
        HUGGINGFACE._loaded = True
        return self.pipeline
