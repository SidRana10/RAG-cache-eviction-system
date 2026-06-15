"""Module to define custom exceptions."""


class ModelLoadError(Exception):
    """Raised when errors occur during model loading."""


class EmbeddingError(Exception):
    """Raised when errors occur in the embedding method."""


class BatchingError(Exception):
    """Raised when errors occur during batching of data."""
