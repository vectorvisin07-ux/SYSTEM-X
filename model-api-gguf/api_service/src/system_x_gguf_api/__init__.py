"""System X GGUF API service with a private llama-server router adapter."""

__version__ = "0.8.0"

from .application import app, create_application

__all__ = ["__version__", "app", "create_application"]
