"""Model wrappers. `Model` is a litellm-backed, native-tool-calling model
(models/litellm.py). Non-Claude focused (no Anthropic thinking/cache-control plumbing)."""

from micro_swe.models.litellm import Model, ModelConfig

__all__ = ["Model", "ModelConfig"]
