"""Configuration for the GCP Model."""
from typing import Optional

from steamship.invocable import Config


class PromptGenerationTrainablePluginConfig(Config):
    openai_api_key: str
    max_words: int
    temperature: float
    tag_kind: str


