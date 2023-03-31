"""Configuration for the GCP Model."""
from typing import Optional

from pydantic import Field
from steamship.invocable import Config


class PromptGenerationTrainablePluginConfig(Config):
    openai_api_key: str = Field("", description="An openAI API key to use. If left default, will use Steamship's API key.")
    max_words: int = Field(description="The maximum number of words to generate per request")
    temperature: float = Field(0.4, description="Controls randomness. Lower values produce higher likelihood / more predictable results; higher values produce more variety. Values between 0-1.")
    tag_kind: str = Field(description="The tag_kind to use for training examples. The block text referenced by the tag will be used as the prompt; the string_value of the tag value will be used as the completion.")
    num_completions: int = Field(1, description="The number of returned completion alternatives per request")



