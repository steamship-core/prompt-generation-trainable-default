"""
Provides an example of how a third-party model can be incorporated into Steamship as a Trainable Tagger.

In this example, we have three classes:

- MockClient       -- Simulates the API client for a service such as Google AutoML
- ThirdPartyModel  -- Demonstrates how to build a model that is simply wrapping usage of the MockClient
- ThirdPartyTrainableTaggerPlugin -- Plugin-wrapper around the ThirdPartyModel
"""
import logging
from typing import Type, Dict, Any

from pydantic import Field
from steamship import SteamshipError, TaskState, Steamship
from steamship.invocable import Config, InvocationContext
from steamship.invocable.invocable_response import InvocableResponse
from steamship.plugin.inputs.block_and_tag_plugin_input import \
    BlockAndTagPluginInput
from steamship.plugin.inputs.train_plugin_input import TrainPluginInput
from steamship.plugin.inputs.training_parameter_plugin_input import \
    TrainingParameterPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import \
    BlockAndTagPluginOutput
from steamship.plugin.outputs.train_plugin_output import TrainPluginOutput
from steamship.plugin.outputs.training_parameter_plugin_output import \
    TrainingParameterPluginOutput
from steamship.plugin.request import PluginRequest
from steamship.plugin.tagger import TrainableTagger
from steamship.plugin.trainable_model import TrainableModel

from model import OpenAIModel

logging.getLogger().setLevel(logging.INFO)


class PromptGenerationTrainablePlugin(TrainableTagger):
    """Plugin Wrapper for the `VertexAIModel`.

    This wrapper class translates the plugin lifecycle requests from Steamship into `VertexAIModel` object calls.
    """

    class PromptGenerationTrainablePluginConfig(Config):
        openai_api_key: str = Field("",
                                    description="An openAI API key to use. If left default, will use Steamship's API key.")
        max_words: int = Field(description="The maximum number of words to generate per request")
        temperature: float = Field(0.4,
                                   description="Controls randomness. Lower values produce higher likelihood / more predictable results; higher values produce more variety. Values between 0-1.")
        tag_kind: str = Field(
            description="The tag_kind to use for training examples. The block text referenced by the tag will be used as the prompt; the string_value of the tag value will be used as the completion.")
        num_completions: int = Field(1, description="The number of returned completion alternatives per request")

    config: PromptGenerationTrainablePluginConfig

    def __init__(
            self,
            client: Steamship = None,
            config: Dict[str, Any] = None,
            context: InvocationContext = None,
    ):
        super().__init__(client, config, context)

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return PromptGenerationTrainablePlugin.PromptGenerationTrainablePluginConfig

    def model_cls(self) -> Type[OpenAIModel]:
        return OpenAIModel

    def run_with_model(
        self, request: PluginRequest[BlockAndTagPluginInput], model: OpenAIModel
    ) -> InvocableResponse[BlockAndTagPluginOutput]:
        """Run the tag request with the given model file."""
        logging.debug(f"run_with_model {request} {model}")
        return model.run(request)

    def get_training_parameters(
        self, request: PluginRequest[TrainingParameterPluginInput]
    ) -> InvocableResponse[TrainingParameterPluginOutput]:
        """Return training parameters for this model."""
        # Since it's AutoML, I don't have to care about this!
        return InvocableResponse(json=TrainingParameterPluginOutput.from_input(request.data))

    def train(self, request: PluginRequest[TrainPluginInput], model: OpenAIModel) -> InvocableResponse[TrainPluginOutput]:
        """Instruct the model to begin training."""
        logging.info(f"Trainable prompt generation received train request {request}")
        model.training_task_id = request.status.task_id
        return model.train(request.data)

    def train_status(
            self, request: PluginRequest[TrainPluginInput], model: TrainableModel
    ) -> InvocableResponse[TrainPluginOutput]:
        """Fetch status of the training job."""
        logging.info(f"Trainable prompt generation received train_status request {request}")

        model.training_task_id = request.status.task_id
        train_plugin_output_response = model.train_status(request)

        if train_plugin_output_response.status.state == TaskState.succeeded:
            # Training has completed
            # Save the model with the `default` handle.
            archive_path_in_steamship = model.save_remote(
                client=self.client, plugin_instance_id=request.context.plugin_instance_id
            )

            # Set the model location on the plugin output.
            train_plugin_output_response.data["archive_path"] = archive_path_in_steamship

        return train_plugin_output_response

