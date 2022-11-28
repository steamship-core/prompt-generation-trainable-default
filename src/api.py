"""
Provides an example of how a third-party model can be incorporated into Steamship as a Trainable Tagger.

In this example, we have three classes:

- MockClient       -- Simulates the API client for a service such as Google AutoML
- ThirdPartyModel  -- Demonstrates how to build a model that is simply wrapping usage of the MockClient
- ThirdPartyTrainableTaggerPlugin -- Plugin-wrapper around the ThirdPartyModel
"""
import logging
from typing import Type

from steamship import SteamshipError, TaskState
from steamship.invocable import Config, create_handler
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

from config import PromptGenerationTrainablePluginConfig
from model import OpenAIModel

logging.getLogger().setLevel(logging.INFO)


class PromptGenerationTrainablePlugin(TrainableTagger):
    """Plugin Wrapper for the `VertexAIModel`.

    This wrapper class translates the plugin lifecycle requests from Steamship into `VertexAIModel` object calls.
    """

    config: PromptGenerationTrainablePluginConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_latebound_config()

    def _validate_latebound_config(self):
        """Validate that the late-bound config is present before using the plugin.

        This isn't strictly necessary, but can help with debugging since there are so many config fields that must
        be present.
        """
        latebound_variables = [
            "openai_api_key",
        ]
        for varname in latebound_variables:
            val = getattr(self.config, varname)
            if val is None or not val:
                raise SteamshipError(message=f"The {varname} parameter was not found but is required for operation.")


    def config_cls(self) -> Type[Config]:
        return PromptGenerationTrainablePluginConfig

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


