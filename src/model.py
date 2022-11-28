"""Model class that wraps OpenAI's fine-tunable generation"""
import io
import json
import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional
import openai

import jsonlines
import requests
from openai import FineTune
from requests_file import FileAdapter
from steamship import Block, File, SteamshipError, Tag, Task, TaskState
from steamship.data import TagValue, TagKind, GenerationTag
from steamship.invocable.invocable_response import InvocableResponse
from steamship.plugin.inputs.block_and_tag_plugin_input import \
    BlockAndTagPluginInput
from steamship.plugin.inputs.train_plugin_input import TrainPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import \
    BlockAndTagPluginOutput
from steamship.plugin.outputs.train_plugin_output import TrainPluginOutput
from steamship.plugin.request import PluginRequest
from steamship.plugin.trainable_model import TrainableModel


class ThirdPartyTrainingStatus(str, Enum):
    TRAINING = "training"
    TRAINED = "trained"


class OpenAIModel(TrainableModel):
    """Example of a trainable model that wraps a third-party training/inference process.

    We separate this class from the `MockClient` above because, in a real setting, the `MockClient` would be a
    pip-installable package (e.g. `pip install google-automl`), where as this class is the Steamship-created
    wrapper of that pacakge.

    From this *Steamship* perspective, the goal of this `ThirdPartyModel` wrapper class is to load and store
    all the necessary state for training and communicating with the remote model. This might be as simple as
    a single `MODEL_ID` parameter returned by the remote service, used for inquiring about training status
    and/or invoking the trained model.
    """

    # We save the remote model ID in this file.
    PARAM_FILE = "model_information.txt"
    fine_tuned_model_name : Optional[str] = None
    training_task_id: Optional[str] = None

    def _prepare_credentials(self):
        openai.api_key = self.config.openai_api_key

    def load_from_folder(self, checkpoint_path: Path):
        """[Required by TrainableModel] Load state from the provided path."""
        with open(checkpoint_path / self.PARAM_FILE, "r") as f:
            self.fine_tuned_model_name = f.read()

    def save_to_folder(self, checkpoint_path: Path):
        """[Required by TrainableModel] Save state to the provided path."""
        with open(checkpoint_path / self.PARAM_FILE, "w") as f:
            f.write(self.fine_tuned_model_name)


    # Which generations for training were on this block.
    # Restricted to tag_kind that was provided in config.
    def _find_training_tags(self, block : Block) -> [Tag]:
        return [tag for tag in block.tags if tag.kind == self.config.tag_kind]

    # Format examples into the json structure expected by google
    def _format_examples(self, tags : Optional[List[Tag]], block : Block) -> Optional[List[dict]]:
        if tags is None or len(tags) == 0:
            logging.warning(f"Warning: Skipping block with no tags. id={block.id} text={block.text}")
            return None
        return [dict(
            completion=tag.value.get(TagValue.STRING_VALUE, None),
            prompt=block.text[tag.start_idx:tag.end_idx]
            ) for tag in tags]


    def _training_job_complete(self, job: FineTune) -> bool:
        status = job.get("status")
        if status in ['pending', 'running']:
            return False
        elif status == "succeeded":
            return True
        else:
            raise SteamshipError(message=f'OpenAI fine tune job returned unknown status {status}')


    def train(self, input: TrainPluginInput) -> InvocableResponse[TrainPluginOutput]:
        self._prepare_credentials()
        #Steps:
        #download from presigned url
        # So that requests can fetch the test training file from a file:// url
        requests_session = requests.session()
        requests_session.mount('file://', FileAdapter())
        response = requests_session.get(input.training_data_url)
        training_files = [File.parse_obj(json.loads(line)) for line in response.content.splitlines()]

        logging.info("Converting training set to appropriate format")
        # Translate a file to correct format
        num_examples = 0
        data_to_upload = io.StringIO()
        with jsonlines.Writer(data_to_upload) as writer:
            for file in training_files:
                for block in file.blocks:
                    #logging.info(f"Processing block: {block}")
                    tags = self._find_training_tags(block)
                    examples = self._format_examples(tags, block)
                    if examples is not None:
                        for example in examples:
                            writer.write(example)
                            num_examples += 1
        logging.info(f"Found {num_examples} training examples")
        training_data_content = data_to_upload.getvalue()

        # Upload it to GCP
        logging.info("Uploading training file to OpenAI")
        file_upload_response = openai.File.create(file=io.StringIO(training_data_content), purpose="fine-tune", user_provided_filename=self.training_task_id)
        logging.info(f"Uploaded training file successfully. ID: {file_upload_response.openai_id}")

        logging.info("Beginning fine-tuning job")
        fine_tune_job = openai.FineTune.create(training_file=file_upload_response.openai_id, suffix=self.training_task_id)
        training_job_id = fine_tune_job.openai_id
        logging.info(f"Fine tuning job started. ID: {training_job_id}")

        training_results = dict(state=ThirdPartyTrainingStatus.TRAINING.value)
        reference_data = dict(training_job_id=training_job_id)

        # The training will take a while, so instead of returning a TrainPluginOutput we return a Task
        # this will signal to the Engine that it must call us back.
        return self._still_training_response(training_results, reference_data)


    def _still_training_response(self, training_results, remote_status_input):
        return InvocableResponse(
            status=Task(
                task_id=self.training_task_id,
                state=TaskState.running,
                remote_status_output=training_results,
                remote_status_input=remote_status_input
            )
        )

    def _log_events(self, fine_tune_job):
        training_events = fine_tune_job.get("events")
        if training_events is not None:
            messages = []
            for event in sorted(training_events, key=lambda x: x.get("created_at")):
                created_at = event.get("created_at")
                message = event.get("message")
                messages.append(f"created_at: {created_at} message: {message}")
            combined_messages = '\n'.join(messages)
            logging.info(f"Training events: {combined_messages}")


    def train_status(self, input: PluginRequest[TrainPluginInput]) -> InvocableResponse[TrainPluginOutput]:
        self._prepare_credentials()
        # When returning to this method, we're waiting for the fine-tuning job to complete.
        if input.status.remote_status_input is None:
            raise SteamshipError(message="`training_reference_data` field of input was empty. Unable to check training status.")

        training_job_id = input.status.remote_status_input.get('training_job_id')

        training_results = dict(
            training_job_id=training_job_id,
        )

        if training_job_id is None:
            raise SteamshipError(message="`training_job_id` field of input.training_reference_data was empty. This indicates an error. Perhaps training hasn't started yet?")

        logging.info(
            f'Checking status of on training job {training_job_id}.')
        fine_tune_job = openai.FineTune.retrieve(training_job_id)

        self._log_events(fine_tune_job)

        if not self._training_job_complete(fine_tune_job):
            # Waiting on training job, not complete.  Just return current status.
            message = 'Job not complete - continuing to wait.'
            logging.info(message)
            training_results["message"] = message
            training_results["state"] = ThirdPartyTrainingStatus.TRAINING.value
            return self._still_training_response(training_results, input.status.remote_status_input)
        else:
            # Training job complete!
            logging.info('Done waiting on training job.')

            self.fine_tuned_model_name = fine_tune_job.get("fine_tuned_model")
            training_results["fine_tuned_model_name"] = self.fine_tuned_model_name
            training_results["state"] = ThirdPartyTrainingStatus.TRAINED.value
            training_results["message"] = f"Successfully trained fine tuned model: {self.fine_tuned_model_name}"

            return InvocableResponse(
                status=Task(
                    task_id=self.training_task_id,
                    state=TaskState.succeeded,
                    output=training_results['message'],
                    remote_status_output=training_results,
                    remote_status_input=input.status.remote_status_input
                ),
                json=TrainPluginOutput(
                    training_complete=True,
                    training_results=training_results,
                    training_reference_data=input.status.remote_status_input
                )
            )

    def _generate_text_for(self, text_prompt: str) -> str:
        """Call the API to generate the next section of text."""
        completion = openai.Completion.create(model=self.fine_tuned_model_name, prompt=text_prompt,
                                              temperature=self.config.temperature, max_tokens=self.config.max_words)
        return completion.choices[0].text

    def run(self, request: PluginRequest[BlockAndTagPluginInput]) -> InvocableResponse[BlockAndTagPluginOutput]:
        """Run the text generator against all Blocks of text.
        """
        self._prepare_credentials()
        if self.fine_tuned_model_name is None:
            raise SteamshipError(
                message="No fine_tuned_model_name was found in model parameter file. Has the model been trained?"
            )
        request_file = request.data.file
        output = BlockAndTagPluginOutput(file=File.CreateRequest(id=request_file.id), tags=[])
        for block in request.data.file.blocks:
            text = block.text
            generated_text = self._generate_text_for(text)
            tags = [Tag.CreateRequest(kind=TagKind.GENERATION, name=GenerationTag.PROMPT_COMPLETION,
                                      value={TagValue.STRING_VALUE: generated_text})]
            output_block = Block.CreateRequest(id=block.id, tags=tags)
            output.file.blocks.append(output_block)

        return InvocableResponse(data=output)
