"""Model class that wraps Google's Vertex AI AutoML"""
import io
import json
import logging
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
import requests
from google.cloud import aiplatform, storage
from google.oauth2 import service_account
from google.protobuf import json_format
from requests_file import FileAdapter
from steamship import Block, File, SteamshipError, Tag, Task, TaskState
from steamship.invocable.invocable_response import InvocableResponse
from steamship.plugin.inputs.block_and_tag_plugin_input import \
    BlockAndTagPluginInput
from steamship.plugin.inputs.train_plugin_input import TrainPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import \
    BlockAndTagPluginOutput
from steamship.plugin.outputs.train_plugin_output import TrainPluginOutput
from steamship.plugin.request import PluginRequest
from steamship.plugin.trainable_model import TrainableModel

# If this isn't present, Localstack won't show logs
logging.getLogger().setLevel(logging.INFO)



class ThirdPartyTrainingStatus(str, Enum):
    PREPARING_DATASET = "preparing_dataset"
    TRAINING = "training"
    DEPLOYING_ENDPOINT = "deploying_endpoint"
    AWAITING_ENDPOINT = "awaiting_endpoint"
    TRAINED = "trained"


class VertexAIModel(TrainableModel):
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
    endpoint_path : Optional[str] = None
    bucket_name: Optional[str] = None
    training_task_id: Optional[str] = None
    scoped_credentials = None

    def _prepare_credentials(self):
        json_acct_info = self.config.dict()
        credentials = service_account.Credentials.from_service_account_info(
            json_acct_info)

        self.scoped_credentials = credentials.with_scopes(
            ['https://www.googleapis.com/auth/cloud-platform'])
        aiplatform.init(project=self.config.project_id, location=self.config.region, credentials=self.scoped_credentials)
        self.bucket_name = self.config.training_data_bucket

    def load_from_folder(self, checkpoint_path: Path):
        """[Required by TrainableModel] Load state from the provided path."""
        with open(checkpoint_path / self.PARAM_FILE, "r") as f:
            self.endpoint_path = f.read()

    def save_to_folder(self, checkpoint_path: Path):
        """[Required by TrainableModel] Save state to the provided path."""
        with open(checkpoint_path / self.PARAM_FILE, "w") as f:
            f.write(self.endpoint_path)


    # Which labels for classification are on this block.
    # Restricted to tag_kind that was provided in config.
    # If multi_class false, only returns first matching tag.
    # If include_tag_names provided, filter returned tag names.
    def _find_training_tags(self, block : Block, multi_class: bool, include_tag_names: [str]) -> [str]:
        result = []
        for tag in block.tags:
            if tag.kind == self.config.tag_kind:
                if include_tag_names is None or tag.name in include_tag_names:
                    result.append(tag.name)
                    if not multi_class:
                        return result
        return list(set(result))

    # Format examples into the json structure expected by google
    def _format_examples(self, tag_names : Optional[List[str]], block : Block, multi_class: bool) -> Optional[dict]:
        if tag_names is None or len(tag_names) == 0:
            logging.warning(f"Warning: Skipping block with no tags. id={block.id} text={block.text}")
            return None
        if not multi_class:
            return dict(
                classificationAnnotation=dict(displayName=tag_names[0]),
                textContent=block.text
            )
        else:
            return dict(
                classificationAnnotations=[dict(displayName=tag_name) for tag_name in tag_names],
                textContent=block.text
            )

    def _get_dataset_handle(self, dataset_display_name: str):
        dataset_list = aiplatform.TextDataset.list(filter=f'display_name={dataset_display_name}')
        if len(dataset_list) > 0:
            logging.info(f"Found dataset for: {dataset_display_name}")
            return dataset_list[0]
        logging.info(f"No dataset found for: {dataset_display_name}")
        return None

    def _get_endpoint_handle(self, endpoint_display_name: str) -> Optional[aiplatform.Endpoint]:
        endpoint_list = aiplatform.Endpoint.list(filter=f'display_name={endpoint_display_name}')
        if len(endpoint_list) > 0:
            logging.info(f"Found endpoint for: {endpoint_display_name}")
            return endpoint_list[0]
        logging.info(f"No endpoint found for: {endpoint_display_name}")
        return None

    def _deploy_endpoint(self, training_job_name: str, model: aiplatform.Model):
        logging.info(f"Deploying endpoint for: {training_job_name}")
        deployed_model_display_name = f"endpoint-{training_job_name}"
        model.deploy(
            deployed_model_display_name=deployed_model_display_name,
            sync=False
        )
        # Because the deployment job is starting on a background async process, we need to wait here
        # and give it time to actually get the job started in the cloud. Otherwise this lambda will
        # terminate upon return, and the dataset conversion job will not have started.
        time.sleep(30)

    def _is_endpoint_deploy_complete(self, endpoint: aiplatform.Endpoint) -> bool:
        """Until the deployment is complete, the endpoint will report an empty attached model list."""
        attached_models = endpoint.list_models()
        return len(attached_models) > 0

    def _training_job_complete(self, job_display_name: str) -> bool:
        job = aiplatform.AutoMLTextTrainingJob.list(filter=f'display_name={job_display_name}')[0]
        if job.done():
            if job.has_failed:
                raise SteamshipError(message='GCP Vertex AI Training job failed.', internal_message=str(job.error))
            else:
                return True
        else:
            return False

    def _get_model_handle(self, model_display_name: str) -> Optional[aiplatform.Model]:
        model_list = aiplatform.Model.list(filter=f'display_name={model_display_name}')
        if len(model_list) > 0:
            logging.info(f"Found model handle for: {model_display_name}")
            return model_list[0]
        logging.info(f"No model handle found for: {model_display_name}")
        return None


    def train(self, input: TrainPluginInput) -> InvocableResponse[TrainPluginOutput]:
        self._prepare_credentials()
        #Steps:
        #download from presigned url
        # So that requests can fetch the test training file from a file:// url
        requests_session = requests.session()
        requests_session.mount('file://', FileAdapter())
        response = requests_session.get(input.training_data_url)
        training_files = [File.parse_obj(json.loads(line)) for line in response.content.splitlines()]

        multi_class = (self.config.single_or_multi_label == 'multi')
        logging.info("Converting training set to appropriate format")
        # Translate a file to correct format
        num_examples = 0
        data_to_upload = io.StringIO()
        include_tag_names = self.config.include_tag_names.split(',') if self.config.include_tag_names is not None and self.config.include_tag_names != "" else None
        logging.info(f"Include tag names: {include_tag_names}")
        with jsonlines.Writer(data_to_upload) as writer:
            for file in training_files:
                for block in file.blocks:
                    #logging.info(f"Processing block: {block}")
                    tag_names = self._find_training_tags(block, multi_class, include_tag_names)
                    example = self._format_examples(tag_names, block, multi_class)
                    if example is not None:
                        writer.write(example)
                        num_examples += len(tag_names)
        if num_examples < 20 or num_examples > 1000000:
            raise SteamshipError(message="Training set not compatible with VertexAI; must provide between 20 and 1M training samples.")
        logging.info(f"Found {num_examples} training examples")
        training_data_content = data_to_upload.getvalue()

        # Upload it to GCP
        logging.info("Uploading training file to GCP")
        storage_client = storage.Client(credentials=self.scoped_credentials)
        bucket = storage_client.bucket(self.bucket_name)
        training_job_name = f'no-task-{uuid.uuid1()}' if self.training_task_id is None else f'training-task-{self.training_task_id}'
        gcp_training_filename = training_job_name + '.jsonl'
        blob = bucket.blob(gcp_training_filename)

        blob.upload_from_string(training_data_content)

        logging.info(
            f"{gcp_training_filename} with content len {len(training_data_content)} uploaded to {self.bucket_name}."
        )

        # Convert it into a vertex ai dataset
        logging.info(f"Converting training file to Vertex AI dataset (this takes 20-30 minutes) : {training_job_name}")
        dataset_start_time = time.time()
        blob_gs_uri = f'gs://{self.bucket_name}/{gcp_training_filename}'
        import_schema = aiplatform.schema.dataset.ioformat.text.multi_label_classification if multi_class else aiplatform.schema.dataset.ioformat.text.single_label_classification
        try:
            _ = aiplatform.TextDataset.create(
                display_name=training_job_name,
                gcs_source=blob_gs_uri,
                import_schema_uri=import_schema,
                sync=False,
                create_request_timeout=1, #Time out the client side request IMMEDIATELY, then do our own polling for when it finishes.
            )
        except Exception as e:
            logging.info('Error thrown while waiting for dataset to be created. Will continue for now; training job will likely fail.')

        training_results = dict(state=ThirdPartyTrainingStatus.PREPARING_DATASET.value)
        reference_data = dict(dataset_start_time=dataset_start_time, training_job_name=training_job_name, training_started=False)

        # Because the dataset conversion job is starting on a background async process, we need to wait here
        # and give it time to actually get the job started in the cloud. Otherwise this lambda will
        # terminate upon return, and the dataset conversion job will not have started.
        time.sleep(30)

        # The training will take a while, so instead of returning a TrainPluginOutput we return a Task
        # this will signal to the Engine that it must call us back.
        return self._still_training_response(training_results, reference_data)

    def _is_awaiting_dataset_deployment(self, dataset_start_time: int) -> bool:
        """Returns whether the model is still awaiting dataset deployment

        There is no apparent way to determine if this is complete;
        instead we'll just wait a total of 35 min and hope for the best.
        """
        dataset_elapsed_time = time.time() - dataset_start_time
        return dataset_elapsed_time < (60*35)

    def _start_training(self, training_job_name: str, multi_class: bool = False) -> str:
        """Begins training, returning the model_display_name."""

        # Get a new handle to the dataset we just created.
        # This allows us to possibly recover from a dropped connection.
        ds = self._get_dataset_handle(training_job_name)
        if ds is None:
            raise SteamshipError(message="Got null dataset handle; perhaps dataset creation did not end?")

        job = aiplatform.AutoMLTextTrainingJob(
            display_name=training_job_name,
            prediction_type="classification",
            multi_label=multi_class,
        )

        # Run the training job - asynchronously.
        model_display_name = 'model-' + training_job_name

        _ = job.run(
            dataset=ds,
            model_display_name=model_display_name,
            training_fraction_split=0.7,
            validation_fraction_split=0.2,
            test_fraction_split=0.1,
            sync=False,
        )
        # Because the training job launch is starting on a background async process, we need to wait here
        # and give it time to actually get the job started in the cloud. Otherwise this lambda will
        # terminate upon return, and the training job will not have started.
        time.sleep(30)
        return model_display_name

    def _still_training_response(self, training_results, remote_status_input):
        return InvocableResponse(
            status=Task(
                task_id=self.training_task_id,
                state=TaskState.running,
                remote_status_output=training_results,
                remote_status_input=remote_status_input
            )
        )

    def _fetch_model_evaluation(self, model) -> Dict:
        try:
            # Get a reference to the Model Service client
            client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
            model_service_client = aiplatform.gapic.ModelServiceClient(
                client_options=client_options
            )
            model_evaluations = model_service_client.list_model_evaluations(
                parent=model.resource_name
            )
            model_evaluations_list = list(model_evaluations)
            if len(model_evaluations_list) == 0:
                logging.warning("Model evaluation list was empty.")
                return {}
            model_evaluation = model_evaluations_list[0]
            model_evaluation_dict = json_format.MessageToDict(model_evaluation._pb)
            return model_evaluation_dict
        except Exception as e:
            logging.error(f"Exception while fetching model evaluation: {e}")
            return {}


    def train_status(self, input: PluginRequest[TrainPluginInput]) -> InvocableResponse[TrainPluginOutput]:
        self._prepare_credentials()
        # When returning to this method, we're in one of four states:
        # 1 - We're currently waiting on the dataset preprocessing.  No apparent way to determine if this is complete;
        #     instead we'll just wait a total of 35 min and hope for the best.
        # 2 - 35 min have elapsed since dataset creation time - need to create the training job and launch it.
        # 3 - Waiting on training job (~5 total hours).  Poll for job completion, assuming not done, return not done.
        # 4 - Training job complete. Create endpoint, get training eval data, return complete.
        if input.status.remote_status_input is None:
            raise SteamshipError(message="`training_reference_data` field of input was empty. Unable to check tarining status.")

        dataset_start_time = input.status.remote_status_input.get('dataset_start_time')
        if dataset_start_time is None:
            raise SteamshipError(message="`dataset_start_time` field of input.training_reference_data was empty. This indicates an error. Perhaps training hasn't started yet?")

        training_job_name = input.status.remote_status_input.get('training_job_name')

        if training_job_name is None:
            raise SteamshipError(message="`training_job_name` field of input.training_reference_data was empty. This indicates an error. Perhaps training hasn't started yet?")

        multi_class = (self.config.single_or_multi_label == 'multi')

        training_started = input.status.remote_status_input.get('training_started')
        if training_started is None:
            raise SteamshipError(message="`training_started` field of input.training_reference_data was empty. This indicates training has not yet started. Unable to check status.")

        dataset_elapsed_time = time.time() - dataset_start_time
        is_awaiting_dataset_deployment = self._is_awaiting_dataset_deployment(dataset_start_time)

        training_results = dict(
            training_started = training_started,
            training_job_name = training_job_name,
        )

        if not training_started and is_awaiting_dataset_deployment:
            # State 1 - Waiting on dataset. Do nothing.
            message = f'Waiting longer for dataset creation job {training_job_name}. Have waited {dataset_elapsed_time}s ({dataset_elapsed_time / 60} minutes) since dataset creation began.'
            training_results["message"] = message
            training_results["state"] = ThirdPartyTrainingStatus.PREPARING_DATASET.value
            logging.info(message)
            return self._still_training_response(training_results, input.status.remote_status_input)
        elif not training_started and not is_awaiting_dataset_deployment:
            # State 2. Dataset complete; Training not Started; Must start training job.
            message = "Running training (this takes 4-5 hours)"
            logging.info(message)

            model_display_name = self._start_training(
                training_job_name=training_job_name,
                multi_class=multi_class
            )

            training_results["message"] = message
            training_results["model_display_name"] = model_display_name
            training_results["state"] = ThirdPartyTrainingStatus.TRAINING.value
            training_results["training_started"] = True
            return self._still_training_response(training_results, input.status.remote_status_input)
        else:
            # Training started; Must check the status of the training job to decide what to do.
            logging.info(
                f'Checking status of on training job {training_job_name}. Have waited {dataset_elapsed_time}s ({dataset_elapsed_time / 60} minutes) since dataset creation began.')
            if not self._training_job_complete(training_job_name):
                #State 3 - waiting on training job, not complete.  Just return current status.
                message = 'Job not complete - continuing to wait. Have waited {dataset_elapsed_time}s ({dataset_elapsed_time / 60} minutes) since dataset creation began.'
                logging.info(message)
                training_results["message"] = message
                training_results["state"] = ThirdPartyTrainingStatus.TRAINING.value
                return self._still_training_response(training_results, input.status.remote_status_input)
            else:
                #State 4 - training job complete, need to get or create endpoint.
                # HOORAY WE MADE IT
                logging.info('Done waiting on training job. Performing endpoint check.')

                #Now get fresh handle to the model that was created
                model_display_name = 'model-' + training_job_name
                model = self._get_model_handle(model_display_name)

                if model is None:
                    raise SteamshipError(message=f"Tried to fetch GCP model handle for model_display_name={model_display_name} but did not find it.")

                training_results["model_display_name"] = model_display_name

                # GCP names the endpoints predictably: it's docs say:
                # """If not specified, endpoint display name will be model display name+'_endpoint'."""
                endpoint_display_name = f"{model_display_name}_endpoint"
                training_results["endpoint_display_name"] = endpoint_display_name
                endpoint = self._get_endpoint_handle(endpoint_display_name)

                if endpoint is None:
                    # State 4a - Create endpoint
                    message = "Determined endpoint deployment has not yet begun."
                    logging.info(message)
                    training_results["message"] = message
                    training_results["state"] = ThirdPartyTrainingStatus.DEPLOYING_ENDPOINT.value
                    self._deploy_endpoint(training_job_name, model)
                    return self._still_training_response(training_results, input.status.remote_status_input)
                elif self._is_endpoint_deploy_complete(endpoint) == False:
                    # State 4b - Creating endpoint
                    message = f"Determined endpoint point deployment is not yet complete: {endpoint.resource_name}"
                    logging.info(message)
                    training_results["message"] = message
                    training_results["state"] = ThirdPartyTrainingStatus.AWAITING_ENDPOINT.value
                    training_results["endpoint_resource_name"] = endpoint.resource_name
                    return self._still_training_response(training_results, input.status.remote_status_input)
                else:
                    # State 4c - Created endpoint
                    # Woohoo!
                    message = f"Determined endpoint point has is complete: {endpoint.resource_name}"
                    logging.info(message)
                    self.endpoint_path = endpoint.resource_name
                    training_results["endpoint_resource_name"] = endpoint.resource_name
                    training_results["state"] = ThirdPartyTrainingStatus.TRAINED.value
                    training_results["message"] = message
                    # Get evaluation info
                    try:
                        model_evaluation = self._fetch_model_evaluation(model)
                        training_results.update(model_evaluation)
                    except Exception as e:
                        try:
                            exception_message = str(e)
                        except:
                            exception_message = "Could not convert exception to string."
                        logging.error(f'Unable to fetch evaluation data from training model. Proceeding without evaluation info.  Exception: {exception_message}')
                        training_results["message"] += '.  Unable to fetch evaluation data from training model.'

                    return InvocableResponse(
                        status=Task(
                            task_id=self.training_task_id,
                            state=TaskState.succeeded,
                            output=training_results,
                            remote_status_output=training_results,
                            remote_status_input=input.status.remote_status_input
                        ),
                        json=TrainPluginOutput(
                            training_complete=True,
                            training_results=training_results,
                            training_reference_data=input.status.remote_status_input
                        )
                    )

    def run(
        self, request: PluginRequest[BlockAndTagPluginInput]
    ) -> InvocableResponse[BlockAndTagPluginOutput]:
        """Runs the mock client"""
        self._prepare_credentials()
        if self.endpoint_path is None:
            raise SteamshipError(
                message="No endpoint path was found in model parameter file. Has the model been trained?"
            )
        endpoint = aiplatform.Endpoint(self.endpoint_path)
        output = BlockAndTagPluginOutput(file=File.CreateRequest(blocks=[]))

        for in_block in request.data.file.blocks:
            #can only pass one block at a time;
            response = endpoint.predict(instances=[{"content": in_block.text}])
            tags : List[Tag.CreateRequest] = []
            #only get one set of predictions from our one input.
            if len(response.predictions) == 1:
                class_predictions = response.predictions[0]
                display_names = class_predictions['displayNames']
                confidence_scores = class_predictions['confidences']
                for count, id in enumerate(class_predictions["ids"]):
                    tag_name = display_names[count]
                    tag_confidence = confidence_scores[count]
                    tags.append(Tag.CreateRequest(name=tag_name, kind='classification', value=dict(confidence=tag_confidence)))

                out_block = Block.CreateRequest(
                    id=in_block.id, tags=tags
                )
                output.file.blocks.append(out_block)
            else:
                raise SteamshipError(
                    message="Got unexpected number of responses for one input."
                )

        return InvocableResponse(json=output)
